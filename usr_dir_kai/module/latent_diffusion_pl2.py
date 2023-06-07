from multiprocessing.pool import Pool
from tqdm import tqdm
import torch, os
import torch.nn as nn
from utils.hparams import hparams

from modules.tts_modules import DurationPredictorLoss

from usr_dir.utils.tensor_utils import sequence_mask
import torch.nn.functional as F
from usr_dir.codec.codec_decoder import CodecDecoder
import soundfile as sf
from utils.text_encoder import TokenTextEncoder
import json

import pytorch_lightning as pl
from usr_dir.module.diffusion_tts import DiffusionTTS
# from tasks.transformer_tts import RSQRTSchedule
import transformers
import utils
import logging
import numpy as np
from utils.world_utils import restore_pitch, process_f0
logger = logging.getLogger(__name__)

class LatentDiff(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        
        # tasks/transformer_tts.py  init
        self.arch = hparams['arch']
        if isinstance(self.arch, str):
            self.arch = list(map(int, self.arch.strip().split()))
        if self.arch is not None:
            self.num_heads = utils.get_num_heads(self.arch[hparams['enc_layers']:])
        self.vocoder = None
        self.phone_encoder = self.build_phone_encoder(hparams['data_dir'])
        self.padding_idx = self.phone_encoder.pad()
        self.eos_idx = self.phone_encoder.eos()
        self.seg_idx = self.phone_encoder.seg()
        self.saving_result_pool = None
        self.saving_results_futures = None
        self.stats = {}
        
        # usr_dir/tasks/latent_diffusion_separate_ref.py init
        self.dur_loss_fn = DurationPredictorLoss()
        self.mse_loss_fn = torch.nn.MSELoss()
        
        # build model
        arch = self.arch
        from usr_dir.module.fs2_cln import FastSpeech2_dis
        fs2_model = FastSpeech2_dis(arch, self.phone_encoder)
        
        model = DiffusionTTS(fs2_model)
        prior_enc_count = self.get_model_stats(fs2_model.encoder)
        ref_enc_count = self.get_model_stats(model.ref_enc) if model.ref_enc else 0
        dur_predictor_params_count = self.get_model_stats(fs2_model.dur_predictor)
        pitch_predictor_params_count = self.get_model_stats(fs2_model.pitch_predictor) if getattr(fs2_model, "pitch_predictor", None) is not None else 0
        prior_params_count = self.get_model_stats(fs2_model)
        full_params_count = self.get_model_stats(model)
        
        logger.info(f"Prior enc model params count: {prior_enc_count / 1000000} M")
        logger.info(f"Ref enc model params count: {ref_enc_count / 1000000} M")
        logger.info(f"Duration predictor params count: {dur_predictor_params_count / 1000000} M")
        logger.info(f"Pitch predictor params count: {pitch_predictor_params_count / 1000000} M")
        logger.info(f"Prior model params count: {prior_params_count / 1000000} M")
        logger.info(f"Diffusion model params count: {(full_params_count - prior_params_count) / 1000000} M")
        logger.info(f"Full model params count: {full_params_count / 1000000} M")
        print(model)
        # print(fs2_model.encoder)
        
        self.model = model
        
    def build_phone_encoder(self, data_dir):
        phone_list_file = os.path.join(data_dir, 'phone_set.json')
        phone_list = json.load(open(phone_list_file))
        return TokenTextEncoder(None, vocab_list=phone_list) 
        
    def training_step(self, sample, batch_idx, optimizer_idx=0):
        
        input = sample['src_tokens']  # [B, T_t]
        target = sample['targets']  # [B, T_s, 80]
        target_len = sample["target_lengths"]
        mel2ph = sample['mel2ph']  # [B, T_s]
        pitch = sample['pitch']
        #energy = sample['energy']
        energy = None
        uv = sample['uv']
        
        ref_codes = sample.get("ref_codes", None)
        ref_codes_mask = sample.get("ref_codes_mask", None)
      
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        
        loss_output, output, logging_info = self.run_model(self.model, input, mel2ph, spk_embed, target,
                                             pitch=pitch, uv=uv, energy=energy,
                                             target_len=target_len,
                                             ref_codes=ref_codes,
                                             ref_codes_mask=ref_codes_mask,
                                             return_output=True)
        
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = target.size()[0]
        if "mel_acc" in output:
            loss_output["mel_acc"] = output["mel_acc"]
            
        loss_output.update(logging_info)
        
        log_outputs = utils.tensors_to_scalars(loss_output)

        log_outputs['all_loss'] = total_loss.item()
        log_outputs['ref_len'] = ref_codes.shape[1] if ref_codes is not None else 0
  
        progress_bar_log = log_outputs
        tb_log = {f'tr/{k}': v for k, v in log_outputs.items()}
        
        self.logger.log_metrics(progress_bar_log)
        
        return {
            'loss': total_loss,
            'progress_bar': progress_bar_log,
            'log': tb_log
        }
    
    def validation_step(self, sample, batch_idx):
        input = sample['src_tokens']
        target = sample['targets']
        target_len = sample["target_lengths"]
        mel2ph = sample['mel2ph']
        pitch = sample['pitch']
        #energy = sample['energy']
        energy = None
        uv = sample['uv']
        #for k in sample.keys():
        #    if hasattr(sample[k], "shape"):
        #        print(k, sample[k].shape)
        
        ref_codes = sample.get("ref_codes", None)
        ref_codes_mask = sample.get("ref_codes_mask", None)
        
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out, logging_info = self.run_model(self.model, input, mel2ph, spk_embed, target,
                                                      pitch=pitch, uv=uv,
                                                      energy=energy,
                                                      target_len=target_len,
                                                      ref_codes=ref_codes,
                                                      ref_codes_mask=ref_codes_mask,
                                                      return_output=True,)
        outputs['total_loss'] = outputs['losses']['diff-mel']
        outputs['nmels'] = sample['nmels']
        outputs['nsamples'] = sample['nsamples']
        if "mel_acc" in model_out:
            outputs['losses']["mel_acc"] = model_out["mel_acc"]
            
        outputs['losses'].update(logging_info)
        
        outputs = utils.tensors_to_scalars(outputs)
        
        outputs.update(logging_info)
        return outputs
    
        
    def run_model(self, model, input, mel2ph, spk_embed, target, ref_codes, ref_codes_mask,
                  return_output=False, ref_mel='tgt', pitch=None, uv=None, energy=None, target_len=None):
        hparams['global_steps'] = self.global_step
        
        y_mask = sequence_mask(target_len, target.size(-1)).unsqueeze(1).to(input.device)
        
        with torch.no_grad():
            target_latent = self.model.convert_code_to_latent(target)
            ref_latent = self.model.convert_code_to_latent(ref_codes)
            ref_latent_mask = ref_codes_mask.bool()
        
        losses = {}
        ret_logging_info = {}
        # if ref_mel == 'tgt':
        #     ref_mel = target
        ref_mel = target_latent
        
        if "pred_pitch_after" in hparams and hparams["pred_pitch_after"] >= 0:
            output, diff_out = model(input, mel2ph, spk_embed, ref_mel, pitch, uv, energy, use_pred_pitch=self.global_step > hparams["pred_pitch_after"], skip_decoder=bool(hparams["skip_decoder"]), reference_spk_latent=ref_latent, ref_latent_mask=ref_latent_mask)
        else:
            output, diff_out = model(input, mel2ph, spk_embed, ref_mel, pitch, uv, energy, skip_decoder=bool(hparams["skip_decoder"]), reference_spk_latent=ref_latent, ref_latent_mask=ref_latent_mask)

        if hparams["prior_weight"] > 0:
            losses['mel'] = self.l1_loss(output['mel_out'], target_latent, y_mask) * hparams['prior_weight']
        else:
            losses["mel"] = 0

        if hparams.get("duration_predictor_type", "conv") == "conv":
            losses['dur'] = self.dur_loss(output['dur'], mel2ph, input)
        elif hparams.get("duration_predictor_type", "conv") == "ar_transformer":
            dur_loss, dur_acc = self.ce_loss(output["dur"], output["dur-gt"], padding_idx=-1)
          
            losses["dur"] = dur_loss
            ret_logging_info["dur_acc"] = dur_acc
            
            pred_dur = output['dur'].argmax(-1)
            
            pred_dur_logv = torch.log(pred_dur.float() + 1.0)
            
            ori_dur_loss = self.dur_loss(pred_dur_logv, mel2ph, input)
            ret_logging_info["ori_dur_loss"] = ori_dur_loss.item()
            

        if hparams['use_pitch_embed']:
            if hparams["pitch_predictor_type"] == "conv":
                p_pred = output['pitch_logits']            
                losses['uv'], losses['f0'] = self.pitch_loss(p_pred, pitch, uv)
                if losses['uv'] is None:
                    del losses['uv']
            elif hparams["pitch_predictor_type"] == 'conv_phone':
                p_pred = output['pitch_logits']
                p_label = output["phoneme_pitch_target"]
                uv_target = output["phoneme_pitch_uv"]
                
                use_uv = hparams["use_uv"]
                
                
                losses['uv'], losses['f0'] = self.pitch_loss(p_pred, p_label, uv_target.float())
                if losses['uv'] is None:
                    del losses['uv']
                                
        # diffusion loss
        if hparams["diffusion_loss_type"] == "l1":
            losses["diff-mel"] = self.l1_loss(diff_out["diff_x0_pred"], target_latent, y_mask) * hparams["diffusion_mel_weight"]
        elif hparams["diffusion_loss_type"] == "l2":
            losses["diff-mel"] = self.mse_loss(diff_out["diff_x0_pred"], target_latent, y_mask) * hparams["diffusion_mel_weight"]
        else:
            raise ValueError(f"Unknown diffusion loss type: {hparams['diffusion_loss_type']}")
        
        losses["diff-noise"] = self.l1_loss(diff_out["diff_noise_pred"], diff_out["diff_noise_gt"], y_mask)  * hparams["diff_loss_noise_weight"]
        
        if hparams["prompt_loss_weight"] > 0:
            mask = ~ diff_out["ref_latent_mask"].unsqueeze(1)
            # losses["diff-prompt-l1"] = self.l1_loss(diff_out["diff_prompt_pred"], diff_out["diff_prompt_gt"], mask) * hparams["prompt_loss_weight"]
            losses["diff-prompt-noise"] = torch.abs(diff_out["diff_prompt_noise_pred"] * mask.transpose(1, 2)).mean() * hparams["prompt_loss_weight"]
            

        
        
        
        if hparams["vq_dist_weight"] > 0:
            require_dist_loss = True
        else:
            require_dist_loss = False
        
        codebook_bp_loss, logging_info = self.codebook_loss(x=diff_out["diff_x0_pred"].transpose(1, 2), discrete_y=target.transpose(1, 2), latent_y=target_latent.transpose(1, 2), y_mask=y_mask, name="posterior", require_dist_loss=require_dist_loss)
        
        losses.update(codebook_bp_loss)
        ret_logging_info.update(logging_info)
        
        # if hparams["apply_pitch_on_x0"]:
        #     x0_pitch_pred = diff_out['x0_pitch_pred']
            
        #     losses['diff_uv'], losses['diff_f0'] = self.pitch_loss(x0_pitch_pred, pitch, uv)
        #     if losses['diff_uv'] is None:
        #         del losses['diff_uv']
        
        
        if not return_output:
            return losses, ret_logging_info
        else:
            return losses, output, ret_logging_info
        
    def codebook_loss(self, x, discrete_y, latent_y, y_mask, name="", require_dist_loss=False):
        
        bp_loss = {}
        
        loss_summary = {}
        
        # print(x.shape, discrete_y.shape, latent_y.shape, y_mask.shape)
        # exit(0)
        
        
        quantized_out, all_indices, commit_loss, discrete_pred_loss, discrete_dist_loss = self.model.quantizer.calculate_loss(x=x, discrete_y=discrete_y, latent_y=latent_y, y_mask=y_mask, require_dist_loss=require_dist_loss)
        
        
        # diff_loss += discrete_pred_loss.mean() * self.cust_opt.vq_pred_weight
        
        bp_loss["x0-quantized-cls"] = discrete_pred_loss.mean() * hparams["vq_quantizer_weight"]
        
        # loss_summary["{}_x0-quantinizer-loss".format(name)] = discrete_pred_loss.mean().item()
        
        if require_dist_loss:
            bp_loss["x0-dist"] = discrete_dist_loss.mean() * hparams["vq_dist_weight"] # share the same weight 
        # loss_summary["{}_x0-dist-loss".format(name)] = discrete_dist_loss.mean().item()
        # all_indices = all_indices.transpose(0, 1)
        
        # print(all_indices.shape, discrete_y.shape, "-------", y_mask.shape)
        # exit(0)
  
        discrete_y = discrete_y.transpose(0, 1)
        
        
        correct = (all_indices == discrete_y).float()
        
        correct_mask = y_mask.transpose(0, 1).expand(all_indices.shape[0], -1, -1)
        
        for layer_idx in range(correct.shape[0]):
            
            correct_layer = correct[layer_idx].view(-1)
            correct_mask_layer = correct_mask[layer_idx].view(-1)
            
            correct_scores = correct_layer[correct_mask_layer.bool()]
            loss_summary["{}_acc_{}".format(name, layer_idx)] = correct_scores.mean().item()
        
        return bp_loss, loss_summary
        
    def l1_loss(self, decoder_output, target, mask):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
               
        l1_loss = F.l1_loss(decoder_output, target, reduction='none')
        weights = self.weights_nonzero_speech(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        return l1_loss

    def mse_loss(self, decoder_output, target, target_len):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        mse_loss = F.mse_loss(decoder_output, target, reduction='none')
        weights_pr = target_len.sum() * target.size(-1)
        mse_loss = mse_loss.sum() / weights_pr.sum()
        return mse_loss

    def ce_loss(self, decoder_output, target, padding_idx=None):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        #assert target.shape[-1] * target.shape[-2] == decoder_output.shape[1]
        #assert (target >= 0 ).all()
        #assert (target < 1025).all()
        #assert decoder_output.shape[-1] == 1025
        
        if padding_idx is None:
            padding_idx = self.padding_idx
        
        ce_loss = F.cross_entropy(decoder_output.reshape(-1, decoder_output.shape[-1]), target.reshape(-1), reduction='none', ignore_index=padding_idx)

        weights = (target != padding_idx).long().reshape(-1)

        is_acc = (decoder_output.max(-1)[1].reshape(-1) == target.reshape(-1)).float()
        acc = (is_acc * weights).sum() / weights.sum() * 100

        ce_loss = (ce_loss * weights).sum() / weights.sum()
        return ce_loss, acc 

    def dur_loss(self, dur_pred, mel2ph, input, split_pause=False, sent_dur_loss=False):
        B, T_t = input.shape
        dur_gt = mel2ph.new_zeros(B, T_t + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
        dur_gt = dur_gt[:, 1:]
        nonpadding = (input != 0).to(dur_pred.dtype)
        if split_pause:
            is_pause = (input == self.phone_encoder.seg()) | (input == self.phone_encoder.unk()) | (
                    input == self.phone_encoder.eos())
            is_pause = is_pause.to(dur_pred.dtype)
            phone_loss = self.dur_loss_fn(dur_pred, dur_gt, (1 - is_pause) * nonpadding) \
                         * hparams['lambda_dur']
            seg_loss = self.dur_loss_fn(dur_pred, dur_gt, is_pause) \
                       * hparams['lambda_dur']
            return phone_loss, seg_loss
        
        ph_dur_loss = self.dur_loss_fn(dur_pred, dur_gt, nonpadding) * hparams['lambda_dur']
        if not sent_dur_loss:
            return ph_dur_loss
        else:
            dur_pred = (dur_pred.exp() - 1).clamp(min=0) * nonpadding
            dur_gt = dur_gt.to(dur_pred.dtype) * nonpadding
            sent_dur_loss = F.l1_loss(dur_pred.sum(-1), dur_gt.sum(-1), reduction='none') / dur_gt.sum(-1)
            sent_dur_loss = sent_dur_loss.mean()
            return ph_dur_loss, sent_dur_loss

    def pitch_loss(self, p_pred, pitch, uv):
        assert p_pred[..., 0].shape == pitch.shape
        assert p_pred[..., 0].shape == uv.shape
        nonpadding = (pitch != -200).to(p_pred.dtype).reshape(-1)
        if hparams['use_uv']:
            uv_loss = (F.binary_cross_entropy_with_logits(
                p_pred[:, :, 1].reshape(-1), uv.reshape(-1), reduction='none') * nonpadding).sum() \
                      / nonpadding.sum() * hparams['lambda_uv']
            nonpadding = (pitch != -200).to(p_pred.dtype) * (uv == 0).to(p_pred.dtype)
            nonpadding = nonpadding.reshape(-1)
        else:
            pitch[uv > 0] = -4
            uv_loss = None

        pitch_loss_fn = F.l1_loss if hparams['pitch_loss'] == 'l1' else F.mse_loss
        pitch_loss = (pitch_loss_fn(
            p_pred[:, :, 0].reshape(-1), pitch.reshape(-1), reduction='none') * nonpadding).sum() \
                     / nonpadding.sum() * hparams['lambda_pitch']
        return uv_loss, pitch_loss

    def energy_loss(self, energy_pred, energy):
        nonpadding = (energy != 0).to(energy_pred.dtype)
        loss = (F.mse_loss(energy_pred, energy, reduction='none') * nonpadding).sum() / nonpadding.sum()
        loss = loss * hparams['lambda_energy']
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=hparams["lr"],
                                      betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
                                      weight_decay=hparams['weight_decay'])
        # num_training_steps, num_warmup_steps = self.compute_warmup(
        #     num_training_steps=100000000,
        #     num_warmup_steps=0.1,
        # )
        # scheduler = RSQRTSchedule(optimizer)
        scheduler = transformers.get_inverse_sqrt_schedule(optimizer=optimizer, num_warmup_steps=hparams["warmup_updates"])
        
        # scheduler = transformers.get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=hparams["warmup_updates"] * 20, num_training_steps=9240250
        # )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }    
            
    @torch.no_grad()
    def convert_code_to_latent(self, codes):
        latent = self.model.quantizer.vq2emb(codes.long())
        return latent
        
    def test_step(self, sample, batch_idx):
        if sample['id'][0]!= 6:
            print(sample['id'][0])
            return
        print(self.model)
        if hparams["in_context_infer"]:
            return self.in_context_inference(sample, batch_idx)
        else:
            return self.conventional_inference(sample, batch_idx)
    
    @torch.inference_mode()
    def conventional_inference(self, sample, batch_idx):
        
        self.model.quantizer.eval()
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        input = sample['src_tokens']
        utt_id = sample['utt_id']
        device = input.device
        if hparams['profile_infer']:
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            mel2ph = sample['mel2ph']
            pitch = sample['pitch']
            uv = sample['uv']
        else:
            mel2ph = None
            pitch = None
            uv = None
            
        # mel2ph = sample['mel2ph']
 
        if hparams["infer_laoban"]:
            import librosa
            from usr_dir.codec.codec_encoder import CodecEncoder
            
            # input_dir = '/blob/v-zeqianju/dataset/tts/Valle_demo'
            # ref_dir = '/blob/v-zeqianju/dataset/tts/Valle_demo'
            # input_dir = '/blob/v-zeqianju/dataset/tts/Product_ZeroShot_V2_infer_text'
            # ref_dir = '/blob/v-zeqianju/dataset/tts/Product_ZeroShot_V2'
            version = int(hparams['infer_laoban'])
            print('version', version)
            root = '/blob/v-zeqianju/dataset/tts/'
            inputs = ['', # v0: pad
                      '', # v1: todo
                    #   'v2/text_punc18',  # v2
                    #   'v3/text_v3', # v3
                      'Product_ZeroShot_V2_infer_text/text_punc18',
                      'Valle_demo/valle_infer_text',
                      'v4/text_v4', # v4
                      'v2/text_punc18', # v5 v2-unnorm
                      'v3/text_v3',   # v6   v3-unnorm
                      'v3/text_v3',   # v7  v3-gt
                      'v3/text_v3',     #v8 v3-gt-full
                      'train/text'        # v9 train-3s
                      ]
            refs = ['', # v0: pad
                    '', # v1: todo
                    # 'v2', # v2
                    # 'v3', # v3
                    'Product_ZeroShot_V2',
                    'Valle_demo',
                    'v4', # v4
                    'v2', # v5 v2-unnorm
                    'v3', # v6   v3-unnorm
                    'v3_gt', # v7  v3-gt
                    'v3_gt_full', #v8  v3-gt-full
                    'train'
                    ]
            input_dir = f'{root}/{inputs[version]}'
            ref_dir = f'{root}/{refs[version]}'
            
            old_sample = sample
            dur_mean = []
            dur_std = []
            pitch_mean = []
            pitch_std = []
            for idx, ref_name in enumerate(os.listdir(ref_dir)):
                if os.path.isdir(f'{ref_dir}/{ref_name}'):
                    continue
                
                
                # if idx >=50:
                #     print('dur_mean_delta', sum(dur_mean)/len(dur_mean))
                #     print('dur_std_delta', sum(dur_std)/len(dur_std))
                #     print('pitch_mean_delta', sum(pitch_mean)/len(pitch_mean))
                #     print('pitch_std_delta', sum(pitch_std)/len(pitch_std))
                #     exit()
                import pickle
                from utils.text_encoder import NUM_RESERVED_TOKENS, UNK_ID, EOS_ID
                ref_id = ref_name.replace('.wav','').replace('.flac','')
                try:
                    with open(f'{input_dir}/{ref_id}.phone_id', 'rb') as f:
                    # with open(f'{input_dir}/{ref_id}.phone_id', 'rb') as f:
                        phone_id = torch.from_numpy(pickle.load(f))
                except:
                    print(f'load error {ref_id}: {input_dir}/{ref_id}.phone_id, skip it')
                    continue
                
                phone_encoded = []
                phone_encoded.append(UNK_ID)
                for i in phone_id:
                    phone_encoded.append(NUM_RESERVED_TOKENS + i)
                phone_encoded.append(EOS_ID)
                
                input = torch.LongTensor(phone_encoded).to(device)
                if hparams["remove_bos"]:
                    input = input[1:] 
                print(input-NUM_RESERVED_TOKENS)
                # exit()
                input = input[None, ...]
                # ref_name = f'{ref_id}.wav'
                ref_fn = os.path.join(ref_dir, ref_name)
                wav_data, sr = librosa.load(ref_fn, sr=16000)
                wav_data = np.pad(wav_data, (0, 600-len(wav_data) % 600))
                
                codec_enc = CodecEncoder()
                codec_dec = CodecDecoder()
                ckpt_path = '/blob/v-shenkai/checkpoints/tts/codec/chanpin_5w/v5/lambda_disc_1_commit_weight_0.25/checkpoint-440000.pt'
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                codec_enc.load_state_dict(checkpoint["model"]['CodecEnc'])
                codec_enc = codec_enc.eval().to(device)
                codec_dec.load_state_dict(checkpoint["model"]['generator'])
                codec_dec = codec_dec.eval().to(device)
                wav_tensor = torch.from_numpy(wav_data).to(device)  
                vq_emb = codec_enc(wav_tensor[None, None, :])
                ref_mel, _, _ = codec_dec(vq_emb, None, 1000000, vq=True)
                # print(ref_mel.shape) torch.Size([1, 256, 3192])
                ref_mel = ref_mel.transpose(1, 2)
                # ref_mel = ref_mel[:, :1000, :]
                # self.preprare_vocoder(hparams)
                # ref_audio, _ = self.vocoder.inference(ref_mel.transpose(1, 2))
                
                # utt_id = ref_id
                # print(ref_audio.shape)
                # ref_audio = ref_audio.cpu().numpy().astype(np.float32)
                
                # gen_dir = os.path.join(hparams['work_dir'],
                #                     f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}_v{hparams["infer_laoban"]}_tp{hparams["temp"]}'+ ('stoc' if hparams['stoc'] else '')+ ('_refnorm' if hparams['ref_norm'] else '')+('_inc' if hparams['in_context_infer'] else '' ))
                # os.makedirs(gen_dir, exist_ok=True)

                # print(gen_dir)
                # # sf.write(f'{gen_dir}/prior_{os.path.basename(utt_id)}.wav', prior_audio[0], 16000, subtype='PCM_24')
                
                # sf.write(f'{gen_dir}/ref_{os.path.basename(utt_id)}.wav', ref_audio[0,0], 16000, subtype='PCM_24')
                # exit()
                # utt_id = [ref_id]
                # try:
                #     with open(f'{ref_dir}/data/{ref_id}.mel2ph', 'rb') as f:
                #         ref_mel2ph = torch.from_numpy(pickle.load(f)).to(device)
                #     with open(f'{ref_dir}/data/{ref_id}.phone_id', 'rb') as f:
                #         ref_phone_id = torch.from_numpy(pickle.load(f))
                #     with open(f'{ref_dir}/data/{ref_id}.f0', 'rb') as f:
                #         ref_f0 = pickle.load(f)
                # except:
                #     continue
                # # ref_mel2ph = torch.LongTensor(ref_mel2ph).to(device)
                # max_phone = max(ref_mel2ph.max().item(), hparams['max_input_tokens'])
                # phone_encoded = []
                # phone_encoded.append(UNK_ID)
                # for i in ref_phone_id:
                #     phone_encoded.append(NUM_RESERVED_TOKENS + i)
                # phone_encoded.append(EOS_ID)
                # ref_input = torch.LongTensor(phone_encoded[:max_phone]).to(device)
                # if hparams["remove_bos"]:
                #     ref_input = ref_input[1:]     
                # ref_input = ref_input[None, ...]
                # ref_mel2ph = ref_mel2ph[None, ...]   
                
                # ref_pitch, ref_uv = process_f0(ref_f0, hparams)
                # ref_pitch = ref_pitch[None, ...].to(device)
                # ref_uv = ref_uv[None, ...].to(device)
                # B = 1
                # ref_tokens = ref_input.shape[1]
                # dur_gt = ref_mel2ph.new_zeros(B, ref_tokens-1 + 1).scatter_add(1, ref_mel2ph, torch.ones_like(ref_mel2ph))
                # dur_gt = dur_gt[:, 1:]
                # # mask_gt = (dur_gt > 0) 
                # # dur_gt[dur_gt > 20] = 20
                
                # d = dur_gt[dur_gt > 0].float()
                # p = restore_pitch(ref_pitch, ref_uv, hparams)
                # # p = p[p>1]
                
                # print('ref_dur_mean', d.mean().item())
                # print('ref_dur_std', d.std().item())
                # print('ref_pitch_mean', p.mean().item())
                # print('ref_pitch_std', p.std().item())
                
                if hparams['n_samples'] == 0:
                    utt_id = [ref_id]
                    
                    with utils.Timer('fs', print_time=hparams['profile_infer']):
                        prior_outputs, spk_embed = self.model.infer_fs2(input, mel2ph=mel2ph, spk_embed=spk_embed, ref_mels=ref_mel, pitch=pitch, uv=uv, skip_decoder=hparams['skip_decoder'])
                    
                    # p_d = prior_outputs['dur'][prior_outputs['dur'] > 0].float()
                    # p_p = prior_outputs['pitch']
                    # # p_p = p_p[p_p > 1]
                    # print('pred_dur_mean', p_d.mean().item())
                    # print('pred_dur_std', p_d.std().item())
                    # print('pred_pitch_mean', p_p.mean().item())
                    # print('pred_pitch_std', p_p.std().item())
                    # dur_mean.append(abs(p_d.mean().item()-d.mean().item()))
                    # dur_std.append(abs(p_d.std().item()-d.std().item()))
                    # pitch_mean.append(abs(p_p.mean().item()-p.mean().item()))
                    # pitch_std.append(abs(p_p.std().item()-p.std().item()))
                    # if idx >=0:
                    #     continue
                    # 
                    
                    prompt = spk_embed if hparams['use_spk_prompt'] else ref_mel 
                    promt_mask = torch.zeros(prompt.shape[0], prompt.shape[1]).bool().to(prompt.device)  
                    with utils.Timer('diffusion', print_time=hparams['profile_infer']):
                        mel_denoised = self.model.infer_diffusion(mu=prior_outputs["mel_out"], n_timesteps=150, spk=spk_embed, temperature=hparams['temp'], prompt=prompt, prompt_mask=promt_mask)
                        
                    self.preprare_vocoder(hparams)
                    # # prior
                    # prior_quant, _, _ = self.vocoder(prior_outputs['mel_out'].transpose(1, 2), None, 100000, vq=True, early_stop=-1)
                    # prior_audio, _ = self.vocoder.inference(prior_quant)
                    # sample['prior_audio'] = prior_audio
                
                    # diffusion
                    denoised_quant, _, _ = self.vocoder(mel_denoised.transpose(1, 2), None, 100000, vq=True, early_stop=-1)
                    
                    
                    denoised_audio, _ = self.vocoder.inference(denoised_quant)
                    sample['denoised_audio'] = denoised_audio
                    
                    
                    # reference
                    ref_audio, _ = self.vocoder.inference(ref_mel.transpose(1, 2))
                    sample["ref_audio"] = ref_audio
                    
                    
                    sample['outputs'] = prior_outputs['mel_out']
                    # sample['pitch_pred'] = prior_outputs.get('pitch')
                    # sample['pitch'] = restore_pitch(sample['pitch'], uv if hparams['use_uv'] else None, hparams)
                    sample['utt_id'] = utt_id
                    self.after_infer(sample)
                else:
                    self.preprare_vocoder(hparams)
                    for sample_id in tqdm(range(hparams['n_samples'])):
                        utt_id = [f'sp{sample_id}_{ref_id}']
                    
                        with utils.Timer('fs', print_time=hparams['profile_infer']):
                            prior_outputs, spk_embed = self.model.infer_fs2(input, mel2ph=mel2ph, spk_embed=spk_embed, ref_mels=ref_mel, pitch=pitch, uv=uv, skip_decoder=hparams['skip_decoder'])
                        
                        
                        prompt = spk_embed if hparams['use_spk_prompt'] else ref_mel 
                        promt_mask = torch.zeros(prompt.shape[0], prompt.shape[1]).bool().to(prompt.device)  
                        with utils.Timer('diffusion', print_time=hparams['profile_infer']):
                            mel_denoised = self.model.infer_diffusion(mu=prior_outputs["mel_out"], n_timesteps=150, spk=spk_embed, temperature=hparams['temp'], prompt=prompt, prompt_mask=promt_mask)
                    
                        # diffusion
                        denoised_quant, _, _ = self.vocoder(mel_denoised.transpose(1, 2), None, 100000, vq=True, early_stop=-1)
                        
                        
                        denoised_audio, _ = self.vocoder.inference(denoised_quant)
                        sample['denoised_audio'] = denoised_audio
                        
                        
                        # reference
                        ref_audio, _ = self.vocoder.inference(ref_mel.transpose(1, 2))
                        sample["ref_audio"] = ref_audio
                        
                        
                        sample['outputs'] = prior_outputs['mel_out']
                        # sample['pitch_pred'] = prior_outputs.get('pitch')
                        # sample['pitch'] = restore_pitch(sample['pitch'], uv if hparams['use_uv'] else None, hparams)
                        sample['utt_id'] = utt_id
                        self.after_infer(sample)
                        
            exit()
            
            ref_name = 'sydney.wav'
            ref_fn = os.path.join(ref_dir, ref_name)
            wav_data, sr = librosa.load(ref_fn, sr=16000)
            wav_data = np.pad(wav_data, (0, 200-len(wav_data) % 200))
            
            codec_enc = CodecEncoder()
            codec_dec = CodecDecoder()
            ckpt_path = '/blob/v-shenkai/checkpoints/tts/codec/chanpin_5w/v5/lambda_disc_1_commit_weight_0.25/checkpoint-440000.pt'
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            codec_enc.load_state_dict(checkpoint["model"]['CodecEnc'])
            codec_enc = codec_enc.eval().to(device)
            codec_dec.load_state_dict(checkpoint["model"]['generator'])
            codec_dec = codec_dec.eval().to(device)
            wav_tensor = torch.from_numpy(wav_data).to(device)  
            vq_emb = codec_enc(wav_tensor[None, None, :])
            ref_mel, _, _ = codec_dec(vq_emb, None, 1000000, vq=True)
            # print(ref_mel.shape) torch.Size([1, 256, 3192])
            ref_mel = ref_mel.transpose(1, 2)
            # ref_mel = ref_mel[:, :1000, :]
           
            print(ref_mel.shape)
            
            utt_id = [ref_name + "_" + utt_id[0]]
            
        else:
            refs = sample['refs'][0]
            print(refs)
            
            if refs and refs[0]['utt_id'] != utt_id[0]:
                ref = refs[0]
                ref_utt_id = ref['utt_id']
                used = min(hparams['max_frames'], 1200)
                utt_id = [f'{used}_' + str(ref['spk_id'])]
                
                print('ref spk id:', ref['spk_id'], 'utt_id:', ref['utt_id'])
                code_dir = '/blob/v-zeqianju/dataset/tts/product_5w/full_code'
                ref_code = torch.load(f'{code_dir}/{ref_utt_id}.code', map_location=device)
                ref_code = ref_code.permute(1, 2, 0)  # C B T  -> B T C
                ref_mel = self.model.convert_code_to_latent(ref_code)
                ref_mel = ref_mel[:, :used, :]
                # ref_mel = ref_mel[:, :1000, :]
                # print(ref_code.shape, ref_mel.shape)
                # exit()
            else:
                # assert False
                hparams['append_sep'] = False
                # if hparams['test_set_name'] == 'valid':
                    # from usr_dir.datasets import latent_dataset
                    # test_dataset = latent_dataset.FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                    #                                 hparams['test_set_name'], hparams, shuffle=False)
                # else:
                #     from usr_dir.datasets import latent_dataset_separate_ref
                #     test_dataset = latent_dataset_separate_ref.FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                #                                     hparams['test_set_name'], hparams, shuffle=False)
                from usr_dir.datasets import latent_dataset
                test_dataset = latent_dataset.FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                                hparams['test_set_name'], hparams, shuffle=False)
                print(len(test_dataset))
                import pickle
                import tarfile
                num = 50
                for i in range(num):
                    sample = test_dataset.collater([test_dataset[i]])
                    input = sample['src_tokens'].to(device)
                    if hparams['test_set_name'] == 'valid' or hparams['test_set_name'] == 'train':
                        prefix = 'seen' if hparams['test_set_name'] == 'valid' else 'train'
                        utt_id = [f'{prefix}_{i}']
                        ref_sample = test_dataset.collater([test_dataset[i+num]])
                        ref_mel = self.model.convert_code_to_latent(ref_sample['targets'].to(device))
                        try:
                            ref_input = sample['src_tokens'].to(device)
                            # ref_mel2ph = ref_sample['mel2ph'].to(device)
                            # ref_pitch = ref_sample['pitch'].to(device)
                            # ref_uv = ref_sample['uv'].to(device)
                            print(ref_mel.shape)
                            ref = ref_sample['refs'][0][0]
                            print(ref)
                            # exit()
                            tar = ref['tar']
                            tar_obj = tarfile.open(tar, mode='r')
                            fn = ref['fn']
                            keys = ['duration', 'phone_id', 'speech', 'mel']
                            file_info = dict()
                            for info in tar_obj:
                                try:
                                    if not info.isfile():
                                        continue
                                    for key in keys:
                                        # if info.name.endswith('.speech'):
                                        #     print(info.name)
                                        if info.name == f'{fn}.{key}':
                                            if key == 'speaker_id':
                                                value = int(info.name.split('.')[1].split('_')[0])
                                            else:
                                                cur_f = tar_obj.extractfile(info)
                                                value = pickle.load(cur_f)
                                            file_info[key] = value
                                            break
                                except:
                                    assert False
                                    continue
                                # print(file_info)
                                
                                gen_dir = os.path.join(hparams['work_dir'],
                                            f'generated_{0}_{hparams["gen_dir_name"]}_v{hparams["infer_laoban"]}_tp{hparams["temp"]}'+ ('stoc' if hparams['stoc'] else '')+ ('_refnorm' if hparams['ref_norm'] else '')+('_inc' if hparams['in_context_infer'] else '' ))
                                os.makedirs(gen_dir, exist_ok=True)

                                print(gen_dir)
                                # sf.write(f'{gen_dir}/prior_{os.path.basename(utt_id)}.wav', prior_audio[0], 16000, subtype='PCM_24')
                                
                                sf.write(f'{gen_dir}/gt_ref_{os.path.basename(utt_id[0])}.wav', file_info['speech'], 16000, subtype='PCM_24')
                        except:
                            pass
                    else: 
                        prefix = 'train'
                        utt_id = [f'{prefix}_{i}']
                        ref_mel = self.model.convert_code_to_latent(sample["ref_codes"].to(device))
                    
                    # exit()
                    
                    with utils.Timer('fs', print_time=hparams['profile_infer']):
                        prior_outputs, spk_embed = self.model.infer_fs2(input, mel2ph=mel2ph, spk_embed=spk_embed, ref_mels=ref_mel, pitch=pitch, uv=uv, skip_decoder=hparams['skip_decoder'])
                    
                    # if idx >=20:
                    #     continue
                    # prior_outputs['mel_out'] = prior_outputs['mel_out'].transpose(1, 2)
                    
                    prompt = spk_embed if hparams['use_spk_prompt'] else ref_mel 
                    promt_mask = torch.zeros(prompt.shape[0], prompt.shape[1]).to(prompt.device)  
                    with utils.Timer('diffusion', print_time=hparams['profile_infer']):
                        mel_denoised = self.model.infer_diffusion(mu=prior_outputs["mel_out"], n_timesteps=150, spk=spk_embed, temperature=hparams['temp'], prompt=prompt, prompt_mask=promt_mask)
                    
                    self.preprare_vocoder(hparams)
                    
                    # # prior
                    # prior_quant, _, _ = self.vocoder(prior_outputs['mel_out'].transpose(1, 2), None, 100000, vq=True, early_stop=-1)
                    # prior_audio, _ = self.vocoder.inference(prior_quant)
                    # sample['prior_audio'] = prior_audio
                
                    # diffusion
                    denoised_quant, _, _ = self.vocoder(mel_denoised.transpose(1, 2), None, 100000, vq=True, early_stop=-1)
                    
                    
                    denoised_audio, _ = self.vocoder.inference(denoised_quant)
                    sample['denoised_audio'] = denoised_audio
                    
                    
                    # reference
                    ref_audio, _ = self.vocoder.inference(ref_mel.transpose(1, 2))
                    sample["ref_audio"] = ref_audio
                    
                    
                    sample['outputs'] = prior_outputs['mel_out']
                    # sample['pitch_pred'] = prior_outputs.get('pitch')
                    # sample['pitch'] = restore_pitch(sample['pitch'], uv if hparams['use_uv'] else None, hparams)
                    sample['utt_id'] = utt_id
                    self.after_infer(sample)
                    # for i in range(100):
                    #     ref_sample = test_dataset.collater([test_dataset[i]])
                    #     ref_mel = self.model.convert_code_to_latent(ref_sample['targets'].to(device))
                    #     spk_embed = self.model.ref_enc(ref_mel.contiguous())
                    #     torch.save(spk_embed, f'spk_emb/zero-shot/{i}.pt')
                exit()

        # print(prompt.shape, promt_mask.shape)
        # print(sample["ref_codes"].shape, sample["ref_codes_mask"].shape)
        # exit()
        with utils.Timer('fs', print_time=hparams['profile_infer']):
            prior_outputs, spk_embed = self.model.infer_fs2(input, mel2ph=mel2ph, spk_embed=spk_embed, ref_mels=ref_mel, pitch=pitch, uv=uv, skip_decoder=hparams['skip_decoder'])

        prompt = ref_mel
        promt_mask = torch.zeros(prompt.shape[0], prompt.shape[1]).to(prompt.device)       
            
        
        with utils.Timer('diffusion', print_time=hparams['profile_infer']):
            mel_denoised = self.model.infer_diffusion(mu=prior_outputs["mel_out"], n_timesteps=150, spk=spk_embed, temperature=hparams['temp'], prompt=prompt, prompt_mask=promt_mask)
            
        self.preprare_vocoder(hparams)
        # # prior
        # prior_quant, _, _ = self.vocoder(prior_outputs['mel_out'].transpose(1, 2), None, 100000, vq=True, early_stop=-1)
        # prior_audio, _ = self.vocoder.inference(prior_quant)
        # sample['prior_audio'] = prior_audio
    
        # diffusion
        denoised_quant, _, _ = self.vocoder(mel_denoised.transpose(1, 2), None, 100000, vq=True, early_stop=-1)
        
        
        denoised_audio, _ = self.vocoder.inference(denoised_quant)
        sample['denoised_audio'] = denoised_audio
        
        
        # reference
        ref_audio, _ = self.vocoder.inference(ref_mel.transpose(1, 2))
        sample["ref_audio"] = ref_audio
        
        
        sample['outputs'] = prior_outputs['mel_out']
        # sample['pitch_pred'] = prior_outputs.get('pitch')
        # sample['pitch'] = restore_pitch(sample['pitch'], uv if hparams['use_uv'] else None, hparams)
        sample['utt_id'] = utt_id
        return self.after_infer(sample)

    def in_context_inference(self, sample, batch_idx):
        self.model.quantizer.eval()
        utt_id = sample['utt_id']
        
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        input = sample['src_tokens']
        device = input.device
        # print(input)
        if hparams['profile_infer']:
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            mel2ph = sample['mel2ph']
            pitch = sample['pitch']
            uv = sample['uv']
        else:
            mel2ph = None
            pitch = None
            uv = None
            
        # in-context learning
        ref_pitch = None
        ref_uv = None
        
        # hparams["infer_laoban"] = False # lack MFA, input
        if hparams["infer_laoban"]:
            import librosa
            from usr_dir.codec.codec_encoder import CodecEncoder
            
            input_dir = '/blob/v-zeqianju/dataset/tts/Product_ZeroShot_V2_infer_text'
            ref_dir = '/blob/v-zeqianju/dataset/tts/Product_ZeroShot_V2'
            old_sample = sample
            for idx, ref_name in enumerate(os.listdir(ref_dir)):
                if os.path.isdir(f'{ref_dir}/{ref_name}'):
                    continue
                if idx >=10:
                    exit()
                import pickle
                from utils.text_encoder import NUM_RESERVED_TOKENS, UNK_ID, EOS_ID
                ref_id = ref_name.replace('.wav','')
                try:
                    with open(f'{input_dir}/text_raw/{ref_id}.phone_id', 'rb') as f:
                        phone_id = torch.from_numpy(pickle.load(f))
                except:
                    print(f'load error {ref_id}, skip it')
                    continue
                # if ref_id=='00001': # insert silence mannually
                #     # br3 punc. sil br0 4150 132 4153 4147
                #     phone_id = [4152, 4147, 1554, 2073,  829,  633, 1901, 2279,  671, 4148,  813,  649,
                #                 1554, 4148, 2923,  671,  649,  822, 4148, 1554,  649,  633, 3323, 2020,
                #                 4148, 3945, 2218, 4148, 3056,  686,  633,  813, 3787, 2522, 4148,  652,
                #                 1554, 4148, 1810,  486, 4150,  4153, 4147, 3400, 3333, 3715, 4148, 1554,  271,  813,
                #                 4148, 1929, 2073, 2218,  633, 1901, 2279, 1554, 3715, 4150,  132, 4148,
                #                 4153, 4147,    4]
                #     phone_id = [i-3 for i in phone_id]
                #     print(phone_id)
                
                phone_encoded = []
                phone_encoded.append(UNK_ID)
                for i in phone_id:
                    phone_encoded.append(NUM_RESERVED_TOKENS + i)
                phone_encoded.append(EOS_ID)
                
                input = torch.LongTensor(phone_encoded).to(device)
                if hparams["remove_bos"]:
                    input = input[1:] 
                print(input-NUM_RESERVED_TOKENS)
                # exit()
                input = input[None, ...]
                ref_name = f'{ref_id}.wav'
                ref_fn = os.path.join(ref_dir, ref_name)
                wav_data, sr = librosa.load(ref_fn, sr=16000)
                wav_data = np.pad(wav_data, (0, 200-len(wav_data) % 200))
                
                codec_enc = CodecEncoder()
                codec_dec = CodecDecoder()
                ckpt_path = '/blob/v-shenkai/checkpoints/tts/codec/chanpin_5w/v5/lambda_disc_1_commit_weight_0.25/checkpoint-440000.pt'
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                codec_enc.load_state_dict(checkpoint["model"]['CodecEnc'])
                codec_enc = codec_enc.eval().to(device)
                codec_dec.load_state_dict(checkpoint["model"]['generator'])
                codec_dec = codec_dec.eval().to(device)
                wav_tensor = torch.from_numpy(wav_data).to(device)  
                vq_emb = codec_enc(wav_tensor[None, None, :])
                ref_mel, _, _ = codec_dec(vq_emb, None, 1000000, vq=True)
                # print(ref_mel.shape) torch.Size([1, 256, 3192])
                ref_mel = ref_mel.transpose(1, 2)
                # ref_mel = ref_mel[:, :1000, :]
                
                utt_id = ['inc_'+ ref_id]

                try:
                    with open(f'{ref_dir}/data/{ref_id}.mel2ph', 'rb') as f:
                        ref_mel2ph = torch.from_numpy(pickle.load(f)).to(device)
                    with open(f'{ref_dir}/data/{ref_id}.phone_id', 'rb') as f:
                        ref_phone_id = torch.from_numpy(pickle.load(f))
                    with open(f'{ref_dir}/data/{ref_id}.f0', 'rb') as f:
                        ref_f0 = pickle.load(f)
                except:
                    continue
                # ref_mel2ph = torch.LongTensor(ref_mel2ph[:used]).to(device)
                max_phone = max(ref_mel2ph.max().item(), hparams['max_input_tokens'])
                phone_encoded = []
                phone_encoded.append(UNK_ID)
                for i in ref_phone_id:
                    phone_encoded.append(NUM_RESERVED_TOKENS + i)
                phone_encoded.append(EOS_ID)
                ref_input = torch.LongTensor(phone_encoded[:max_phone]).to(device)
                if hparams["remove_bos"]:
                    ref_input = ref_input[1:]     
                ref_input = ref_input[None, ...]
                ref_mel2ph = ref_mel2ph[None, ...]   
                ref_pitch, ref_uv = process_f0(ref_f0, hparams)
                ref_pitch = ref_pitch[None, ...].to(device)
                ref_uv = ref_uv[None, ...].to(device)
                with utils.Timer('fs', print_time=hparams['profile_infer']):
                    prior_outputs, spk_embed = self.model.infer_fs2_hack(src_tokens_ref=ref_input, src_tokens_in=input, mel2ph=ref_mel2ph,
                                                                        spk_embed=None, ref_mels=ref_mel, pitch=ref_pitch, uv=ref_uv,
                                                                        use_pred_pitch=True)
                
                print(prior_outputs.keys())
                ref_mel = prior_outputs['ref_mels']
                with utils.Timer('diffusion', print_time=hparams['profile_infer']):
                    mel_denoised = self.model.infer_diffusion(mu=prior_outputs["mel_out"], n_timesteps=150, spk=spk_embed, temperature=hparams['temp'], ref_x=ref_mel)
                    
                self.preprare_vocoder(hparams)
                # prior
                prior_quant, _, _ = self.vocoder(prior_outputs['mel_out'].transpose(1, 2), None, 100000, vq=True, early_stop=-1)
                prior_audio, _ = self.vocoder.inference(prior_quant)
                sample['prior_audio'] = prior_audio
            
                # diffusion
                denoised_quant, _, _ = self.vocoder(mel_denoised.transpose(1, 2), None, 100000, vq=True, early_stop=-1)
                
                
                denoised_audio, _ = self.vocoder.inference(denoised_quant)
                sample['denoised_audio'] = denoised_audio
                
                
                # reference
                # ref_audio, _ = self.vocoder.inference(ref_mel.transpose(1, 2))
                sample["ref_audio"] = wav_data
                
                
                sample['outputs'] = prior_outputs['mel_out']
                sample['pitch_pred'] = prior_outputs.get('pitch')
                # sample['pitch'] = restore_pitch(sample['pitch'], uv if hparams['use_uv'] else None, hparams)
                sample['utt_id'] = utt_id
                self.after_infer(sample)
            exit()
            
            ref_id = '00044'
            ref_name = f'{ref_id}.wav'
            ref_fn = os.path.join(ref_dir, ref_name)
            wav_data, sr = librosa.load(ref_fn, sr=16000)
            wav_data = np.pad(wav_data, (0, 200-len(wav_data) % 200))
            
            codec_enc = CodecEncoder()
            codec_dec = CodecDecoder()
            ckpt_path = '/blob/v-shenkai/checkpoints/tts/codec/chanpin_5w/v5/lambda_disc_1_commit_weight_0.25/checkpoint-440000.pt'
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            codec_enc.load_state_dict(checkpoint["model"]['CodecEnc'])
            codec_enc = codec_enc.eval().to(device)
            codec_dec.load_state_dict(checkpoint["model"]['generator'])
            codec_dec = codec_dec.eval().to(device)
            wav_tensor = torch.from_numpy(wav_data).to(device)  
            vq_emb = codec_enc(wav_tensor[None, None, :])
            ref_mel, _, _ = codec_dec(vq_emb, None, 1000000, vq=True)
            # print(ref_mel.shape) torch.Size([1, 256, 3192])
            ref_mel = ref_mel.transpose(1, 2)
            ref_mel = ref_mel[:, :1000, :]
            
            utt_id = ['product_'+ ref_id + "_" + utt_id[0]]

            import pickle
            from utils.text_encoder import NUM_RESERVED_TOKENS, UNK_ID, EOS_ID
            with open(f'{ref_dir}/data/{ref_id}.mel2ph', 'rb') as f:
                ref_mel2ph = torch.from_numpy(pickle.load(f)).to(device)
            with open(f'{ref_dir}/data/{ref_id}.phone_id', 'rb') as f:
                ref_phone_id = torch.from_numpy(pickle.load(f))
            with open(f'{ref_dir}/data/{ref_id}.f0', 'rb') as f:
                ref_f0 = pickle.load(f)
            # ref_mel2ph = torch.LongTensor(ref_mel2ph[:used]).to(device)
            max_phone = max(ref_mel2ph.max().item(), hparams['max_input_tokens'])
            phone_encoded = []
            phone_encoded.append(UNK_ID)
            for i in ref_phone_id:
                phone_encoded.append(NUM_RESERVED_TOKENS + i)
            phone_encoded.append(EOS_ID)
            ref_input = torch.LongTensor(phone_encoded[:max_phone]).to(device)
            # if hparams["remove_bos"]:
            #     ref_input = ref_input[1:]     
            ref_input = ref_input[None, ...]
            ref_mel2ph = ref_mel2ph[None, ...]   
            ref_pitch, ref_uv = process_f0(ref_f0, hparams)
            ref_pitch = ref_pitch[None, ...].to(device)
            ref_uv = ref_uv[None, ...].to(device)
        else:
            refs = sample['refs'][0]
            print(refs)
            if refs and refs[0]['utt_id'] != utt_id[0]:
                # raise NotImplementedError()
                ref = refs[0]
                ref_utt_id = ref['utt_id']
                used = hparams['max_frames']
                # used = min(used, 1200)
                utt_id = [f'inc_{used}_' + str(ref['spk_id'])]
                print('ref spk id:', ref['spk_id'], 'utt_id:', ref['utt_id'])
                code_dir = '/blob/v-zeqianju/dataset/tts/product_5w/full_code'
                ref_code = torch.load(f'{code_dir}/{ref_utt_id}.code', map_location=device)
                ref_code = ref_code.permute(1, 2, 0)  # C B T  -> B T C
                ref_mel = self.model.convert_code_to_latent(ref_code)
                ref_mel = ref_mel[:, :used, :]
                import tarfile
                import pickle
                from utils.text_encoder import NUM_RESERVED_TOKENS, UNK_ID, EOS_ID
                tar = ref['tar']
                tar_obj = tarfile.open(tar, mode='r')
                fn = ref['fn']
                keys = ['duration', 'phone_id', 'speech', 'mel']
                file_info = dict()
                for info in tar_obj:
                    try:
                        if not info.isfile():
                            continue
                        for key in keys:
                            if info.name == f'{fn}.{key}':
                                if key == 'speaker_id':
                                    value = int(info.name.split('.')[1].split('_')[0])
                                else:
                                    cur_f = tar_obj.extractfile(info)
                                    value = pickle.load(cur_f)
                                file_info[key] = value
                                break
                    except:
                        continue
                from datasets.tts.product.gen_fs2_product_full_wotrain import dur_to_mel2ph
                from utils.preprocessor import get_pitch
                ref_mel2ph = dur_to_mel2ph(file_info['duration'])
                ref_mel2ph = torch.LongTensor(ref_mel2ph[:used]).to(device)
                hparams["min_delta_l"] = 3
                ref_f0, _ = get_pitch(file_info["speech"], file_info["mel"], hparams)
                ref_pitch, ref_uv = process_f0(ref_f0, hparams)
                ref_pitch = ref_pitch[None, ...]
                ref_uv = ref_uv[None, ...]
                # print(ref_pitch.shape, sample['pitch'].shape)
                # print(ref_pitch, sample['pitch'])
                # exit()
                max_phone = max(ref_mel2ph.max().item(), hparams['max_input_tokens'])
                phone_encoded = []
                phone_encoded.append(UNK_ID)
                for i in file_info['phone_id']:
                    phone_encoded.append(NUM_RESERVED_TOKENS + i)
                phone_encoded.append(EOS_ID)
                ref_input = torch.LongTensor(phone_encoded[:max_phone]).to(device)
                # if hparams["remove_bos"]:
                #     ref_input = ref_input[1:]     
                ref_input = ref_input[None, ...]
                ref_mel2ph = ref_mel2ph[None, ...]   
                # print(ref_code.shape, ref_mel.shape)
                # print(ref_input.shape, ref_mel2ph.shape)
                # print(sample['src_tokens'].shape, sample['mel2ph'].shape)
                # exit()
            else:
                # 
                pass
            
            
        with utils.Timer('fs', print_time=hparams['profile_infer']):
            prior_outputs, spk_embed = self.model.infer_fs2_hack(src_tokens_ref=ref_input, src_tokens_in=input, mel2ph=ref_mel2ph,
                                                                 spk_embed=None, ref_mels=ref_mel, pitch=ref_pitch, uv=ref_uv,
                                                                 use_pred_pitch=True)
        
        print(prior_outputs.keys())
        ref_mel = prior_outputs['ref_mels']
        with utils.Timer('diffusion', print_time=hparams['profile_infer']):
            mel_denoised = self.model.infer_diffusion(mu=prior_outputs["mel_out"], n_timesteps=150, spk=spk_embed, temperature=1.2, ref_x=ref_mel)
             
        self.preprare_vocoder(hparams)
        # prior
        prior_quant, _, _ = self.vocoder(prior_outputs['mel_out'].transpose(1, 2), None, 100000, vq=True, early_stop=-1)
        prior_audio, _ = self.vocoder.inference(prior_quant)
        sample['prior_audio'] = prior_audio
    
        # diffusion
        denoised_quant, _, _ = self.vocoder(mel_denoised.transpose(1, 2), None, 100000, vq=True, early_stop=-1)
        
        
        denoised_audio, _ = self.vocoder.inference(denoised_quant)
        sample['denoised_audio'] = denoised_audio
        
        
        # reference
        ref_audio, _ = self.vocoder.inference(ref_mel.transpose(1, 2))
        sample["ref_audio"] = ref_audio
        
        
        sample['outputs'] = prior_outputs['mel_out']
        sample['pitch_pred'] = prior_outputs.get('pitch')
        # sample['pitch'] = restore_pitch(sample['pitch'], uv if hparams['use_uv'] else None, hparams)
        sample['utt_id'] = utt_id
        return self.after_infer(sample)
        
    def preprare_vocoder(self, hparams):
        if self.vocoder is None:
            vocoder = CodecDecoder()
            
            vocoder.load_state_dict(torch.load(hparams["vocoder_ckpt"], map_location="cpu"))
            _ = vocoder.cuda().eval()
            self.vocoder = vocoder
    
    def after_infer(self, predictions):
        if self.saving_result_pool is None and not hparams['profile_infer']:
            self.saving_result_pool = Pool(8)
            self.saving_results_futures = []
        # self.prepare_vocoder()
        predictions = utils.unpack_dict_to_list(predictions)
        t = tqdm(predictions)
        for num_predictions, prediction in enumerate(t):
            for k, v in prediction.items():
                if type(v) is torch.Tensor:
                    prediction[k] = v.cpu().numpy()

            utt_id = prediction.get('utt_id')
            text = prediction.get('text')
            targets = self.remove_padding(prediction.get("targets"))
            outputs = self.remove_padding(prediction["outputs"])
            print("outputs shape:", outputs.shape)
            
            # prior_audio = prediction["prior_audio"]
            denoised_audio = prediction["denoised_audio"].astype(np.float32)
            reference_audio = prediction["ref_audio"].astype(np.float32)
            cat_audio = np.concatenate([reference_audio, denoised_audio], axis=1)

            gen_dir = os.path.join(hparams['work_dir'],
                                   f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}_v{hparams["infer_laoban"]}_tp{hparams["temp"]}'+ ('stoc' if hparams['stoc'] else '')+ ('_refnorm' if hparams['ref_norm'] else '')+('_inc' if hparams['in_context_infer'] else '' ))
            os.makedirs(gen_dir, exist_ok=True)

            print(gen_dir)
            # sf.write(f'{gen_dir}/prior_{os.path.basename(utt_id)}.wav', prior_audio[0], 16000, subtype='PCM_24')
            
            sf.write(f'{gen_dir}/denoised_{os.path.basename(utt_id)}.wav', denoised_audio[0], 16000, subtype='PCM_24')
            
            sf.write(f'{gen_dir}/ref_{os.path.basename(utt_id)}.wav', reference_audio[0], 16000, subtype='PCM_24')
            
            sf.write(f'{gen_dir}/{os.path.basename(utt_id)}.wav', cat_audio[0], 16000, subtype='PCM_24')

        return {}
    
    def remove_padding(self, x, padding_idx=0):
        return utils.remove_padding(x, padding_idx)
    
    def weights_nonzero_speech(self, target):
        # target : B x T x mel
        # Assign weight 1.0 to all labels except for padding (id=0).
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)

    def get_model_stats(self, model):
        trainable_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return trainable_parameter_count