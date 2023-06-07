
import torch, os
import torch.nn as nn
from utils.hparams import hparams

from usr_dir.huggingface_lib.gpt2 import GPT2Model
from usr_dir.huggingface_lib.configure import GPT2Config

from utils.chanpin_utils import chanpin_phone_dict
from transformers.modeling_utils import ModuleUtilsMixin
from typing import IO, Any, Callable, Dict, Optional, Tuple, Type, Union
import transformers
from usr_dir.utils.tensor_utils import sequence_mask
import torch.nn.functional as F
from usr_dir.codec.codec_decoder import CodecDecoder
import soundfile as sf

import pytorch_lightning as pl
import utils

class TTSLatentGPT(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        
        self.in_proj = nn.Linear(512, 512)
        
        if hparams["vq_ckpt"] is not None:
            print("Loading VQ model from {}".format(hparams["vq_ckpt"]))
            from usr_dir.codec.residual_vq import ResidualVQ
            
            quantizer = ResidualVQ(
                num_quantizers=hparams['audio_num_mel_bins'], dim=hparams['vqemb_size'], codebook_size=hparams['codebook_size'], commitment=0.25,
                threshold_ema_dead_code=2
            )
            quantizer.load_state_dict(torch.load(hparams["vq_ckpt"], map_location="cpu"))
            quantizer.train()
            quantizer.requires_grad_(False)
            self.quantizer = quantizer
        
        self.gpt2_config = GPT2Config()
        self.gpt2_config.vocab_size = 6000
        
        self.gpt2 = GPT2Model(self.gpt2_config)
        
        self.vocoder = None

        
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        input = batch['src_tokens']  # [B, T_t]
        target = batch['targets']  # [B, T_s, 80]
        target_len = batch["target_lengths"]
        mel2ph = batch['mel2ph']  # [B, T_s]
        pitch = batch['pitch']
        #energy = sample['energy']
        energy = None
        uv = batch['uv']

        spk_embed = batch.get('spk_embed') if not hparams['use_spk_id'] else batch.get('spk_ids')
        loss_output, output, logging_info = self.run_model(input, mel2ph, spk_embed, target,
                                             pitch=pitch, uv=uv, energy=energy,
                                             target_len=target_len,
                                             return_output=True)
        
        total_loss = sum([v for v in loss_output.values() if v.requires_grad])
        loss_output['batch_size'] = target.size()[0]
        if "mel_acc" in output:
            loss_output["mel_acc"] = output["mel_acc"]
            
        loss_output.update(logging_info)
        

        log_outputs = utils.tensors_to_scalars(loss_output)


        log_outputs['all_loss'] = total_loss.item()
  
        progress_bar_log = log_outputs
        tb_log = {f'tr/{k}': v for k, v in log_outputs.items()}
        
        self.logger.log_metrics(progress_bar_log)
        
        return {
            'loss': total_loss,
            'progress_bar': progress_bar_log,
            'log': tb_log
        }
    
    def validation_step(self, batch, batch_idx):
        input = batch['src_tokens']
        target = batch['targets']
        target_len = batch["target_lengths"]
        mel2ph = batch['mel2ph']
        pitch = batch['pitch']
        #energy = sample['energy']
        energy = None
        uv = batch['uv']
        #for k in sample.keys():
        #    if hasattr(sample[k], "shape"):
        #        print(k, sample[k].shape)
        

        spk_embed = batch.get('spk_embed') if not hparams['use_spk_id'] else batch.get('spk_ids')
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out, logging_info = self.run_model(input, mel2ph, spk_embed, target,
                                                      pitch=pitch, uv=uv,
                                                      energy=energy,
                                                      target_len=target_len,
                                                      return_output=True)
        outputs['total_loss'] = outputs['losses']['mel']
        outputs['nmels'] = batch['nmels']
        outputs['nsamples'] = batch['nsamples']
        if "mel_acc" in model_out:
            outputs['losses']["mel_acc"] = model_out["mel_acc"]
            
        outputs['losses'].update(logging_info)
        
        outputs = utils.tensors_to_scalars(outputs)
        
        outputs.update(logging_info)
        return outputs
    
        
    def run_model(self, input, mel2ph, spk_embed, target,
                  return_output=False, ref_mel='tgt', pitch=None, uv=None, energy=None, target_len=None):
        hparams['global_steps'] = self.global_step
        
        y_mask = sequence_mask(target_len, target.size(-1)).unsqueeze(1).to(input.device)
        
        with torch.no_grad():
            target_latent = self.convert_code_to_latent(target)
        
        losses = {}
        ret_logging_info = {}
            
        pred_latent, stop_logits = self._forward(src_tokens=input.long(), speech_latent=target_latent, speech_length=target_len)

        losses['mel'] = self.l1_loss(pred_latent, target_latent, y_mask) * hparams['loss_latent_weight']
        
        # stop token loss
        # uv_loss = (F.binary_cross_entropy_with_logits(
        #         p_pred[:, :, 1].reshape(-1), uv.reshape(-1), reduction='none') * nonpadding).sum() \
        #               / nonpadding.sum() * hparams['lambda_uv']
        target_stop = F.one_hot(target_len - 1, num_classes=target_latent.shape[1])          

        target_stop_weights = target_stop * hparams["stop_token_weight"]
        target_stop_weights[target_stop_weights == 0] = 1

        losses["stop_token"] = (F.binary_cross_entropy_with_logits(
            stop_logits, target_stop.float(), reduction="none", weight=target_stop_weights
        ) * y_mask.squeeze(1)).sum() / y_mask.sum() * hparams["loss_stop_weight"]


        
        codebook_bp_loss, logging_info = self.codebook_loss(x=pred_latent.transpose(1, 2), discrete_y=target.transpose(1, 2), latent_y=target_latent.transpose(1, 2), y_mask=y_mask, name="posterior", require_dist_loss=False)
        
        losses.update(codebook_bp_loss)
        ret_logging_info.update(logging_info)

        
        
        if not return_output:
            return losses, ret_logging_info
        else:
            return losses, {}, ret_logging_info
        
    def codebook_loss(self, x, discrete_y, latent_y, y_mask, name="", require_dist_loss=False):
        
        bp_loss = {}
        
        loss_summary = {}
        
        # print(x.shape, discrete_y.shape, latent_y.shape, y_mask.shape)
        # exit(0)
        
        
        quantized_out, all_indices, commit_loss, discrete_pred_loss, discrete_dist_loss = self.quantizer.calculate_loss(x=x, discrete_y=discrete_y, latent_y=latent_y, y_mask=y_mask, require_dist_loss=require_dist_loss)
        
        
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
        
        
        pass

    def l1_loss(self, decoder_output, target, mask):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        
        
        
        
        l1_loss = F.l1_loss(decoder_output, target, reduction='none')
        weights = self.weights_nonzero_speech(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        return l1_loss

    def weights_nonzero_speech(self, target):
        # target : B x T x mel
        # Assign weight 1.0 to all labels except for padding (id=0).
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=hparams["lr"],
                                      betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
                                      weight_decay=hparams['weight_decay'])
        # num_training_steps, num_warmup_steps = self.compute_warmup(
        #     num_training_steps=100000000,
        #     num_warmup_steps=0.1,
        # )
        
        scheduler = transformers.get_inverse_sqrt_schedule(optimizer=optimizer, num_warmup_steps=hparams["warmup_updates"])
        
        # scheduler = transformers.get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=hparams["warmup_updates"] * 20, num_training_steps=9240250
        # )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
        
    def compute_warmup(self, num_training_steps: int, num_warmup_steps: Union[int, float]) -> Tuple[int, int]:
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return num_training_steps, num_warmup_steps
    
            
    @torch.no_grad()
    def convert_code_to_latent(self, codes):
        latent = self.quantizer.vq2emb(codes.long())
        return latent
        
    def _forward(self, src_tokens, speech_latent, speech_length):
        
        # print(src_tokens.shape)
        # print(src_tokens[0].detach().cpu().numpy().tolist())
        
        # padding src_tokens to enable hacking for simplicity
        src_tokens_with_pad = torch.cat((src_tokens, torch.zeros((src_tokens.shape[0], speech_latent.shape[1]), dtype=torch.long, device=src_tokens.device)), dim=1)
        
        # phonemes = chanpin_phone_dict.decode_list(src_tokens[0].detach().cpu().numpy().tolist())

        gpt_results = self.gpt2(
            input_ids=src_tokens_with_pad,
            speech_latent=speech_latent,
            speech_length=speech_length
        )
        
        pred_latent = gpt_results["predicted_speech_latent"]
        
        stop_logits = pred_latent[:, :, -1]
        
        pred_latent = pred_latent[:, :, :-1]
        
        return pred_latent, stop_logits

    def test_step(self, sample, batch_idx):
        if hparams["in_context_infer"]:
            return self.in_context_inference(sample, batch_idx)
        else:
            return self.conventional_inference(sample, batch_idx)
    
    @torch.inference_mode()
    def conventional_inference(self, sample, batch_idx):
        self.eval()
        src_tokens = sample['src_tokens']
        utt_id = sample['utt_id'][0]
        target_codes = sample["targets"]
        
        target_latent = self.convert_code_to_latent(target_codes)
        
        print(src_tokens.long())
        
        infer_frames = 3000
        start_idx = 500
        pred_latent = None
        src_tokens_with_pad = torch.cat((src_tokens, torch.zeros((src_tokens.shape[0], infer_frames), dtype=torch.long, device=src_tokens.device)), dim=1)
        
        output_latent_collect = [target_latent[:, :start_idx, :]]
        for i in range(start_idx, infer_frames):
            
            if i == start_idx + 200:
                break
            
            input_latent = torch.cat(output_latent_collect, dim=1)

            print("step", i, input_latent.shape)
            gpt_results = self.gpt2(
                input_ids=src_tokens_with_pad.long(),
                speech_latent=input_latent.clone(),
                speech_length=torch.Tensor([input_latent.shape[1]]).long().to(input_latent.device),
                enable_dropout=True
            )
            last_hidden_state = gpt_results["last_hidden_state"]
            
            ar_pred = last_hidden_state[:, src_tokens.shape[1] - 2: src_tokens.shape[1] + i - 1, :-1].view(1, -1, 256)
            
            print(ar_pred.shape, target_latent[:, :i, :].shape)
            exit(0)
            
            predicted_logits = last_hidden_state[:, src_tokens.shape[1] + i - 1, :].view(1, 1, 257)
            
            is_stop = predicted_logits[0, 0, -1].item()
            
            if is_stop > 0:
                print("stop....................", i)
                break
            
            pred_latent = predicted_logits[:, :, :-1]
            
            gt_latent = target_latent[:, i, :].unsqueeze(1)
            print(F.l1_loss(pred_latent, gt_latent).item(), "loss l1")
            
            output_latent_collect.append(pred_latent)
        predicted_latent = torch.cat(output_latent_collect, dim=1)
        
        print(pred_latent.shape, target_latent.shape)
        print("ok......")
        self.preprare_vocoder(hparams)
        
        predicted_quant, _, _ = self.vocoder(predicted_latent.transpose(1, 2), None, 100000, vq=True, early_stop=-1)
        predict_audio, _ = self.vocoder.inference(predicted_quant)
        predict_audio = predict_audio.detach().cpu().numpy()
        
        
        ar_pred_quant, _, _ = self.vocoder(ar_pred.transpose(1, 2), None, 100000, vq=True, early_stop=-1)
        ar_audio, _ = self.vocoder.inference(ar_pred_quant)
        ar_audio = ar_audio.detach().cpu().numpy()
        
        gen_dir = os.path.join(hparams['work_dir'],
                                   f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}_')
        os.makedirs(gen_dir, exist_ok=True)

        print(gen_dir)
        # sf.write(f'{gen_dir}/prior_{os.path.basename(utt_id)}.wav', prior_audio[0], 16000, subtype='PCM_24')
        
        sf.write(f'{gen_dir}/ar_{os.path.basename(utt_id)}.wav', predict_audio[0, 0], 16000, subtype='PCM_24')
        sf.write(f'{gen_dir}/ar_1_{os.path.basename(utt_id)}.wav', ar_audio[0, 0], 16000, subtype='PCM_24')
        
        
        
        
        
        
        
        exit(0)
        
        
        
        refs = sample['refs'][0]
    
        ref = refs[0]
        ref_utt_id = ref['utt_id']
        used = min(hparams['max_frames'], 1200)
        ref_utt_id = [f'{used}_' + str(ref['spk_id'])]
        
        print('ref spk id:', ref['spk_id'], 'utt_id:', ref['utt_id'])
        code_dir = '/blob/v-zeqianju/dataset/tts/product_5w/full_code'
        ref_code = torch.load(f'{code_dir}/{ref_utt_id}.code', map_location="cpu")
        ref_code = ref_code.permute(1, 2, 0)  # C B T  -> B T C
        ref_mel = self.convert_code_to_latent(ref_code.to(input.device))
        ref_mel = ref_mel[:, :used, :]
        
    def preprare_vocoder(self, hparams):
        if self.vocoder is None:
            vocoder = CodecDecoder()
            
            vocoder.load_state_dict(torch.load(hparams["vocoder_ckpt"], map_location="cpu"))
            _ = vocoder.cuda().eval()
            self.vocoder = vocoder