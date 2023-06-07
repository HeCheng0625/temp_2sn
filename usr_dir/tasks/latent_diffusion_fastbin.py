import matplotlib

matplotlib.use('Agg')
from utils.pl_utils import data_loader
import os
from multiprocessing.pool import Pool
from tqdm import tqdm

from modules.tts_modules import DurationPredictorLoss
from utils.hparams import hparams

from utils.plot import plot_to_figure
from utils.world_utils import restore_pitch, process_f0

import numpy as np

from  usr_dir.module.fs2_dis import FastSpeech2_dis
from usr_dir.module.diffusion_tts import DiffusionTTS
from tasks.transformer_tts import TransformerTtsTask
from usr_dir.datasets.latent_dataset_fastbin import FastSpeechDataset


import torch
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import utils

from usr_dir.codec.codec_decoder import CodecDecoder
import soundfile as sf
from usr_dir.utils.tensor_utils import sequence_mask


class FastSpeech2Task(TransformerTtsTask):
    def __init__(self):
        super(FastSpeech2Task, self).__init__()
        self.dur_loss_fn = DurationPredictorLoss()
        self.mse_loss_fn = torch.nn.MSELoss()

    @data_loader
    def train_dataloader(self):
        train_dataset = FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                          hparams['train_set_name'], hparams, shuffle=True)
        return self.build_dataloader(train_dataset, True, self.max_tokens, self.max_sentences,
                                     endless=hparams['endless_ds'])

    @data_loader
    def val_dataloader(self):
        valid_dataset = FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                          hparams['valid_set_name'], hparams,
                                          shuffle=False)
        return self.build_dataloader(valid_dataset, False, self.max_eval_tokens, self.max_eval_sentences)

    @data_loader
    def test_dataloader(self):
        test_dataset = FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                         hparams['test_set_name'], hparams, shuffle=False)
        return self.build_dataloader(test_dataset, False, self.max_eval_tokens, self.max_eval_sentences)

    def build_model(self):
        arch = self.arch
        fs2_model = FastSpeech2_dis(arch, self.phone_encoder)
        
        model = DiffusionTTS(fs2_model)
        
        return model

    def _training_step(self, sample, batch_idx, _):
        input = sample['src_tokens']  # [B, T_t]
        target = sample['targets']  # [B, T_s, 80]
        target_len = sample["target_lengths"]
        mel2ph = sample['mel2ph']  # [B, T_s]
        pitch = sample['pitch']
        #energy = sample['energy']
        energy = None
        uv = sample['uv']

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        
            
        loss_output, output, logging_info = self.run_model(self.model, input, mel2ph, spk_embed, target,
                                             pitch=pitch, uv=uv, energy=energy,
                                             target_len=target_len,
                                             return_output=True)
        
        total_loss = sum([v for v in loss_output.values() if v.requires_grad])
        loss_output['batch_size'] = target.size()[0]
        if "mel_acc" in output:
            loss_output["mel_acc"] = output["mel_acc"]
            
        loss_output.update(logging_info)

        return total_loss, loss_output

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
        

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out, logging_info = self.run_model(self.model, input, mel2ph, spk_embed, target,
                                                      pitch=pitch, uv=uv,
                                                      energy=energy,
                                                      target_len=target_len,
                                                      return_output=True)
        outputs['total_loss'] = outputs['losses']['diff-mel']
        outputs['nmels'] = sample['nmels']
        outputs['nsamples'] = sample['nsamples']
        if "mel_acc" in model_out:
            outputs['losses']["mel_acc"] = model_out["mel_acc"]
        outputs = utils.tensors_to_scalars(outputs)
        
        outputs.update(logging_info)
        
        if batch_idx < 10:
            if 'pitch_logits' in model_out:
                pitch[uv > 0] = -4
                pitch_pred = model_out['pitch_logits'][:, :, 0]
                pitch_pred[model_out['pitch_logits'][:, :, 1] > 0] = -4
                self.logger.experiment.add_figure(f'pitch_{batch_idx}', plot_to_figure({
                    'gt': pitch[0].detach().cpu().numpy(),
                    'pred': pitch_pred[0].detach().cpu().numpy()
                }), self.global_step)
        return outputs

    def _validation_end(self, outputs):
        all_losses_meter = {
            'total_loss': utils.AvgrageMeter(),
        }
        for output in outputs:
            n = output['nsamples']
            for k, v in output['losses'].items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = utils.AvgrageMeter()
                all_losses_meter[k].update(v, n)
            all_losses_meter['total_loss'].update(output['total_loss'], n)
        return {k: round(v.avg, 4) for k, v in all_losses_meter.items()}

    def run_model(self, model, input, mel2ph, spk_embed, target,
                  return_output=False, ref_mel='tgt', pitch=None, uv=None, energy=None, target_len=None):
        hparams['global_steps'] = self.global_step
        
        y_mask = sequence_mask(target_len, target.size(-1)).unsqueeze(1).to(input.device)
        
        with torch.no_grad():
            target_latent = self.model.convert_code_to_latent(target)
        
        losses = {}
        # if ref_mel == 'tgt':
        #     ref_mel = target
        ref_mel = target_latent
        
        if "pred_pitch_after" in hparams and hparams["pred_pitch_after"] >= 0:
            output, diff_out = model(input, mel2ph, spk_embed, ref_mel, pitch, uv, energy, use_pred_pitch=self.global_step > hparams["pred_pitch_after"], skip_decoder=bool(hparams["skip_decoder"]))
        else:
            output, diff_out = model(input, mel2ph, spk_embed, ref_mel, pitch, uv, energy, skip_decoder=bool(hparams["skip_decoder"]))

        losses['mel'] = self.l1_loss(output['mel_out'], target_latent, y_mask) * hparams['prior_weight']

        losses['dur'] = self.dur_loss(output['dur'], mel2ph, input)

        if hparams['use_pitch_embed']:
            p_pred = output['pitch_logits']
            
            
            losses['uv'], losses['f0'] = self.pitch_loss(p_pred, pitch, uv)
            if losses['uv'] is None:
                del losses['uv']
                
        # print(p_pred.shape, diff_out["x0_pitch_pred"].shape)
        # exit(0)
                
        # diffusion loss
        if hparams["diffusion_loss_type"] == "l1":
            losses["diff-mel"] = self.l1_loss(diff_out["diff_x0_pred"], target_latent, y_mask) * hparams["diffusion_mel_weight"]
        elif hparams["diffusion_loss_type"] == "l2":
            losses["diff-mel"] = self.mse_loss(diff_out["diff_x0_pred"], target_latent, y_mask) * hparams["diffusion_mel_weight"]
        else:
            raise ValueError(f"Unknown diffusion loss type: {hparams['diffusion_loss_type']}")

        losses["diff-noise"] = self.l1_loss(diff_out["diff_noise_pred"], diff_out["diff_noise_gt"], y_mask)  * hparams["diff_loss_noise_weight"]
        
        
        if hparams["vq_dist_weight"] > 0:
            require_dist_loss = True
        else:
            require_dist_loss = False
        
        codebook_bp_loss, logging_info = self.codebook_loss(x=diff_out["diff_x0_pred"].transpose(1, 2), discrete_y=target.transpose(1, 2), latent_y=target_latent.transpose(1, 2), y_mask=y_mask, name="posterior", require_dist_loss=require_dist_loss)
        
        losses.update(codebook_bp_loss)
        
        
        if hparams["apply_pitch_on_x0"]:
            x0_pitch_pred = diff_out['x0_pitch_pred']
            
            losses['diff_uv'], losses['diff_f0'] = self.pitch_loss(x0_pitch_pred, pitch, uv)
            if losses['diff_uv'] is None:
                del losses['diff_uv']
        
        
        if not return_output:
            return losses, logging_info
        else:
            return losses, output, logging_info
        
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
        
        
        pass

    def l1_loss(self, decoder_output, target, mask):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        
        
        
        
        l1_loss = F.l1_loss(decoder_output, target, reduction='none')
        weights = self.weights_nonzero_speech(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        return l1_loss
    
    
        # l1_loss = F.l1_loss(decoder_output, target, reduction='none')
        
        # mask = mask.transpose(1, 2)
        
        # weights_pr = mask.float().sum() * l1_loss.size(-1)

        # l1_loss = (l1_loss * mask.float()).sum() / weights_pr.sum()
        # return l1_loss

    def mse_loss(self, decoder_output, target, target_len):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        mse_loss = F.mse_loss(decoder_output, target, reduction='none')
        weights_pr = target_len.sum() * target.size(-1)
        mse_loss = mse_loss.sum() / weights_pr.sum()
        return mse_loss

    def ce_loss(self, decoder_output, target):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        #assert target.shape[-1] * target.shape[-2] == decoder_output.shape[1]
        #assert (target >= 0 ).all()
        #assert (target < 1025).all()
        #assert decoder_output.shape[-1] == 1025
        ce_loss = F.cross_entropy(decoder_output.reshape(-1, decoder_output.shape[-1]), target.reshape(-1), reduction='none', ignore_index=self.padding_idx)

        weights = (target != self.padding_idx).long().reshape(-1)

        is_acc = (decoder_output.max(-1)[1].reshape(-1) == target.reshape(-1)).float()
        acc = (is_acc * weights).sum() / weights.sum() * 100

        ce_loss = (ce_loss * weights).sum() / weights.sum()
        return ce_loss, acc 

    def dur_loss(self, dur_pred, mel2ph, input, split_pause=False, sent_dur_loss=False):
        B, T_t = input.shape
        dur_gt = mel2ph.new_zeros(B, T_t + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
        dur_gt = dur_gt[:, 1:]
        nonpadding = (input != 0).float()
        if split_pause:
            is_pause = (input == self.phone_encoder.seg()) | (input == self.phone_encoder.unk()) | (
                    input == self.phone_encoder.eos())
            is_pause = is_pause.float()
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
            dur_gt = dur_gt.float() * nonpadding
            sent_dur_loss = F.l1_loss(dur_pred.sum(-1), dur_gt.sum(-1), reduction='none') / dur_gt.sum(-1)
            sent_dur_loss = sent_dur_loss.mean()
            return ph_dur_loss, sent_dur_loss

    def pitch_loss(self, p_pred, pitch, uv):
        assert p_pred[..., 0].shape == pitch.shape
        assert p_pred[..., 0].shape == uv.shape
        nonpadding = (pitch != -200).float().reshape(-1)
        if hparams['use_uv']:
            uv_loss = (F.binary_cross_entropy_with_logits(
                p_pred[:, :, 1].reshape(-1), uv.reshape(-1), reduction='none') * nonpadding).sum() \
                      / nonpadding.sum() * hparams['lambda_uv']
            nonpadding = (pitch != -200).float() * (uv == 0).float()
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
        nonpadding = (energy != 0).float()
        loss = (F.mse_loss(energy_pred, energy, reduction='none') * nonpadding).sum() / nonpadding.sum()
        loss = loss * hparams['lambda_energy']
        return loss

    def test_step(self, sample, batch_idx):
        
        self.model.quantizer.eval()
        utt_id = sample['utt_id']
        
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        input = sample['src_tokens']
        device = input.device
        print(input)
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
        test_dataset = FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                         hparams['test_set_name'], hparams, shuffle=False)
        ref_sample = test_dataset.collater([test_dataset[1]])
        ref_input = ref_sample['src_tokens'].to(device)
        
        ref_mel = self.model.convert_code_to_latent(ref_sample['targets'].to(device)).transpose(1, 2)
        ref_mel2ph = ref_sample['mel2ph'].to(device)
        ref_pitch = ref_sample['pitch'].to(device)
        ref_uv = ref_sample['uv'].to(device)
        
        if self.model.ref_enc is not None:
            # # gt ref
            # target = ref_sample['targets'].to(device)
            # # ood valid
            # #fn = '/blob/v-zeqianju/dataset/tts/product_5w/valid_test_code/006794c9a88333f2f4cd3c94b48ff844074d5c4395c0993063f73a31192066d0.code'
            # #in domain train
            # # fn = '/blob/v-zeqianju/dataset/tts/product_5w/24h_code/0002c6c9d19fd22bdba6c4b0caa683ef5c33642808a209154317c90495f76a7e.code'
            # # target = torch.load(fn)
            # # target = target.squeeze(1).transpose(0, 1)[None, ...]
            # with torch.no_grad():
            #     target = self.model.convert_code_to_latent(target)
            spk_embed = self.model.ref_enc(ref_mel[None, ...].contiguous()).to(device)
        
        # input format: UNK + BOS + br0 + .... + '~' + EOS
        input = input[:, 2:] # 保留开头br0
        if hparams['profile_infer']:
            raise NotImplementedError()
            # mel2ph = mel2ph[:, 2:]
            # pitch = pitch[:, 2:]
            # uv = uv[:, 2:]
        
        # 删除结尾br0
        # to do 删除结尾token对应mel
        ref_input = ref_input[:, :-2] 
        input_length = ref_input.shape[1]
        frame_length = (ref_mel2ph < input_length).sum().item()
        
        ref_mel2ph = ref_mel2ph[:, :frame_length]
        ref_pitch = ref_pitch[:, :frame_length]
        ref_uv = ref_uv[:, :frame_length]
        ref_mel = ref_mel[:, :frame_length]
        print(frame_length)
            
            
        with utils.Timer('fs', print_time=hparams['profile_infer']):
            prior_outputs = self.model.infer_fs2(input, mel2ph, spk_embed, None, pitch, uv)

        with utils.Timer('fs', print_time=hparams['profile_infer']):
            ref_prior_outputs = self.model.infer_fs2(ref_input, ref_mel2ph, spk_embed, None, ref_pitch, ref_uv)
            
            
        with utils.Timer('diffusion', print_time=hparams['profile_infer']):
            mel_denoised = self.model.infer_diffusion(mu=prior_outputs["mel_out"], n_timesteps=1000, spk=spk_embed, temperature=2.0, ref_mu=ref_prior_outputs["mel_out"], ref_x=ref_mel)
            

            
        self.preprare_vocoder(hparams)
        # prior
        prior_quant, _, _ = self.vocoder(prior_outputs['mel_out'].transpose(1, 2), None, 100000, vq=True, early_stop=-1)
        prior_audio, _ = self.vocoder.inference(prior_quant)
        sample['prior_audio'] = prior_audio
    
        # diffusion
        denoised_quant, _, _ = self.vocoder(mel_denoised.transpose(1, 2), None, 100000, vq=True, early_stop=-1)
        
        
        denoised_audio, _ = self.vocoder.inference(denoised_quant)
        sample['denoised_audio'] = denoised_audio
        
        
        sample['outputs'] = prior_outputs['mel_out']
        sample['pitch_pred'] = prior_outputs.get('pitch')
        sample['pitch'] = restore_pitch(sample['pitch'], uv if hparams['use_uv'] else None, hparams)
        sample['utt_id'] = utt_id
        return self.after_infer(sample)
    
    def preprare_vocoder(self, hparams):
        if self.vocoder is None:
            vocoder = CodecDecoder()
            
            vocoder.load_state_dict(torch.load(hparams["vocoder_ckpt"]))
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
            
            prior_audio = prediction["prior_audio"]
            denoised_audio = prediction["denoised_audio"]


            gen_dir = os.path.join(hparams['work_dir'],
                                   f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
            os.makedirs(gen_dir, exist_ok=True)

            print(gen_dir)
            sf.write(f'{gen_dir}/prior_{os.path.basename(utt_id)}.wav', prior_audio[0], 16000, subtype='PCM_24')
            
            sf.write(f'{gen_dir}/denoised_{os.path.basename(utt_id)}.wav', denoised_audio[0], 16000, subtype='PCM_24')
            
            exit(0) 
            # np.save(os.path.join(gen_dir, utt_id + ".npy"), outputs)
            
            # wav_pred = self.inv_spec(outputs, pitch_pred, noise_outputs)
            # if not hparams['profile_infer']:
            #     os.makedirs(gen_dir, exist_ok=True)
            #     os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
            #     os.makedirs(f'{gen_dir}/spec_plot', exist_ok=True)
            #     os.makedirs(f'{gen_dir}/pitch_plot', exist_ok=True)
            #     self.saving_results_futures.append(
            #         self.saving_result_pool.apply_async(self.save_result, args=[
            #             wav_pred, outputs, f'P', utt_id, text, gen_dir, [pitch_pred, pitch_gt], noise_outputs]))

            #     wav_gt = self.inv_spec(targets, pitch_gt, noise_outputs)
            #     if targets is not None:
            #         self.saving_results_futures.append(
            #             self.saving_result_pool.apply_async(self.save_result, args=[
            #                 wav_gt, targets, 'G', utt_id, text, gen_dir, pitch_gt, noise_outputs]))
            #     t.set_description(
            #         f"Pred_shape: {outputs.shape}, gt_shape: {targets.shape}")
            # else:
            #     if 'gen_wav_time' not in self.stats:
            #         self.stats['gen_wav_time'] = 0
            #     self.stats['gen_wav_time'] += len(wav_pred) / hparams['audio_sample_rate']
            #     print('gen_wav_time: ', self.stats['gen_wav_time'])

        return {}


if __name__ == '__main__':
    FastSpeech2Task.start()
