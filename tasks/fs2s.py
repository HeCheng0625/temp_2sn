import os

import numpy as np
import torch
import torch.distributed as dist

import utils
from modules.fs2s import FastSpeech2s
from parallel_wavegan.models import ParallelWaveGANDiscriminator
from parallel_wavegan.optimizers import RAdam
from tasks.base_task import BaseDataset
from tasks.fs2 import FastSpeech2Task
from tasks.pwg import PwgTask
from tasks.transformer_tts import RSQRTSchedule
from utils import audio
from utils.hparams import hparams, set_hparams
from utils.indexed_datasets import IndexedDataset
from utils.pl_utils import data_loader
from utils.tts_utils import GeneralDenoiser
from utils.world_utils import process_f0


class FastSpeech2sDataset(BaseDataset):
    """A dataset that provides helpers for batching."""

    def __init__(self, data_dir, prefix, shuffle=False):
        super().__init__(data_dir, prefix, hparams, shuffle)
        self.utt_ids = []
        self.texts = []
        self.phones = []
        self.mels = []
        self.mel2ph = []
        self.is_infer = prefix == 'test'
        self.batch_max_frames = 0 if self.is_infer else hparams['max_samples'] // hparams['hop_size']
        self.aux_context_window = hparams['generator_params'].get('aux_context_window', 0)
        self.hop_size = hparams['hop_size']
        self.use_pitch_embed = hparams['use_pitch_embed']
        self.indexed_bs = None
        self.idx2key = np.load(f'{self.data_dir}/{self.prefix}_all_keys.npy')
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        f0s = np.load(f'{self.data_dir}/train_f0s.npy', allow_pickle=True)
        f0s = np.concatenate(f0s, 0)
        f0s = f0s[f0s != 0]
        hparams['f0_mean'] = self.f0_mean = np.mean(f0s).item()
        hparams['f0_std'] = self.f0_std = np.std(f0s).item()

    def _get_item(self, index):
        if self.indexed_bs is None:
            self.indexed_bs = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        item = self.indexed_bs[index]
        return item

    def __getitem__(self, index):
        hparams = self.hparams
        key = self.idx2key[index]
        item = self._get_item(index)
        spec = torch.Tensor(item['mel'])[:hparams['max_frames']]
        energy = (spec.exp() ** 2).sum(-1).sqrt()[:hparams['max_frames']]
        mel2ph = torch.LongTensor(item['mel2ph'])[:hparams['max_frames']]
        f0, uv = process_f0(item["f0"], hparams)
        phone = torch.LongTensor(item['phone'][:hparams['max_input_tokens']])

        # energy_ph = torch.zeros_like(phone).float()
        # frame_cnt = torch.zeros_like(phone).float()
        # ones = torch.ones_like(energy).float()
        # energy_ph.scatter_add_(0, mel2ph - 1, energy)
        # frame_cnt.scatter_add_(0, mel2ph - 1, ones)
        # energy_ph = energy_ph / frame_cnt
        sample = {
            "id": index,
            "utt_id": key,
            "text": item['txt'],
            "source": phone,
            "mel2ph": mel2ph,
            "mel": spec,
            "pitch": f0[:hparams['max_frames']],
            "energy": energy,
            # "energy": energy_ph,
            "uv": uv[:hparams['max_frames']],
            "wav": torch.Tensor(item['wav'])[:hparams['max_frames'] * hparams['hop_size']],
        }
        return sample

    def collater(self, batch):
        if len(batch) == 0:
            return {}

        wav_batch, mel_batch, mel2ph_batch, source_batch, p_batch, uv_batch, energy_batch = \
            [], [], [], [], [], [], []
        text_batch, uttid_batch = [], []
        mel_start_end = []
        for idx in range(len(batch)):
            wav = batch[idx]['wav']
            mel = batch[idx]['mel']
            mel2ph = batch[idx]['mel2ph']
            source = batch[idx]['source']
            pitch = batch[idx]['pitch']
            uv = batch[idx]['uv']
            energy = batch[idx]['energy']
            self._assert_ready_for_upsampling(wav, mel, self.hop_size, 0)
            if len(mel) - 2 * self.aux_context_window > self.batch_max_frames:
                # randomly pickup with the batch_max_steps length of the part
                interval_start = self.aux_context_window
                batch_max_frames = self.batch_max_frames if self.batch_max_frames != 0 else len(
                    mel) - 2 * self.aux_context_window - 1
                batch_max_steps = batch_max_frames * self.hop_size

                interval_end = len(mel) - batch_max_frames - self.aux_context_window
                start_frame = np.random.randint(interval_start, interval_end)
                start_step = start_frame * self.hop_size
                wav = wav[start_step: start_step + batch_max_steps]
                # mel = mel[start_frame - self.aux_context_window:
                #           start_frame + self.aux_context_window + self.batch_max_frames]
                mel2ph_ = mel2ph[start_frame - self.aux_context_window:
                                 start_frame + self.aux_context_window + batch_max_frames]
                mel_start_end.append([
                    start_frame - self.aux_context_window,
                    start_frame + self.aux_context_window + batch_max_frames
                ])
                self._assert_ready_for_upsampling(wav, mel2ph_, self.hop_size, self.aux_context_window)
            else:
                # print(f"Removed short sample from batch (length={len(x)}).")
                continue
            wav_batch += [wav]
            mel_batch += [mel]
            p_batch += [pitch]
            uv_batch += [uv]
            mel2ph_batch += [mel2ph]
            source_batch += [source]
            energy_batch += [energy]
            text_batch += [batch[idx]['text']]
            uttid_batch += [batch[idx]['utt_id']]

        # convert each batch to tensor, asuume that each item in batch has the same length
        wav_batch = utils.collate_1d(wav_batch)
        mel_batch = utils.collate_2d(mel_batch)
        p_batch = utils.collate_1d(p_batch, -200)
        uv_batch = utils.collate_1d(uv_batch)
        mel2ph_batch = utils.collate_1d(mel2ph_batch)
        source_batch = utils.collate_1d(source_batch)
        energy_batch = utils.collate_1d(energy_batch)
        src_lengths = torch.LongTensor([s.numel() for s in source_batch])

        return {
            'wavs': wav_batch[:, None, :],
            'mels': mel_batch,
            'pitch': p_batch,
            'uv': uv_batch,
            'energy': energy_batch,
            'mel2ph': mel2ph_batch,
            'src_tokens': source_batch,
            'src_lengths': src_lengths,
            'text': text_batch,
            'utt_id': uttid_batch,
            'mel_start_end': mel_start_end,
        }

    @staticmethod
    def _assert_ready_for_upsampling(x, c, hop_size, context_window):
        """Assert the audio and feature lengths are correctly adjusted for upsamping."""
        assert len(x) == (len(c) - 2 * context_window) * hop_size, (len(x), (len(c) - 2 * context_window) * hop_size)

    @property
    def num_workers(self):
        return 1


class FastSpeech2sTask(PwgTask, FastSpeech2Task):
    def __init__(self):
        super(FastSpeech2sTask, self).__init__()
        self.denoiser = None

    @data_loader
    def train_dataloader(self):
        train_dataset = FastSpeech2sDataset(hparams['data_dir'], 'train', shuffle=True)
        return self.build_dataloader(train_dataset, True, self.max_tokens, self.max_sentences,
                                     endless=hparams['endless_ds'])

    @data_loader
    def val_dataloader(self):
        valid_dataset = FastSpeech2sDataset(hparams['data_dir'], 'valid', shuffle=True)
        return self.build_dataloader(valid_dataset, False, self.max_eval_tokens, self.max_eval_sentences)

    @data_loader
    def test_dataloader(self):
        test_dataset = FastSpeech2sDataset(hparams['data_dir'], 'test', shuffle=True)
        return self.build_dataloader(test_dataset, False, self.max_eval_tokens, self.max_eval_sentences)

    def build_dataloader(self, dataset, shuffle, max_tokens=None, max_sentences=None,
                         required_batch_size_multiple=-1, endless=False):
        return FastSpeech2Task.build_dataloader(
            self, dataset, shuffle, max_tokens, max_sentences,
            required_batch_size_multiple, endless)

    def configure_optimizers(self):
        set_hparams()
        self.build_model()
        optimizer_gen = torch.optim.AdamW(
            self.model_gen.parameters(),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        optimizer_disc = RAdam(self.model_disc.parameters(),
                               **hparams["discriminator_optimizer_params"])
        self.scheduler = self.build_scheduler({'gen': optimizer_gen, 'disc': optimizer_disc})
        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return {
            "gen": RSQRTSchedule(optimizer['gen']),
            "disc": torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer["disc"],
                **hparams["discriminator_scheduler_params"]),
        }

    def build_model(self):
        arch = self.arch
        self.model_gen = FastSpeech2s(arch, self.phone_encoder)
        print(self.model_gen)
        self.model_disc = ParallelWaveGANDiscriminator()

    def _training_step(self, sample, batch_idx, optimizer_idx):
        mel2ph = sample['mel2ph']
        y = sample['wavs']
        pitch = sample['pitch']
        uv = sample['uv']
        energy = sample['energy']
        src_tokens = sample['src_tokens']
        mel_start_end = sample['mel_start_end']
        mels = sample['mels']

        losses = {}
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            # calculate generator loss
            gen_out = self.model_gen(src_tokens, mel2ph, mel_start_end, pitch=pitch, uv=uv, energy=energy)
            y_ = gen_out['wav']
            y, y_ = y.squeeze(1), y_.squeeze(1)
            sc_loss, mag_loss = self.stft_loss(y_, y)

            losses['sc'] = sc_loss
            losses['mag'] = mag_loss
            if hparams['use_aux_mel_loss']:
                losses['mel'] = self.mse_loss(gen_out['mel_out'], mels) * 0.1
            losses["dur"] = self.dur_loss(gen_out['dur'], mel2ph, src_tokens)
            if hparams['use_pitch_embed']:
                losses['uv'], losses['pitch'] = self.pitch_loss(gen_out['pitch_logits'], pitch, uv)
            if hparams['use_energy_embed']:
                losses['energy'] = self.energy_loss(gen_out['energy_pred'], energy)
            total_loss = sum(losses.values())

            if self.global_step > hparams["discriminator_train_start_steps"]:
                # keep compatibility
                total_loss *= hparams.get("lambda_aux_after_introduce_adv_loss", 1.0)
                p_ = self.model_disc(y_.unsqueeze(1))
                if not isinstance(p_, list):
                    # for standard discriminator
                    adv_loss = self.mse_loss(p_, p_.new_ones(p_.size()))
                    losses['adv'] = adv_loss
                else:
                    # for multi-scale discriminator
                    adv_loss = 0.0
                    for i in range(len(p_)):
                        adv_loss += self.mse_loss(
                            p_[i][-1], p_[i][-1].new_ones(p_[i][-1].size()))
                    adv_loss /= (i + 1)
                    losses['adv'] = adv_loss

                    # feature matching loss
                    if hparams["use_feat_match_loss"]:
                        # no need to track gradients
                        with torch.no_grad():
                            p = self.model_disc(y.unsqueeze(1))
                        fm_loss = 0.0
                        for i in range(len(p_)):
                            for j in range(len(p_[i]) - 1):
                                fm_loss += self.l1_loss_fn(p_[i][j], p[i][j].detach())
                        fm_loss /= (i + 1) * (j + 1)
                        losses["fm"] = fm_loss
                        adv_loss += hparams["lambda_feat_match"] * fm_loss
                total_loss += hparams["lambda_adv"] * adv_loss
        else:
            #######################
            #    Discriminator    #
            #######################
            if self.global_step > hparams["discriminator_train_start_steps"]:
                # calculate discriminator loss
                with torch.no_grad():
                    gen_out = self.model_gen(src_tokens, mel2ph, mel_start_end, pitch=pitch, uv=uv, energy=energy)
                y_pred = gen_out['wav']
                p = self.model_disc(y)
                p_ = self.model_disc(y_pred)
                if not isinstance(p, list):
                    # for standard discriminator
                    real_loss = self.mse_loss_fn(p, p.new_ones(p.size()))
                    fake_loss = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
                else:
                    # for multi-scale discriminator
                    real_loss = 0.0
                    fake_loss = 0.0
                    for i in range(len(p)):
                        real_loss += self.mse_loss_fn(p[i][-1], p[i][-1].new_ones(p[i][-1].size()))
                        fake_loss += self.mse_loss_fn(p_[i][-1], p_[i][-1].new_zeros(p_[i][-1].size()))
                    real_loss /= (i + 1)
                    fake_loss /= (i + 1)
                losses["real"] = real_loss
                losses["fake"] = fake_loss
                total_loss = real_loss + fake_loss
            else:
                # skip disc training
                return None
        return total_loss, losses

    def validation_step(self, sample, batch_idx):
        mel2ph = sample['mel2ph']
        y = sample['wavs']
        pitch = sample['pitch']
        uv = sample['uv']
        energy = sample['energy']
        src_tokens = sample['src_tokens']
        mel_start_end = sample['mel_start_end']
        mels = sample['mels']
        losses = {}

        gen_out = self.model_gen(src_tokens, mel2ph, mel_start_end, pitch, uv, energy=energy)
        y_ = gen_out['wav']
        y, y_ = y.squeeze(1), y_.squeeze(1)
        sc_loss, mag_loss = self.stft_loss(y_, y)

        losses["sc"] = sc_loss
        losses["mag"] = mag_loss
        if hparams['use_aux_mel_loss']:
            losses['mel'] = self.mse_loss(gen_out['mel_out'], mels)
        losses["dur"] = self.dur_loss(gen_out['dur'], mel2ph, src_tokens)
        if hparams['use_pitch_embed']:
            losses['uv'], losses['pitch'] = self.pitch_loss(gen_out['pitch_logits'], pitch, uv)
        if hparams['use_energy_embed']:
            losses['energy'] = self.energy_loss(gen_out['energy_pred'], energy)

        if dist.is_initialized():
            losses = utils.reduce_tensors(losses)
        losses = utils.tensors_to_scalars(losses)

        # for idx, (wav_pred, wav_gt) in enumerate(zip(y_, y)):
        #     wav_gt = wav_gt / wav_gt.abs().max()
        #     wav_pred = wav_pred / wav_pred.abs().max()
        #     self.logger.experiment.add_audio(f'valid/wav_{batch_idx}_{idx}_gt', wav_gt,
        #                                      self.global_step, sample_rate=22050)
        #     self.logger.experiment.add_audio(f'valid/wav_{batch_idx}_{idx}_pred', wav_pred,
        #                                      self.global_step, sample_rate=22050)
        return losses

    def test_step(self, sample, batch_idx):
        src_tokens = sample['src_tokens']
        if hparams['profile_infer']:
            mel2ph = sample['mel2ph']
            energy = sample['energy']
            pitch = sample['pitch']
            uv = sample['uv']
        else:
            mel2ph = None
            pitch = None
            energy = None
            uv = None
        loss_output = {}
        with utils.Timer('fs2', print_time=hparams['profile_infer']):
            outputs = self.model_gen(src_tokens, mel2ph, pitch=pitch, uv=uv, energy=energy, infer=True)

        if hparams['profile_infer']:
            return {}

        if hparams['gen_wav_denoise']:
            mel2ph_pred = outputs['mel2ph']
            input_noise = torch.ones_like(src_tokens[:, :1]).long() * 3
            mel2ph_noise = torch.ones_like(mel2ph_pred)
            mel2ph_noise = mel2ph_noise * (mel2ph_pred > 0).long()
            mel2ph_noise = mel2ph_noise[:, :40]
            pitch_noise = torch.zeros_like(mel2ph_pred).float()[:, :40]
            uv_noise = torch.ones_like(mel2ph_pred)[:, :40]
            noise_wavs = self.model_gen(input_noise, mel2ph_noise, pitch=pitch_noise, uv=uv_noise)['wav']
        else:
            noise_wavs = [None for _ in outputs['wav']]

        gen_dir = os.path.join(hparams['work_dir'], f'generated_{self.trainer.global_step}')
        os.makedirs(gen_dir, exist_ok=True)
        if self.denoiser is None:
            self.denoiser = GeneralDenoiser()

        for idx, (wav_pred, noise_wav, wav_gt, utt_id, text) in enumerate(
                zip(outputs['wav'], noise_wavs, sample['wavs'], sample['utt_id'], sample['text'])):
            if hparams['gen_wav_denoise']:
                wav_pred = self.denoiser(wav_pred.view(-1)[None, :], noise_wav)[0, 0].cpu().numpy()
                wav_gt = self.denoiser(wav_gt.view(-1)[None, :], noise_wav)[0, 0].cpu().numpy()
            else:
                wav_pred = wav_pred.view(-1).cpu().numpy()
                wav_gt = wav_gt.view(-1).cpu().numpy()
            audio.save_wav(wav_gt, f'{gen_dir}/[G][{utt_id}]{text.replace(":", "%3A")}.wav',
                           22050, norm=False)
            audio.save_wav(wav_pred, f'{gen_dir}/[P][{utt_id}]{text.replace(":", "%3A")}.wav',
                           22050, norm=False)
        return loss_output


if __name__ == '__main__':
    FastSpeech2sTask.start()
