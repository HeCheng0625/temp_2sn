from modules.operations import *
from modules.transformer_tts import TransformerEncoder
from modules.tts_modules import FastspeechDecoder, DurationPredictor, LengthRegulator, PitchPredictor, EnergyPredictor
from utils.world_utils import f0_to_coarse_torch, restore_pitch


class FastSpeech2_dis(nn.Module):
    def __init__(self, arch, dictionary, out_dims=None):
        super().__init__()
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        if isinstance(arch, str):
            self.arch = list(map(int, arch.strip().split()))
        else:
            assert isinstance(arch, (list, tuple))
            self.arch = arch
        self.enc_layers = hparams['enc_layers']
        self.dec_layers = hparams['dec_layers']
        self.enc_arch = self.arch[:self.enc_layers]
        self.dec_arch = self.arch[self.enc_layers:self.enc_layers + self.dec_layers]
        self.hidden_size = hparams['hidden_size']
        self.encoder_embed_tokens = nn.Embedding(len(self.dictionary), self.hidden_size, self.padding_idx)
        self.encoder = TransformerEncoder(self.enc_arch, self.encoder_embed_tokens)
        self.decoder = FastspeechDecoder(self.dec_arch) if hparams['dec_layers'] > 0 else None
        self.parallel_predict = hparams.get("parallel_predict", False)
        self.vqemb_predict = hparams.get("vqemb_predict", False)
        if self.vqemb_predict:
            assert not self.parallel_predict
            self.mel_out = Linear(self.hidden_size,
                                    hparams['vqemb_size'],
                                    bias=False)
        
        else:
            if self.parallel_predict:
                self.mel_out_list = torch.nn.ModuleList([Linear(self.hidden_size,
                                    hparams['codebook_size'] + 1,
                                    bias=False) for _ in range(hparams["predict_first_ndim"])])
            else:
                out_dims = hparams['audio_num_mel_bins'] * hparams['codebook_size'] + 1
                assert self.padding_idx == 0
                self.mel_out = Linear(self.hidden_size,
                                    out_dims,
                                    bias=False)
        if hparams['use_spk_id']:
            self.spk_embed_proj = nn.Embedding(hparams['num_spk'], self.hidden_size)
        else:
            self.spk_embed_proj = Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dur_predictor = DurationPredictor(
            self.hidden_size,
            n_layers=hparams['duration_layers'],
            n_chans=hparams['predictor_hidden'],
            dropout_rate=0.5, padding=hparams['ffn_padding'],
            kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        self.predict_first_ndim = hparams["predict_first_ndim"]
        if hparams['use_pitch_embed']:
            self.pitch_embed = nn.Embedding(300, self.hidden_size, self.padding_idx)
            self.pitch_predictor = PitchPredictor(
                self.hidden_size, n_layers=hparams['pitch_layers'], n_chans=hparams['predictor_hidden'], dropout_rate=0.5,
                padding=hparams['ffn_padding'], odim=2)
            self.pitch_do = nn.Dropout(0.5)
        # if hparams['use_energy_embed']:
        #     self.energy_predictor = EnergyPredictor(
        #         self.hidden_size, n_chans=hparams['predictor_hidden'], dropout_rate=0.5, odim=1,
        #         padding=hparams['ffn_padding'])
        #     self.energy_embed = nn.Embedding(256, self.hidden_size, self.padding_idx)
        #     self.energy_do = nn.Dropout(0.5)
        
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
            
             
    def forward_hack(self, src_tokens_ref, src_tokens_in, mel2ph, spk_embed=None, spk_mask=None,
                ref_mels=None, pitch=None, uv=None, energy=None, skip_decoder=False, use_pred_pitch=False):
        """

        :param src_tokens: [B, T]
        :param mel2ph:
        :param spk_embed:
        :param ref_mels:
        :return: {
            'mel_out': [B, T_s, 80], 'dur': [B, T_t],
            'w_st_pred': [heads, B, tokens], 'w_st': [heads, B, tokens],
            'encoder_out_noref': [B, T_t, H]
        }
        """
        
        ref_tokens = src_tokens_ref.shape[1]
        in_tokens = src_tokens_in.shape[1]
        
        for_enc_tokens = torch.cat([src_tokens_ref[:, :-2], src_tokens_in[:, 2:]], dim=1)
        
        ret = {}
        encoder_outputs = self.encoder(for_enc_tokens)
        encoder_out = encoder_outputs['encoder_out']  # [T, B, C]
        src_nonpadding = (for_enc_tokens > 0).float().permute(1, 0)[:, :, None]
        if hparams['use_spk_embed'] and spk_embed is not None:
            # spk_embed = self.spk_embed_proj(spk_embed)[None, :, :]
            encoder_out += spk_embed[None, :, :]
        encoder_out = encoder_out * src_nonpadding  # [T, B, C]

        dur_input = encoder_out.transpose(0, 1)
        if hparams['predictor_sg']:
            dur_input = dur_input.detach()
            
        
        dur = self.dur_predictor.inference(dur_input, for_enc_tokens == 0, condition=spk_embed, condition_mask=spk_mask)
        
        dur_in = dur[:, ref_tokens-2:]
        
        in_mel2ph = self.length_regulator(dur_in, (src_tokens_in[:, 2:] != 0).sum(-1))[..., 0]
        
        in_mel2ph += (ref_tokens - 2)
        
        # print("0------------")
        # print(src_tokens_ref.shape, src_tokens_in.shape, for_enc_tokens.shape, encoder_out.shape)
        
        # print(dur.shape, dur_in.shape)
        # print(mel2ph.shape, in_mel2ph.shape)
        # print(mel2ph)
        # print(in_mel2ph)
        
        # mel2ph = self.length_regulator(dur, (for_enc_tokens != 0).sum(-1))[..., 0]
        # print(mel2ph.shape)
        # exit(0)
        
        ret['mel2ph'] = mel2ph = torch.cat([mel2ph, in_mel2ph], dim=1)
        
            
        # if mel2ph is None:
        #     dur = self.dur_predictor.inference(dur_input, src_tokens == 0)
        #     if not hparams['sep_dur_loss']:
        #         dur[src_tokens == self.dictionary.seg()] = 0
        #     ret['mel2ph'] = mel2ph = self.length_regulator(dur, (src_tokens != 0).sum(-1))[..., 0]
        # else:
        #     ret['dur'] = self.dur_predictor(dur_input, src_tokens == 0)

        # expand encoder out to make decoder inputs
        decoder_inp = F.pad(encoder_out, [0, 0, 0, 0, 1, 0])  #[T, B, C]
        mel2ph_unrepeat = mel2ph
        mel2ph_unrepeat_ = mel2ph.permute([1, 0])[..., None].repeat([1, 1, encoder_out.shape[-1]]).contiguous()  #[T', B, C]
        decoder_inp_unrepeat = torch.gather(decoder_inp, 0, mel2ph_unrepeat_).transpose(0, 1)  # [B, T, H]
        
        if self.parallel_predict or self.vqemb_predict:
            ret['decoder_inp_origin'] = decoder_inp_origin = decoder_inp = decoder_inp_unrepeat
        else:
            mel2ph = mel2ph[:, :, None].repeat(1, 1, self.predict_first_ndim).reshape(mel2ph.shape[0], mel2ph.shape[1] * self.predict_first_ndim)
            mel2ph_ = mel2ph.permute([1, 0])[..., None].repeat([1, 1, encoder_out.shape[-1]]).contiguous()  #[T', B, C]
            decoder_inp = torch.gather(decoder_inp, 0, mel2ph_).transpose(0, 1)  # [B, T, H]
            ret['decoder_inp_origin'] = decoder_inp_origin = decoder_inp



        # add pitch embed
        if hparams['use_pitch_embed']:
            pitch_emb = self.add_pitch(decoder_inp_unrepeat, pitch, uv, mel2ph_unrepeat, ret, use_pred_pitch=use_pred_pitch)
            if not self.parallel_predict and not self.vqemb_predict:
                pitch_emb = pitch_emb[:, :, None, :].repeat(1, 1, self.predict_first_ndim, 1).reshape(pitch_emb.shape[0], -1, pitch_emb.shape[-1])

            decoder_inp = decoder_inp + pitch_emb
        # add energy embed
        # if hparams['use_energy_embed']:
        #     decoder_inp = decoder_inp + self.add_energy(decoder_inp_origin, energy, ret)

        decoder_inp = decoder_inp * (mel2ph != 0).float()[:, :, None]
        ret['decoder_inp'] = decoder_inp

        if skip_decoder:
            ret['mel_out'] = decoder_inp
            return ret
        x = decoder_inp
        if hparams['dec_layers'] > 0:
            x = self.decoder(x)
        if self.parallel_predict and not self.vqemb_predict:
            result_list = []
            for i in range(hparams["predict_first_ndim"]):
                result_list.append((self.mel_out_list[i](x) * (mel2ph != 0).float()[:, :, None])[:, :, None, :])
            x = torch.cat(result_list, dim=2).reshape(x.shape[0], -1, result_list[0].shape[-1])
        else:
            x = self.mel_out(x)
            x = x * (mel2ph != 0).float()[:, :, None]
        ret['mel_out'] = x
        return ret

        

    def forward(self, src_tokens, mel2ph, spk_embed=None, spk_mask=None,
                ref_mels=None, pitch=None, uv=None, energy=None, skip_decoder=False, use_pred_pitch=False):
        """

        :param src_tokens: [B, T]
        :param mel2ph:
        :param spk_embed:
        :param ref_mels:
        :return: {
            'mel_out': [B, T_s, 80], 'dur': [B, T_t],
            'w_st_pred': [heads, B, tokens], 'w_st': [heads, B, tokens],
            'encoder_out_noref': [B, T_t, H]
        }
        """
        ret = {}
        encoder_outputs = self.encoder(src_tokens)
        encoder_out = encoder_outputs['encoder_out']  # [T, B, C]
        src_nonpadding = (src_tokens > 0).float().permute(1, 0)[:, :, None]
        if hparams['use_spk_embed'] and spk_embed is not None and hparams['use_ref_enc']:
            spk_embed = self.spk_embed_proj(spk_embed)[None, :, :]
            encoder_out += spk_embed
        encoder_out = encoder_out * src_nonpadding  # [T, B, C]
        print('encoder_out', torch.isnan(encoder_out).any())
        dur_input = encoder_out.transpose(0, 1)
        if hparams['predictor_sg']:
            dur_input = dur_input.detach()
        if mel2ph is None:
            dur = self.dur_predictor.inference(dur_input, src_tokens == 0, condition=spk_embed, condition_mask=spk_mask)
            if not hparams['sep_dur_loss']:
                dur[src_tokens == self.dictionary.seg()] = 0
            ret['mel2ph'] = mel2ph = self.length_regulator(dur, (src_tokens != 0).sum(-1))[..., 0]
        else:
            ret['dur'] = self.dur_predictor(dur_input, src_tokens == 0, condition=spk_embed, condition_mask=spk_mask)
        print('dur', torch.isnan(ret['dur']).any())
        # expand encoder out to make decoder inputs
        decoder_inp = F.pad(encoder_out, [0, 0, 0, 0, 1, 0])  #[T, B, C]
        mel2ph_unrepeat = mel2ph
        mel2ph_unrepeat_ = mel2ph.permute([1, 0])[..., None].repeat([1, 1, encoder_out.shape[-1]]).contiguous()  #[T', B, C]
        decoder_inp_unrepeat = torch.gather(decoder_inp, 0, mel2ph_unrepeat_).transpose(0, 1)  # [B, T, H]
        
        if self.parallel_predict or self.vqemb_predict:
            ret['decoder_inp_origin'] = decoder_inp_origin = decoder_inp = decoder_inp_unrepeat
        else:
            mel2ph = mel2ph[:, :, None].repeat(1, 1, self.predict_first_ndim).reshape(mel2ph.shape[0], mel2ph.shape[1] * self.predict_first_ndim)
            mel2ph_ = mel2ph.permute([1, 0])[..., None].repeat([1, 1, encoder_out.shape[-1]]).contiguous()  #[T', B, C]
            decoder_inp = torch.gather(decoder_inp, 0, mel2ph_).transpose(0, 1)  # [B, T, H]
            ret['decoder_inp_origin'] = decoder_inp_origin = decoder_inp



        # add pitch embed
        if hparams['use_pitch_embed']:
            pitch_emb = self.add_pitch(decoder_inp_unrepeat, pitch, uv, mel2ph_unrepeat, ret, use_pred_pitch=use_pred_pitch, condition=spk_embed, condition_mask=spk_mask)
            if not self.parallel_predict and not self.vqemb_predict:
                pitch_emb = pitch_emb[:, :, None, :].repeat(1, 1, self.predict_first_ndim, 1).reshape(pitch_emb.shape[0], -1, pitch_emb.shape[-1])

            decoder_inp = decoder_inp + pitch_emb
        
        print('pitch', torch.isnan(ret['pitch_logits']).any())
        # add energy embed
        # if hparams['use_energy_embed']:
        #     decoder_inp = decoder_inp + self.add_energy(decoder_inp_origin, energy, ret)

        decoder_inp = decoder_inp * (mel2ph != 0).float()[:, :, None]
        ret['decoder_inp'] = decoder_inp

        if skip_decoder:
            ret['mel_out'] = decoder_inp
            return ret
        x = decoder_inp
        if hparams['dec_layers'] > 0:
            x = self.decoder(x)
        if self.parallel_predict and not self.vqemb_predict:
            result_list = []
            for i in range(hparams["predict_first_ndim"]):
                result_list.append((self.mel_out_list[i](x) * (mel2ph != 0).float()[:, :, None])[:, :, None, :])
            x = torch.cat(result_list, dim=2).reshape(x.shape[0], -1, result_list[0].shape[-1])
        else:
            x = self.mel_out(x)
            x = x * (mel2ph != 0).float()[:, :, None]
        ret['mel_out'] = x
        return ret

    def decode_with_pred_pitch(self, decoder_inp, mel2ph):
        #if hparams['use_ref_enc']:
        #    assert False
        pitch_embed = self.add_pitch(decoder_inp, None, None, mel2ph, {})
        decoder_inp = decoder_inp + self.pitch_do(pitch_embed)
        decoder_inp = decoder_inp * (mel2ph != 0).float()[:, :, None]
        x = decoder_inp
        x = self.decoder(x)
        x = self.mel_out(x)
        x = x * (mel2ph != 0).float()[:, :, None]
        return x

    # run other modules
    def add_energy(self, decoder_inp, energy, ret):
        if hparams['predictor_sg']:
            decoder_inp = decoder_inp.detach()
        ret['energy_pred'] = energy_pred = self.energy_predictor(decoder_inp)[:, :, 0]
        if energy is None:
            energy = energy_pred
        energy = torch.clamp(energy * 256 // 4, max=255).long()
        energy_embed = self.energy_embed(energy)
        return energy_embed

    def add_pitch(self, decoder_inp_origin, pitch, uv, mel2ph, ret, use_pred_pitch=False, condition=None, condition_mask=None):
        pp_inp = decoder_inp_origin
        if hparams['predictor_sg']:
            pp_inp = pp_inp.detach()
        ret['pitch_logits'] = pitch_logits = self.pitch_predictor(pp_inp, condition=condition, condition_mask=condition_mask)
        if pitch is not None and not use_pred_pitch:  # train
            pitch_padding = pitch == -200
            pitch_restore = restore_pitch(pitch, uv if hparams['use_uv'] else None, hparams,
                                          pitch_padding=pitch_padding)
            ret['pitch'] = pitch_restore
            pitch_restore = f0_to_coarse_torch(pitch_restore)
            pitch_embed = self.pitch_embed(pitch_restore)
        else:  # test
            pitch_padding = (mel2ph == 0)
            pitch = pitch_logits[:, :, 0]
            uv = pitch_logits[:, :, 1] > 0
            if not hparams['use_uv']:
                uv = pitch < -3.5
            pitch_restore = restore_pitch(pitch, uv, hparams, pitch_padding=pitch_padding)
            ret['pitch'] = pitch_restore
            pitch_restore = f0_to_coarse_torch(pitch_restore)
            pitch_embed = self.pitch_embed(pitch_restore)
        return self.pitch_do(pitch_embed)
