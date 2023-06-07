from modules.operations import *
from modules.transformer_tts import TransformerEncoder
from modules.tts_modules import FastspeechDecoder, DurationPredictor, LengthRegulator, PitchPredictor, EnergyPredictor
from utils.world_utils import f0_to_coarse_torch, restore_pitch, process_f0_nointerp


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
        self.encoder = TransformerEncoder(self.enc_arch, self.encoder_embed_tokens, use_cln=hparams["prior_use_cln"])
        self.decoder = FastspeechDecoder(self.dec_arch, use_cln=hparams["prior_use_cln"]) if hparams['dec_layers'] > 0 else None
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
        
        
        self.duration_predictor_type = hparams.get("duration_predictor_type", "conv")
        
        if self.duration_predictor_type == "conv":
            self.dur_predictor = DurationPredictor(
                self.hidden_size,
                n_layers=hparams['duration_layers'],
                n_chans=hparams['predictor_hidden'],
                dropout_rate=0.5, padding=hparams['ffn_padding'],
                kernel_size=hparams['dur_predictor_kernel'])
        elif self.duration_predictor_type == "ar_transformer":
            from .ar_duration_predictor import ARDurationPredictor
            self.dur_predictor = ARDurationPredictor(
                arch=hparams['duration_transformer_arch'],
                idim=self.hidden_size,
                n_chans=hparams['predictor_hidden'],
                dropout_rate=0.3,
            )
            
            pass
        else:
            raise NotImplementedError("Unknown duration predictor type: {}".format(self.duration_predictor_type))
        
        
        
        
        self.length_regulator = LengthRegulator()
        self.predict_first_ndim = hparams["predict_first_ndim"]
        
        
        if hparams['use_pitch_embed']:
            self.pitch_embed = nn.Embedding(300, self.hidden_size, self.padding_idx)
            nn.init.kaiming_normal_(self.pitch_embed.weight, mode='fan_out', nonlinearity='relu')
            self.pitch_do = nn.Dropout(0.5)
            
            self.pitch_predictor_type = hparams.get("pitch_predictor_type", "conv")
            if self.pitch_predictor_type == "conv" or self.pitch_predictor_type == "conv_phone":
                self.pitch_predictor = PitchPredictor(
                    self.hidden_size, n_layers=hparams['pitch_layers'], n_chans=hparams['predictor_hidden'], dropout_rate=0.5,
                    padding=hparams['ffn_padding'], odim=2)
            elif self.pitch_predictor_type == "ar_transformer":
                from .ar_pitch_predictor import ARPitchPredictor
                self.dur_predictor = ARPitchPredictor(
                    arch=hparams['pitch_transformer_arch'],
                    idim=self.hidden_size,
                    n_chans=hparams['predictor_hidden'],
                    dropout_rate=0.3,
                )
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
            
            
    def forward_hack(self, src_tokens_ref, src_tokens_in, mel2ph, spk_embed=None,
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
        ref_remove_tokens= 4  # 'br1', '-sil-', 'br0', '~'  
        for_enc_tokens = torch.cat([src_tokens_ref[:, :-ref_remove_tokens], src_tokens_in[:, 1:]], dim=1)
        clip_len = (mel2ph <= ref_tokens-ref_remove_tokens).sum().item() # 1-idx 
        # print(ref_tokens, clip_len, src_tokens_ref.shape)
        if pitch is not None:
            print(pitch.shape)
            pitch = pitch[:, :clip_len]
        if uv is not None:
            print(uv.shape)
            uv= uv[:, :clip_len]
            
        mel2ph = mel2ph[:, :clip_len]
        # print(ref_mels.shape)
        # print(mel2ph)
        ret = {}
        ret['ref_mels'] = ref_mels[:, :clip_len, :]
        # exit()
        encoder_outputs = self.encoder(for_enc_tokens, condition=spk_embed)
        encoder_out = encoder_outputs['encoder_out']  # [T, B, C]
        src_nonpadding = (for_enc_tokens > 0).to(encoder_out.dtype).permute(1, 0)[:, :, None]
        if hparams['use_spk_embed'] and spk_embed is not None:
            # spk_embed = self.spk_embed_proj(spk_embed)[None, :, :]
            encoder_out += spk_embed[None, :, :]
        encoder_out = encoder_out * src_nonpadding  # [T, B, C]

        dur_input = encoder_out.transpose(0, 1)
        if hparams['predictor_sg']:
            dur_input = dur_input.detach()
            
        
        dur = self.dur_predictor.inference(dur_input, for_enc_tokens == 0, condition=spk_embed)
        
        dur_in = dur[:, ref_tokens -  ref_remove_tokens:]
        alpha=1.0
        if hparams['ref_norm']:
            B = dur.shape[0]
            dur_gt = mel2ph.new_zeros(B, ref_tokens - ref_remove_tokens + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
            dur_gt = dur_gt[:, 1:]
            dur[:, :ref_tokens-ref_remove_tokens] = dur_gt
            mask = (dur_in > 0) 
            mask_gt = (dur_gt > 0) 
            dur_in[dur_in > 20] = 20
            dur_gt[dur_gt > 20] = 20
            pred_mean = dur_in[mask].float().mean()
            pred_std = dur_in[mask].float().std()
            ref_mean = dur_gt[mask_gt].float().mean()
            ref_std = dur_gt[mask_gt].float().std()
            lmabda = 0.4
            mean = lmabda * pred_mean + (1-lmabda) * ref_mean
            std = lmabda * pred_std + (1-lmabda) * ref_std
            # mean = pred_mean
            # std = pred_std
            print('dur infer:', pred_mean, pred_std)
            print('dur ref:', ref_mean, ref_std)
            print('dur used:', mean, std)
            dur_in_float = dur_in.float()
            
            dur_in_float[mask] = (dur_in_float[mask] - pred_mean) / pred_std
            dur_in_float[mask] = dur_in_float[mask] * std + mean
            # exit()
            dur_in_float[dur_in_float < 0] = 1
            dur_in_float[dur_in_float > 20] = 20
            print(dur_in_float)
            dur_in = dur_in_float.long()
            alpha=1.2

            
        
        
        in_mel2ph = self.length_regulator(dur_in, (src_tokens_in[:, 1:] != 0).sum(-1), alpha=alpha)[..., 0]
        
        in_mel2ph += (ref_tokens - ref_remove_tokens)
        
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
            pitch_emb = self.add_pitch(decoder_inp_unrepeat, pitch, uv, mel2ph_unrepeat, ret, use_pred_pitch=use_pred_pitch, condition=spk_embed)
            if not self.parallel_predict and not self.vqemb_predict:
                pitch_emb = pitch_emb[:, :, None, :].repeat(1, 1, self.predict_first_ndim, 1).reshape(pitch_emb.shape[0], -1, pitch_emb.shape[-1])
            #print('pitch norm', torch.norm(pitch_emb), 'decoder_inp norm', torch.norm(decoder_inp))
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
            x = self.decoder(x, condition=spk_embed)
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
        encoder_outputs = self.encoder(src_tokens, condition=spk_embed)
        encoder_out = encoder_outputs['encoder_out']  # [T, B, C]

        src_nonpadding = (src_tokens > 0).to(encoder_out.dtype).permute(1, 0)[:, :, None]
        if hparams['use_spk_embed'] and spk_embed is not None and spk_embed.dim() == 2:
            spk = self.spk_embed_proj(spk_embed)
            encoder_out += spk[None, :, :]
        encoder_out = encoder_out * src_nonpadding  # [T, B, C]

        dur_input = encoder_out.transpose(0, 1)
        if hparams['predictor_sg']:
            dur_input = dur_input.detach()
        if mel2ph is None:
            if self.duration_predictor_type == "conv":
                dur = self.dur_predictor.inference(dur_input, src_tokens == 0, condition=spk_embed, condition_mask=spk_mask)
            elif self.duration_predictor_type == "ar_transformer":
                dur = self.dur_predictor.inference(dur_input, src_tokens == 0, condition=spk_embed)
            else:
                raise NotImplementedError("duration predictor type not supported")
                
            if not hparams['sep_dur_loss']:
                dur[src_tokens == self.dictionary.seg()] = 0
            ret['mel2ph'] = mel2ph = self.length_regulator(dur, (src_tokens != 0).sum(-1))[..., 0]
            ret['dur'] = dur
        else:
            if self.duration_predictor_type == "conv":
                ret['dur'] = self.dur_predictor(dur_input, src_tokens == 0, condition=spk_embed, condition_mask=spk_mask)
            elif self.duration_predictor_type == "ar_transformer":
                B, T_t = src_tokens.shape
                dur_gt = mel2ph.new_zeros(B, T_t + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
                
                dur_gt = dur_gt[:, 1:]

                ret['dur'], ret["dur-gt"] = self.dur_predictor(dur_input, src_tokens == 0, dur_gt, condition=spk_embed)
        
        mel2ph_unrepeat = mel2ph
        mel2ph_unrepeat_ = mel2ph.permute([1, 0])[..., None].repeat([1, 1, encoder_out.shape[-1]]).contiguous()  #[T', B, C]
       
        # add phone-level pitch embed
      
        if hparams['use_pitch_embed'] and self.pitch_predictor_type == "conv_phone":
            pitch_emb = self.add_pitch_phone(encoder_out, pitch, uv, mel2ph_unrepeat, ret, use_pred_pitch=use_pred_pitch, condition=spk_embed, mask=(src_tokens == 0))
            if not self.parallel_predict and not self.vqemb_predict:
                pitch_emb = pitch_emb[:, :, None, :].repeat(1, 1, self.predict_first_ndim, 1).reshape(pitch_emb.shape[0], -1, pitch_emb.shape[-1])
            #print('pitch norm', torch.norm(pitch_emb), 'decoder_inp norm', torch.norm(decoder_inp))
            encoder_out = encoder_out + pitch_emb.transpose(0, 1)
        # expand encoder out to make decoder inputs

        decoder_inp = F.pad(encoder_out, [0, 0, 0, 0, 1, 0])  #[T, B, C]        
        decoder_inp_unrepeat = torch.gather(decoder_inp, 0, mel2ph_unrepeat_).transpose(0, 1)  # [B, T, H]
        
        if self.parallel_predict or self.vqemb_predict:
            ret['decoder_inp_origin'] = decoder_inp_origin = decoder_inp = decoder_inp_unrepeat
        else:
            mel2ph = mel2ph[:, :, None].repeat(1, 1, self.predict_first_ndim).reshape(mel2ph.shape[0], mel2ph.shape[1] * self.predict_first_ndim)
            mel2ph_ = mel2ph.permute([1, 0])[..., None].repeat([1, 1, encoder_out.shape[-1]]).contiguous()  #[T', B, C]
            decoder_inp = torch.gather(decoder_inp, 0, mel2ph_).transpose(0, 1)  # [B, T, H]
            ret['decoder_inp_origin'] = decoder_inp_origin = decoder_inp



        # add pitch embed
        if hparams['use_pitch_embed'] and self.pitch_predictor_type == "conv":
            pitch_emb = self.add_pitch(decoder_inp_unrepeat, pitch, uv, mel2ph_unrepeat, ret, use_pred_pitch=use_pred_pitch, condition=spk_embed, condition_mask=spk_mask)
            if not self.parallel_predict and not self.vqemb_predict:
                pitch_emb = pitch_emb[:, :, None, :].repeat(1, 1, self.predict_first_ndim, 1).reshape(pitch_emb.shape[0], -1, pitch_emb.shape[-1])
            #print('pitch norm', torch.norm(pitch_emb), 'decoder_inp norm', torch.norm(decoder_inp))
            decoder_inp = decoder_inp + pitch_emb
        # add energy embed
        # if hparams['use_energy_embed']:
        #     decoder_inp = decoder_inp + self.add_energy(decoder_inp_origin, energy, ret)

        decoder_inp = decoder_inp * (mel2ph != 0).to(decoder_inp.dtype)[:, :, None]
        ret['decoder_inp'] = decoder_inp

        if skip_decoder:
            ret['mel_out'] = decoder_inp
            return ret
        x = decoder_inp
        if hparams['dec_layers'] > 0:
            x = self.decoder(x, condition=spk_embed)
        if self.parallel_predict and not self.vqemb_predict:
            result_list = []
            for i in range(hparams["predict_first_ndim"]):
                result_list.append((self.mel_out_list[i](x) * (mel2ph != 0).to(x.dtype)[:, :, None])[:, :, None, :])
            x = torch.cat(result_list, dim=2).reshape(x.shape[0], -1, result_list[0].shape[-1])
        else:
            x = self.mel_out(x)
            x = x * (mel2ph != 0).to(x.dtype)[:, :, None]
        ret['mel_out'] = x
        return ret

    def decode_with_pred_pitch(self, decoder_inp, mel2ph):
        #if hparams['use_ref_enc']:
        #    assert False
        pitch_embed = self.add_pitch(decoder_inp, None, None, mel2ph, {})
        decoder_inp = decoder_inp + self.pitch_do(pitch_embed)
        decoder_inp = decoder_inp * (mel2ph != 0).to(decoder_inp.dtype)[:, :, None]
        x = decoder_inp
        x = self.decoder(x)
        x = self.mel_out(x)
        x = x * (mel2ph != 0).to(x.dtype)[:, :, None]
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
        # print(pitch_logits.shape, pitch.shape)
        # exit()
        if pitch is not None and not use_pred_pitch:  # train
            pitch_padding = pitch == -200
            pitch_restore = restore_pitch(pitch, uv if hparams['use_uv'] else None, hparams,
                                          pitch_padding=pitch_padding)
            ret['pitch'] = pitch_restore
            pitch_restore = f0_to_coarse_torch(pitch_restore)
            pitch_embed = self.pitch_embed(pitch_restore)
        else:  # test
            ref_pitch = pitch
            ref_uv = uv
            pitch_padding = (mel2ph == 0)
            pitch = pitch_logits[:, :, 0]
            uv = pitch_logits[:, :, 1] > 0
            if not hparams['use_uv']:
                uv = pitch < -3.5
            if ref_pitch is not None:
                pitch[:, :ref_pitch.shape[1]] = ref_pitch
                pitch[:, ref_pitch.shape[1]:] = pitch[:, ref_pitch.shape[1]:] 
                      
            if ref_uv is not None:
                uv[:, :ref_uv.shape[1]] = ref_uv
                
            pitch_restore = restore_pitch(pitch, uv, hparams, pitch_padding=pitch_padding)
            if hparams['ref_norm'] and ref_pitch is not None:
                # trick ref-normalized  
                ref_pitch = pitch_restore[:, :ref_pitch.shape[1]]
                infer_pitch = pitch_restore[:, ref_pitch.shape[1]:]
                padding = (infer_pitch > 1)
                pred_mean = infer_pitch[infer_pitch > 1].mean()
                pred_std = infer_pitch[infer_pitch > 1].std()
                gt_mean = ref_pitch[ref_pitch > 1].mean()
                gt_std = ref_pitch[ref_pitch > 1].std()
                lmabda=0.2
                
                mean = lmabda * pred_mean + (1 - lmabda) * gt_mean
                std = lmabda * pred_std + (1 - lmabda) * gt_std
                # mean = pitch_restore[pitch_restore>1].mean()
                # std = pitch_restore[pitch_restore>1].std()
                # mean = pred_mean
                # std = pred_std
                
                print('pitch infer:', pred_mean, pred_std)
                print('pitch ref:', gt_mean, gt_std)
                print('pitch used:', mean, std)
                
                # print(infer_pitch[infer_pitch > 1])
                infer_pitch_ = infer_pitch.clone()
                infer_pitch_[infer_pitch > 1] = (infer_pitch_[infer_pitch > 1] - pred_mean) / pred_std
                infer_pitch_[infer_pitch > 1] = infer_pitch_[infer_pitch > 1] * std + mean
                # print('-----------------')
                infer_pitch = infer_pitch_
                # print(infer_pitch[infer_pitch > 1])
                pitch_restore[:, ref_pitch.shape[1]:] = infer_pitch
            
            ret['pitch'] = pitch_restore
            pitch_restore = f0_to_coarse_torch(pitch_restore)
            pitch_embed = self.pitch_embed(pitch_restore)
        return self.pitch_do(pitch_embed)
    
    def add_pitch_phone(self, encoder_output, pitch, uv, mel2ph, ret, use_pred_pitch=False, condition=None, mask=None):
        pp_inp = encoder_output.transpose(0 , 1)  # B, T_t, D
        
        if hparams['predictor_sg']:
            pp_inp = pp_inp.detach()
            
        if pitch is not None and not use_pred_pitch:  # train
            pitch_padding = pitch == -200
            pitch_restore = restore_pitch(pitch, uv if hparams['use_uv'] else None, hparams,
                                          pitch_padding=pitch_padding) # frame-level
            

                    
            B, T_t, D = pp_inp.shape
            dur_gt = mel2ph.new_zeros(B, T_t + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
            dur_gt = dur_gt[:, 1:]
            
            pitch_restore = torch.cat([pitch_restore.new_zeros(B, 1), pitch_restore], dim=1)
            
            cum_pitch = pitch_restore.new_zeros(B, T_t + 1).scatter_add(1, mel2ph, pitch_restore)
            cum_pitch = cum_pitch[:, 1:]
            # exit()
            pitch_restore = mel2ph.new_zeros(B, T_t)
            cum_pitch[dur_gt > 0] = cum_pitch[dur_gt > 0] / dur_gt[dur_gt > 0]
            
            phoneme_f0_target = process_f0_nointerp(cum_pitch.clone(), hparams)
            

            uv_phoneme = cum_pitch == 1.0
            ret["phoneme_pitch_target"] = phoneme_f0_target
            ret["phoneme_pitch_uv"] = uv_phoneme

            # print("------------------------------")
            # print(cum_pitch[0].detach().cpu().numpy().tolist())
            # print(uv_phoneme[0].detach().cpu().numpy().tolist())
            
            pitch_restore[dur_gt > 0] = f0_to_coarse_torch(cum_pitch[dur_gt > 0], f0_min=hparams["phoneme_f0_min"], f0_max=hparams["phoneme_f0_max"], f0_bin=hparams["phoneme_f0_bin"])  # dur >0
            
            
            if hparams["phoneme_pitch_uv_shift_1"]:
                pitch_restore += 1
                pitch_restore[uv_phoneme] = 1
                pitch_restore[dur_gt == 0] = 256 + 2 # special token for padding
            else:
                pitch_restore[dur_gt == 0] = 256 + 1 # special token for padding
            # print(pitch_restore[0].detach().cpu().numpy().tolist())
            # exit(0)
            # print(pitch_restore)
            # exit(0)
            
            ret['pitch'] = pitch_restore 
            pitch_embed = self.pitch_embed(pitch_restore)
            # exit()
            if self.pitch_predictor_type == "ar_transformer_phone" or self.pitch_predictor_type == "ar_transformer_phone_continue":
                ret['pitch_logits'], ret['pitch_gt'] = self.pitch_predictor(enc_output=pp_inp, dur_gt=pitch_restore.long(), condition=condition, mask=mask)
                ret['pitch_gt'] = ret['pitch_gt'].masked_fill(mask, -200)
                
                # exit()
            elif self.pitch_predictor_type == "conv_phone":
                ret['pitch_logits'] = pitch_logits = self.pitch_predictor(pp_inp, condition=condition)
                # phoneme_f0 = restore_pitch(pitch_logits[:,:,0].clone(), uv=None, hparams=hparams)
                # uv_phoneme = pitch_logits[:,:,1] > 0
                # print('train uv acc:', (uv_phoneme==hparams['gt_uv']).float().mean())
                
                
            else:
                raise NotImplementedError("Unknown pitch predictor type: {}".format(self.pitch_predictor_type))
            
        else:  # test
            B, T_t, D = pp_inp.shape
            dur_gt = mel2ph.new_zeros(B, T_t + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
            dur_gt = dur_gt[:, 1:]
            print(dur_gt)
            dur_mask = (dur_gt == 0)
            # exit()
            if self.pitch_predictor_type == "ar_transformer_phone" or self.pitch_predictor_type == "ar_transformer_phone_continue":
                pitch = self.pitch_predictor.inference(enc_output=pp_inp, condition=condition, mask=mask, dur_mask=dur_mask)
                pitch_embed = self.pitch_embed(pitch)
                return self.pitch_do(pitch_embed)
            elif self.pitch_predictor_type == "conv_phone":
                pitch_logits = self.pitch_predictor(pp_inp, condition=condition)
                
                cum_pitch = restore_pitch(pitch_logits[:,:,0].clone(), uv=None, hparams=hparams)
                uv_phoneme = pitch_logits[:,:,1] > 0
                pitch_restore = mel2ph.new_zeros(B, T_t)
                pitch_restore[dur_gt > 0] = f0_to_coarse_torch(cum_pitch[dur_gt > 0], f0_min=hparams["phoneme_f0_min"], f0_max=hparams["phoneme_f0_max"], f0_bin=hparams["phoneme_f0_bin"])  # dur >0
                if hparams["phoneme_pitch_uv_shift_1"]:
                    pitch_restore += 1
                    pitch_restore[uv_phoneme] = 1
                    pitch_restore[dur_gt == 0] = 256 + 2 # special token for padding
                else:
                    pitch_restore[dur_gt == 0] = 256 + 1 # special token for padding
                # print(pitch_restore, 'infer pitch')
                # print('infer uv acc:', (uv_phoneme==hparams['gt_uv']).float().mean())
                # print('infer pitch acc:', (pitch_restore==hparams['gt_pitch']).float().mean())
                pitch_embed = self.pitch_embed(pitch_restore)
                return self.pitch_do(pitch_embed)
            else:
                raise NotImplementedError("Unknown pitch predictor type: {}".format(self.pitch_predictor_type))
            
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
