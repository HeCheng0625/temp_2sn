
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion import Diffusion


from utils.hparams import hparams
from modules.operations import MultiheadAttention, LayerNorm, MultiheadAttentionIndependentW

AD_PAD_IDX = 0




class DiffusionTTS(nn.Module):
    def __init__(self, prior_encoder) -> None:
        super().__init__()
        
        self.priority_encoder = prior_encoder
        
        self.quantizer = self.priority_encoder.quantizer
        
        self.diffusion = Diffusion(n_feats=hparams["vqemb_size"], dim=hparams["vqemb_size"], n_spks=hparams["num_spk"], spk_emb_dim=hparams['hidden_size'], 
                                   beta_min=hparams["beta_min"], beta_max=hparams["beta_max"], pe_scale=hparams["pe_scale"])
        
        if hparams['use_ref_enc']:
            
                
            if isinstance(hparams['reference_encoder_filters'], str):
                self.ref_filters = ref_filters = list(map(int, hparams['reference_encoder_filters'].strip().split()))
            else:
                assert isinstance(hparams['reference_encoder_filters'], (list, tuple))
                self.ref_filters = ref_filters = hparams['reference_encoder_filters']
            if hparams["ref_enc_type"] == "residual_ref_enc":
                from .ref_enc_residual import ReferenceEncoder
                self.ref_enc = ReferenceEncoder(ref_speaker_embedding_dim=hparams['hidden_size'], audio_num_mel_bins=hparams['vqemb_size'], reference_encoder_filters=ref_filters)

            elif hparams["ref_enc_type"] == "default":
                from .ref_enc import ReferenceEncoder
                self.ref_enc = ReferenceEncoder(ref_speaker_embedding_dim=hparams['hidden_size'], audio_num_mel_bins=hparams['vqemb_size'], reference_encoder_filters=ref_filters)
            else:
                raise NotImplementedError("Not implemented yet")
            
            if hparams['spk_dropout'] > 0.0:
                if not hparams['old_empty_spk']:
                    self.empty_spk = torch.nn.Parameter(torch.randn(1, hparams['hidden_size'], requires_grad=True))
                    nn.init.kaiming_normal_(self.empty_spk, mode='fan_out',nonlinearity='relu')
                else:
                    self.empty_spk = torch.nn.Parameter(torch.randn(hparams['hidden_size'], requires_grad=True))
        else:
            arch = hparams['ref_enc_arch']
            print(arch, isinstance(arch, str))
            if isinstance(arch, str):
                self.ref_enc_arch = list(map(int, arch.strip().split()))
            else:
                assert isinstance(arch, (list, tuple))
                self.ref_enc_arch = arch
            if len(self.ref_enc_arch) == 0:
                self.ref_enc = None
                self.empty_spk = None
            else:
                from modules.tts_modules import FastspeechDecoder
                from modules.operations import Linear
                self.ref_enc_mlp = Linear(hparams['vqemb_size'],
                                    hparams['hidden_size'],
                                    bias=False)
                self.ref_enc =  FastspeechDecoder(self.ref_enc_arch, use_cln=False)
                
                self.empty_spk = None
                if hparams['ref_query_tokens'] > 0:
                    self.query_emb = nn.Embedding(hparams['ref_query_tokens'], hparams['hidden_size'])
                    if hparams['ref_query_norm']:
                        self.layer_norm = LayerNorm(hparams['hidden_size'])
                    kvdim = hparams['hidden_size']
                    
                    if hparams["query_attn_type"] == "vanilla_mha":
                        self.query_attn = MultiheadAttention(hparams['hidden_size'], 8, self_attention=False, dropout=0.1, kdim=kvdim, vdim=kvdim)
                    elif hparams["query_attn_type"] == "independent_w_mha":
                        self.query_attn = MultiheadAttentionIndependentW(hparams['hidden_size'], 8, self_attention=False, dropout=0.1, kdim=kvdim, vdim=kvdim,
                                                                         query_token=hparams['ref_query_tokens'])
                    else:
                        raise NotImplementError()
                    
                    if hparams['predictor_type'] == 'sawavnet':
                        self.ref_enc_out = Linear(hparams['hidden_size'] * hparams['ref_query_tokens'], hparams['hidden_size'])
                    if hparams['spk_dropout'] > 0.0:
                        self.empty_query_emb = nn.Embedding(hparams['ref_query_tokens'], hparams['hidden_size'])
            # print(self.ref_enc)
                
                
        
        self.set = False
        
        if hparams["apply_pitch_on_x0"]:
            self.pitch_predictor_on_x0 = torch.nn.Sequential(torch.nn.Linear(hparams['vqemb_size'], ), 
                                                             torch.nn.ReLU(),
                                                             torch.nn.Dropout(0.3),
                                                             torch.nn.Linear(hparams['hidden_size'], 2))
        
    @torch.no_grad()
    def convert_code_to_latent(self, codes):
        latent = self.quantizer.vq2emb(codes.long())
        return latent
        
    def forward(self, src_tokens, mel2ph, spk_embed=None,
                ref_mels=None, pitch=None, uv=None, energy=None, skip_decoder=False, use_pred_pitch=False, reference_spk_latent=None, ref_latent_mask=None):
        
        if not self.set:
            self.set = True
            self.quantizer.requires_grad_(False)
        
        if hparams['use_ref_enc']:
            ref_mels_input = ref_mels if not hparams["use_random_segment_as_ref"] else reference_spk_latent
            B, T, C = ref_mels_input.shape
            
            if ref_latent_mask is None:
                
                mask = torch.zeros(B, T, 1).to(dtype=torch.bool, device=ref_mels.device)
                pad = 0
                indices = torch.where(ref_mels_input != pad)
                mask[indices[0], indices[1]] = True
            else:
                mask = (~ ref_latent_mask).unsqueeze(2)

            # print(ref_latent_mask.shape, mask.shape)
            
            if hparams['ref_random_clip']:
                min_t = 2 ** len(self.ref_enc.convs)
                print(min_t, "minimum length", flush=True)
                t = torch.randint(min_t, T + 1, (1,) ,dtype=torch.long).item() # sample random target lengths 
                start = torch.randint(0, T - t + 1, (B,), dtype=torch.long) # sample random start indices for each batch element that are within the range [0,T-t]
                end = start + t # compute the end indices for each batch element
                ref_mels_input = [ref_mels_input[b,start[b]:end[b],:] for b in range(B)] # slice the data tensor according to the start and end indices
                ref_mels_input = torch.stack(ref_mels_input) 
                mask = [mask[b,start[b]:end[b],:] for b in range(B)] 
                mask = torch.stack(mask)
                # print(ref_mels.shape, mask.shape)
                # exit()
            spk_embed = self.ref_enc(ref_mels_input, mask=mask)
            # print(spk_embed.shape, mask.shape)
            # mask = mask[:, ::16, :]
            if not hparams['ref_enc_pooling']:
                stride = int((4 ** len(self.ref_filters)) / hparams['vqemb_size'])
                mask = mask[:, ::stride, :]
                # print(stride, mask.shape, spk_embed.shape, ref_mels_input.shape, flush=True)
                # exit()
                mask = F.pad(mask, (0, 0, 0, spk_embed.shape[1]-mask.shape[1], 0, 0))
                ref_latent_mask = ~ mask.squeeze(-1)
            
            if hparams['spk_dropout'] > 0.0:
                B, C = spk_embed.shape
                is_empty = torch.rand((B, 1)) < hparams['spk_dropout']
                is_empty = is_empty.to(spk_embed.device)
                # use is_empty to replace spk_embed
                spk_embed = torch.where(is_empty, self.empty_spk.expand(B, D).to(spk_embed.dtype), spk_embed)
            # print(spk_embed.shape)
            # exit()
        elif self.ref_enc is not None:
            ref_mels_input = ref_mels if not hparams["use_random_segment_as_ref"] else reference_spk_latent
            B, T, C = ref_mels_input.shape
            
            if ref_latent_mask is None:    
                mask = torch.zeros(B, T, 1).to(dtype=torch.bool, device=ref_mels.device)
                pad = 0
                indices = torch.where(ref_mels_input != pad)
                mask[indices[0], indices[1]] = True
            else:
                mask = (~ ref_latent_mask).unsqueeze(2)
                
            ref_mels_input = self.ref_enc_mlp(ref_mels_input)
            # if hparams['ref_query_tokens'] > 0:
            #     query = self.query_emb(torch.arange(hparams['ref_query_tokens']).to(ref_mels_input.device)).repeat(B, 1, 1)
            #     # print(query.shape, ref_mels_input.shape)
            #     ref_mels_input = torch.cat([query, ref_mels_input], dim=1)
            #     mask = F.pad(mask, (0, 0, 0, hparams['ref_query_tokens'], 0, 0))
            spk_embed = self.ref_enc(ref_mels_input)
            spk_embed = spk_embed * mask.to(spk_embed)

            if hparams['ref_query_tokens'] > 0:
                query = self.query_emb(torch.arange(hparams['ref_query_tokens']).to(ref_mels_input.device)).repeat(B, 1, 1)
                if hparams['ref_query_norm']:
                    query = self.layer_norm(query)   
                query, _ = self.query_attn(query.transpose(0, 1), spk_embed.transpose(0,1), spk_embed.transpose(0,1), key_padding_mask=ref_latent_mask)
                query = query.transpose(0, 1)
                if hparams['predictor_type'] == 'sawavnet':
                    query = query.reshape(B, -1)
                    query = self.ref_enc_out(query)
                elif hparams['spk_dropout'] > 0.0:
                    B, T, C = query.shape
                    empty_query = self.empty_query_emb(torch.arange(hparams['ref_query_tokens']).to(ref_mels_input.device)).repeat(B, 1, 1)
                    is_empty = torch.rand((B, hparams['ref_query_tokens'], 1)) < hparams['spk_dropout']
                    is_empty = is_empty.to(spk_embed.device)
                    # use is_empty to replace spk_embed
                    query = torch.where(is_empty, empty_query, query)
                if hparams['ref_query_norm']:
                    query = self.layer_norm(query)
                # print(query.shape, reference_spk_latent.shape)
                
            # print(spk_embed.shape)
            # is nan
            # print('spk_embed', torch.isnan(spk_embed).any())
            # exit()            
        else:
            ref_mels_input = ref_mels if not hparams["use_random_segment_as_ref"] else reference_spk_latent
            B, T, C = ref_mels_input.shape
            
            if ref_latent_mask is None:
                
                mask = torch.zeros(B, T, 1).to(dtype=torch.bool, device=ref_mels.device)
                pad = 0
                indices = torch.where(ref_mels_input != pad)
                mask[indices[0], indices[1]] = True
            else:
                mask = (~ ref_latent_mask).unsqueeze(2)
            
            spk_embed = ref_mels_input
            # print(spk_embed.shape, mask.shape)

        prior_out = self.priority_encoder(src_tokens=src_tokens, mel2ph=mel2ph, spk_embed=spk_embed, spk_mask=mask, ref_mels=ref_mels, pitch=pitch, uv=uv, energy=energy, skip_decoder=skip_decoder, use_pred_pitch=use_pred_pitch)
        # print(prior_out.keys(), prior_out["mel_out"].shape)
        # exit(0)
        
        x_mask = (mel2ph == 0)
        # print(x_mask.float().mean())
        mu_x = prior_out["mel_out"]
        
        if bool(hparams["detach_mu"]):
            mu_x = mu_x.detach()
        
        mu_x = mu_x.transpose(1, 2)
        ref_mels = ref_mels.transpose(1, 2)
        
        if hparams['ref_query_tokens'] > 0:
            spk_embed = query
            ref_latent_mask = torch.zeros(B, hparams['ref_query_tokens']).bool().to(query.device)
        
        if not hparams["incontext"]:
            x0_pred, noise_pred, noise_gt, prompt_pred, noise_prompt_pred, noise_prompt_gt = self.diffusion.compute_loss_prompt(x0=ref_mels, x_mask=x_mask, mu=mu_x, spk=spk_embed, prompt=spk_embed if hparams['use_spk_prompt'] else reference_spk_latent, prompt_mask=ref_latent_mask)
        else:
            x0_pred, noise_pred, noise_gt = self.diffusion.compute_loss_incontext(x0=ref_mels, mu=mu_x, spk=spk_embed)
            
        diff_out = {}
        diff_out["diff_x0_pred"] = x0_pred.transpose(1, 2)
        diff_out["diff_noise_pred"] = noise_pred.transpose(1, 2)
        diff_out["diff_noise_gt"] = noise_gt.transpose(1, 2)
        
        diff_out["diff_prompt_pred"] = prompt_pred
        diff_out["diff_prompt_gt"] = spk_embed
        diff_out["ref_latent_mask"] = ref_latent_mask
        diff_out["diff_prompt_noise_pred"] = noise_prompt_pred
        diff_out["diff_prompt_noise_gt"] = noise_prompt_gt
        
        
        if hparams["apply_pitch_on_x0"]:
            pitch_pred = self.pitch_predictor_on_x0(x0_pred.transpose(1, 2))
            diff_out["x0_pitch_pred"] = pitch_pred
        
        return prior_out, diff_out
    

    def infer_fs2(self, src_tokens, mel2ph, spk_embed=None,
                ref_mels=None, pitch=None, uv=None, energy=None, skip_decoder=False, use_pred_pitch=False):
        
        if self.ref_enc is not None:
            B, T, C = ref_mels.shape
            try:
                ref_mels = self.ref_enc_mlp(ref_mels)
            except:
                pass
            ref_mel_for_enc = ref_mels
            # if hparams['ref_query_tokens'] > 0:
            #     query = self.query_emb(torch.arange(hparams['ref_query_tokens']).to(ref_mel_for_enc.device)).repeat(B, 1, 1)
            #     # print(query.shape, ref_mels_input.shape)
            #     ref_mel_for_enc = torch.cat([query, ref_mel_for_enc], dim=1)
            spk_embed = self.ref_enc(ref_mel_for_enc.contiguous())

            if hparams['ref_query_tokens'] > 0:
                query = self.query_emb(torch.arange(hparams['ref_query_tokens']).to(ref_mel_for_enc.device)).repeat(B, 1, 1)
                if hparams['ref_query_norm']:
                    query = self.layer_norm(query)
                query, _ = self.query_attn(query.transpose(0, 1), spk_embed.transpose(0,1), spk_embed.transpose(0,1), key_padding_mask= torch.zeros(B, T).bool().to(ref_mel_for_enc.device))
                query = query.transpose(0, 1)
                if hparams['predictor_type'] == 'sawavnet':
                    query = query.reshape(B, -1)
                    query = self.ref_enc_out(query)
                if hparams['ref_query_norm']:
                    query = self.layer_norm(query)
            print(spk_embed.shape)
        else:
            ref_mels_input = ref_mels 
            B, T, C = ref_mels_input.shape
            spk_embed = ref_mels_input
      
        ret = self.priority_encoder(src_tokens=src_tokens, mel2ph=mel2ph, spk_embed=spk_embed, ref_mels=ref_mels, pitch=pitch, uv=uv, energy=energy, skip_decoder=skip_decoder, use_pred_pitch=use_pred_pitch)
        if hparams['ref_query_tokens'] > 0:
            spk_embed = query
        return ret, spk_embed
    
    def infer_fs2_hack(self, src_tokens_ref, src_tokens_in, mel2ph, spk_embed=None,
                ref_mels=None, pitch=None, uv=None, energy=None, skip_decoder=False, use_pred_pitch=False):
        
        if self.ref_enc is not None:
            spk_embed = self.ref_enc(ref_mels.contiguous())
        else:
            spk_embed = None

        
        ret = self.priority_encoder.forward_hack(src_tokens_ref=src_tokens_ref, src_tokens_in=src_tokens_in, mel2ph=mel2ph, spk_embed=spk_embed, ref_mels=ref_mels, pitch=pitch, uv=uv, energy=energy, skip_decoder=skip_decoder, use_pred_pitch=use_pred_pitch)
        return ret, spk_embed

    def infer_diffusion(self, mu, n_timesteps, spk=None,temperature=1.0, ref_x=None, prompt=None, prompt_mask=None):
        mu_x = mu.transpose(1, 2)
        z = torch.randn(mu_x.shape[0], hparams['vqemb_size'], mu_x.shape[2]).to(mu_x) * hparams['noise_factor'] / temperature
        if hparams['diffusion_from_prior']:
            z = mu_x + z
        # Generate sample by performing reverse dynamics
        x_mask = torch.zeros(mu_x.shape[0], mu_x.shape[2]).bool().to(z.device)

        decoder_outputs = self.diffusion(z, x_mask, mu_x, n_timesteps, spk=spk, stoc=hparams['stoc'], ref_x=ref_x, prompt=prompt, prompt_mask=prompt_mask)
        return decoder_outputs.transpose(1, 2)     
        pass
