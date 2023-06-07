
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.tts_modules import FastspeechDecoder
from utils.hparams import hparams

from modules.tts_modules import DEFAULT_MAX_TARGET_POSITIONS, SinusoidalPositionalEmbedding, TransformerDecoderLayer

from modules.operations import OPERATIONS_DECODER, LayerNorm, MultiheadAttention, NewTransformerFFNLayer, TransformerFFNLayer, utils

from .wavnet import AttrDict, SinusoidalPosEmb, Mish
from usr_dir_kai.conformer_fairseq.conformer_wavnet import S2TConformerEncoder
from torch.nn.utils.rnn import pad_sequence

class TransformerEstimator(nn.Module):
    def __init__(self, arch) -> None:
        super().__init__()

        self.decoder = S2TConformerEncoder()
        
        self.params = params = AttrDict(
            # Model params
            transformer_hidden=hparams['transformer_hidden'],
            condition_hidden=hparams["hidden_size"],
            latent_dim=hparams['vqemb_size']
        )
        
        
        dim = params.latent_dim
        
        self.diffusion_embedding = SinusoidalPosEmb(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, params.transformer_hidden)
        )
        
        self.spec_linear = nn.Linear(dim, params.transformer_hidden)
        self.cond_linear = nn.Linear(params.condition_hidden, params.transformer_hidden)
        
        self.output_linear = nn.Linear(params.transformer_hidden, dim)
        
        self.prompt_linear = nn.Linear(params.latent_dim, params.transformer_hidden)
        
        self.embed_positions = SinusoidalPositionalEmbedding(
            params.transformer_hidden, 0,
            init_size=4000 + 0 + 1,
        )
        
        nn.init.constant_(self.output_linear.weight, 0)
        nn.init.constant_(self.output_linear.bias, 0)
    
    def forward(self, spec, x_mask, diffusion_step, prompt_step, cond, spk, prompt, prompt_mask):
        
        
        diffusion_step = self.diffusion_embedding(diffusion_step)
        prompt_step = self.diffusion_embedding(prompt_step)

        diffusion_step = self.mlp(diffusion_step)
        prompt_step = self.mlp(prompt_step)

        
        spec = spec.transpose(1, 2)
        cond = cond.transpose(1, 2)
        diffusion_step = diffusion_step.unsqueeze(1)
        prompt_step = prompt_step.unsqueeze(1)
        
        spec = self.spec_linear(spec)
        cond = self.cond_linear(cond)
        
        prompt = self.prompt_linear(prompt)
        
        
        cond_xt = cond
        
        
        # ------- merge prompt and spec
        prompt_seq_len = (1 - prompt_mask.long()).sum(1).long()
        spec_seq_len = (1 - x_mask.long()).sum(1).long()
        batch_size = spec.shape[0]
        
        input_speech_latent_collect = []
        cond_collect = []
        step_collect = []
        input_seq_len = []

        for i in range(spec.shape[0]):
            
            prompt_latent = prompt[i, :prompt_seq_len[i].item(), :]
            spec_latent = spec[i, :spec_seq_len[i].item(), :]
            
            input_speech_latent_collect.append(torch.cat((prompt_latent, spec_latent), dim=0))
            input_seq_len.append(prompt_latent.shape[0] + spec_latent.shape[0])
            
            cond_spec = cond_xt[i, :spec_seq_len[i].item(), :]
            cond_collect.append(torch.cat((cond_spec.new_zeros((prompt_latent.shape[0], cond_spec.shape[-1])), cond_spec), dim=0))
            
            step_collect.append(
                torch.cat((
                    prompt_step[i, :, :].expand(prompt_latent.shape[0], -1), diffusion_step[i, :, :].expand(spec_latent.shape[0], -1)
                ), dim=0)
            )
            
        
        input_latent = pad_sequence(input_speech_latent_collect, batch_first=True)
        
        cond = pad_sequence(cond_collect, batch_first=True)
        input_len = torch.Tensor(input_seq_len).to(input_latent.device).long()
        step_emb = pad_sequence(step_collect, batch_first=True)
        
        input_latent = input_latent.transpose(0, 1)
        cond = cond.transpose(0, 1)
        step_emb = step_emb.transpose(0, 1)
        
        ret_conformer = self.decoder(speech_latent=input_latent, condition=cond, diffusion_step=step_emb, input_lengths=input_len)
        
        x = ret_conformer["encoder_out"][0]
        
        x = self.output_linear(x)
        
        ret = []
        prompt_ret = []
        for i in range(batch_size):
            ret.append(
                x[i, prompt_seq_len[i].item():, :]
            )
            prompt_ret.append(
                x[i, :prompt_seq_len[i].item(), :]
            )
            
        x = pad_sequence(ret, batch_first=True)
        x = x[:, :spec.shape[1], :]
        
        prompt_ret = pad_sequence(prompt_ret, batch_first=True)
        prompt_ret = prompt_ret[:, :prompt.shape[1], :]
        
        x = x.transpose(1, 2)
        
        return prompt_ret, x