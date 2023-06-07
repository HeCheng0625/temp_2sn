import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.tts_modules import FastspeechDecoder
from utils.hparams import hparams
from math import sqrt

from modules.tts_modules import DEFAULT_MAX_TARGET_POSITIONS, SinusoidalPositionalEmbedding, TransformerDecoderLayer

from modules.operations import OPERATIONS_DECODER, LayerNorm, MultiheadAttention, NewTransformerFFNLayer, TransformerFFNLayer, utils

from .wavnet import AttrDict, SinusoidalPosEmb, Mish

class DiffLayer(nn.Module):
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1, kernel_size=9, use_cln=False, need_skip=False):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.layer_norm1 = LayerNorm(c, use_cln=use_cln)
        self.self_attn = MultiheadAttention(c, num_heads, self_attention=True, dropout=attention_dropout, bias=True)

        self.layer_norm2 = LayerNorm(c, use_cln=use_cln)
        if hparams['use_new_ffn']:
            self.ffn = NewTransformerFFNLayer(c, 2 * c, padding='SAME', kernel_size=kernel_size, dropout=relu_dropout)
        else:
            self.ffn = TransformerFFNLayer(c, 2 * c, padding='SAME', kernel_size=kernel_size, dropout=relu_dropout)

        self.diffusion_projection = nn.Linear(c, c)
        self.cond_projection = nn.Linear(c, c)

        self.need_skip = need_skip

        if self.need_skip:
            self.skip_linear = nn.Linear(c, c)
            self.layer_norm3 = LayerNorm(c)

    def forward(
            self,
            x,
            cond,
            diffusion_step,
            spk,
            skip=None,
            encoder_padding_mask=None,
            incremental_state=None,
            self_attn_mask=None,
            self_attn_padding_mask=None,
            **kwargs,
    ):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm3.training = layer_norm_training

        if self.need_skip:
            x = x + self.skip_linear(skip)

        # x = x + self.diffusion_projection(diffusion_step)
        # a bug version: x = x + self.diffusion_projection(diffusion_step) + self.diffusion_projection(diffusion_step), can also work...     
        x = x + self.diffusion_projection(diffusion_step) + self.cond_projection(cond)

        # self attention
        residual = x
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask, incremental_state=incremental_state, attn_mask=self_attn_mask)
        x = F.dropout(x, self.dropout, training=self.training)
        self_attn_skip_result = x
        x = residual + x
        x = self.layer_norm1(x, spk)

        # ffn
        residual = x
        x = self.ffn(x, incremental_state=incremental_state)
        ffn_skip_result = x
        x = residual + x
        x = self.layer_norm2(x, spk)

        return x, self_attn_skip_result, ffn_skip_result

    def clear_buffer(self, input, cond=None, encoder_padding_mask=None, incremental_state=None):
        self.encoder_attn.clear_buffer(incremental_state)
        self.ffn.clear_buffer(incremental_state)

    def set_buffer(self, name, tensor, incremental_state):
        return utils.set_incremental_state(self, incremental_state, name, tensor)
    

diff_transformer_num_head = 8

OPERATIONS_DECODER[13] = lambda c, dropout, use_cln, need_skip: DiffLayer(c, diff_transformer_num_head, dropout=dropout,
                                     attention_dropout=0.0, relu_dropout=dropout,
                                     use_cln=use_cln,
                                     need_skip=need_skip,
                                     kernel_size=hparams['dec_ffn_kernel_size'])

class TransformerDecoder(nn.Module):
    def __init__(self, arch, hidden_size=None, dropout=None, use_cln=False):
        super().__init__()
        self.arch = arch  # '13 13 13 13 13 13 13 13 13 13 13 13'
        self.num_layers = len(arch)
        if hidden_size is not None:
            embed_dim = self.hidden_size = hidden_size
        else:
            # embed_dim = self.hidden_size = hparams['hidden_size']
            embed_dim = self.hidden_size = hparams['transformer_hidden']
        if dropout is not None:
            self.dropout = dropout
        else:
            self.dropout = hparams['dropout']

        if self.num_layers % 2 != 0:
            raise ValueError('num_layers must be even')
        
        self.in_layers = nn.ModuleList([])
        self.in_layers.extend([
            TransformerDecoderLayer(self.arch[i], self.hidden_size, self.dropout, use_cln=use_cln)
            for i in range(self.num_layers // 2)
        ])
        
        self.out_layers = nn.ModuleList([])
        
        # if use long skip connection like unet
        # self.out_layers.extend([
        #     TransformerDecoderLayer(self.arch[i], self.hidden_size, self.dropout, use_cln=use_cln, need_skip=True)
        #     for i in range(self.num_layers // 2)
        # ])
        self.out_layers.extend([
            TransformerDecoderLayer(self.arch[i], self.hidden_size, self.dropout, use_cln=use_cln)
            for i in range(self.num_layers // 2)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_dim)   # for prenorm transformer
        self.skip_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, cond, diffusion_step, mask):
        """
        :param x: [B, T, C]
        :param cond: [B, T, C]
        :param diffusion_step: [B, T, C]
        """
        padding_mask = x.abs().sum(-1).eq(0).data
        bs, seq_len = mask.shape
        slf_attn_mask = mask.view(bs, 1, 1, seq_len).expand(-1, diff_transformer_num_head, seq_len, -1).contiguous().view(bs * diff_transformer_num_head, seq_len, seq_len).bool()

        x = x + cond
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        cond = cond.transpose(0, 1)
        diffusion_step = diffusion_step.transpose(0, 1)
        x = x + diffusion_step
        # x = x + cond + diffusion_step   # add too much cond

        # encoder layers
        in_x_collect = []
        self_attn_skip_collect = []
        ffn_skip_collect = []
        for layer in self.in_layers:
            x, self_attn_skip_result, ffn_skip_result = layer(x, cond=cond, diffusion_step=diffusion_step, spk=None, encoder_padding_mask=padding_mask, self_attn_mask=slf_attn_mask)
            in_x_collect.append(x)
            self_attn_skip_collect.append(self_attn_skip_result)
            ffn_skip_collect.append(ffn_skip_result)

        for layer in self.out_layers:
            x, self_attn_skip_result, ffn_skip_result = layer(x, cond=cond, diffusion_step=diffusion_step, spk=None, encoder_padding_mask=padding_mask, self_attn_mask=slf_attn_mask, skip=in_x_collect.pop())
            self_attn_skip_collect.append(self_attn_skip_result)
            ffn_skip_collect.append(ffn_skip_result)
        
        # for prenorm transformer
        # x = self.layer_norm(x)

        # # similar to dual residual transformer
        # x = x + self.layer_norm(torch.sum(torch.stack(ffn_skip_collect), dim=0))
        
        # similar to wavnet
        x = torch.sum(torch.stack(ffn_skip_collect), dim=0) / sqrt(len(ffn_skip_collect))
        # x = self.layer_norm(torch.sum(torch.stack(ffn_skip_collect), dim=0))

        # skip projection
        x = self.skip_projection(x)
        x = F.relu(x)

        x = x.transpose(0, 1)

        return x


class TransformerEstimator(nn.Module):
    def __init__(self, arch) -> None:
        super().__init__()
        if isinstance(arch, str):
            self.arch = list(map(int, arch.strip().split()))
        
        # if LayerNorm
        self.decoder = TransformerDecoder(self.arch, use_cln=False)
        # # if AdaNorm:
        # self.decoder = TransformerDecoder(self.arch, use_cln=True)
        
        self.params = params = AttrDict(
            # Model params
            transformer_hidden=hparams['transformer_hidden'],   # default: 512
            condition_hidden=hparams['hidden_size'],   # 512
            latent_dim=hparams['vqemb_size']   # 256
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
        self.prompt_linear = nn.Linear(params.condition_hidden, params.transformer_hidden)
        # self.prompt_linear = nn.Linear(params.latent_dim, params.transformer_hidden)

        self.embed_positions = SinusoidalPositionalEmbedding(
            params.transformer_hidden, 0, init_size=4000 + 0 + 1,
        )

    def forward(self, spec, x_mask, diffusion_step, cond, spk, prompt, prompt_mask):

        prompt_step = diffusion_step.new_zeros(diffusion_step.shape[0])
        
        diffusion_step = self.diffusion_embedding(diffusion_step).to(spec.dtype)
        
        prompt_step = self.diffusion_embedding(prompt_step).to(spec.dtype)

        diffusion_step = self.mlp(diffusion_step)   # (B, transformer_hidden)
        prompt_step = self.mlp(prompt_step)   # (B, transformer_hidden)

        spec = spec.transpose(1, 2)   # (B, latent_dim, T_x) -> (B, T_x, latent_dim)
        cond = cond.transpose(1, 2)   # (B, condition_hidden, T_x) -> (B, T_x, condition_hidden)

        diffusion_step = diffusion_step.unsqueeze(1).expand(-1, cond.shape[1], -1)   # (B, 1, transformer_hidden) -> (B, T_x, transformer_hidden)
        prompt_step = prompt_step.unsqueeze(1).expand(-1, prompt.shape[1], -1)   # (B, 1, transformer_hidden) -> (B, T_p, transformer_hidden)

        spec = self.spec_linear(spec)   # (B, T_x, transformer_hidden)
        cond = self.cond_linear(cond)   # (B, T_x, transformer_hidden)

        cond_xt = cond
        cond_prompt = cond.new_zeros(cond.size(0), prompt.shape[1], cond.size(2))   # (B, T_p, transformer_hidden)
        cond = torch.cat((cond_prompt, cond_xt), dim=1)   # (B, T_p + T_x, transformer_hidden)
        
        step_emb = torch.cat((prompt_step, diffusion_step), dim=1)
        
        
        cond = cond + step_emb
        
        prompt = self.prompt_linear(prompt)   # (B, T_p, transformer_hidden)
        
        mask_full = torch.cat((prompt_mask, x_mask), dim=1)

        spec_pos = self.embed_positions(spec[..., 0])
        prompt_pos = self.embed_positions(prompt[..., 0])
        
        input_pos = torch.cat((prompt_pos, spec_pos), dim=1)
        input = torch.cat((prompt, spec), dim=1)   # (B, T_p + T_x, transformer_hidden)
        
        input = input + input_pos   # (B, T_p + T_x, transformer_hidden); input + position embedding

        input = F.dropout(input, p=0.2, training=self.training)
        
        x = self.decoder(x=input, cond=cond, diffusion_step=step_emb, mask=mask_full)
        
        x = self.output_linear(x)
        
        x = x[:, prompt.shape[1]:, :]
        
        x = x.transpose(1, 2)
        
        return x