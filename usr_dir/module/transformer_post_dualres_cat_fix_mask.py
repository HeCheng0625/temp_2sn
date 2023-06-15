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
    

diff_transformer_num_head = hparams['diff_transformer_num_head']

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
            self.dropout = hparams['diffusion_dropout']

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
        # padding_mask = x.abs().sum(-1).eq(0).data   # padding_mask = mask = self_attn_padding_mask
        padding_mask = mask
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
            x, self_attn_skip_result, ffn_skip_result = layer(x, cond=cond, diffusion_step=diffusion_step, spk=None, self_attn_padding_mask=padding_mask, self_attn_mask=slf_attn_mask)
            in_x_collect.append(x)
            self_attn_skip_collect.append(self_attn_skip_result)
            ffn_skip_collect.append(ffn_skip_result)

        for layer in self.out_layers:
            x, self_attn_skip_result, ffn_skip_result = layer(x, cond=cond, diffusion_step=diffusion_step, spk=None, self_attn_padding_mask=padding_mask, self_attn_mask=slf_attn_mask, skip=in_x_collect.pop())
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
        self.prompt_linear = nn.Linear(dim, params.transformer_hidden)
        # self.prompt_linear = nn.Linear(params.latent_dim, params.transformer_hidden)

        self.embed_positions = SinusoidalPositionalEmbedding(
            params.transformer_hidden, 0, init_size=4000 + 0 + 1,
        )

    def forward(self, spec, x_mask, diffusion_step, cond, spk, prompt, prompt_mask):

        spec = spec.transpose(1, 2)
        cond = cond.transpose(1, 2)
        spec = self.spec_linear(spec)
        cond = self.cond_linear(cond)

        prompt = self.prompt_linear(prompt)

        prompt_step = diffusion_step.new_zeros(diffusion_step.shape[0])
        diffusion_step = self.diffusion_embedding(diffusion_step).to(spec.dtype)
        prompt_step = self.diffusion_embedding(prompt_step).to(spec.dtype)
        diffusion_step = self.mlp(diffusion_step)
        prompt_step = self.mlp(prompt_step)

        diffusion_step = diffusion_step.unsqueeze(1).expand(-1, spec.shape[1], -1)   # (B, T_x, d)
        prompt_step = prompt_step.unsqueeze(1).expand(-1, prompt.shape[1], -1)   # (B, T_p, d)

        max_prompt_len = prompt.size(1)   # T_p
        max_spec_len = spec.size(1)   # T_x

        input_len = max_prompt_len + max_spec_len   # T_p + T_x

        input = []
        input_pos = []
        input_cond = []
        input_diffusion_step = []
        input_mask = []
        x_indices = []

        input_spec = spec
        for i in range(input_spec.size(0)):

            prompt_len_i = (~prompt_mask[i]).int().sum().item()
            spec_len_i = (~x_mask[i]).int().sum().item()

            input_prompt_i = prompt[i, :prompt_len_i, :]
            input_spec_i = spec[i, :spec_len_i, :]
            input_pad_i = spec.new_zeros(input_len - prompt_len_i - spec_len_i, spec.size(2))
            input_i = torch.cat((input_prompt_i, input_spec_i, input_pad_i), dim=0)
            input.append(input_i)

            prompt_step_i = prompt_step[i, :prompt_len_i, :]
            diffusion_step_i = diffusion_step[i, :spec_len_i, :]
            pad_step_i = diffusion_step.new_zeros(input_len - prompt_len_i - spec_len_i, diffusion_step.size(2))
            input_diffusion_step_i = torch.cat((prompt_step_i, diffusion_step_i, pad_step_i), dim=0)
            input_diffusion_step.append(input_diffusion_step_i)

            spec_cond_i = cond[i, :spec_len_i, :]
            prompt_cond_i = spec_cond_i.new_zeros(prompt_len_i, spec_cond_i.size(1))
            pad_cond_i = spec_cond_i.new_zeros(input_len - prompt_len_i - spec_len_i, spec_cond_i.size(1))
            input_cond_i = torch.cat((prompt_cond_i, spec_cond_i, pad_cond_i), dim=0)
            input_cond.append(input_cond_i)

            prompt_mask_i = prompt.new_zeros(prompt_len_i).bool()
            x_mask_i = x_mask[i, :spec_len_i]
            pad_mask_i = prompt.new_ones(input_len - prompt_len_i - spec_len_i).bool()
            input_mask_i = torch.cat((prompt_mask_i, x_mask_i, pad_mask_i), dim=0)
            input_mask.append(input_mask_i)

            prompt_pos_i = self.embed_positions(prompt[i,:prompt_len_i,0].unsqueeze(0))
            spec_pos_i = self.embed_positions(spec[i,:spec_len_i,0].unsqueeze(0))
            if prompt_len_i + spec_len_i < input_len:
                pad_pos_i = self.embed_positions(spec[i,:,0].new_zeros(1, input_len - prompt_len_i - spec_len_i))
                input_pos_i = torch.cat((prompt_pos_i, spec_pos_i, pad_pos_i), dim=1).squeeze(0)
            else:
                input_pos_i = torch.cat((prompt_pos_i, spec_pos_i), dim=1).squeeze(0)
            input_pos.append(input_pos_i)

            x_index_i = torch.arange(prompt_len_i, prompt_len_i + max_spec_len).to(spec.device)
            x_indices.append(x_index_i)

        input = torch.stack(input, dim=0)
        input_pos = torch.stack(input_pos, dim=0)
        input_cond = torch.stack(input_cond, dim=0)
        input_diffusion_step = torch.stack(input_diffusion_step, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        x_indices = torch.stack(x_indices, dim=0)


        input = input + input_pos
        input = F.dropout(input, p=0.2, training=self.training)
        x = self.decoder(x=input, cond=input_cond, diffusion_step=input_diffusion_step, mask=input_mask)

        x = self.output_linear(x)

        # extract x index mask matrix
        x = x.gather(dim=1, index=x_indices.unsqueeze(-1).expand(-1, -1, x.size(2)))

        x = x.transpose(1, 2)
        return x