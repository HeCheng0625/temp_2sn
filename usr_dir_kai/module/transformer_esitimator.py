
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.tts_modules import FastspeechDecoder
from utils.hparams import hparams

from modules.tts_modules import DEFAULT_MAX_TARGET_POSITIONS, SinusoidalPositionalEmbedding, TransformerDecoderLayer

from modules.operations import OPERATIONS_DECODER, LayerNorm, MultiheadAttention, NewTransformerFFNLayer, TransformerFFNLayer, utils

from .wavnet import AttrDict, SinusoidalPosEmb, Mish



class DiffLayer(nn.Module):
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1, kernel_size=9, use_cln=False, need_skip=False):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.layer_norm1 = LayerNorm(c, use_cln=use_cln)
        self.self_attn = MultiheadAttention(
            c, num_heads, self_attention=True, dropout=attention_dropout, bias=True
        )
        self.layer_norm2 = LayerNorm(c, use_cln=use_cln)
        self.self_attn2 = MultiheadAttention(
            c, num_heads, self_attention=True, dropout=attention_dropout, bias=True
        )
        self.layer_norm3 = LayerNorm(c, use_cln=use_cln)
        # if hparams['use_new_ffn']:
        #     self.ffn = NewTransformerFFNLayer(c, 4 * c, padding='LEFT', kernel_size=kernel_size, dropout=relu_dropout)
        # else:
        #     self.ffn = TransformerFFNLayer(c, 4 * c, padding='LEFT', kernel_size=kernel_size, dropout=relu_dropout)
        
        self.fc1 = nn.Linear(c, 4 * c)
        self.fc2 = nn.Linear(4 * c, c)
        
        self.diffusion_projection = nn.Linear(c, c)
        self.enc_projection = nn.Linear(c, c)
        
        self.need_skip = need_skip
        
        if self.need_skip:
            self.skip_linear = nn.Linear(c, c)
            self.layer_norm4 = LayerNorm(c)

    def forward(
            self,
            x,
            encoder_out,
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
            self.layer_norm2.training = layer_norm_training
            self.layer_norm3.training = layer_norm_training
            
        if self.need_skip:
            x = x + self.skip_linear(skip)

        
        x = x + self.diffusion_projection(diffusion_step)
            
            
        residual = x
        x = self.layer_norm1(x, spk)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            attn_mask=self_attn_mask
        )
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        
        x = x + self.enc_projection(encoder_out)

        residual = x
        x = self.layer_norm2(x, spk)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            attn_mask=self_attn_mask
        )
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norm3(x, spk)
        # x = self.ffn(x, incremental_state=incremental_state)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        x = residual + x

        # if len(attn_logits.size()) > 3:
        #    indices = attn_logits.softmax(-1).max(-1).values.sum(-1).argmax(-1)
        #    attn_logits = attn_logits.gather(1, 
        #        indices[:, None, None, None].repeat(1, 1, attn_logits.size(-2), attn_logits.size(-1))).squeeze(1)
        return x

    def clear_buffer(self, input, encoder_out=None, encoder_padding_mask=None, incremental_state=None):
        self.encoder_attn.clear_buffer(incremental_state)
        self.ffn.clear_buffer(incremental_state)

    def set_buffer(self, name, tensor, incremental_state):
        return utils.set_incremental_state(self, incremental_state, name, tensor)


OPERATIONS_DECODER[13] = lambda c, dropout, use_cln, need_skip: DiffLayer(c, 4, dropout=dropout,
                                     attention_dropout=0.0, relu_dropout=dropout,
                                     use_cln=use_cln,
                                     need_skip=need_skip,
                                     kernel_size=hparams['dec_ffn_kernel_size'])


class TransformerDecoder(nn.Module):
    def __init__(self, arch, hidden_size=None, dropout=None, use_cln=False):
        super().__init__()
        self.arch = arch  # arch  = encoder op code
        self.num_layers = len(arch)
        if hidden_size is not None:
            embed_dim = self.hidden_size = hidden_size
        else:
            embed_dim = self.hidden_size = hparams['hidden_size']
        if dropout is not None:
            self.dropout = dropout
        else:
            self.dropout = hparams['dropout']
        self.max_source_positions = DEFAULT_MAX_TARGET_POSITIONS
        self.padding_idx = 0
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))
        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim, self.padding_idx,
            init_size=self.max_source_positions + self.padding_idx + 1,
        )
        
        
        if self.num_layers % 2 != 0:
            raise ValueError('num_layers must be even')
        
        self.in_layers = nn.ModuleList([])
        self.in_layers.extend([
            TransformerDecoderLayer(self.arch[i], self.hidden_size, self.dropout, use_cln=use_cln)
            for i in range(self.num_layers // 2)
        ])
        
        self.out_layers = nn.ModuleList([])
        self.out_layers.extend([
            TransformerDecoderLayer(self.arch[i], self.hidden_size, self.dropout, use_cln=use_cln, need_skip=True)
            for i in range(self.num_layers // 2)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, cond, diffusion_step, spk, require_w=False):
        """
        :param x: [B, T, C]
        :param require_w: True if this module needs to return weight matrix
        :return: [B, T, C]
        """
        padding_mask = x.abs().sum(-1).eq(0).data
        positions = self.pos_embed_alpha * self.embed_positions(x[..., 0])
        x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        cond = cond.transpose(0, 1)
        diffusion_step = diffusion_step.transpose(0, 1)

        # encoder layers
        in_x_collect = []
        for layer in self.in_layers:
            x = layer(x, encoder_out=cond, diffusion_step=diffusion_step, spk=spk, encoder_padding_mask=padding_mask)
            in_x_collect.append(x)
        
        for layer in self.out_layers:
            x = layer(x, encoder_out=cond, diffusion_step=diffusion_step, spk=spk, encoder_padding_mask=padding_mask, skip=in_x_collect.pop())

        x = self.layer_norm(x)
        x = x.transpose(0, 1)

        return x


class TransformerEstimator(nn.Module):
    def __init__(self, arch) -> None:
        super().__init__()
        if isinstance(arch, str):
            self.arch = list(map(int, arch.strip().split()))
        self.decoder = TransformerDecoder(self.arch, use_cln=True)
        
        self.params = params = AttrDict(
            # Model params
            transformer_hidden=hparams['transformer_hidden'],
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
        self.cond_linear = nn.Linear(dim, params.transformer_hidden)
        
        self.output_linear = nn.Linear(params.transformer_hidden, dim)
    
    def forward(self, spec, diffusion_step, cond, spk):
        diffusion_step = self.diffusion_embedding(diffusion_step)

        diffusion_step = self.mlp(diffusion_step)

        
        spec = spec.transpose(1, 2)
        cond = cond.transpose(1, 2)
        diffusion_step = diffusion_step.unsqueeze(1)
        
        spec = self.spec_linear(spec)
        cond = self.cond_linear(cond)
        
        cond = cond + diffusion_step
        
        # print(spec.shape, cond.shape, diffusion_step.shape, spk.shape, "11111111111111")
        # exit(0)

        x = self.decoder(x=spec, diffusion_step=diffusion_step, cond=cond, spk=spk)
        x = self.output_linear(x)
        
        x = x.transpose(1, 2)
        
        return x