
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
       
        self.layer_norm3 = LayerNorm(c, use_cln=use_cln)
        if hparams['use_new_ffn']:
            self.ffn = NewTransformerFFNLayer(c, 2 * c, padding='SAME', kernel_size=kernel_size, dropout=relu_dropout)
        else:
            self.ffn = TransformerFFNLayer(c, 2 * c, padding='SAME', kernel_size=kernel_size, dropout=relu_dropout)
        
        # self.fc1 = nn.Linear(c, 4 * c)
        # self.fc2 = nn.Linear(4 * c, c)
        
        self.diffusion_projection = nn.Linear(c, c)
        self.enc_projection = nn.Linear(c, c)
        
        self.need_skip = need_skip
        
        if self.need_skip:
            self.skip_linear = nn.Linear(c, c)
            self.layer_norm4 = LayerNorm(c)

    def forward(
            self,
            x,
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

        residual = x
        x = self.layer_norm3(x, spk)
        x = self.ffn(x, incremental_state=incremental_state)
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

diff_transformer_num_head = 8

OPERATIONS_DECODER[13] = lambda c, dropout, use_cln, need_skip: DiffLayer(c, diff_transformer_num_head, dropout=dropout,
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
            self.dropout = hparams['diffusion_dropout']
        # self.max_source_positions = DEFAULT_MAX_TARGET_POSITIONS
        # self.padding_idx = 0
        # self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))
        # self.embed_positions = SinusoidalPositionalEmbedding(
        #     embed_dim, self.padding_idx,
        #     init_size=self.max_source_positions + self.padding_idx + 1,
        # )
        
        
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

    def forward(self, x, cond, diffusion_step, mask):
        """
        :param x: [B, T, C]
        :param require_w: True if this module needs to return weight matrix
        :return: [B, T, C]
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
        x = x + cond + diffusion_step

        # encoder layers
        in_x_collect = []
        for layer in self.in_layers:
            x = layer(x, diffusion_step=diffusion_step, spk=None, encoder_padding_mask=None, self_attn_mask=slf_attn_mask)
            in_x_collect.append(x)
        
        for layer in self.out_layers:
            x = layer(x, diffusion_step=diffusion_step, spk=None, encoder_padding_mask=None, self_attn_mask=slf_attn_mask, skip=in_x_collect.pop())

        x = self.layer_norm(x)
        x = x.transpose(0, 1)
        

        return x


class TransformerEstimator(nn.Module):
    def __init__(self, arch) -> None:
        super().__init__()
        if isinstance(arch, str):
            self.arch = list(map(int, arch.strip().split()))
        self.decoder = TransformerDecoder(self.arch, use_cln=False)
        
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
        
        
    
    def forward(self, spec, x_mask, diffusion_step, prompt_step, cond, spk, prompt, prompt_mask):
        
        diffusion_step = self.diffusion_embedding(diffusion_step)
        prompt_step = self.diffusion_embedding(prompt_step)

        diffusion_step = self.mlp(diffusion_step)
        prompt_step = self.mlp(prompt_step)

        
        spec = spec.transpose(1, 2)
        cond = cond.transpose(1, 2)
        diffusion_step = diffusion_step.unsqueeze(1).expand(-1, cond.shape[1], -1)
        prompt_step = prompt_step.unsqueeze(1).expand(-1, prompt.shape[1], -1)
        
        spec = self.spec_linear(spec)
        cond = self.cond_linear(cond)
        
        
        cond_xt = cond
        
        cond_prompt = cond.new_zeros(cond.size(0), prompt.shape[1], cond.size(2))
        cond = torch.cat((cond_prompt, cond_xt), dim=1)
        
        step_emb = torch.cat((prompt_step, diffusion_step), dim=1)
        
        
        cond = cond + step_emb
        
        prompt = self.prompt_linear(prompt)
        
        mask_full = torch.cat((prompt_mask, x_mask), dim=1)
        
        
        spec_pos = self.embed_positions(spec[..., 0])
        prompt_pos = self.embed_positions(prompt[..., 0])
        
        input_pos = torch.cat((prompt_pos, spec_pos), dim=1)
        input = torch.cat((prompt, spec), dim=1)
        
        input = input + input_pos
        
        input = F.dropout(input, p=0.2, training=self.training)
        
        x = self.decoder(x=input, cond=cond, diffusion_step=step_emb, mask=mask_full)
        
        x = self.output_linear(x)
        
        x = x[:, prompt.shape[1]:, :]
        
        x = x.transpose(1, 2)
        
        return x