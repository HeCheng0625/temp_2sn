import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

from utils.hparams import hparams

from modules.operations import MultiheadAttention, LayerNorm

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d





class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] * hparams["pe_scale"]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class FiLM(nn.Module):
    def __init__(self, in_channels, condition_channels):
        super().__init__()
        self.gain = Linear(condition_channels, in_channels)
        self.bias = Linear(condition_channels, in_channels)
        
        nn.init.xavier_uniform(self.gain.weight)
        nn.init.constant(self.gain.bias, 1)
        
        nn.init.xavier_uniform(self.bias.weight)
        nn.init.constant(self.bias.bias, 0)
    
    def forward(self, x, condition):
        return x * self.gain(condition).unsqueeze(-1) + self.bias(condition).unsqueeze(-1)

class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation, use_filn=False, has_sattn=False):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(residual_channels, residual_channels)
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)
        
        self.has_sattn = has_sattn
        
        if self.has_sattn:
            self.attn = MultiheadAttention(residual_channels, 4, self_attention=True, dropout=0.1)
            self.layer_norm = LayerNorm(residual_channels)
            self.dropout = nn.Dropout(0.2)
        
        if use_filn:
            self.film = FiLM(residual_channels * 2, residual_channels)
        self.use_filn = use_filn

    def forward(self, x, conditioner, diffusion_step, spk=None, prompt=None, prompt_mask=None):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step
        
        if self.has_sattn:
            residual = y_ = y.transpose(1, 2)
            y_ = self.layer_norm(y_)
            y_, _, = self.attn(y_, y_, y_)
            
            y_ = self.dropout(y_)
            
            y_ = residual + y_
            
            y = y_.transpose(1, 2)

        y = self.dilated_conv(y) + conditioner
        
        if self.use_filn:
            y = self.film(y, spk)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class WavNet(nn.Module):
    def __init__(self, in_dims=80):
        super().__init__()
        self.params = params = AttrDict(
            # Model params
            encoder_hidden=hparams['vqemb_size'] if not hparams["skip_decoder"] else hparams["hidden_size"],
            residual_layers=hparams['residual_layers'],
            residual_channels=hparams['residual_channels'],
            dilation_cycle_length=hparams['dilation_cycle_length'],
            use_filn=hparams['diffusion_use_film'],
            has_sattn=hparams['diffusion_has_sattn'],
            sa_per_layer=hparams['diffusion_sa_per_layer'],
        )
        self.input_projection = Conv1d(in_dims, params.residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(params.residual_channels)
        dim = params.residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.encoder_hidden, params.residual_channels, 2 ** (i % params.dilation_cycle_length), use_filn=self.params.use_filn, 
                          has_sattn=self.params.has_sattn and (i % self.params.sa_per_layer == 0))
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, x_mask, diffusion_step, cond, spk, prompt=None, prompt_mask=None):
        """

        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        
        if bool(hparams["detach_wavenet"]):
            cond = cond.detach()
        
        x = spec
        x = self.input_projection(x)  # x [B, residual_channel, T]

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step, spk)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, 80, T]
        return x
