import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.operations import SinusoidalPositionalEmbedding, OPERATIONS_ENCODER, OPERATIONS_DECODER, LayerNorm, MultiheadAttention
from utils.hparams import hparams
from math import sqrt


DEFAULT_MAX_SOURCE_POSITIONS = 2000
DEFAULT_MAX_TARGET_POSITIONS = 2000


class TransformerEncoderLayer(nn.Module):
    def __init__(self, layer, hidden_size, dropout, use_cln):
        super().__init__()
        self.layer = layer
        self.hidden_size = hidden_size
        self.dropout = dropout
        if layer == 13:
            self.op = OPERATIONS_ENCODER[layer](hidden_size, dropout, hparams['gaus_bias'], hparams['gaus_tao'])
        else:
            self.op = OPERATIONS_ENCODER[layer](hidden_size, dropout, use_cln)

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, layer, hidden_size, dropout, use_cln, need_skip=False):
        super().__init__()
        self.layer = layer
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        if layer == 13:
            self.op = OPERATIONS_DECODER[layer](hidden_size, dropout, use_cln=use_cln, need_skip=need_skip)
        else:
        
            self.op = OPERATIONS_DECODER[layer](hidden_size, dropout, use_cln=use_cln)

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)

    def clear_buffer(self, *args):
        return self.op.clear_buffer(*args)

    def set_buffer(self, *args):
        return self.op.set_buffer(*args)


######################
# fastspeech modules
######################
class old_LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(old_LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(old_LayerNorm, self).forward(x)
        return super(old_LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class DurationPredictor(torch.nn.Module):
    """Duration predictor module.
    This is a module of duration predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`, those are calculated in linear domain.
    """

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0, padding='SAME'):
        """Initilize duration predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.cattn = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                # LayerNorm(n_chans, dim=1),
                LayerNorm(n_chans, use_cln=hparams["prior_use_cln"]) if hparams["dur_cln"] else old_LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
            if hparams['predictor_use_cattention'] and idx % hparams['predictor_ca_per_layer'] == 0:
                kv_dim = hparams['hidden_size'] if hparams['use_ref_enc'] or len(hparams['ref_enc_arch']) else hparams['vqemb_size']
                self.cattn.append(torch.nn.Sequential(
                    MultiheadAttention(hparams['residual_channels'], 8, self_attention=False, dropout=0.1, kdim=kv_dim, vdim=kv_dim),
                    LayerNorm(hparams['residual_channels']),
                    nn.Dropout(0.2),
                    ))
        self.linear = torch.nn.Linear(n_chans, 1)

    def _forward(self, xs, x_masks=None, is_inference=False, condition=None, condition_mask=None):
        # print(hparams["prior_use_cln"], 'condition', condition.shape)
        if condition_mask is not None:
            condition_mask = condition_mask.squeeze(2)
            condition_mask = ~ condition_mask
            # print(condition_mask)
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for idx, (conv, act, norm, dropout) in enumerate(self.conv):
            if idx != 0 and hparams['predictor_use_res']: 
                res = xs
            if hparams['predictor_use_cattention'] and idx % hparams['predictor_ca_per_layer'] == 0:
                attn_idx= idx // hparams['predictor_ca_per_layer']
                attn, ln, drop = self.cattn[attn_idx]
                residual = y_ = xs.transpose(1, 2)
                y_ = ln(y_)
                # print(y_.transpose(0, 1).shape, condition.transpose(0, 1).shape, condition_mask.shape)
                # print('before', idx, attn_idx, torch.isnan(y_).any(),torch.isnan(xs).any(), torch.isnan(condition).any(), torch.isnan(condition_mask).any())
                y_, _, = attn(y_.transpose(0, 1), condition.transpose(0, 1), condition.transpose(0, 1), key_padding_mask=condition_mask)
                # print('after', idx, attn_idx, torch.isnan(y_).any(),torch.isnan(xs).any(), torch.isnan(condition).any(), torch.isnan(condition_mask).any())
                y_ = drop(y_.transpose(0, 1))
                
                y_ = (residual + y_) / sqrt(2.0)
                xs = y_.transpose(1, 2)
                
                
            if self.padding == 'SAME':
                xs = F.pad(xs, [self.kernel_size // 2, self.kernel_size // 2])
            elif self.padding == 'LEFT':
                xs = F.pad(xs, [self.kernel_size - 1, 0])
                
            
                # exit()
                
            xs = conv(xs)
            xs = act(xs)  
            if hparams["dur_cln"]:
                xs = xs.permute(2, 0, 1)  # B C T -> T B C
                xs = norm(xs, condition=condition)
                xs = xs.permute(1, 2, 0)  # T B C -> B C T
            else:
                xs = norm(xs)
            xs = dropout(xs) # (B, C, Tmax)
            # xs = f(xs)  # (B, C, Tmax)
            if idx != 0 and hparams['predictor_use_res']: 
                xs = res + xs
            if x_masks is not None:
                xs = xs * (1 - x_masks.to(xs.dtype))[:, None, :]
            
            

        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)  # (B, Tmax)

        if is_inference:
            # NOTE: calculate in linear domain
            xs = torch.clamp(torch.round(xs.exp() - self.offset), min=0).long()  # avoid negative value

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)
        return xs

    def forward(self, xs, x_masks=None, condition=None, condition_mask=None):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).
        """
        return self._forward(xs, x_masks, False, condition=condition, condition_mask=condition_mask)

    def inference(self, xs, x_masks=None, condition=None, condition_mask=None):
        """Inference duration.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).
        """
        return self._forward(xs, x_masks, True, condition=condition, condition_mask=condition_mask)


class DurationPredictorLoss(torch.nn.Module):
    """Loss function module for duration predictor.
    The loss value is Calculated in log domain to make it Gaussian.
    """

    def __init__(self, offset=1.0, reduction="none"):
        """Initilize duration predictor loss module.
        Args:
            offset (float, optional): Offset value to avoid nan in log domain.
            reduction (str): Reduction type in loss calculation.
        """
        super(DurationPredictorLoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction=reduction)
        self.offset = offset

    def forward(self, outputs, targets, nonpadding):
        """Calculate forward propagation.
        Args:
            outputs (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Batch of groundtruth durations in linear domain (B, T)
        Returns:
            Tensor: Mean squared error loss value.
        Note:
            `outputs` is in log domain but `targets` is in linear domain.
        """
        # NOTE: outputs is in log domain while targets in linear
        targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets.to(outputs.dtype))
        loss = (loss * nonpadding).sum() / nonpadding.sum()
        return loss


def pad_list(xs, pad_value, max_len=None):
    """Perform padding for the list of tensors.
    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.
    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).
    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])
    """
    n_batch = len(xs)
    if max_len is None:
        max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, :min(xs[i].size(0), max_len)] = xs[i][:max_len]

    return pad


class LengthRegulator(torch.nn.Module):
    """Length regulator module for feed-forward Transformer.
    This is a module of length regulator described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The length regulator expands char or phoneme-level embedding features to frame-level by repeating each
    feature based on the corresponding predicted durations.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    """

    def __init__(self, pad_value=0.0):
        """Initilize length regulator module.
        Args:
            pad_value (float, optional): Value used for padding.
        """
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value

    def forward(self, ds, ilens, alpha=1.0, max_len=None):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
            ds (LongTensor): Batch of durations of each frame (B, T).
            ilens (LongTensor): Batch of input lengths (B,).
            alpha (float, optional): Alpha value to control speed of speech.
        Returns:
            Tensor: replicated input tensor based on durations (B, T*, D).
        """
        assert alpha > 0
        if alpha != 1.0:
            ds = torch.round(ds.float() * alpha).long()
        ds = [d[:ilen] for d, ilen in zip(ds, ilens)]
        mel2ph = [self._repeat_one_sequence(torch.arange(len(d)).to(d.device), d) + 1 for d in ds]
        return pad_list(mel2ph, 0, max_len).long()

    def _repeat_one_sequence(self, x, d):
        """Repeat each frame according to duration.
        Examples:
            >>> x = torch.tensor([[1], [2], [3]])
            tensor([[1],
                    [2],
                    [3]])
            >>> d = torch.tensor([1, 2, 3])
            tensor([1, 2, 3])
            >>> self._repeat_one_sequence(x, d)
            tensor([[1],
                    [2],
                    [2],
                    [3],
                    [3],
                    [3]])
        """
        if d.sum() == 0:
            logging.warn("all of the predicted durations are 0. fill 0 with 1.")
            d = d.fill_(1)
        return torch.cat([x_.repeat(int(d_), 1) for x_, d_ in zip(x, d) if d_ != 0], dim=0)


class PitchPredictor(torch.nn.Module):
    def __init__(self, idim, n_layers=5, n_chans=384, odim=2, kernel_size=5,
                 dropout_rate=0.1, padding='SAME'):
        """Initilize pitch predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(PitchPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.cattn = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.padding = padding
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                # torch.nn.ConstantPad1d(((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                #                        if padding == 'SAME'
                #                        else (kernel_size - 1, 0), 0),
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=0),
                torch.nn.ReLU(),
                LayerNorm(n_chans, use_cln=hparams["prior_use_cln"]) if hparams["pitch_cln"] else old_LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
            if hparams['predictor_use_cattention'] and idx % hparams['predictor_ca_per_layer'] == 0:
                kv_dim = hparams['hidden_size'] if hparams['use_ref_enc'] or len(hparams['ref_enc_arch']) else hparams['vqemb_size']
                self.cattn.append(torch.nn.Sequential(
                    MultiheadAttention(hparams['residual_channels'], 8, self_attention=False, dropout=0.1, kdim=kv_dim, vdim=kv_dim),
                    LayerNorm(hparams['residual_channels']),
                    nn.Dropout(0.2),
                    ))
            
        self.linear = torch.nn.Linear(n_chans, odim)
        self.embed_positions = SinusoidalPositionalEmbedding(idim, 0, init_size=4096)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))
        

    def forward(self, xs, condition=None, condition_mask=None):
        """

        :param xs: [B, T, H]
        :return: [B, T, H]
        """
        if condition_mask is not None:
            condition_mask = condition_mask.squeeze(2)
            condition_mask = ~ condition_mask
        positions = self.pos_embed_alpha * self.embed_positions(xs[..., 0])
        xs = xs + positions
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for idx, (conv, act, norm, dropout) in enumerate(self.conv):
            if idx != 0 and hparams['predictor_use_res']: 
                res = xs
            if hparams['predictor_use_cattention'] and idx % hparams['predictor_ca_per_layer'] == 0:
                attn_idx= idx // hparams['predictor_ca_per_layer']
                attn, ln, drop = self.cattn[attn_idx]
                residual = y_ = xs.transpose(1, 2)
                y_ = ln(y_)
                
                y_, _, = attn(y_.transpose(0, 1), condition.transpose(0, 1), condition.transpose(0, 1), key_padding_mask=condition_mask)
                
                y_ = drop(y_.transpose(0, 1))
                
                y_ = (residual + y_) / sqrt(2.0)
                xs = y_.transpose(1, 2)
                # print(xs.shape)
                # exit()
            
            if self.padding == 'SAME':
                xs = F.pad(xs, [self.kernel_size // 2, self.kernel_size // 2])
            elif self.padding == 'LEFT':
                xs = F.pad(xs, [self.kernel_size - 1, 0])
            xs = conv(xs)
            xs = act(xs)
            if hparams["pitch_cln"]:
                xs = xs.permute(2, 0, 1)  # B C T -> T B C
                xs = norm(xs, condition=condition)
                xs = xs.permute(1, 2, 0)  # T B C -> B C T
            else:
                xs = norm(xs)
            xs = dropout(xs) # (B, C, Tmax)
            if idx != 0 and hparams['predictor_use_res']: 
                xs = res + xs
            # xs = f(xs)  # (B, C, Tmax)

        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1))  # (B, Tmax, H)
        return xs


class EnergyPredictor(PitchPredictor):
    pass


class FastspeechDecoder(nn.Module):
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
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(self.arch[i], self.hidden_size, self.dropout, use_cln=use_cln)
            for i in range(self.num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, require_w=False, condition=None):
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
        # encoder layers
        attn_w = []
        if require_w:
            for layer in self.layers:
                x, attn_w_i = layer(x, encoder_padding_mask=padding_mask, require_w=require_w, condition=condition)
                attn_w.append(attn_w_i)
        else:
            # modules/operations.py:122, modules.operations.EncSALayer
            for layer in self.layers:
                x = layer(x, encoder_padding_mask=padding_mask, condition=condition)  # remember to assign back to x
        x = self.layer_norm(x)
        x = x.transpose(0, 1)

        return (x, attn_w) if require_w else x
