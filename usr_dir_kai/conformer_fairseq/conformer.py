# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn

from .conformer_layer import ConformerEncoderLayer

from .position_encoding import RelPositionalEncoding

from usr_dir.module.wavnet import AttrDict
from utils.hparams import hparams

logger = logging.getLogger(__name__)

def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask



class S2TConformerEncoder(nn.Module):
    """Conformer Encoder for speech translation based on https://arxiv.org/abs/2005.08100"""

    def __init__(self,):
        super().__init__()

        self.encoder_freezing_updates = 0
        self.num_updates = 0
        
        self.args = args = AttrDict(
            # Model params
            encoder_embed_dim=hparams['transformer_hidden'],
            encoder_ffn_embed_dim=2048,
            encoder_attention_heads=getattr(hparams, "conformer_attention_heads", 8),
            condition_hidden=hparams["hidden_size"],
            latent_dim=hparams['vqemb_size'],
            pos_enc_type="rel_pos",
            depthwise_conv_kernel_size=getattr(hparams, "depthwise_conv_kernel_size", 31),
            encoder_layers=getattr(hparams, "encoder_layers", 24),
            dropout=0.2,
            max_source_positions=4000,
        )

        self.embed_scale = math.sqrt(args.encoder_embed_dim)

        self.padding_idx = 1

        self.pos_enc_type = args.pos_enc_type

        self.embed_positions = RelPositionalEncoding(
            args.max_source_positions, args.encoder_embed_dim
        )
        

        self.linear = torch.nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.dropout = torch.nn.Dropout(args.dropout)
        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    embed_dim=args.encoder_embed_dim,
                    ffn_embed_dim=args.encoder_ffn_embed_dim,
                    attention_heads=args.encoder_attention_heads,
                    dropout=args.dropout,
                    depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,
                    pos_enc_type=self.pos_enc_type,
                )
                for _ in range(args.encoder_layers)
            ]
        )

    def forward(self, speech_latent, condition, diffusion_step, input_lengths, return_all_hiddens=False):
        """
        Args:
            src_tokens: Input source tokens Tensor of shape B X T X C
            src_lengths: Lengths Tensor corresponding to input source tokens
            return_all_hiddens: If true will append the self attention states to the encoder states
        Returns:
            encoder_out: Tensor of shape B X T X C
            encoder_padding_mask: Optional Tensor with mask
            encoder_embedding: Optional Tensor. Always empty here
            encoder_states: List of Optional Tensors wih self attention states
            src_tokens: Optional Tensor. Always empty here
            src_lengths: Optional Tensor. Always empty here
        """
        
        x = speech_latent
        
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        x = self.embed_scale * x
        
        positions = self.embed_positions(x)

        

        x = self.linear(x) + condition + diffusion_step
        x = self.dropout(x)
        encoder_states = []
        
        encoder_padding_mask = encoder_padding_mask.transpose(0, 1) # [T, B]
        
        # print(x.shape, encoder_padding_mask.shape, positions.shape, diffusion_step.shape, "------------")
        # exit(0)

        # x is T X B X C
        for layer in self.conformer_layers:
            x, _ = layer(x, encoder_padding_mask, positions, diffusion_step=diffusion_step)
            if return_all_hiddens:
                encoder_states.append(x)

        return {
            "encoder_out": [x.transpose(0, 1)],  # B, T, C
            "encoder_padding_mask": [encoder_padding_mask]
            if encoder_padding_mask.any()
            else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }



# @register_model("s2t_conformer")
# class S2TConformerModel(S2TTransformerModel):
#     def __init__(self, encoder, decoder):
#         super().__init__(encoder, decoder)

#     @staticmethod
#     def add_args(parser):
#         S2TTransformerModel.add_args(parser)
#         parser.add_argument(
#             "--input-feat-per-channel",
#             type=int,
#             metavar="N",
#             help="dimension of input features per channel",
#         )
#         parser.add_argument(
#             "--input-channels",
#             type=int,
#             metavar="N",
#             help="number of chennels of input features",
#         )
#         parser.add_argument(
#             "--depthwise-conv-kernel-size",
#             type=int,
#             metavar="N",
#             help="kernel size of depthwise convolution layers",
#         )
#         parser.add_argument(
#             "--attn-type",
#             type=str,
#             metavar="STR",
#             help="If not specified uses fairseq MHA. Other valid option is espnet",
#         )
#         parser.add_argument(
#             "--pos-enc-type",
#             type=str,
#             metavar="STR",
#             help="Must be specified in addition to attn-type=espnet for rel_pos and rope",
#         )

#     @classmethod
#     def build_encoder(cls, args):
#         encoder = S2TConformerEncoder(args)
#         pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
#         if pretraining_path is not None:
#             if not Path(pretraining_path).exists():
#                 logger.warning(
#                     f"skipped pretraining because {pretraining_path} does not exist"
#                 )
#             else:
#                 encoder = checkpoint_utils.load_pretrained_component_from_model(
#                     component=encoder, checkpoint=pretraining_path
#                 )
#                 logger.info(f"loaded pretrained encoder from: {pretraining_path}")
#         return encoder


# @register_model_architecture("s2t_conformer", "s2t_conformer")
# def conformer_base_architecture(args):
#     args.attn_type = getattr(args, "attn_type", None)
#     args.pos_enc_type = getattr(args, "pos_enc_type", "abs")
#     args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
#     args.input_channels = getattr(args, "input_channels", 1)
#     args.max_source_positions = getattr(args, "max_source_positions", 6000)
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
#     args.dropout = getattr(args, "dropout", 0.1)
#     args.encoder_layers = getattr(args, "encoder_layers", 16)
#     args.depthwise_conv_kernel_size = getattr(args, "depthwise_conv_kernel_size", 31)
#     transformer_base_architecture(args)