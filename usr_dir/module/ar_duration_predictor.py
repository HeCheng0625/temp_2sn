import torch
import torch.nn as nn
from modules.tts_modules import TransformerEncoderLayer, TransformerDecoderLayer, \
    DEFAULT_MAX_SOURCE_POSITIONS, DEFAULT_MAX_TARGET_POSITIONS
from modules.operations import *


class TransformerDecoder(nn.Module):
    def __init__(self, arch, vocab_size, padding_idx=0, causal=True, dropout=None, out_dim=None):
        super().__init__()
        self.arch = arch
        self.num_layers = len(arch)
        self.hidden_size = hparams['hidden_size']
        self.prenet_hidden_size = hparams['prenet_hidden_size']
        self.padding_idx = padding_idx
        self.causal = causal
        self.dropout = hparams['dropout'] if dropout is None else dropout
        self.in_dim = hparams['audio_num_mel_bins']
        self.out_dim = hparams['audio_num_mel_bins'] + 1 if out_dim is None else out_dim
        self.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
        self.embed_positions = SinusoidalPositionalEmbedding(
            self.hidden_size, self.padding_idx,
            init_size=self.max_target_positions + self.padding_idx + 1,
        )
        self.layers = nn.ModuleList([])
        # self.layers.extend([
        #     TransformerDecoderLayer(self.arch[i], self.hidden_size, self.dropout)
        #     for i in range(self.num_layers)
        # ])
        
        self.layers.extend([
            TransformerDecoderLayer(self.arch[i], self.hidden_size, self.dropout, use_cln=hparams['use_ref_enc'])
            for i in range(self.num_layers)
        ])
        
        self.layer_norm = LayerNorm(self.hidden_size, use_cln=True)
        self.project_out_dim = Linear(self.hidden_size, vocab_size, bias=True)
        
        self.embed_scale = math.sqrt(self.hidden_size)
        self.token_emb = nn.Embedding(vocab_size + 3, self.hidden_size)


    def forward(
            self,
            encoder_out=None,  # T x B x C
            encoder_padding_mask=None,  # B x T x C
            previous_target_seq=None,
            incremental_state=None,
            condition=None
    ):
        # embed positions
        if incremental_state is not None:
            positions = self.embed_positions(
                previous_target_seq,
                incremental_state=incremental_state
            )
            
            previous_tgt_emb = self.token_emb(previous_target_seq) * self.embed_scale

        else:
            positions = self.embed_positions(
                previous_target_seq,
                incremental_state=incremental_state
            )
            
            previous_tgt_emb = self.token_emb(previous_target_seq) * self.embed_scale
            
        x = encoder_out    

        # convert mels through prenet
        
        # embed positions
        x += positions
        
        x += previous_tgt_emb
        
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        for layer in self.layers:
            if incremental_state is None and self.causal:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
            x, _ = layer(x, encoder_padding_mask=encoder_padding_mask, self_attn_mask=self_attn_mask, condition=condition)

        x = self.layer_norm(x, condition=condition)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # B x T x C -> B x T x 81
        x = self.project_out_dim(x)
        
        return x

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
                not hasattr(self, '_future_mask')
                or self._future_mask is None
                or self._future_mask.device != tensor.device
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]



class ARDurationPredictor(nn.Module):
    def __init__(self, arch, idim, n_chans, dropout_rate) -> None:
        super().__init__()
        if isinstance(arch, str):
            self.arch = list(map(int, arch.strip().split()))
        else:
            assert isinstance(arch, (list, tuple))
            self.arch = arch
        
        self.max_duration = 30
        
        self.sos_duration = self.max_duration + 1
        
        self.decoder = TransformerDecoder(arch=self.arch, out_dim=n_chans, vocab_size=self.max_duration + 3)
        
        
    def forward(self, enc_output, mask, dur_gt, condition):
        
        dur_gt = torch.clamp(dur_gt, 0, self.max_duration)
        
        previous_duration = torch.cat(
            ((torch.zeros(dur_gt.shape[0], 1, dtype=dur_gt.dtype, device=dur_gt.device).fill_(self.sos_duration), dur_gt[:, :-1])), dim=1
        )
        
        x = self.decoder(encoder_out=enc_output, encoder_padding_mask=mask, previous_target_seq=previous_duration, condition=condition)
        
        return x, dur_gt
    
    def inference(self, enc_output, mask, condition):
        incremental_state = {}
        
        bsz, decode_length, _ = enc_output.shape

        decoder_input = enc_output.new(bsz, decode_length + 1).fill_(
            self.sos_duration).long()
        
        print(mask.shape)
        
        output = []
        for step in range(decode_length):
            print(step)
            decoder_output = self.decoder(encoder_out=enc_output[:, :step + 1, :], encoder_padding_mask=mask[:, :step + 1], incremental_state=None,
                                                       previous_target_seq=decoder_input[:, :step + 1], condition=condition)
            
            pred = decoder_output[:, -1, :].argmax(-1)
            decoder_input[:, step + 1] = pred
            
            output.append(pred.view(bsz, 1))
        
        output_dur = torch.cat(output, dim=1)
        
        print(output_dur.shape)
        print(output_dur)
        exit(0)
            
        
        
        pass


