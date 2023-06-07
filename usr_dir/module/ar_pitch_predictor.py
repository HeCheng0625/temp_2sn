import torch
import torch.nn as nn
from modules.tts_modules import TransformerEncoderLayer, TransformerDecoderLayer, \
    DEFAULT_MAX_SOURCE_POSITIONS, DEFAULT_MAX_TARGET_POSITIONS
from modules.operations import *
from usr_dir.module.ar_duration_predictor import TransformerDecoder




class ARPitchPredictor(nn.Module):
    def __init__(self, arch, idim, n_chans, dropout_rate) -> None:
        super().__init__()
        if isinstance(arch, str):
            self.arch = list(map(int, arch.strip().split()))
        else:
            assert isinstance(arch, (list, tuple))
            self.arch = arch
        
        self.max_pitch = 300
        
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


