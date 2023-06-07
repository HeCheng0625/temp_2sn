import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from utils.hparams import hparams


class ReferenceEncoder(nn.Module):
    '''
        inputs --- [N, Ty/r, n_mels*r]  mels
        outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, ref_speaker_embedding_dim, reference_encoder_filters=[32, 32, 64, 64, 128, 128], audio_num_mel_bins=80):
        super().__init__()
        K = len(reference_encoder_filters)  # the length of layer 6
        filters = [1] + reference_encoder_filters
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.audio_num_mel_bins = audio_num_mel_bins
        
        if hparams["use_new_refenc"]:
            print('use new ref enc')
            self.norms = nn.ModuleList(
                [nn.GroupNorm(num_groups=32, num_channels=reference_encoder_filters[i]) for i in range(K)])
            self.pool = nn.AdaptiveAvgPool1d(output_size=1)
            self.out_channels = ref_speaker_embedding_dim
        else:
            self.bns = nn.ModuleList(
                [nn.BatchNorm2d(num_features=reference_encoder_filters[i]) for i in range(K)])
            out_channels = self.calculate_channels(audio_num_mel_bins, 3, 2, 1, K)
            self.gru = nn.GRU(input_size=reference_encoder_filters[-1] * out_channels,
                            hidden_size=ref_speaker_embedding_dim,
                            batch_first=True)

    def forward(self, inputs, mask=None):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.audio_num_mel_bins)  # [N, 1, Ty, n_mels]
        if mask is not None:
            mask = mask.view(N, 1, -1, 1)   # [N, 1, Ty, 1]
        norms = self.norms if hparams["use_new_refenc"] else self.bns
        for conv, norm in zip(self.convs, norms):  
            res = out
            out = norm(conv(out))
            # res torch.Size([20, 1, 1000, 256])
            # out torch.Size([20, 32, 500, 128])
            # out = res + out
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]
            if mask is not None:
                mask = mask[..., ::2, :]
                out = out * mask

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        # print(out.shape)
        T = out.size(1)
        N = out.size(0)
        
        if hparams["use_new_refenc"]:
            out = out.contiguous().view(N, -1, self.out_channels)  
            out = out.transpose(1, 2)
            # print('before pooling', out.shape)
            out = self.pool(out)   # N, E, 1
            # print('after pooling', out.shape)
            out = out.squeeze(-1)
            # exit()
        else:
            out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]
            self.gru.flatten_parameters()
            memory, out = self.gru(out)  # out --- [1, N, E]
            out = out.squeeze(0)  # [N, E]

        return out

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

