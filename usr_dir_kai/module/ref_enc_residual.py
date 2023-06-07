import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from utils.hparams import hparams

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

class ReferenceEncoder(nn.Module):
    '''
        inputs --- [N, Ty/r, n_mels*r]  mels
        outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, ref_speaker_embedding_dim, reference_encoder_filters=[32, 32, 64, 64, 128, 128], sub_layers=4, audio_num_mel_bins=80):
        super().__init__()
        K = len(reference_encoder_filters)  # the length of layer 6
        filters = [1] + reference_encoder_filters
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        
        # self.short_cuts = nn.ModuleList([
        #     nn.Conv2d(in_channels=filters[i], 
        #               out_channels=filters[i + 1],
        #               kernel_size=1,
        #               stride=(2, 2))
        #     for i in range(K)
        # ])
        
        self.stride1_convs = nn.ModuleList([
            nn.Sequential(
                *[nn.Sequential(nn.Conv2d(in_channels=filters[i + 1],
                        out_channels=filters[i + 1],
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1)),
                nn.GroupNorm(num_groups=32, num_channels=filters[i + 1]),
                nn.ReLU() if _ != sub_layers -1 else nn.Identity()) for _ in range(sub_layers)
                ]
            ) for i in range(K)
        ])
        
        
        self.audio_num_mel_bins = audio_num_mel_bins
        

        self.norms = nn.ModuleList(
            [nn.GroupNorm(num_groups=32, num_channels=reference_encoder_filters[i]) for i in range(K)])
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.out_channels = ref_speaker_embedding_dim
        

    def forward(self, inputs, mask=None):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.audio_num_mel_bins)  # [N, 1, Ty, n_mels]
        if mask is not None:
            mask = mask.view(N, 1, -1, 1)   # [N, 1, Ty, 1]
        norms = self.norms if hparams["use_new_refenc"] else self.bns
        for conv, norm, sublayer in zip(self.convs, norms, self.stride1_convs):  
            
            out = norm(conv(out))
            out = F.relu(out)
            res = out
            out_sublayer = sublayer(out)

            out = out_sublayer + res

            
            # res torch.Size([20, 1, 1000, 256])
            # out torch.Size([20, 32, 500, 128])
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]
            if mask is not None:
                mask = mask[..., ::2, :]
                out = out * mask
            # print('out', out.shape)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        # print(out.shape)
        T = out.size(1)
        N = out.size(0)
        
        out = out.contiguous().view(N, -1, self.out_channels)  
        if hparams['ref_enc_pooling']:
            out = out.transpose(1, 2)
            # print('before pooling', out.shape)
            out = self.pool(out)   # N, E, 1
            # print('after pooling', out.shape)
            out = out.squeeze(-1)
        else:
            pass
            # print(out.shape)
            # exit()
            # out = out.transpose(1, 2)
            # out = torch.max_pool1d(out, kernel_size=2**len(self.convs))
            # out = out.transpose(1, 2)
            # exit()
        return out

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

