import torch
from torch import nn
import numpy as np
from .codec_decoder import ResnetBlock

class CodecEncoder(nn.Module):
    def __init__(self,
                 in_channels=1,
                 ngf=32,
                 n_residual_layers=(3, 3, 3, 3),
                 dropout=0.1,
                 use_rnn=True,
                 rnn_bidirectional=False,
                 rnn_num_layers=1,
                 rnn_dropout=0.,
                 up_ratios=(4, 5, 5, 2),
                 out_channels=256,
                 use_weight_norm=True):
        super().__init__()
        self.hop_length = np.prod(up_ratios)   # 200
        self.ngf = ngf
        self.up_ratios = up_ratios
        #mul = int(2 ** len(up_ratios))

        self.model = nn.ModuleDict()

        pre_model = [
            nn.ReflectionPad1d(3),
            #nn.Conv1d(in_channels, mul * ngf, kernel_size=7),
            nn.Conv1d(in_channels, ngf, kernel_size=7),
            nn.LeakyReLU(0.2),
        ]

        self.model['pre_model'] = nn.Sequential(*pre_model)   # (b, 1, d) -> (b, 32, d)

        # Upsample to raw audio scale
        for i, r in enumerate(up_ratios):
            upsample_model = []
            for j in range(n_residual_layers[i]):   # dilation = 1, 3, 9, kernel_size = 7
                upsample_model += [ResnetBlock(ngf, kernel_size=7, dilation=3 ** j)]
            upsample_model += [nn.Conv1d(in_channels=ngf,
                                         out_channels=ngf * 2,
                                         kernel_size=r * 2,
                                         stride=r,
                                         padding=r // 2 + r % 2,)]
            upsample_model += [nn.LeakyReLU(0.2)]
            # (kernel_size, stride) = (8, 4), (10, 5), (10, 5), (4, 2)
            #mul //= 2
            ngf *= 2

            self.model[f'upsample_{i}'] = nn.Sequential(*upsample_model)

        post_model = []
        '''
        post_model = [
            nn.Dropout(dropout),
            GRU(in_size=ngf,
                out_size=ngf,
                num_layers=rnn_num_layers,
                bidirectional=rnn_bidirectional,
                dropout=rnn_dropout)
        ]
        '''

        post_model += [nn.ReflectionPad1d(3),
                       nn.Conv1d(ngf, out_channels, kernel_size=7),
                       #nn.Tanh(),
                       ]   # (b, 256, d/200)

        self.model['post_model'] = nn.Sequential(*post_model)


        if use_weight_norm:
            self.apply_weight_norm()
        self.reset_parameters()

    def forward(self, x):
        #print(x.size())
        hiddes=[]
        for _, layer in self.model.items():
            x = layer(x)
            if _.split("_")[0] == "upsample":
                hiddes.append(x)
            #print(x.size())
        #print(x.size())
        return x

    def inference(self, x):
        for _, layer in self.model.items():
            x = layer(x)
        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0.0, 0.02)
        self.apply(_reset_parameters)