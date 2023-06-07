import numpy as np
import torch
import torch.nn as nn
from .residual_vq import ResidualVQ



class ResnetBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1, causal=False):
        super().__init__()
        if 3 == kernel_size:
          pad_dilation = dilation
        elif 5 == kernel_size:
          pad_dilation = 2*dilation
        elif 7 == kernel_size:
          pad_dilation = 3*dilation

        if causal is True:
          pad_dilation = (2 * pad_dilation, 0)
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(pad_dilation),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, dilation=dilation),
            nn.LeakyReLU(0.2),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

        self.shortcut = nn.Conv1d(dim, dim, kernel_size=1)
        self.alpha = nn.Parameter(torch.Tensor(dim, 1))
        with torch.no_grad():
            self.alpha.fill_(0)

    def forward(self, x):
        #print(x.size())
        return self.shortcut(x) + self.block(x) * self.alpha




class CodecDecoder(nn.Module):
    def __init__(self,
                 in_channels=256,
                 ngf=32,
                 n_residual_layers=(3, 3, 3, 3),
                 dropout=0.1,
                 use_rnn=False,
                 rnn_bidirectional=False,
                 rnn_num_layers=1,
                 rnn_dropout=0.,
                 up_ratios=(2, 5, 5, 4),
                 vq_num_quantizers=16,
                 vq_dim=256,
                 vq_commit_weight=0.005,
                 vq_weight_init=False,
                 vq_full_commit_loss=False,
                 codebook_size=1024,
                 
                 use_weight_norm=True):
        super().__init__()
        self.hop_length = np.prod(up_ratios)   # 200
        self.ngf = ngf
        self.up_ratios = up_ratios
        mul = int(2 ** len(up_ratios))   # 2**4 = 16

        use_rnn = True
        rnn_bidirectional = True

            #num_quantizers=16, dim=128, codebook_size=1024,
            #num_quantizers=4, dim=1024, codebook_size=1024,
            # default: num_quantizers=16, dim=256, codebook_size=1024
        self.quantizer = ResidualVQ(
            num_quantizers=vq_num_quantizers, dim=vq_dim, codebook_size=codebook_size,
            threshold_ema_dead_code=2, commitment=vq_commit_weight, weight_init=vq_weight_init, full_commit_loss=vq_full_commit_loss
        )

        self.model = nn.ModuleDict()

        pre_model = [
            #nn.ReflectionPad1d((6,0)),
            nn.ReflectionPad1d(3),
            nn.Conv1d(in_channels, mul * ngf, kernel_size=7),
            nn.LeakyReLU(0.2),
        ]

        '''
        if use_rnn:
            pre_model += [
                nn.Dropout(dropout),
                GRU(in_size=mul * ngf,
                    out_size=mul * ngf,
                    num_layers=rnn_num_layers,
                    bidirectional=rnn_bidirectional,
                    dropout=rnn_dropout)
            ]
        '''

        self.model['pre_model'] = nn.Sequential(*pre_model)

        # Upsample to raw audio scale
        for i, r in enumerate(up_ratios):
            upsample_model = [
                nn.ConvTranspose1d(
                    in_channels=mul * ngf,
                    out_channels=mul * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]
            '''
            upsample_model = [
                nn.ConvTranspose1d(
                    in_channels=mul * ngf,
                    out_channels=mul * ngf // 2,
                    kernel_size=r,
                    stride=r),
            ]
            '''

            for j in range(n_residual_layers[i]):
                upsample_model += [ResnetBlock(mul * ngf // 2, dilation=3 ** j, causal = False)]

            upsample_model += [nn.LeakyReLU(0.2)]

            mul //= 2

            self.model[f'upsample_{i}'] = nn.Sequential(*upsample_model)

        self.model['post_model'] = nn.Sequential(
            #nn.ReflectionPad1d((6,0)),
            nn.ReflectionPad1d(3),
            nn.Conv1d(ngf, 1, kernel_size=7),
            nn.Tanh(),
        )
        '''
        self.speaker_embedding = torch.nn.Embedding(num_embeddings=1024, embedding_dim=1024,
                                                    padding_idx=0)
        self.speaker_embeddinglinear = torch.nn.Linear(1024, 1024)
        self.speaker_embedding_proj = torch.nn.Linear(1024 + 1024, 1024)
        self.emb_act = nn.Softsign()
        '''

        if use_weight_norm:
            self.apply_weight_norm()
        self.reset_parameters()

    def forward(self, x, hid_in=None, global_steps=100000, vq=True, get_vq=False, eval_vq=True, early_stop=-1):
        if get_vq:
            return self.quantizer.get_emb()
        if vq is True:
            if eval_vq:
                self.quantizer.eval()
            x, q, commit_loss = self.quantizer(x)   # (b, 256, d/200), (16, b, d/200), (16,1) 
            return x, q, commit_loss

        '''
        x = x.transpose(1, 2)
        spembs = self.speaker_embedding(speaker_id)
        if len(spembs.size()) == 2:
            spembs = spembs.unsqueeze(1)
        spembs = self.emb_act(self.speaker_embeddinglinear(spembs))
        spembs = spembs.repeat(1, x.size(1), 1)
        x = self.speaker_embedding_proj(torch.cat([x, spembs], -1))
        x = x.transpose(1, 2)
        '''

        if hid_in != None:
            hid_idx = len(hid_in)-2
        for _, layer in self.model.items():
            x = layer(x)
            if global_steps<=1000:
              #if _ == "pre_model":
                #x += hid_in[hid_idx]
              if _.split("_")[0] == "upsample" and int(_.split("_")[1])<3:
                x += hid_in[hid_idx]
                hid_idx -= 1
        return x   # (b, 1, d)

    def vq2emb(self, vq):
        self.quantizer = self.quantizer.eval()
        x = self.quantizer.vq2emb(vq)
        return x

    def get_emb(self):
        self.quantizer = self.quantizer.eval()
        embs = self.quantizer.get_emb()
        return embs

    # def inference_vq(self, vq):

    #     x = self.vq2emb(vq)
    #     x = torch.transpose(x, 1, 2)

    #     for _, layer in self.model.items():
    #         x = layer(x)
    #     return x

    def inference_0(self, x):
        x, q, loss = self.quantizer(x)
        for _, layer in self.model.items():
            x = layer(x)
        return x, None
    
    def inference(self, x):
        for _, layer in self.model.items():
            
            x = layer(x)

        return x, None


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
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.02)
        self.apply(_reset_parameters)
