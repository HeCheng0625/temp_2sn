import torch
import numpy as np
import math
from utils.hparams import hparams

class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        """
        Returns number of trainable parameters of the module.
        """
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params


    def relocate_input(self, x: list):
        """
        Relocates provided tensors to the same device set for the module.
        """
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x

class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class Diffusion(BaseModule):
    def __init__(self, n_feats, dim,
                 n_spks=1, spk_emb_dim=64,
                 beta_min=0.05, beta_max=20, pe_scale=1000):
        super(Diffusion, self).__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        self.diffusion_loss_type = hparams["diffusion_loss_type"]

        self.predictor_type = hparams["predictor_type"]

        if self.predictor_type == "transformer_post_dualres_cat_fix_mask":
            from .transformer_post_dualres_cat_fix_mask import TransformerEstimator
            self.estimator = TransformerEstimator(arch=hparams["transformer_esitimator_arch"])
        elif self.predictor_type == "transformer_pre_fix_mask":
            from .transformer_pre_fix_mask import TransformerEstimator
            self.estimator = TransformerEstimator(arch=hparams["transformer_esitimator_arch"])
        elif self.predictor_type == "wavnet_crossattn":
            from .slfattnwavnet_cross_attention import WavNet
            self.estimator = WavNet(in_dims=n_feats)
        else:
            raise NotImplementedError("unknown predictor type: {}".format(self.predictor_type))
    
    def forward_diffusion(self, x0, mu, t):
        """t \in [0, 1], alpha_t = cos(0.5*pi*t), sigmod_t = sin(0.5*pi*t)"""
        time = t.unsqueeze(-1).unsqueeze(-1)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                requires_grad=False)
        xt = torch.cos(0.5*math.pi*time) * x0 + torch.sin(0.5*math.pi*time) * z
        v_gt = torch.cos(0.5*math.pi*time) * z - torch.sin(0.5*math.pi*time) * x0
        return xt, z, v_gt

    @torch.no_grad()
    def reverse_diffusion(self, z, x_mask, mu, n_timesteps, spk=None, stoc=False, prompt=None, prompt_mask=None):
        t_min = 1e-5
        t_max = 1 - 1e-5
        h = (t_max - t_min) / n_timesteps
        h = h * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
        xt = z
        for i in range(n_timesteps):
            t = t_max - i * h
            v_predict = self.estimator(spec=xt, x_mask=x_mask, cond=mu, diffusion_step=t, spk=spk, prompt=prompt, prompt_mask=prompt_mask)
            xt = math.cos(0.5*math.pi*h) * xt - math.sin(0.5*math.pi*h) * v_predict
        return xt

    @torch.no_grad()
    def forward(self, z, x_mask, mu, n_timesteps, spk=None, stoc=False, ref_x=None, prompt=None, prompt_mask=None):
        return self.reverse_diffusion(z=z, x_mask=x_mask, mu=mu, n_timesteps=n_timesteps, spk=spk, stoc=stoc, prompt=prompt, prompt_mask=prompt_mask)

    def loss_t(self, x0, x_mask, mu, t, spk=None, prompt=None, prompt_mask=None):
        if hparams['diffusion_from_prior']:
            xt, z, v_gt = self.forward_diffusion(x0, mu, t)
        else:
            xt, z, v_gt = self.forward_diffusion(x0, 0, t)

        v_predict = self.estimator(spec=xt, x_mask=x_mask, cond=mu, diffusion_step=t, spk=spk, prompt=prompt, prompt_mask=prompt_mask)
        x0_predict = torch.cos(0.5*math.pi*t.unsqueeze(-1).unsqueeze(-1)) * xt  - torch.sin(0.5*math.pi*t.unsqueeze(-1).unsqueeze(-1)) * v_predict

        return v_predict, v_gt, z, x0_predict
    
    def compute_loss(self, x0, x_mask, mu, spk=None, offset=1e-5, pos_ids=None, prompt=None, prompt_mask=None):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1 - offset)
        return self.loss_t(x0, x_mask, mu, t, spk=spk, prompt=prompt, prompt_mask=prompt_mask)
