import torch
import numpy as np

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
    
def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        noise = beta_init*t + 0.5*(beta_term - beta_init)*(t**2)
    else:
        noise = beta_init + (beta_term - beta_init)*t
    return noise

class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))



class Diffusion(BaseModule):
    def __init__(self, n_feats, dim,
                 n_spks=1, spk_emb_dim=64,
                ):
        super(Diffusion, self).__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.sigma_data = 0.5
        self.P_mean = -1.2
        self.P_std = 1.2
        
        # self.difformer_f = getattr(cust_opt, "difformer_f", 1)
        
        self.diffusion_loss_type = hparams["diffusion_loss_type"]
        
        self.spk_mlp = torch.nn.Sequential(torch.nn.Linear(hparams['hidden_size'], hparams['hidden_size'] * 4), Mish(),
                                               torch.nn.Linear(hparams['hidden_size'] * 4, hparams['vqemb_size']))
        self.predictor_type = hparams["predictor_type"]
        if self.predictor_type == "wavnet":
            from .wavnet import WavNet
            
            self._estimator = WavNet(in_dims=n_feats)
            
        elif self.predictor_type == "transformer":

            from .transformer_esitimator import TransformerEstimator
            self._estimator = TransformerEstimator(arch=hparams["transformer_esitimator_arch"])
        elif self.predictor_type == "sawavnet":
            from .slfattnwavnet import WavNet
            
            self._estimator = WavNet(in_dims=n_feats)
            
        elif self.predictor_type == "wavnet_crossattn":
            from .slfattnwavnet_cross_attention import WavNet
            self._estimator = WavNet(in_dims=n_feats)
        elif self.predictor_type == "wavnet_refcat":  
            from .catwavnet import WavNet
            self._estimator = WavNet(in_dims=n_feats)
        else:
            raise NotImplementedError("unknown predictor type: {}".format(self.predictor_type))
        
        assert not hparams['diffusion_from_prior']
        # print(self.estimator)

    def forward_diffusion(self, x0, sigma):
        sigma = sigma.unsqueeze(-1).unsqueeze(-1)
        
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                        requires_grad=False)
        xt = x0 +  z * sigma
        return xt, z
    
    def estimator(self, spec, x_mask, cond, sigma, spk, prompt=None, prompt_mask=None):
        c_skip = (self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2))[:, None, None]
        c_out = (sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt())[:, None, None]
        c_in = (1 / (self.sigma_data ** 2 + sigma ** 2).sqrt())[:, None, None]
        c_noise = sigma.log() / 4
        # print(c_in.shape, spec.shape)
        
        F_x = self._estimator(spec=c_in * spec, diffusion_step=c_noise, x_mask=x_mask, cond=cond,spk=spk, prompt=prompt, prompt_mask=prompt_mask)
        D_x = c_skip * spec + c_out * F_x
        return D_x
        
    
    @torch.no_grad()
    def reverse_diffusion(self, z, x_mask, mu, n_timesteps, spk=None, stoc=False, prompt=None, prompt_mask=None,
                          S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
        
        step_indices = torch.arange(n_timesteps).to(z.device)
        t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (n_timesteps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        t = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
        # Main sampling loop.
        x_next = z * t_steps[0]  # sample x0 from N(0, sigma(t0)I)
        
        
        
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            print(t_cur, t_next)
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / n_timesteps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = t_cur + gamma * t_cur
            print(t_hat)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
            

            # Euler step.
            denoised = self.estimator(spec=x_hat, x_mask=x_mask, cond=mu, sigma=t_hat[None], spk=spk, prompt=prompt, prompt_mask=prompt_mask)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < n_timesteps - 1:
                denoised = self.estimator(spec=x_next, x_mask=x_mask, cond=mu, sigma=t_next[None], spk=spk, prompt=prompt, prompt_mask=prompt_mask)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    
    @torch.no_grad()
    def forward(self, z, x_mask, mu, n_timesteps, spk=None, stoc=False, ref_x=None, prompt=None, prompt_mask=None):
        return self.reverse_diffusion(z=z, x_mask=x_mask, mu=mu, n_timesteps=n_timesteps, spk=spk, stoc=stoc, prompt=prompt, prompt_mask=prompt_mask)

    def loss_t(self, x0, x_mask, mu, sigma, spk=None, pos_ids=None, prompt=None, prompt_mask=None):
        # print('x0:', x0.shape, 'mu:', mu.shape)
        
        xt, z = self.forward_diffusion(x0, sigma) # xt and noise
        
        # time = t.unsqueeze(-1).unsqueeze(-1)
        # if spk is not None:
        #     xt = xt + self.spk_mlp(spk)[..., None]
        # print(prompt.shape, prompt_mask.shape)
 
        x0_estimation = self.estimator(spec=xt, x_mask=x_mask, cond=mu, sigma=sigma, spk=spk, prompt=prompt, prompt_mask=prompt_mask)
        
        noise_pred = (xt - x0_estimation) / sigma[:, None, None]
        weight = ((sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2)[:, None, None]
        
        return x0_estimation, noise_pred, z, weight
    

    def compute_loss(self, x0, x_mask, mu, spk=None, offset=1e-5, pos_ids=None, prompt=None, prompt_mask=None):
        rnd_normal = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        # t = torch.clamp(t, offset, 1.0 - offset)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        
        
        return self.loss_t(x0, x_mask, mu, sigma, spk=spk, pos_ids=pos_ids, prompt=prompt, prompt_mask=prompt_mask)
    

