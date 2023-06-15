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
    if cumulative:   # sum(beta_0: beta_t)
        noise = beta_init*t + 0.5*(beta_term - beta_init)*(t**2)
    else:   # beta_t
        noise = beta_init + (beta_term - beta_init)*t
    return noise

class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))



class Diffusion(BaseModule):
    def __init__(self, n_feats, dim,
                 n_spks=1, spk_emb_dim=64,
                 beta_min=0.05, beta_max=20, pe_scale=1000):
        super(Diffusion, self).__init__()
        self.n_feats = n_feats
        self.dim = dim   # not used
        self.n_spks = n_spks   # not used
        self.spk_emb_dim = spk_emb_dim   # not used
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        
        # self.difformer_f = getattr(cust_opt, "difformer_f", 1)
        
        self.diffusion_loss_type = hparams["diffusion_loss_type"]
        
        self.spk_mlp = torch.nn.Sequential(torch.nn.Linear(hparams['hidden_size'], hparams['hidden_size'] * 4), Mish(),
                                               torch.nn.Linear(hparams['hidden_size'] * 4, hparams['vqemb_size']))
        self.predictor_type = hparams["predictor_type"]
        if self.predictor_type == "wavnet":
            from .wavnet import WavNet
            
            self.estimator = WavNet(in_dims=n_feats)

        elif self.predictor_type == "conformer_post_dualres_cat":
            from .conformer_post_dualres_cat import TransformerEstimator
            self.estimator = TransformerEstimator(arch=hparams["transformer_esitimator_arch"])

        elif self.predictor_type == "conformer_pre":
            from .conformer_pre import TransformerEstimator
            self.estimator = TransformerEstimator(arch=hparams["transformer_esitimator_arch"])

        elif self.predictor_type == "transformer_pre":
            from .transformer_pre import TransformerEstimator
            self.estimator = TransformerEstimator(arch=hparams["transformer_esitimator_arch"])

        elif self.predictor_type == "transformer_post_dualres_2_cat_fix_mask":
            from .transformer_post_dualres_2_cat_fix_mask import TransformerEstimator
            self.estimator = TransformerEstimator(arch=hparams["transformer_esitimator_arch"])

        elif self.predictor_type == "transformer_post_dualres_cat_fix_mask":
            from .transformer_post_dualres_cat_fix_mask import TransformerEstimator
            self.estimator = TransformerEstimator(arch=hparams["transformer_esitimator_arch"])

        elif self.predictor_type == "transformer_pre_fix_mask":
            from .transformer_pre_fix_mask import TransformerEstimator
            self.estimator = TransformerEstimator(arch=hparams["transformer_esitimator_arch"])

        # A bug version
        elif self.predictor_type == "transformer_post_dualres_cat_old":
            from .transformer_post_dualres_cat_old import TransformerEstimator
            self.estimator = TransformerEstimator(arch=hparams["transformer_esitimator_arch"])

        elif self.predictor_type == "transformer_post_dualres_cat":
            from .transformer_post_dualres_cat import TransformerEstimator
            self.estimator = TransformerEstimator(arch=hparams["transformer_esitimator_arch"])

        elif self.predictor_type == "transformer_post_dualres_cat_adaprompt":
            from .transformer_post_dualres_cat_adaprompt import TransformerEstimator
            self.estimator = TransformerEstimator(arch=hparams["transformer_esitimator_arch"])

        elif self.predictor_type == "transformer_post_dualres_cat_gru":
            from .transformer_post_dualres_cat_gru import TransformerEstimator
            self.estimator = TransformerEstimator(arch=hparams["transformer_esitimator_arch"])

        elif self.predictor_type == "sawavnet":
            from .slfattnwavnet import WavNet
            
            self.estimator = WavNet(in_dims=n_feats)
            
        elif self.predictor_type == "wavnet_crossattn":
            from .slfattnwavnet_cross_attention import WavNet
            self.estimator = WavNet(in_dims=n_feats)
        elif self.predictor_type == "wavnet_refcat":  
            from .catwavnet import WavNet
            self.estimator = WavNet(in_dims=n_feats)
        else:
            raise NotImplementedError("unknown predictor type: {}".format(self.predictor_type))
        # print(self.estimator)

    def forward_diffusion(self, x0, mu, t):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        mean = x0*torch.exp(-0.5*cum_noise/ (hparams['sigma'] ** 2)) + mu*(1.0 - torch.exp(-0.5*cum_noise/ (hparams['sigma'] ** 2)))
        variance = (hparams['sigma'] ** 2) * (1.0 - torch.exp(-cum_noise/ ((hparams['sigma'] ** 2))))
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance) * hparams['noise_factor']
        return xt, z
    
    @torch.no_grad()
    def cal_dxt(self, xt, x_mask, mu, t, spk, prompt, prompt_mask, stoc, h): 
        time = t.unsqueeze(-1).unsqueeze(-1)
        noise_t = get_noise(time, self.beta_min, self.beta_max, 
                            cumulative=False)   # beta_t
        
        x0_ = self.estimator(spec=xt, x_mask=x_mask, cond=mu, diffusion_step=t, spk=spk, prompt=prompt, prompt_mask=prompt_mask)
        if hparams['spk_dropout'] > 0.0:
            cfgs=1.
            if cfgs != 1.:
                x0_null = self.estimator(spec=xt, x_mask=x_mask, cond=mu, diffusion_step=t, spk=spk, prompt=hparams['empty_query'], prompt_mask=prompt_mask)
                x0_ = x0_null + cfgs * (x0_ - x0_null)
            
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        if hparams['diffusion_from_prior']:
            rho = x0_*torch.exp(-0.5*cum_noise/ (hparams['sigma'] ** 2)) + mu*(1.0 - torch.exp(-0.5*cum_noise)/ (hparams['sigma'] ** 2))
        else:
            rho = x0_*torch.exp(-0.5*cum_noise/ (hparams['sigma'] ** 2))   # x_t_predict
        noise_pred = xt - rho   # z * sqrt(lambda)
        
        lambda_ = (hparams['sigma'] ** 2) * (1.0 - torch.exp(-cum_noise / (hparams['sigma'] ** 2)) )
        
        logp = - noise_pred / (lambda_ + 1e-8)
        if stoc:  # SDE: dx_t = - (1/2 * x_t + logp) * beta_t * h + sqrt(beta_t) * sqrt(h) * z; z ~ N(0, 1)
            if hparams['diffusion_from_prior']:
                dxt_det = 0.5 * (mu - xt) - logp
            else:
                dxt_det = - 0.5 * xt - logp             
            
            dxt_det = dxt_det * noise_t * h
            dxt_stoc = torch.randn(dxt_det.shape, dtype=dxt_det.dtype, device=dxt_det.device,
                                    requires_grad=False) / hparams["temp"]
            dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
            dxt = dxt_det + dxt_stoc
        else:   # ODE: dx_t = -1/2 * (x_t + logp) * beta_t * h
            
            if hparams['diffusion_from_prior']:
                dxt = 0.5 * ((mu - xt) / (hparams['sigma'] ** 2) -  logp)
            else:
                dxt = 0.5 * (- xt /(hparams['sigma'] ** 2) - logp)
            dxt = dxt * noise_t * h
        return dxt
        
    @torch.no_grad()
    def reverse_diffusion(self, z, x_mask, mu, n_timesteps, spk=None, stoc=False, prompt=None, prompt_mask=None):
        h = 1.0 / n_timesteps
        xt = z
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            dxt = self.cal_dxt(xt, x_mask, mu, t, spk, prompt, prompt_mask, stoc, h)
            if hparams['infer_style'] == "default":
                xt = xt - dxt 
            elif hparams['infer_style'] == "heun":
                if i == n_timesteps - 1:
                    xt = xt - dxt 
                else:
                    xt_step = xt - dxt
                    t_step = t - h
                    dxt_step = self.cal_dxt(xt_step, x_mask, mu, t_step, spk, prompt, prompt_mask, stoc, h)
                    xt = xt - (dxt + dxt_step) / 2   # x_{t-1} = x_t - (dx_t + dx_{t-1}) / 2
            else:
                raise NotImplementedError()
        return xt
    
    
    def forward_diffusion_fromt(self, xt, mu, t, delta_t):
        
        # return x_{t+delta_t} given mu and x_t
        
        time = t.unsqueeze(-1).unsqueeze(-1)
        new_t = (t+delta_t).unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        cum_noise_new = get_noise(new_t, self.beta_min, self.beta_max, cumulative=True)  # != cumulative=False
        cum_noise_delta = cum_noise_new - cum_noise
        
        mean = xt*torch.exp(-0.5*cum_noise_delta) + mu*(1.0 - torch.exp(-0.5*cum_noise_delta))
        variance = 1.0 - torch.exp(-cum_noise_delta)
        z = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, 
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance) * hparams['noise_factor']
        return xt, z
    
    @torch.no_grad()
    def reverse_diffusion_incontext(self, z, mu, n_timesteps, spk=None, stoc=False, ref_x=None):
        print(stoc)
        ref_x = ref_x.transpose(1, 2)

        ref_cnt = ref_x.shape[2]
        ref_mu = mu[:, :, :ref_cnt]
        
        h = 1.0 / n_timesteps
        xt = z
        # ref_zs = []
        # for i in range(n_timesteps):
        #     t = ((i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
        #                                          device=z.device)
        #     if i == 0:
        #         ref_z, _ = self.forward_diffusion(ref_x, ref_mu, t)
        #     else:
        #         ref_z, _ = self.forward_diffusion_fromt(ref_z, ref_mu, t - h, h)
        #     ref_zs.append(ref_z)
        # print(len(ref_zs), n_timesteps)
        # exit()
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            
            delta_t = 0
            if i == 0:
                if hparams['diffusion_from_prior']:
                    ref_z, _ = self.forward_diffusion(ref_x, ref_mu, t = t + delta_t)
                else:
                    ref_z, _ = self.forward_diffusion(ref_x, 0, t = t + delta_t)
                xt[:, :, :ref_cnt] = ref_z
                
            
            
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max, 
                                cumulative=False)
            if stoc:  # adds stochastic term
                if self.predictor_type == "wavnet":
                    x0_ = self.estimator(spec=xt, cond=mu, diffusion_step=t)
                    x0_[:, :, :ref_cnt] = ref_x
                else:
                    x0_ = self.estimator(spec=xt, cond=mu, diffusion_step=t, spk=spk)
                x0_[..., :ref_cnt] = ref_x
                cfgs = 1
                if cfgs !=1:
                    x0_null = self.estimator(spec=xt[..., ref_cnt:], cond=mu[..., ref_cnt:], diffusion_step=t, spk=spk)
                    x0_[..., ref_cnt:] = x0_null + (cfgs - 1) * (x0_[..., ref_cnt:] - x0_null)
                    
                
                cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
                if hparams['diffusion_from_prior']:
                    rho = x0_*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
                else:
                    rho = x0_*torch.exp(-0.5*cum_noise)
                noise_pred = xt - rho
                
                lambda_ = (1.0 - torch.exp(-cum_noise)) * hparams['noise_factor']
                 
                logp = - noise_pred / (lambda_ + 1e-8)
                if hparams['diffusion_from_prior']:
                    dxt_det = 0.5 * (mu - xt) - logp
                else:
                    dxt_det = 0.5 * (- xt) - logp
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(dxt_det.shape, dtype=dxt_det.dtype, device=dxt_det.device,
                                       requires_grad=False)
                temp = 1.2
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h) / temp
                dxt = dxt_det + dxt_stoc
            else:
                if self.predictor_type == "wavnet":
                    x0_ = self.estimator(spec=xt, cond=mu, diffusion_step=t)
                    x0_[:, :, :ref_cnt] = ref_x
                else:
                    x0_ = self.estimator(spec=xt, cond=mu, diffusion_step=t, spk=spk)
                x0_[..., :ref_cnt] = ref_x
                cfgs = 1
                if cfgs !=1:
                    x0_null = self.estimator(spec=xt[..., ref_cnt:], cond=mu[..., ref_cnt:], diffusion_step=t, spk=spk)
                    x0_[..., ref_cnt:] = x0_null + (cfgs - 1) * (x0_[..., ref_cnt:] - x0_null)
                    
                
                cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
                if delta_t !=0:
                    cum_noise_delta = get_noise(time + delta_t, self.beta_min, self.beta_max, cumulative=True)
                    cum_noise = cum_noise.expand(x0_.shape)
                    cum_noise[:, :, :ref_cnt] = cum_noise_delta.item()
                
                if hparams['diffusion_from_prior']:
                    rho = x0_*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
                else:
                    rho = x0_*torch.exp(-0.5*cum_noise)
                noise_pred = xt - rho
                lambda_ = 1.0 - torch.exp(-cum_noise)
                 
                logp = - noise_pred / (lambda_ + 1e-8)
                if hparams['diffusion_from_prior']:
                    dxt = 0.5 * (mu - xt - logp)
                else:
                    dxt = 0.5 * (- xt - logp)
                dxt = dxt * noise_t * h
            xt = xt - dxt
        return xt[..., ref_cnt:]

    @torch.no_grad()
    def forward(self, z, x_mask, mu, n_timesteps, spk=None, stoc=False, ref_x=None, prompt=None, prompt_mask=None):
        if ref_x is None:
            return self.reverse_diffusion(z=z, x_mask=x_mask, mu=mu, n_timesteps=n_timesteps, spk=spk, stoc=stoc, prompt=prompt, prompt_mask=prompt_mask)
        else:
            return self.reverse_diffusion_incontext(z=z, mu=mu, n_timesteps=n_timesteps, spk=spk, stoc=stoc, ref_x=ref_x)

    def loss_t(self, x0, x_mask, mu, t, spk=None, pos_ids=None, prompt=None, prompt_mask=None):
        # print('x0:', x0.shape, 'mu:', mu.shape)
        
        if hparams['diffusion_from_prior']:
            xt, z = self.forward_diffusion(x0, mu, t) # xt and noise
        else:
            xt, z = self.forward_diffusion(x0, 0, t) # xt and noise
            
        time = t.unsqueeze(-1).unsqueeze(-1)
        # if spk is not None:
        #     xt = xt + self.spk_mlp(spk)[..., None]
        # print(prompt.shape, prompt_mask.shape)
 
        x0_estimation = self.estimator(spec=xt, x_mask=x_mask, cond=mu, diffusion_step=t, spk=spk, prompt=prompt, prompt_mask=prompt_mask)
        
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        if hparams['diffusion_from_prior']:
            rho = x0_estimation*torch.exp(-0.5*cum_noise/ (hparams['sigma'] ** 2)) + mu*(1.0 - torch.exp(-0.5*cum_noise)/ (hparams['sigma'] ** 2))
        else:
            rho = x0_estimation*torch.exp(-0.5*cum_noise/ (hparams['sigma'] ** 2))
        noise_pred = xt - rho
        
        lambda_ = (hparams['sigma'] ** 2) * (1.0 - torch.exp(-cum_noise / (hparams['sigma'] ** 2)) )
        
        
        if hparams['noise_std']:
            noise_pred = (xt - rho) / (torch.sqrt(lambda_) * hparams['noise_factor'])
        else:
            noise_pred = (xt - rho) / hparams['noise_factor']
            z = z * torch.sqrt(lambda_)
        
        return x0_estimation, noise_pred, z
    
    def loss_t_incontext(self, x0, mu, t, spk=None, pos_ids=None):
        
        
        # build mask matrix for context
        
        num_f = x0.shape[2]
        num_in_context = torch.randint(10, num_f // 2, (x0.shape[0], 1), dtype=torch.int64, device=x0.device)
        
        in_context_mask = torch.arange(num_f, device=x0.device).unsqueeze(0) < num_in_context
        
        xt_small_noise, z_small = self.forward_diffusion(x0, mu, t / 2)
        xt_normal_noise, z_normal = self.forward_diffusion(x0, mu, t)
        
        xt_pertube = torch.where(in_context_mask.unsqueeze(1), xt_small_noise, xt_normal_noise)
        z_pertube = torch.where(in_context_mask.unsqueeze(1), z_small, z_normal)
        
        time = t.unsqueeze(-1).unsqueeze(-1)
        
        x0_estimation = self.estimator(spec=xt_pertube, cond=mu, diffusion_step=t, spk=spk)
        
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        rho = x0_estimation * torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        lambda_ = 1.0 - torch.exp(-cum_noise)
        
        noise_pred = (xt_pertube - rho) / (torch.sqrt(lambda_))
        
        return x0_estimation, noise_pred, z_pertube


    def compute_loss(self, x0, x_mask, mu, spk=None, offset=1e-5, pos_ids=None, prompt=None, prompt_mask=None):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, x_mask, mu, t, spk=spk, pos_ids=pos_ids, prompt=prompt, prompt_mask=prompt_mask)   # return x0_estimation, noise_pred, z
    
    
    def compute_loss_incontext(self, x0, mu, spk=None, offset=1e-5, pos_ids=None):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        
        
        return self.loss_t_incontext(x0, mu, t, spk, pos_ids=pos_ids)
