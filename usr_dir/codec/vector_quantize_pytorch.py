import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

import logging
logger = logging.getLogger(__name__)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))   # (1-decay) * new + decay * moving_avg

def laplace_smoothing(x, n_categories, eps = 1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def kmeans(samples, num_clusters, num_iters = 10, use_cosine_sim = False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)   # return num_clusters samples (N, d) -> (m, d)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()   # (N, d) * (d, m) -> (N, m)
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                    - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim = -1)

        buckets = dists.max(dim = -1).indices   # (N, )
        bins = torch.bincount(buckets, minlength = num_clusters)   # (m, )
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype = dtype)   # (m, d)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)   # new_means[buckets[i][j] \in [0, m-1]][j] += samples[i][j]
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins

# distance types

class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init = False,
        kmeans_iters = 10,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        weight_init=False,
    ):
        super().__init__()
        self.decay = decay
        init_fn = torch.randn if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)   # (m, d)
        
        if weight_init:
            nn.init.uniform_(embed, -1 / codebook_size, 1 / codebook_size)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))   # (m, )
        self.register_buffer('embed', embed)
        self.register_buffer('embed_avg', embed.clone())

    def init_embed_(self, data):
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples, mask):
        # Replace the dead codebooks.
        modified_codebook = torch.where(
            mask[..., None],
            sample_vectors(samples, self.codebook_size),
            self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace(batch_samples, mask = expired_codes)

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        #print(x.size())
        flatten = rearrange(x, '... d -> (...) d')   # (N, d)
        #print(flatten.size())
        embed = self.embed.t()   # (d, m)

        if not self.initted:
            self.init_embed_(flatten)   # (m, d) if not initted

        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )   # (N, m)

        embed_ind = dist.max(dim = -1).indices   # (N,)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)   # (N, m)
        embed_ind = embed_ind.view(*shape[:-1]) # (N,) -> (...,)
        quantize = F.embedding(embed_ind, self.embed)   # (..., d)

        # if self.training:
        #     ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)   # (m, )
        #     embed_sum = flatten.t() @ embed_onehot   # (d, m)
        #     ema_inplace(self.embed_avg, embed_sum.t(), self.decay)   # (d, m)
        #     cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
        #     embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        #     self.embed.data.copy_(embed_normalized)
        #     self.expire_codes_(x)
            
        dist = dist.masked_fill(dist >= -1e-6, 0)

        return quantize, embed_ind, dist   # (..., d), (...,), (N, m)

    def vq2emb(self, vq):
        quantize = F.embedding(vq, self.embed)
        return quantize

class CosineSimCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init = False,
        kmeans_iters = 10,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2
    ):
        super().__init__()
        self.decay = decay

        if not kmeans_init:
            embed = l2norm(torch.randn(codebook_size, dim))
        else:
            embed = torch.zeros(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed', embed)

    def init_embed_(self, data):
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters,
                       use_cosine_sim = True)
        self.embed.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples, mask):
        samples = l2norm(samples)
        modified_codebook = torch.where(
            mask[..., None],
            sample_vectors(samples, self.codebook_size),
            self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace(batch_samples, mask = expired_codes)

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')
        flatten = l2norm(flatten)

        if not self.initted:
            self.init_embed_(flatten)

        embed = l2norm(self.embed)
        dist = flatten @ embed.t()   # (N, d) * (d, m) -> (N, m)
        embed_ind = dist.max(dim = -1).indices   # (N, )
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)   # (N, m)
        embed_ind = embed_ind.view(*shape[:-1])   # (...,)

        quantize = F.embedding(embed_ind, self.embed)

        if self.training:
            bins = embed_onehot.sum(0)   # (m, )
            ema_inplace(self.cluster_size, bins, self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = flatten.t() @ embed_onehot   
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()   # (m, d)
            embed_normalized = l2norm(embed_normalized)
            embed_normalized = torch.where(zero_mask[..., None], embed,
                                           embed_normalized)
            ema_inplace(self.embed, embed_normalized, self.decay)
            self.expire_codes_(x)

        return quantize, embed_ind   # (..., d), (...,)

    def vq2emb(self, vq):
        quantize = F.embedding(vq, self.embed)
        return quantize
   

# main class

class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        n_embed = None,
        codebook_dim = None,
        decay = 0.8,
        commitment = 0.005,
        eps = 1e-5,
        kmeans_init = False,
        kmeans_iters = 10,
        use_cosine_sim = False,
        threshold_ema_dead_code = 0,
        channel_last = False,
        weight_init = False,
        full_commit_loss = False
    ):
        super().__init__()
        
        logger.info("Full commit loss: {}, commit weight: {}, weight init: {}".format(full_commit_loss, commitment, weight_init))
        
        n_embed = default(n_embed, codebook_size)

        codebook_dim = default(codebook_dim, dim)
        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection \
                          else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection \
                           else nn.Identity()

        self.eps = eps
        self.commitment = commitment

        codebook_class = EuclideanCodebook 
        # if not use_cosine_sim \
        #                  else CosineSimCodebook

        self._codebook = EuclideanCodebook(
            dim = codebook_dim,
            codebook_size = n_embed,
            kmeans_init = kmeans_init,   # default false
            kmeans_iters = kmeans_iters,
            decay = decay,
            eps = eps,
            threshold_ema_dead_code = threshold_ema_dead_code,
            weight_init = weight_init
        )

        self.codebook_size = codebook_size
        self.channel_last = channel_last   # default false
        self.full_commit_loss = full_commit_loss   # default false

    @property
    def codebook(self):
        return self._codebook.codebook

    def forward(self, x):   # x: (batch_size, channel, dim)
        need_transpose = not self.channel_last

        if need_transpose:
            x = rearrange(x, 'b n d -> b d n')

        x = self.project_in(x)

        quantize, embed_ind, dist = self._codebook(x)

        if self.training:
            
            if self.full_commit_loss:   # (x_q - sg(x))^2 + commitment * (x - xg(x_q))^2
                commit_loss = F.mse_loss(quantize, x.detach()) + F.mse_loss(quantize.detach(), x) * self.commitment
            else:
                commit_loss = F.mse_loss(quantize.detach(), x) * self.commitment
            quantize = x + (quantize - x).detach()   # for optimizing encoder
        else:
            commit_loss = torch.tensor([0.], device = x.device)

        quantize = self.project_out(quantize)

        if need_transpose:
            quantize = rearrange(quantize, 'b d n -> b n d')

        return quantize, embed_ind, commit_loss, dist

    def vq2emb(self, vq):
        return self._codebook.vq2emb(vq)

    def get_emb(self):
        return self._codebook.embed
