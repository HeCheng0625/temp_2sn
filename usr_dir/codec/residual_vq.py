import torch
from torch import nn
from .vector_quantize_pytorch import VectorQuantize

class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        *,
        num_quantizers,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantize(**kwargs) for _ in range(num_quantizers)])

    def forward(self, x):
        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []

        for idx, layer in enumerate(self.layers):
            quantized, indices, loss, _ = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            #print(idx, torch.abs(x - quantized_out).mean())

            all_indices.append(indices)
            all_losses.append(loss)

        all_losses, all_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, all_indices, all_losses

    def vq2emb(self, vq):
        quantized_out = 0.
        for idx, layer in enumerate(self.layers):
            quantized = layer.vq2emb(vq[:, :, idx])
            # quantized = layer.vq2emb(vq[idx, :, :])
            quantized_out = quantized_out + quantized
        return quantized_out
    
    
    
    def get_emb(self):
        embs = [] 
        for idx, layer in enumerate(self.layers):
            embs.append(layer.get_emb())
        return embs

    def calculate_loss(self, x, discrete_y, latent_y, y_mask, require_dist_loss=True):
        
        quantized_out = 0.
        residual = x
        residual_y = latent_y

        all_losses = []
        all_indices = []
        gt_indices = []
        
        all_losses_discrete = []
        
        all_losses_distance = []
        
        y_mask_seq = y_mask.contiguous().view(-1)
        
        for idx, layer in enumerate(self.layers):
            quantized, indices, loss, dist = layer(residual)
            
            
            gt_quantized = layer.vq2emb(discrete_y[:, idx, :]).transpose(1, 2)
            

            # with torch.no_grad():
            #     gt_quantized, indices_gt, _, dist_gt = layer(residual_y)
            
            


            all_indices.append(indices)

            all_losses.append(loss)
            
            
        
            target_y_cur = discrete_y[:, idx, :].contiguous().view(-1)
            
            dist_masked = dist[y_mask_seq]
            target_y_cur_masked = target_y_cur[y_mask_seq]
            
            loss_discrete = torch.nn.functional.cross_entropy(dist_masked, target_y_cur_masked)
            
            all_losses_discrete.append(loss_discrete)

            
            
        
            
            # print(dist.shape, dist_gt.shape)
            # exit(0)
            if require_dist_loss:
                with torch.no_grad():
                    quant_h, gt_c, _, dist_gt = layer(residual_y)
                    residual_y = residual_y - quant_h.detach()
                    
                dist_mask = dist[y_mask_seq]
                dist_gt_mask = dist_gt[y_mask_seq]
                
                dist_mask = torch.sqrt( - dist_mask)
                dist_gt_mask = torch.sqrt( - dist_gt_mask)
                
                loss_distance = torch.nn.functional.mse_loss(dist_mask, dist_gt_mask.detach())
                all_losses_distance.append(loss_distance)
            
            residual = residual - gt_quantized.detach()
            
            quantized_out = quantized_out + quantized

        all_losses, all_indices, all_losses_discrete = map(torch.stack, (all_losses, all_indices, all_losses_discrete))
        
        if require_dist_loss:
            all_losses_distance = torch.stack(all_losses_distance)
        else:
            all_losses_distance = None
        
        return quantized_out, all_indices, all_losses, all_losses_discrete, all_losses_distance

        pass