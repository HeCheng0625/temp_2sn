
import torch
import torch.nn as nn
from utils.hparams import hparams

from usr_dir.huggingface_lib.gpt2 import GPT2Model
from usr_dir.huggingface_lib.configure import GPT2Config

from utils.chanpin_utils import chanpin_phone_dict
from transformers.modeling_utils import ModuleUtilsMixin


class TTSLatentGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.in_proj = nn.Linear(512, 512)
        
        if hparams["vq_ckpt"] is not None:
            print("Loading VQ model from {}".format(hparams["vq_ckpt"]))
            from usr_dir.codec.residual_vq import ResidualVQ
            
            quantizer = ResidualVQ(
                num_quantizers=hparams['audio_num_mel_bins'], dim=hparams['vqemb_size'], codebook_size=hparams['codebook_size'], commitment=0.25,
                threshold_ema_dead_code=2
            )
            quantizer.load_state_dict(torch.load(hparams["vq_ckpt"], map_location="cpu"))
            quantizer.train()
            quantizer.requires_grad_(False)
            self.quantizer = quantizer
        
        self.gpt2_config = GPT2Config()
        self.gpt2_config.vocab_size = 6000
        
        self.gpt2 = GPT2Model(self.gpt2_config)
        
 
    
            
    @torch.no_grad()
    def convert_code_to_latent(self, codes):
        latent = self.quantizer.vq2emb(codes.long())
        return latent
        
    def forward(self, src_tokens, speech_latent, speech_length):
        
        # print(src_tokens.shape)
        # print(src_tokens[0].detach().cpu().numpy().tolist())
        
        # padding src_tokens to enable hacking for simplicity
        src_tokens_with_pad = torch.cat((src_tokens, torch.zeros((src_tokens.shape[0], speech_latent.shape[1]), dtype=torch.long, device=src_tokens.device)), dim=1)
        
        # phonemes = chanpin_phone_dict.decode_list(src_tokens[0].detach().cpu().numpy().tolist())

        gpt_results = self.gpt2(
            input_ids=src_tokens_with_pad,
            speech_latent=speech_latent,
            speech_length=speech_length
        )
        
        pred_latent = gpt_results["predicted_speech_latent"]
        
        stop_logits = pred_latent[:, :, -1]
        
        pred_latent = pred_latent[:, :, :-1]
        
        return pred_latent, stop_logits

