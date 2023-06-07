from utils.text_encoder import TokenTextEncoder
import json
import torch

# chanpin_phone_dict = TokenTextEncoder(None, vocab_list=json.load(open("/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/phone_set_chanpin_librilight.json")))
chanpin_phone_dict = TokenTextEncoder(None, vocab_list=json.load(open("/home/v-yuancwang/naturalspeech2/phone_set_chanpin_librilight.json")))


def calculate_dur_from_mel2ph(mel2ph, max_phones=None):
    if max_phones == None:
        T_t = int(mel2ph.max().item())
    else:
        T_t = max_phones
    
    dur_gt = mel2ph.new_zeros(T_t + 1).scatter_add(0, mel2ph.long(), torch.ones_like(mel2ph))
    dur_gt = dur_gt[1:]
    return dur_gt