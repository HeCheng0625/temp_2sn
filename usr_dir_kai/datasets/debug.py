import numpy as np
import torch

from utils.chanpin_utils import chanpin_phone_dict, calculate_dur_from_mel2ph



def extact_random_ref_from_item_separately(phone_ids, mel2ph, code, pitch=None, uv=None):
    ret = {}
    
    phone_id_list = phone_ids.numpy().tolist()
    break_idx = [4148, 4149, 4150, 4151, 4152, 1]
    
    # phonemes = chanpin_phone_dict.decode_list(phone_id_list)
    
    break_list = ["br1", "br2", "br3", "br4", "-BOS-", "<EOS>"]
    
    break_idx = [4148, 4149, 4150, 4151, 4152, 1] # [chanpin_phone_dict._token_to_id[x] for i, x in enumerate(break_list)]
    
    break_interval = []
    
    for ph_idx, ph in enumerate(phone_id_list):
        if ph in break_idx:
            
            break_interval.append(ph_idx)

    assert len(break_interval) >= 3, "break_interval: {}".format(break_interval)
    a, b = np.random.choice(break_interval, 2, replace=False)
    # exit()
    start_idx = min(a, b)
    end_idx = max(a, b)  # [start_idx, end_idx]

    while (end_idx - start_idx + 1) / len(phone_id_list) > 0.5:
        a, b = np.random.choice(break_interval, 2, replace=False)
    # exit()
        start_idx = min(a, b)
        end_idx = max(a, b)  # [start_idx, end_idx]
    # print(start_idx, end_idx, (end_idx - start_idx + 1) / len(phone_id_list), "222222222")
    
    # start_idx = 1
    # end_idx = 3
    
    durs = calculate_dur_from_mel2ph(mel2ph, max_phones=len(phone_id_list))
    durs = durs.squeeze(0)
    
    durs_sumcur = torch.cumsum(durs, dim=0)
    
    
    start_frame_idx = int(durs_sumcur[start_idx - 1].item() if start_idx > 0 else 0)
    end_frame_idx = int(durs_sumcur[end_idx].item())
    
    
    mel2ph_new = torch.hstack((mel2ph[:start_frame_idx], mel2ph[end_frame_idx:] - (end_idx - start_idx + 1)))
    
    ret["mel2ph_new"] = mel2ph_new
    # print(code.shape)
    phone_ids_new = torch.cat((phone_ids[:start_idx], phone_ids[end_idx + 1:]), dim=0)
    ret["phone_ids_new"] = phone_ids_new
    
    code_new = torch.cat((code[:start_frame_idx, :], code[end_frame_idx:, :]), dim=0)
    ret["code_new"] = code_new
    
    
    code_ref = code[start_frame_idx:end_frame_idx, :]
    ret["code_ref"] = code_ref
    
    
    if pitch is not None:
        pitch_new = torch.cat((pitch[:start_frame_idx], pitch[end_frame_idx:]), dim=0)
        ret["pitch_new"] = pitch_new
    
    if uv is not None:
        uv_new = torch.cat((uv[:start_frame_idx], uv[end_frame_idx:]), dim=0)
        ret["uv_new"] = uv_new
    
    # for k, v in ret.items():
    #     print(k, v.shape)
    
    return ret


if __name__ == "__main__":
    phone_id = [4152, 4147, 1554,  522, 4148,  829, 3307, 1554, 4148, 1554,  522, 2218,
        4148, 2279, 1554,  633,  813, 3945, 3323, 4148, 1929, 2521, 4148,  813,
         671,  271, 2218, 4148, 1929, 2967, 4148, 3056, 2521,  633,  671, 2020,
         633,  827, 2279,  829, 4148, 3307, 1554, 4148, 3715, 3787,  827,  633,
        1929,  652,  671,  813,  633, 1554, 3787, 2522, 4148, 3317, 3079, 4148,
        1901, 3029,  671,  633, 3317, 2279, 1554, 4148, 2073, 1176, 4148, 3056,
        2073, 1554, 4148, 2073, 1176, 4148, 3317, 3079,  829, 4148, 3945, 2218,
        4150, 4153, 4147, 3307, 1554, 4148, 1929, 2967, 4148, 3317, 3945, 4148,
        2279,  633, 1554,  436,  633, 2279, 1554,  827, 4148, 3400,  671,  522,
        2218, 4148, 3307, 1554, 4148,  827,  813,  671,  649, 2522, 2658,  633,
        2801, 2279, 1554, 2218, 4148, 2923,  671,  652,  829, 4148, 1810, 3945,
         671, 4148,  813,  989, 4148, 1810, 3945,  671, 4150, 4153, 4147, 3307,
        1554, 4148, 3317, 3079, 4148, 2658, 3029,  633,  671, 3787, 2212, 4148,
        1901, 3787,  633, 2658, 2073,  829, 2218, 4148, 3056, 2521,  633, 2658,
        2279,  671, 4148,  813,  989, 4148, 1901,  649,  671, 4148, 3945,  813,
        4150,  132, 4148, 4153, 4147,    4,    1]
    
    
    mel2ph = [  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
          1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
          1,   1,   1,   1,   1,   1,   1,   1,   3,   3,   3,   3,   3,   3,
          3,   3,   3,   3,   3,   3,   3,   4,   4,   4,   4,   4,   4,   4,
          4,   4,   4,   5,   6,   6,   6,   6,   6,   6,   7,   7,   7,   7,
          7,   7,   7,   7,   7,   7,   8,   8,   8,   8,   8,   8,   9,  10,
         10,  10,  10,  10,  10,  11,  11,  11,  11,  11,  11,  11,  11,  11,
         12,  12,  12,  12,  12,  12,  12,  13,  14,  14,  14,  14,  15,  15,
         17,  17,  17,  17,  17,  17,  18,  18,  18,  18,  18,  18,  19,  19,
         20,  21,  21,  21,  21,  21,  21,  22,  22,  22,  22,  22,  22,  22,
         22,  23,  24,  24,  24,  24,  24,  24,  24,  24,  24,  24,  25,  25,
         25,  25,  25,  26,  26,  26,  26,  26,  26,  26,  26,  26,  26,  26,
         26,  26,  27,  27,  27,  27,  27,  27,  28,  29,  29,  29,  29,  29,
         29,  29,  29,  29,  29,  29,  29,  29,  29,  29,  29,  29,  30,  30,
         30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  30,  31,
         32,  32,  32,  32,  32,  32,  33,  33,  33,  33,  33,  33,  33,  33,
         35,  35,  35,  35,  35,  35,  36,  36,  36,  36,  36,  38,  38,  38,
         38,  38,  38,  38,  38,  38,  39,  39,  39,  39,  39,  40,  40,  40,
         40,  41,  42,  42,  42,  43,  43,  43,  43,  43,  44,  45,  45,  45,
         45,  45,  46,  46,  46,  46,  47,  47,  47,  47,  47,  47,  49,  49,
         49,  49,  49,  49,  49,  49,  49,  49,  50,  50,  50,  50,  51,  51,
         51,  51,  52,  52,  52,  52,  52,  52,  52,  54,  54,  54,  54,  54,
         54,  54,  55,  55,  56,  56,  56,  56,  56,  56,  56,  56,  56,  57,
         58,  58,  58,  59,  59,  59,  59,  59,  59,  59,  59,  60,  61,  61,
         61,  61,  61,  62,  62,  62,  62,  62,  62,  63,  63,  63,  63,  63,
         63,  65,  65,  65,  65,  66,  66,  66,  66,  66,  67,  67,  67,  68,
         69,  69,  69,  69,  70,  70,  70,  70,  71,  72,  72,  72,  72,  73,
         73,  73,  73,  73,  74,  74,  75,  76,  76,  76,  77,  77,  77,  77,
         77,  78,  79,  79,  79,  80,  80,  80,  80,  81,  81,  81,  81,  82,
         83,  83,  83,  83,  83,  83,  83,  83,  83,  83,  83,  83,  83,  83,
         83,  84,  84,  84,  84,  84,  84,  84,  84,  84,  84,  84,  84,  84,
         84,  84,  84,  84,  84,  84,  84,  84,  84,  84,  85,  85,  85,  85,
         85,  85,  85,  85,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,
         86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,  86,
         86,  86,  86,  86,  88,  88,  88,  88,  88,  88,  88,  88,  88,  88,
         88,  88,  88,  89,  89,  89,  89,  90,  91,  91,  91,  91,  91,  91,
         91,  91,  91,  91,  91,  92,  92,  92,  92,  92,  92,  92,  92,  93,
         94,  94,  94,  94,  95,  95,  95,  95,  96,  97,  97,  97,  97,  99,
         99,  99,  99,  99,  99, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 102, 102, 103, 103, 103, 103, 103, 103, 104, 104,
        104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 105, 106, 106,
        106, 106, 106, 106, 106, 106, 106, 107, 107, 107, 107, 107, 107, 107,
        107, 107, 107, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
        108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 109, 109, 109,
        109, 109, 109, 109, 109, 109, 109, 109, 109, 110, 111, 111, 111, 111,
        112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 113, 114, 114, 114,
        114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 115, 115, 115, 115,
        115, 115, 115, 115, 115, 116, 116, 116, 116, 117, 117, 117, 117, 117,
        117, 117, 118, 118, 118, 118, 119, 119, 119, 119, 121, 121, 121, 121,
        121, 121, 121, 121, 121, 121, 122, 122, 122, 122, 122, 122, 123, 123,
        123, 123, 123, 123, 123, 123, 124, 124, 124, 124, 124, 124, 125, 126,
        126, 126, 126, 126, 126, 127, 127, 127, 128, 128, 129, 129, 129, 129,
        129, 129, 130, 131, 131, 131, 131, 131, 131, 131, 131, 131, 132, 132,
        132, 132, 132, 132, 132, 132, 132, 133, 133, 133, 133, 133, 133, 134,
        135, 135, 135, 136, 136, 137, 138, 138, 138, 138, 138, 138, 138, 138,
        138, 138, 138, 138, 138, 138, 138, 139, 139, 139, 139, 139, 139, 139,
        139, 139, 139, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140,
        140, 140, 140, 140, 140, 140, 140, 140, 141, 141, 141, 141, 141, 141,
        141, 141, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142,
        142, 142, 142, 142, 144, 144, 144, 144, 144, 144, 144, 144, 144, 144,
        144, 145, 145, 146, 147, 147, 147, 147, 147, 147, 147, 147, 147, 148,
        148, 148, 148, 148, 148, 148, 149, 150, 150, 150, 150, 150, 150, 150,
        150, 150, 151, 151, 151, 151, 153, 153, 153, 153, 153, 153, 153, 153,
        154, 154, 154, 154, 154, 155, 155, 155, 155, 155, 155, 155, 155, 155,
        155, 156, 157, 157, 157, 158, 158, 160, 160, 160, 160, 160, 160, 160,
        160, 160, 161, 161, 161, 161, 162, 162, 162, 162, 162, 163, 163, 163,
        163, 163, 163, 164, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165,
        166, 166, 166, 166, 166, 166, 166, 166, 166, 168, 168, 168, 168, 168,
        169, 169, 169, 169, 170, 170, 170, 170, 170, 171, 172, 172, 172, 172,
        173, 173, 173, 173, 173, 173, 173, 174, 175, 175, 175, 175, 175, 176,
        176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 177, 177,
        177, 177, 177, 177, 178, 179]
    

    code = torch.range(1, 1000).unsqueeze(1).expand(-1, 16)
    print(code.shape)
    
    mel2ph = torch.Tensor([mel2ph])
    
    
    phonemes = chanpin_phone_dict.decode_list([idx  for idx in phone_id])
    
    break_list = ["br1", "br2", "br3", "br4", "-BOS-", "<EOS>"]
    
    break_idx = [4148, 4149, 4150, 4151, 4152, 1] # [chanpin_phone_dict._token_to_id[x] for i, x in enumerate(break_list)]
    
    print(phonemes)
    print(break_idx)
    
    break_interval = []
    
    for ph_idx, ph in enumerate(phone_id):
        if ph in break_idx:
            
            break_interval.append(ph_idx)

    print(break_interval)

    a, b = np.random.choice(break_interval, 2, replace=False)
    
    # start_idx = min(a, b)
    # end_idx = max(a, b)  # [start_idx, end_idx]
    
    start_idx = 1
    end_idx = 3
    
    durs = calculate_dur_from_mel2ph(mel2ph, max_frames=len(phonemes))
    durs = durs.squeeze(0)
    
    durs_sumcur = torch.cumsum(durs, dim=0)
    print(durs)

    
    start_frame_idx = int(durs_sumcur[start_idx - 1].item() if start_idx > 0 else 0)
    end_frame_idx = int(durs_sumcur[end_idx].item())
    
    
    mel2ph_new = torch.hstack((mel2ph[0, :start_frame_idx], mel2ph[0, end_frame_idx:] - (end_idx - start_idx + 1)))
    
    code_new = torch.cat((code[:start_frame_idx, :], code[end_frame_idx:, :]), dim=0)
    print(code_new.shape)
    
    
    code_ref = code[start_frame_idx:end_frame_idx, :]
    print(code_ref.shape)
    
    
    
    
    
    