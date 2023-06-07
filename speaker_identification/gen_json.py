import json 
import os
import argparse

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="v3_gt", help="The dataset name")
    parser.add_argument("--gen_dir", type=str, default="demo_data/wav", help="The input json file")
    parser.add_argument("--gen_prefix", type=str, default="denoised_", help="The input json file")
    parser.add_argument("--json_dir", type=str, default="demo_data", help="The output json file")
    
    return parser.parse_args()


if __name__ == "__main__":
    opt = options()
    ds = opt.dataset_name
    gen_dir = opt.gen_dir
    gen_prefix = opt.gen_prefix
    os.makedirs(opt.json_dir, exist_ok=True)
    out_json_fn = f'{opt.json_dir}/{ds}.json'
    res = {
        'dataset' : ds,
    }
    test_cases = []
    ref_dirs = {
        'v3': '/blob/v-zeqianju/dataset/tts/v3',
        'v3_norm': '/blob/v-zeqianju/dataset/tts/Valle_demo', 
        'v3_norm_rec': '/blob/v-zeqianju/dataset/tts/v3_norm_rec',
        'v3_rec': '/blob/v-zeqianju/dataset/tts/v3_rec',
        'v3_gt': '/blob/v-zeqianju/dataset/tts/v3_gt',
        'v3_gt_rec': '/blob/v-zeqianju/dataset/tts/v3_gt_rec',
        'valid': '/blob/v-zeqianju/dataset/tts/valid',
        'valid_3s': '/blob/v-zeqianju/dataset/tts/valid_3s',
        'v3_gt_full': '/blob/v-zeqianju/dataset/tts/v3_gt_full',
        'train_randomclip_rec': '/blob/v-zeqianju/dataset/tts/train_randomclip_rec',
        'train_rec': '/blob/v-zeqianju/dataset/tts/train_rec',
        'train': '/blob/v-zeqianju/dataset/tts/train',
    }
    assert ds in ref_dirs
    ref_dir = ref_dirs[ds]
    for fn in os.listdir(ref_dir):
        if not fn.endswith('.wav') and not fn.endswith('.flac'):
            continue
        # {
        #     "reference_wav_path": "demo_data/wav/david_faustino/hn8GyCJIfLM_0000012.wav",
        #     "synthesized_wav_path": "demo_data/wav/david_faustino/xTOk1Jz-F_g_0000015.wav",
        #     "synthesize_phoneme_seq": [""]
        # }
        test_case = {
            'reference_wav_path' : f'{ref_dir}/{fn}',
            'synthesized_wav_path':  f'{gen_dir}/{gen_prefix}{fn.replace(".flac", ".wav")}',
            'synthesize_phoneme_seq': [],
            }
        
        test_cases.append(test_case)
    
    res["test_cases"] = test_cases
    with open(out_json_fn, "w") as f:
        json.dump(res, f, indent=4)

    
        