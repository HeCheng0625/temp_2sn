import json 
import os
import argparse
from copy import deepcopy
import tqdm
import shutil
def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="", help="The input json file")
    parser.add_argument("--gen_dir", type=str, default="demo_data/wav", help="The input json file")
    parser.add_argument("--gen_prefix", type=str, default="denoised_", help="The input json file")
    parser.add_argument("--output_json", type=str, default="demo_data", help="The output json file")
    parser.add_argument("--nsample", type=int, default=0, help="The output json file")
    return parser.parse_args()


if __name__ == "__main__":
    
    opt = options()
    gen_dir = opt.gen_dir
    gen_prefix = opt.gen_prefix
    nsample = int(opt.nsample)
    
    with open(opt.input_json, "r") as f:
        context = json.load(f)
        
        dataset_name = context["dataset"]
        
        output_cases = []
        score_collection = []
        
        for case in tqdm.tqdm(context["test_cases"]): 
            output_case = deepcopy(case)
            wav1_path = case["reference_wav_path"]
            fn = os.path.basename(wav1_path)
            if nsample > 0:
                output_case["synthesized_wav_paths"] = [f'{gen_dir}/{gen_prefix}sp{idx}_{fn.replace(".flac", ".wav")}'  for idx in range(nsample)]
            else:
                output_case["synthesized_wav_paths"] = [f'{gen_dir}/{gen_prefix}_{fn.replace(".flac", ".wav")}']
            # shutil.copyfile(wav1_path, f'{gen_dir}/{fn}')
            
            output_cases.append(output_case)

        
        os.makedirs(os.path.split(opt.output_json)[0], exist_ok=True)
        with open(opt.output_json, "w") as f:
            json.dump({
                "dataset": dataset_name,
                "test_cases": output_cases
            }, f, indent=4)
            pass
    
    
    
        