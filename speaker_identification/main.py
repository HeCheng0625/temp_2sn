from verification import init_model, verification_case

import argparse
import json
import tqdm
from copy import deepcopy
import numpy as np
import os


def options():
    parser = argparse.ArgumentParser(description='Speaker Identification')
    parser.add_argument("--model_name", type=str, default="wavlm_large", help="The model name")
    parser.add_argument("--checkpoint", type=str, default="/blob/v-shenkai/code/tts/evaluation/checkpoint/wavlm_large_finetune.pth", help="The checkpoint path")
    parser.add_argument("--input_json", type=str, default="demo_data/test.json", help="The input json file")
    parser.add_argument("--output_json", type=str, default="demo_data/test_out.json", help="The output json file")
    
    return parser.parse_args()


if __name__ == "__main__":
    opt = options()
    model = init_model(model_name=opt.model_name, checkpoint=opt.checkpoint)
    model = model.cuda("cuda:0")
    model.eval()
    
    with open(opt.input_json, "r") as f:
        context = json.load(f)
        
        dataset_name = context["dataset"]
        
        output_cases = []
        score_collection = []
        
        for case in tqdm.tqdm(context["test_cases"]):
            wav1_path = case["reference_wav_path"]
            wav2_path = case["synthesized_wav_path"]
            
            scores = verification_case(model, wav1_path, wav2_path, use_gpu=True)
            score = scores[0].item()
            
            output_case = deepcopy(case)
            output_case["speaker_verification_score"] = score
            score_collection.append(score)
            output_cases.append(output_case)

        
        print("There are {} cases, mean similarity score: {}".format(len(output_cases), np.mean(score_collection)))
        os.makedirs(os.path.split(opt.output_json)[0], exist_ok=True)
        with open(opt.output_json, "w") as f:
            json.dump({
                "dataset": dataset_name,
                "test_cases": output_cases,
                "mean_speaker_verification_score": np.mean(score_collection),
            }, f, indent=4)
            pass




