import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch

from datasets.tts.utils import build_phone_encoder
from utils.indexed_datasets import IndexedDatasetBuilder
import glob
import json
import logging
import sys
import traceback
from multiprocessing.pool import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.hparams import hparams, set_hparams
from utils.preprocessor import process_utterance, get_pitch, get_mel2ph, f0_to_coarse

from usr_dir.codec.residual_vq import ResidualVQ
import tarfile
import pickle
import hashlib
from utils.text_encoder import NUM_RESERVED_TOKENS, UNK_ID, EOS_ID

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')



def process_data(tar_file, code_data_dir, emb_data_dir, data_dir, prefix):
    process_num = int(os.getenv('N_PROC', os.cpu_count()))
    
    tars = []
    with open(tar_file, 'r') as f:
        for line in f:
            tars.append(line.strip())
    
    
    from utils.binarizer import LatentDiffusionBinarizer, FileBinarizer
    
    binarizer = LatentDiffusionBinarizer(
        code_data_dir=code_data_dir,
        emb_data_dir=emb_data_dir,
        hparams=hparams
    )
    
    FileBinarizer.multiprocess_dataset(
        binarizer=binarizer,
        tar_list=tars,
        output_prefix=f'{data_dir}/{prefix}',
        num_workers=process_num,
        dataset_impl="index_ds",
        tar_keys=['speaker_id', 'duration', 'phone_id', 'f0', "speech", "mel"],
        except_set=set(),
    )
    

if __name__ == "__main__":
    #torch.multiprocessing.set_start_method('spawn')
    set_hparams()
    #vq_ckpt = hparams['vq_ckpt']
    #vq_ckpt_weight = torch.load(vq_ckpt)
    
    
    #quantizer = ResidualVQ(token
    #    num_quantizers=hparams['audio_num_mel_bins'], dim=hparams['vqemb_size'], codebook_size=hparams['codebook_size'], commitment=0.25,
    #    threshold_ema_dead_code=2
    #)
    #quantizer.load_state_dict(vq_ckpt_weight)
    #quantizer.eval()
    #quantizer.requires_grad_(False)
    #quantizer = quantizer.cuda()
    
    train_file = hparams['train_file']
    valid_file = hparams['valid_file']
    test_file = hparams['test_file']
    print('TRAIN:', train_file)
    print('VALID:', valid_file)
    print('TEST:', test_file)
    raw_data_dir = hparams['raw_data_dir']
    code_data_dir = hparams['code_data_dir']
    emb_data_dir = hparams['emb_data_dir']

    valid_code_data_dir = hparams['valid_code_data_dir']
    valid_emb_data_dir = hparams['valid_code_data_dir']
    test_code_data_dir = valid_code_data_dir
    test_emb_data_dir = valid_emb_data_dir
    # all_wav_fns = sorted(glob.glob(f'{raw_data_dir}/wavs/*.wav'))
    # all_tokens = []
    # print("Token file:", os.path.join(raw_data_dir, hparams["token_filename"]))
    # with open(os.path.join(raw_data_dir, hparams["token_filename"]), 'r', encoding='utf-8') as infile:
    #     for line in infile.readlines():
    #
    #         line_token = list(map(int, line.split("|")[1].strip().split()))
    #         line_token = np.array(line_token).reshape(-1, hparams['audio_num_mel_bins'])
    #
    #         for bin_idx in range(hparams['audio_num_mel_bins']):
    #             line_token[:, bin_idx] = line_token[:, bin_idx] - bin_idx * hparams['codebook_size']
    #
    #         all_tokens.append(line_token)
    # logging.info("load {} discrete token items".format(len(all_tokens)))
    # logging.info("train {}".format(len(all_wav_fns)))

    ph_set = [x.split(' ')[0] for x in open(f'{raw_data_dir}/{hparams["dict_file"]}.txt').readlines()]
    print(ph_set)
    os.makedirs(hparams['data_dir'], exist_ok=True)
    json.dump(ph_set, open(f"{hparams['data_dir']}/phone_set.json", 'w'))
    process_data(train_file, code_data_dir, emb_data_dir, hparams['data_dir'], 'train')
    process_data(test_file, test_code_data_dir, test_emb_data_dir, hparams['data_dir'], 'test')
    process_data(valid_file, valid_code_data_dir, valid_emb_data_dir, hparams['data_dir'], 'valid')
    
    

