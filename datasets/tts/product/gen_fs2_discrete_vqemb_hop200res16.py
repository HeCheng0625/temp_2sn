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

def dur_to_mel2ph(dur):
    dur = torch.from_numpy(dur)
    mel2ph = dur.cumsum(dim=-1)
    mel2ph = torch.nn.functional.one_hot(mel2ph, num_classes=mel2ph.max() + 1)[:-1, :-1].sum(-2).cumsum(dim=-1)
    mel2ph = mel2ph + 1
    return mel2ph.numpy()



def process_item(item_name, file_info):
    # ph = "<UNK> " + ph + " <EOS>"
    phone_encoded = []
    phone_encoded.append(UNK_ID)
    for i in file_info['phone_id']:
        phone_encoded.append(NUM_RESERVED_TOKENS + i)
    phone_encoded.append(EOS_ID)

    token = torch.load(file_info['code'], map_location='cpu')
    token = token.squeeze(1).transpose(0, 1).numpy()
    mel = file_info['emb']

    spk_id = file_info['speaker_id']
    dur = file_info['duration']
    mel2ph = dur_to_mel2ph(dur)
    
    if hparams["f0_use_product"]:
        f0 = file_info['f0']
        pitch_coarse = f0_to_coarse(f0) + 1
    else:
        f0, pitch_coarse = get_pitch(file_info["speech"], file_info["mel"], hparams)

    return item_name, phone_encoded, mel, mel2ph, spk_id, pitch_coarse, f0, dur, token


def process_data(tar_file, code_data_dir, emb_data_dir, data_dir, prefix):
    p = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    tars = []
    with open(tar_file, 'r') as f:
        for line in f:
            tars.append(line.strip())
    
<<<<<<< HEAD
    keys = ['speaker_id', 'duration', 'phone_id', 'f0', "speech", "mel"]
    futures = []

    for tar in tqdm(tars):
=======
    keys = ['speaker_id', 'duration', 'phone_id', 'f0']
    futures = []

    for tar in tars:
>>>>>>> fix
        tar_item = dict()
        tar_obj = tarfile.open(tar, mode='r')
        for info in tar_obj:
            try:
                if not info.isfile():
                    continue
                for key in keys:
                    if info.name.endswith(f'.{key}'):
                        hash_value = hashlib.sha256(
                            os.path.join(tar, info.name.replace(f'.{key}', '')).encode('utf-8')).hexdigest()
                        if key == 'speaker_id':
                            # spk_id not from file, but from filename
                            # e.g. mls_0.{spk_id}_{}_{}.speaker_id
                            value = int(info.name.split('.')[1].split('_')[0])
                        else:
                            cur_f = tar_obj.extractfile(info)
                            value = pickle.load(cur_f)
                        if hash_value not in tar_item:
                            tar_item[hash_value] = dict() 
                        tar_item[hash_value][key] = value
                        break
            except:
                continue
        for hash_value, file_info in tar_item.items():
            if len(file_info) != len(keys):
                print(file_info.keys())
                continue
            if 'code' not in file_info:
                file_info['code'] = os.path.join(code_data_dir, f'{hash_value}.code')
            if 'emb' not in file_info:
                file_info['emb'] = os.path.join(emb_data_dir, f'{hash_value}.emb')
            futures.append(
                p.apply_async(process_item, args=(hash_value, file_info)))

    p.close()
    builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
    all_keys = []
    lengths = []
    f0s = []
    durs = []
    for future in tqdm(futures):
        res = future.get()
        if res is None:
            continue
        item_name, phone_encoded, mel, mel2ph, spk_id, pitch, f0, dur, token = res
        # item_name = f'lj_{item_name}'
        builder.add_item({
            'item_name': item_name,
            # 'txt': txt,
            'phone': phone_encoded,
<<<<<<< HEAD
            # 'mel': mel,
            'code': token.astype(np.uint16),
            'mel2ph': mel2ph,
            'spk_id': spk_id,
            # 'pitch': pitch,
=======
            'mel': mel,
            'code': token,
            'mel2ph': mel2ph,
            'spk_id': spk_id,
            'pitch': pitch,
>>>>>>> fix
            'f0': f0,
        })
        lengths.append(token.shape[0])
        all_keys.append(item_name)
        f0s.append(f0)
        durs.append(dur)
    p.join()
    builder.finalize()
    np.save(f'{data_dir}/{prefix}_all_keys.npy', all_keys)
    np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
    np.save(f'{data_dir}/{prefix}_f0s.npy', f0s)
    np.save(f'{data_dir}/{prefix}_durs.npy', durs)

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
    process_data(valid_file, valid_code_data_dir, valid_emb_data_dir, hparams['data_dir'], 'valid')
    process_data(test_file, test_code_data_dir, test_emb_data_dir, hparams['data_dir'], 'test')
    process_data(train_file, code_data_dir, emb_data_dir, hparams['data_dir'], 'train')

