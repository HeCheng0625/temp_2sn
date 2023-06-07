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
import shutil
from utils.binarizer import LatentDiffusionBinarizer, FileBinarizer
import pandas as pd 
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
import random

def dur_to_mel2ph(dur):
    dur = torch.from_numpy(dur)
    mel2ph = dur.cumsum(dim=-1)
    mel2ph = torch.nn.functional.one_hot(mel2ph, num_classes=mel2ph.max() + 1)[:-1, :-1].sum(-2).cumsum(dim=-1)
    mel2ph = mel2ph + 1
    return mel2ph.numpy()



def process_item(item_name, file_info, refs):
    try:
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

        return item_name, phone_encoded, mel, mel2ph, spk_id, pitch_coarse, f0, dur, token, refs
    except:
        print('failed item:', item_name)
        return None


def process_data(tar_file, code_data_dir, emb_data_dir, data_dir, prefix, utt_ids):
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
        utt_ids=utt_ids,
    )
    
    
def get_test_data(entry, refs, code_data_dir, emb_data_dir):
    tar, fn, utt_id, spk_id = entry['tar'], entry['fn'], entry['utt_id'], entry['spk_id']
    tar_obj = tarfile.open(tar, mode='r')
    keys = ['speaker_id', 'duration', 'phone_id', 'f0', "speech", "mel"]
    file_info = dict()
    for info in tar_obj:
        try:
            if not info.isfile():
                continue
            for key in keys:
                if info.name == f'{fn}.{key}':
                    hash_value = hashlib.sha256(
                        os.path.join(tar, info.name.replace(f'.{key}', '')).encode('utf-8')).hexdigest()
                    assert hash_value == utt_id
                    if key == 'speaker_id':
                        value = int(info.name.split('.')[1].split('_')[0])
                    else:
                        cur_f = tar_obj.extractfile(info)
                        value = pickle.load(cur_f)
                    file_info[key] = value
                    break
        except:
            continue
        
    hash_value = utt_id
    if len(file_info) != len(keys):
        print(file_info.keys())
        return None
    if 'code' not in file_info:
        file_info['code'] = os.path.join(code_data_dir, f'{hash_value}.code')
    if 'emb' not in file_info:
        file_info['emb'] = os.path.join(emb_data_dir, f'{hash_value}.emb')
    return process_item(hash_value, file_info, refs)


def process_data_test(test_meta_fn, spk_dict, zero_shot_spk_ids, ref_data_num, data_dir, code_data_dir, emb_data_dir, prefix, zero_shot):
    used_utt_ids = set()
    with open(test_meta_fn, 'w') as f:
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        p = Pool(int(os.getenv('N_PROC', os.cpu_count())))
        key = ['tar', 'fn', 'utt_id', 'spk_id']
        print('\t'.join(key), file=f)
        futures = []
        for spk_id in zero_shot_spk_ids:
            entries = spk_dict[spk_id]
            ref_entries = entries[:ref_data_num]
            entries = entries[ref_data_num:]
            for idx, entry in enumerate(entries):
                used_utt_ids.add(entry['utt_id'])
                print('\t'.join([str(entry[k]) for k in key]), file=f)
                futures.append(p.apply_async(get_test_data, args=(entry, ref_entries, code_data_dir, emb_data_dir)))
                if not zero_shot and idx >= len(entries) // 2:
                    break
        
        p.close()
        all_keys = []
        lengths = []
        f0s = []
        durs = []
        for future in tqdm(futures):
            res = future.get()
            if res is None:
                continue
            item_name, phone_encoded, mel, mel2ph, spk_id, pitch, f0, dur, token, refs = res
            builder.add_item({
                'item_name': item_name,
                'phone': phone_encoded,
                'code': token.astype(np.uint16),
                'mel2ph': mel2ph,
                'spk_id': spk_id,
                'f0': f0,
                'refs': refs,
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
    return used_utt_ids


def process_data_valid(valid_entries, code_data_dir, emb_data_dir, data_dir, prefix='valid'):
    valid_meta_fn = f'{data_dir}/{prefix}_metadata.tsv'
    used_utt_ids = set()
    with open(valid_meta_fn, 'w') as f:
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        p = Pool(int(os.getenv('N_PROC', os.cpu_count())))
        key = ['tar', 'fn', 'utt_id', 'spk_id']
        print('\t'.join(key), file=f)
        futures = []
        for idx, entry in enumerate(valid_entries):
            used_utt_ids.add(entry['utt_id'])
            print('\t'.join([str(entry[k]) for k in key]), file=f)
            ref_entries = [entry]
            futures.append(p.apply_async(get_test_data, args=(entry, ref_entries, code_data_dir, emb_data_dir)))
        
        p.close()
        all_keys = []
        lengths = []
        f0s = []
        durs = []
        for future in tqdm(futures):
            res = future.get()
            if res is None:
                continue
            item_name, phone_encoded, mel, mel2ph, spk_id, pitch, f0, dur, token, refs = res
            builder.add_item({
                'item_name': item_name,
                'phone': phone_encoded,
                'code': token.astype(np.uint16),
                'mel2ph': mel2ph,
                'spk_id': spk_id,
                'f0': f0,
                'refs': refs,
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
    return used_utt_ids

if __name__ == "__main__":
    set_hparams()
    
    full_file = hparams['full_file']
    code_data_dir = hparams['code_data_dir']  # contain all codes train + valid + test
    emb_data_dir = hparams['emb_data_dir']
    data_dir =  hparams['data_dir']
    print('data_dir:', data_dir)

    # os.makedirs(hparams['data_dir'], exist_ok=True)
    # shutil.copy('phone_set.json', f"{data_dir}/phone_set.json")

    # json.dump(ph_set, open(, 'w'))
    
    # gen metadata
    
    print('generate metadata ...')
    
    tars = []
    with open(full_file, 'r') as f:
        for line in f:
            tars.append(line.strip())
            
    meta_fn = f'{hparams["data_dir"]}/metadata.tsv'
    
    utt_ids = set(pd.read_csv(meta_fn, sep='\t', header=0)['utt_id'])
    test_meta_fn = f'{hparams["data_dir"]}/zero_shot_test_metadata.tsv'  
    used_utt_ids = set(pd.read_csv(test_meta_fn, sep='\t', header=0)['utt_id'])
    utt_ids = utt_ids - used_utt_ids
    
    test_meta_fn = f'{hparams["data_dir"]}/non_zero_shot_test_metadata.tsv'
    used_utt_ids = set(pd.read_csv(test_meta_fn, sep='\t', header=0)['utt_id'])
    utt_ids = utt_ids - used_utt_ids
    
    valid_meta_fn = f'{hparams["data_dir"]}/valid_metadata.tsv'
    used_utt_ids = set(pd.read_csv(valid_meta_fn, sep='\t', header=0)['utt_id'])
    utt_ids = utt_ids - used_utt_ids
   
    
    train_utt_ids = utt_ids
    print('train num:', len(train_utt_ids))
    process_data(full_file, code_data_dir, emb_data_dir, data_dir, 'train', train_utt_ids)

