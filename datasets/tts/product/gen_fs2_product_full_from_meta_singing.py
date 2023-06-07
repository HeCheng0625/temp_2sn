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
        
        # if hparams["f0_use_product"]:
        f0 = file_info['f0']
        pitch_coarse = f0_to_coarse(f0) + 1
        # else:
        #     f0, pitch_coarse = get_pitch(file_info["speech"], file_info["mel"], hparams)

        return item_name, phone_encoded, mel, mel2ph, spk_id, pitch_coarse, f0, dur, token, refs
    except:
        # assert False
        print('failed item:', item_name)
        return None


def process_data(tar_file, code_data_dir, emb_data_dir, data_dir, prefix, sids):
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
        utt_ids=sids,
    )
    
    
def get_test_data(sid, refs, code_data_dir):
    try:     
        hash_value = sid
        file_info = dict()
        for key in ['phone_id', 'duration', 'f0']:
            with open (f'{code_data_dir}/{sid}.{key}', 'rb') as f:
                file_info[key] = pickle.load(f)
        
        file_info['code'] = os.path.join(code_data_dir, f'{hash_value}.code')
        file_info['emb'] = None
        file_info['speaker_id'] = None
        
        return process_item(hash_value, file_info, refs)
    except:
        # assert False
        return None


def process_data_test(test_meta_fn, data_dir, code_data_dir, prefix, zero_shot):
    used_sids = 0
    test_meta = pd.read_csv(test_meta_fn, sep='\t', header=0, dtype=str)
    futures = []
    builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
    p = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    key = ['tar', 'fn', 'sid', 'spk_id']
    for idx, row in test_meta.iterrows():
        # print(row)
        sid = row['sid']
        ref_sid = row.get('ref_sid', sid)
        used_sids += 1
        futures.append(p.apply_async(get_test_data, args=(sid, [ref_sid], code_data_dir)))
            
    p.close()
    all_keys = []
    lengths = []
    f0s = []
    durs = []
    for future in tqdm(futures):
        res = future.get()
        if res is None:
            used_sids -= 1
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
    # np.save(f'{data_dir}/{prefix}_f0s.npy', f0s)
    np.save(f'{data_dir}/{prefix}_durs.npy', durs)
    return used_sids


def process_data_valid(valid_entries, code_data_dir, emb_data_dir, data_dir, prefix='valid'):
    valid_meta_fn = f'{data_dir}/{prefix}_metadata.tsv'
    used_sids = set()
    with open(valid_meta_fn, 'w') as f:
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        p = Pool(int(os.getenv('N_PROC', os.cpu_count())))
        key = ['tar', 'fn', 'sid', 'spk_id']
        print('\t'.join(key), file=f)
        futures = []
        for idx, entry in enumerate(valid_entries):
            used_sids.add(entry['sid'])
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
    return used_sids

if __name__ == "__main__":
    set_hparams()
    
    # full_file = hparams['full_file']
    code_data_dir = '/blob/v-yuancwang/TTS_Data/11lab'
    # emb_data_dir = hparams['emb_data_dir']
    data_dir =  '/blob/v-yuancwang/TTS_Data/11lab_process'
    print('data_dir:', data_dir)

    os.makedirs(data_dir, exist_ok=True)
    shutil.copy('phone_set.json', f"{data_dir}/phone_set.json")

    # json.dump(ph_set, open(, 'w'))
    
    # gen metadata
    
    print('generate metadata ...')
    
    test_meta_fn = f'{data_dir}/zero_shot_test_metadata.tsv'
      
    
    used_sids = process_data_test(test_meta_fn, data_dir, code_data_dir, prefix='zero_shot_test', zero_shot=True)
    print('zero-shot test num:', used_sids)
    
    # generate non-zero-shot test
    test_meta_fn = f'{data_dir}/non_zero_shot_test_metadata.tsv'
    used_sids = process_data_test(test_meta_fn, data_dir, code_data_dir, prefix='non-zero_shot_test', zero_shot=False)
    print('non-zero-shot test num:', used_sids)
    
    
    valid_meta_fn = f'{data_dir}/valid.tsv'
    used_sids = process_data_test(valid_meta_fn, data_dir, code_data_dir, prefix='valid', zero_shot=False)
    print('valid num:', used_sids)
    
    train_meta_fn = f'{data_dir}/train.tsv'
    used_sids = process_data_test(train_meta_fn, data_dir, code_data_dir, prefix='train', zero_shot=False)
    print('train num:', used_sids)
    

    # # process valid data
    # print('valid num:', len(valid_sids))
    # valid_entries = []
    # for spk_id in spk_dict.keys():
    #     for entry in spk_dict[spk_id]:
    #         if entry['sid'] in valid_sids:
    #             valid_entries.append(entry)
    # process_data_valid(valid_entries, code_data_dir, emb_data_dir, data_dir, prefix='valid')
    
    # print('train num:', len(train_sids))
    # process_data(full_file, code_data_dir, emb_data_dir, data_dir, 'train', train_sids)

