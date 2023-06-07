import pandas as pd
import os
# data_dir = '~/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s'
# meta_fn = f'{data_dir}/metadata.tsv'
# total = pd.read_csv(meta_fn, sep='\t', header=0)
# meta_fn = f'{data_dir}/zero_shot_test_metadata.tsv'
# zs = pd.read_csv(meta_fn, sep='\t', header=0)
# meta_fn = f'{data_dir}/non_zero_shot_test_metadata.tsv'
# nzs = pd.read_csv(meta_fn, sep='\t', header=0)
# meta_fn = f'{data_dir}/valid_metadata.tsv'
# valid = pd.read_csv(meta_fn, sep='\t', header=0)


# train = pd.concat([total, zs, zs, nzs, nzs, valid, valid]).drop_duplicates(keep=False)
# train = train.sample(frac=1.)
# print(train.head(10))
# output_dir = '/blob/v-zeqianju/dataset/tts/train'
# os.makedirs(output_dir, exist_ok=True)
# train.head(50).to_csv(f'{output_dir}/meta.tsv', sep='\t', index=False)
# # with open(f'{output_dir}/meta.tsv', 'w') as f:
# #     for idx, row in train.iterrows():
# #         # print(row)
        
# #         tar = row['tar']
# #         fn  = row['fn']
# #         print(f'{tar}\t{fn}', file=f)
# #         if idx > 50:
# #             break
        
import tarfile
import pickle
import soundfile as sf
import numpy as np
output_dir = '/blob/v-zeqianju/dataset/tts/train'
train = pd.read_csv(f'{output_dir}/meta.tsv', sep='\t', header=0)
# print(train)
os.makedirs(f'{output_dir}/text', exist_ok=True)
for idx, row in train.iterrows():
    tar = row['tar']
    fn = row['fn']
    print(tar, fn)
    tar_obj = tarfile.open(tar, mode='r')

    keys = ['duration', 'phone_id', 'speech', 'mel']
    file_info = dict()
    for info in tar_obj:
        if not info.isfile():
            continue
        # print(info.name)
        for key in keys:
            # print(info.name)
            # if info.name.endswith('.speech'):
            #     print(info.name)
            # if info.name.endswith('speech'):
            #     print(info.name)
            if info.name == f'{fn}.{key}':
                if key == 'speaker_id':
                    value = int(info.name.split('.')[1].split('_')[0])
                else:
                    cur_f = tar_obj.extractfile(info)
                    value = pickle.load(cur_f)
                file_info[key] = value
                break

    # print(file_info)
    speech = file_info['speech']
    sec = 3
    sr = 16000
    l = len(speech)
    start = np.random.randint(0, l - sec * sr)
    end = start + sec * sr
    sf.write(f'{output_dir}/train_{idx}.wav', speech[start:end], 16000, subtype='PCM_24')
    with open(f'{output_dir}/text/train_{idx}.phone_id', 'wb') as f:
        pickle.dump(file_info['phone_id'], f)