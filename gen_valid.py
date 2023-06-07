# import pandas as pd
# import os
# data_dir = '~/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s'
# # meta_fn = f'{data_dir}/metadata.tsv'
# # total = pd.read_csv(meta_fn, sep='\t', header=0)
# # meta_fn = f'{data_dir}/zero_shot_test_metadata.tsv'
# # zs = pd.read_csv(meta_fn, sep='\t', header=0)
# # meta_fn = f'{data_dir}/non_zero_shot_test_metadata.tsv'
# # nzs = pd.read_csv(meta_fn, sep='\t', header=0)
# meta_fn = f'{data_dir}/valid_metadata.tsv'
# valid = pd.read_csv(meta_fn, sep='\t', header=0)


# # train = pd.concat([total, zs, zs, nzs, nzs, valid, valid]).drop_duplicates(keep=False)
# # train = train.sample(frac=1.)
# # print(train.head(10))
# output_dir = '/blob/v-zeqianju/dataset/tts/in_domain'
# os.makedirs(output_dir, exist_ok=True)



# group = valid.groupby('spk_id')
# idx = 0
# max_sample = 50
# with open(f'{output_dir}/meta.tsv', 'w') as f:
#     k = ['tar', 'fn', 'spk_id', 'ref_tar', 'ref_fn', 'ref_spk_id']
#     print('\t'.join(k), file=f)
#     for key, value in group:
#         if idx >= max_sample:
#             break
#         if len(value) < 2:
#             continue
#         # print(key, len(value))
        
#         value = value.reset_index(drop=True)
#         # print(value)
#         sample = value.iloc[0]
#         ref_sample = value.iloc[1]
#         line = [sample['tar'], sample['fn'], sample['spk_id'], ref_sample['tar'], ref_sample['fn'], ref_sample['spk_id']]
#         line = [str(i) for i in line]
#         print('\t'.join(line), file=f)
#         print('\t'.join(line))
#         idx += 1
        
        # print(len(value))
        # print(value.head(2))
# train.head(50).to_csv(f'{output_dir}/meta.tsv', sep='\t', index=False)
# with open(f'{output_dir}/meta.tsv', 'w') as f:
#     for idx, row in train.iterrows():
#         # print(row)
        
#         tar = row['tar']
#         fn  = row['fn']
#         print(f'{tar}\t{fn}', file=f)
#         if idx > 50:
# #             break


    
# import pandas as pd
# import os
# import tarfile
# import pickle
# import soundfile as sf
# import numpy as np

# output_dir = '/blob/v-zeqianju/dataset/tts/in_domain'
# valid = pd.read_csv(f'{output_dir}/meta.tsv', sep='\t', header=0)
# # print(valid)
# os.makedirs(f'{output_dir}/ref_wavs', exist_ok=True)
# os.makedirs(f'{output_dir}/ref_phone', exist_ok=True)
# os.makedirs(f'{output_dir}/gt_wavs', exist_ok=True)
# os.makedirs(f'{output_dir}/gt_phone', exist_ok=True)
# # print(train)
# # os.makedirs(f'{output_dir}/text', exist_ok=True)
# def get_from_tar(tar, fn):
#     tar_obj = tarfile.open(tar, mode='r')

#     keys = ['duration', 'phone_id', 'speech', 'mel']
#     file_info = dict()
#     for info in tar_obj:
#         if not info.isfile():
#             continue
#         # print(info.name)
#         for key in keys:
#             # print(info.name)
#             # if info.name.endswith('.speech'):
#             #     print(info.name)
#             # if info.name.endswith('speech'):
#             #     print(info.name)
#             if info.name == f'{fn}.{key}':
#                 if key == 'speaker_id':
#                     value = int(info.name.split('.')[1].split('_')[0])
#                 else:
#                     cur_f = tar_obj.extractfile(info)
#                     value = pickle.load(cur_f)
#                 file_info[key] = value
#                 break
#     return file_info

# sr = 16000
# for idx, row in valid.iterrows():
#     tar = row['tar']
#     fn = row['fn']
#     ref_tar = row['ref_tar']
#     ref_fn = row['ref_fn']
#     # print(tar, fn, ref_tar, ref_fn)
    
#     ref_sample = get_from_tar(ref_tar, ref_fn)
#     print('ref_secs', len(ref_sample['speech']) / sr)
#     with open(f'{output_dir}/ref_phone/{idx}.phone_id', 'wb') as f:
#         pickle.dump(ref_sample['phone_id'], f)
#     sf.write(f'{output_dir}/{idx}.wav', ref_sample['speech'], 16000, subtype='PCM_24')
#     tot_len = len(ref_sample['speech'])
#     for ref_sec in [3,5,10]:
#         if tot_len < ref_sec * sr:
#            print(f'{idx}, tot_len:{tot_len / sr} < {ref_sec}s, skip')
#            continue
#         start = np.random.randint(0, tot_len - sr * ref_sec)
#         end = start + sr * ref_sec
#         sf.write(f'{output_dir}/ref_wavs/{idx}_{ref_sec}s.wav', ref_sample['speech'][start:end], 16000, subtype='PCM_24')
    
    
#     sample = get_from_tar(tar, fn)
    
#     sf.write(f'{output_dir}/gt_wavs/{idx}.wav', sample['speech'], 16000, subtype='PCM_24')
#     with open(f'{output_dir}/gt_phone/{idx}.phone_id', 'wb') as f:
#         pickle.dump(sample['phone_id'], f)
#     exit()

import librosa
import pickle
import json
output_dir = '/blob/v-zeqianju/dataset/tts/in_domain'

phone_set_fn = '/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s/phone_set.json'
with open(phone_set_fn, 'r') as f:
    phone_set = json.load(f)
for ref_sec in [3,5,10]:
    name = f'in_domain_{ref_sec}s'
    output_cases =  []
    for idx in range(50):
        out_case = dict()
        out_case['uid'] = f"in_domain_{idx}"
        out_case['transcription'] = ""
        out_case['ref_duration'] = ref_sec
        out_case['reference_wav_path'] = f'{output_dir}/ref_wavs/{idx}_{ref_sec}s.wav'
        wav, sr = librosa.load(f'{output_dir}/{idx}.wav', sr=None)
        out_case["ref_origin_sr"] = sr
        out_case["ref_origin_transcription"] = ""
        out_case["ref_origin_uid"] = f"{idx}"
        out_case["ref_origin_duration"] = len(wav) / sr
        with open(f'{output_dir}/gt_phone/{idx}.phone_id', 'rb') as f:
            phone_id = pickle.load(f)
        phone_id =phone_id.tolist()
        out_case["synthesize_phoneme_seq"] = [phone_set[i] for i in phone_id]
        out_case["synthesize_phone_id_seq"] = phone_id
        
        output_cases.append(out_case)
        # print(out_case)
    with open(f"{output_dir}/{name}.json", "w") as f:
        json.dump({
            "dataset": name,
            "test_cases": output_cases,
        }, f, indent=4)
            
    