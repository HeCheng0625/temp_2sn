import json
import os
import shutil
import librosa
import soundfile as sf
input_json ='query32-cln-sepw/libritts_3s.json.290k_sp50'
with open(input_json, 'r') as f:
    meta = json.load(f)
   
rerank = True
out_dir = 'cmos_libri_3s'
if rerank:
    out_dir +='_rerank'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(f'{out_dir}/voice1', exist_ok=True)  # gt
os.makedirs(f'{out_dir}/voice2', exist_ok=True)  # infer

n_sample = 20
idx = 0
with open(f'{out_dir}/voice1/ChitChat1.txt', 'w', encoding='utf-16') as f1, open(f'{out_dir}/voice2/ChitChat1.txt', 'w', encoding='utf-16') as f2:
    for i, case in enumerate(meta['test_cases']):
        if ',' in case['transcription'] or 'dinner-' in case['transcription']:
            continue
        if idx >= n_sample:
            break
        idx += 1
        fn = str(idx).zfill(10)
        wav, _ = librosa.load(case['wav_path'], sr=16000)
        sf.write(f'{out_dir}/voice1/{fn}.wav', wav, 16000, subtype='PCM_24')
        # shutil.copyfile(, )
        print(case['transcription'], file=f1)
        print(case['transcription'], file=f2)
        if not rerank:
            s_idx = 0
        else:
            max_score= case['speaker_verification_score_rerank']
            scores = case['speaker_verification_score']
            s_idx = scores.index(max_score)
        shutil.copyfile(case['synthesized_wav_paths'][s_idx], f'{out_dir}/voice2/{fn}.wav')
        