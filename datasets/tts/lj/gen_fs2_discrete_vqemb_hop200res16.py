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
from utils.preprocessor import process_utterance, get_pitch, get_mel2ph

from usr_dir.codec.residual_vq import ResidualVQ

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def process_item(raw_data_dir, encoder, tg_fn, wav_fn, token):
    item_name = os.path.basename(wav_fn)[:-4].replace("lj_", "")
    spk_id = 0
    ph_fn = f'{raw_data_dir}/mfa_input_txt/{item_name}.ph'
    # spk_embed_fn = f'{raw_data_dir}/spk_embeds/{item_name}.npy'
    ph = open(ph_fn).readlines()[0].strip()
    ph = "<UNK> " + ph + " <EOS>"
    try:
        phone_encoded = encoder.encode(ph)
        wav_data, mel = process_utterance(
            wav_fn, fft_size=hparams['n_fft'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            loud_norm=hparams['loud_norm'],
            min_level_db=hparams['min_level_db'],
            return_linear=False, vocoder=hparams['vocoder'])
        mel = mel.T  # [T, 80]
    except:
        traceback.print_exc()
        print("| invalid data", item_name)
        return None

    assert mel.shape[0] - token.shape[0] < 10

    mel = token + 1

    #token_tensor = torch.from_numpy(token)
    #token_emb = 0
    #for i in range(hparams['audio_num_mel_bins']):
    #    token_emb = token_emb + torch.nn.functional.embedding(token_tensor[:, i], vqemb_list[i])

    mel2ph, dur = get_mel2ph(tg_fn, ph, mel, hparams)
    f0, pitch_coarse = get_pitch(wav_data, mel, hparams)
    return item_name, phone_encoded, mel, mel2ph, spk_id, pitch_coarse, f0, dur, token


def process_data(raw_data_dir, encoder, wav_fns, tokens, data_dir, prefix):
    data_df = pd.read_csv(os.path.join(raw_data_dir, 'metadata_phone.csv'))
    fn2txt = {k: v for k, v in zip(data_df['wav'], data_df['txt1'])}

    p = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    #p = Pool(1)
    futures = []

    tg_fn = glob.glob(f"{raw_data_dir}/mfa_outputs/*/*.TextGrid")
    item2tgfn = {os.path.splitext(os.path.basename(v))[0]: v for v in tg_fn}
    for wav_fn, token in zip(wav_fns, tokens):
        item_name = os.path.splitext(os.path.basename(wav_fn))[0].replace("lj_", "")
        if item_name not in item2tgfn.keys():
            print("skip {} due to not found MFA results".format(item_name))
            continue
        # token_tensor = torch.from_numpy(token).cuda()
        # token_emb = 0
        
        
        # token_emb = quantizer.vq2emb(token_tensor)
        
        
        # for i in range(hparams['audio_num_mel_bins']):
        #     token_emb = token_emb + torch.nn.functional.embedding(token_tensor[:, i] - i * hparams['codebook_size'], vqemb_list[i])
        futures.append(p.apply_async(process_item, args=(raw_data_dir, encoder, item2tgfn[item_name], wav_fn, token)))
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
        txt = fn2txt[item_name]
        item_name = f'lj_{item_name}'
        builder.add_item({
            'item_name': item_name,
            'txt': txt,
            'phone': phone_encoded,
            'mel': mel,
            'code': token,
            'mel2ph': mel2ph,
            'spk_id': spk_id,
            'pitch': pitch,
            'f0': f0,
        })
        lengths.append(mel.shape[0])
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
    set_hparams()
    vq_ckpt = hparams['vq_ckpt']
    vq_ckpt_weight = torch.load(vq_ckpt)
    
    
    quantizer = ResidualVQ(
        num_quantizers=hparams['audio_num_mel_bins'], dim=hparams['vqemb_size'], codebook_size=hparams['codebook_size'], commitment=0.25,
        threshold_ema_dead_code=2
    )
    quantizer.load_state_dict(vq_ckpt_weight)
    quantizer.eval()
    quantizer.requires_grad_(False)
    quantizer = quantizer.cuda()
    
    
    raw_data_dir = hparams['raw_data_dir']
    all_wav_fns = sorted(glob.glob(f'{raw_data_dir}/wavs/*.wav'))
    all_tokens = []
    print("Token file:", os.path.join(raw_data_dir, hparams["token_filename"]))
    with open(os.path.join(raw_data_dir, hparams["token_filename"]), 'r', encoding='utf-8') as infile:
        for line in infile.readlines():
            
            line_token = list(map(int, line.split("|")[1].strip().split()))
            line_token = np.array(line_token).reshape(-1, hparams['audio_num_mel_bins'])
            
            for bin_idx in range(hparams['audio_num_mel_bins']):
                line_token[:, bin_idx] = line_token[:, bin_idx] - bin_idx * hparams['codebook_size']
                
            all_tokens.append(line_token)
    logging.info("load {} discrete token items".format(len(all_tokens)))
    logging.info("train {}".format(len(all_wav_fns)))

    ph_set = [x.split(' ')[0] for x in open(f'{raw_data_dir}/{hparams["dict_file"]}.txt').readlines()]
    print(ph_set)
    os.makedirs(hparams['data_dir'], exist_ok=True)
    json.dump(ph_set, open(f"{hparams['data_dir']}/phone_set.json", 'w'))
    encoder = build_phone_encoder(hparams['data_dir'])

    # encoder = build_phone_encoder(raw_data_dir)
    os.makedirs(hparams['data_dir'], exist_ok=True)
    assert len(all_tokens) == len(all_wav_fns)
    process_data(raw_data_dir, encoder, all_wav_fns[:523], all_tokens[:523], hparams['data_dir'], 'valid')
    process_data(raw_data_dir, encoder, all_wav_fns[:523], all_tokens[:523], hparams['data_dir'], 'test')
    process_data(raw_data_dir, encoder, all_wav_fns[523:], all_tokens[523:], hparams['data_dir'], 'train')
