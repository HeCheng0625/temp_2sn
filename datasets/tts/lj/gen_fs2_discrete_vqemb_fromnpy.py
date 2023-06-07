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

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def process_item(raw_data_dir, encoder, tg_fn, wav_fn, npy_dir, need_transpose):
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

    token_emb = np.load(os.path.join(npy_dir, item_name + '.npy'))
    if need_transpose:
        token_emb = token_emb.T

    assert mel.shape[0] - token_emb.shape[0] < 5

    mel = token_emb

    #token_tensor = torch.from_numpy(token)
    #token_emb = 0
    #for i in range(hparams['audio_num_mel_bins']):
    #    token_emb = token_emb + torch.nn.functional.embedding(token_tensor[:, i], vqemb_list[i])

    mel2ph, dur = get_mel2ph(tg_fn, ph, mel, hparams)
    f0, pitch_coarse = get_pitch(wav_data, mel, hparams)
    return item_name, phone_encoded, mel, mel2ph, spk_id, pitch_coarse, f0, dur, token_emb


def process_data(raw_data_dir, encoder, wav_fns, data_dir, npy_dir, need_transpose, prefix):
    data_df = pd.read_csv(os.path.join(raw_data_dir, 'metadata_phone.csv'))
    fn2txt = {k: v for k, v in zip(data_df['wav'], data_df['txt1'])}

    p = Pool(int(os.getenv('N_PROC', os.cpu_count())))
    #p = Pool(1)
    futures = []

    tg_fn = glob.glob(f"{raw_data_dir}/mfa_outputs/*/*.TextGrid")
    item2tgfn = {os.path.splitext(os.path.basename(v))[0]: v for v in tg_fn}
    for wav_fn in wav_fns:
        item_name = os.path.splitext(os.path.basename(wav_fn))[0].replace("lj_", "")
        if item_name not in item2tgfn.keys():
            print("skip {} due to not found MFA results".format(item_name))
            continue
        # token_tensor = torch.from_numpy(token).cuda()
        # token_emb = 0
        # for i in range(hparams['audio_num_mel_bins']):
        #     token_emb = token_emb + torch.nn.functional.embedding(token_tensor[:, i] - i * hparams['codebook_size'], vqemb_list[i])
        futures.append(p.apply_async(process_item, args=(raw_data_dir, encoder, item2tgfn[item_name], wav_fn, npy_dir, need_transpose)))
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
        item_name, phone_encoded, mel, mel2ph, spk_id, pitch, f0, dur, token_emb = res
        txt = fn2txt[item_name]
        item_name = f'lj_{item_name}'
        builder.add_item({
            'item_name': item_name,
            'txt': txt,
            'phone': phone_encoded,
            'mel': mel,
            'vq_emb': token_emb,
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
    # vq_ckpt = hparams['vq_ckpt']
    npy_dir = hparams['npy_dir']
    need_transpose = hparams['need_transpose']
    # vq_ckpt_weight = torch.load(vq_ckpt)
    # vqemb_list = []
    # for i in range(hparams['audio_num_mel_bins']):
    #     weight_name = 'quantizer.layers.' + str(i) + '._codebook.embed'
    #     vqemb_list.append(vq_ckpt_weight['model']['codec_decoder'][weight_name].clone().cuda())
    #     print("load {} vqemb with a shape of {}".format(weight_name, vqemb_list[-1].shape))
    raw_data_dir = hparams['raw_data_dir']
    all_wav_fns = sorted(glob.glob(f'{raw_data_dir}/wavs/*.wav'))
    # all_tokens = []
    # print("Token file:", os.path.join(raw_data_dir, hparams["token_filename"]))
    # with open(os.path.join(raw_data_dir, hparams["token_filename"]), 'r', encoding='utf-8') as infile:
    #     for line in infile.readlines():
    #         line_token = list(map(int, line.strip().split()))
    #         line_token = np.array(line_token).reshape(-1, hparams['audio_num_mel_bins'])
    #         all_tokens.append(line_token)
    # logging.info("load {} discretee token items".format(len(all_tokens)))
    logging.info("train {}".format(len(all_wav_fns)))

    ph_set = [x.split(' ')[0] for x in open(f'{raw_data_dir}/{hparams["dict_file"]}.txt').readlines()]
    print(ph_set)
    os.makedirs(hparams['data_dir'], exist_ok=True)
    json.dump(ph_set, open(f"{hparams['data_dir']}/phone_set.json", 'w'))
    encoder = build_phone_encoder(hparams['data_dir'])

    # encoder = build_phone_encoder(raw_data_dir)
    os.makedirs(hparams['data_dir'], exist_ok=True)
    # assert len(all_tokens) == len(all_wav_fns)
    process_data(raw_data_dir, encoder, all_wav_fns[:523], hparams['data_dir'], npy_dir, need_transpose, 'valid')
    process_data(raw_data_dir, encoder, all_wav_fns[:523], hparams['data_dir'], npy_dir, need_transpose, 'test')
    process_data(raw_data_dir, encoder, all_wav_fns[523:], hparams['data_dir'], npy_dir, need_transpose, 'train')
