
from tasks.base_task import BaseDataset
from utils.indexed_datasets import IndexedDataset
from utils.world_utils import process_f0
import utils


import numpy as np
import torch
import pickle

from utils.chanpin_utils import chanpin_phone_dict
from .debug import extact_random_ref_from_item_separately

class FastSpeechDataset(BaseDataset):
    """A dataset that provides helpers for batching."""

    def __init__(self, data_dir, phone_encoder, prefix, hparams, shuffle=False):
        super().__init__(data_dir, prefix, hparams, shuffle)
        self.phone_encoder = phone_encoder
        self.data = None
        self.idx2key = np.load(f'{self.data_dir}/{self.prefix}_all_keys.npy')
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        self.num_spk = hparams['num_spk']
        self.use_indexed_ds = hparams['indexed_ds']
        self.indexed_bs = None

        # filter out items with no pitch
        # f0s = np.load(f'{self.data_dir}/{prefix}_f0s.npy', allow_pickle=True)
        # self.avail_idxs = [i for i, f0 in enumerate(f0s) if sum(f0) > 0]
        self.avail_idxs = [i for i in range(self.idx2key.shape[0])] # use full dataset
        self.sizes = [self.sizes[i] for i in self.avail_idxs]

        if hparams.get("f0_mean", None) is None:
            f0s = np.load(f'{self.data_dir}/train_f0s.npy', allow_pickle=True)
            f0s = np.concatenate(f0s, 0)
            f0s = f0s[f0s != 0]
            hparams['f0_mean'] = self.f0_mean = np.mean(f0s).item()
            hparams['f0_std'] = self.f0_std = np.std(f0s).item()
            del f0s
        
        
    def __del__(self):
        if self.indexed_bs:
            self.indexed_bs.data_file.close()
            del self.indexed_bs
        pass
    

    def _get_item(self, index):
        if not self.use_indexed_ds:
            key = self.idx2key[index]
            item = np.load(f'{self.data_dir}/{self.prefix}/{key}.npy', allow_pickle=True).item()
        else:
            if self.indexed_bs is None:
                self.indexed_bs = IndexedDataset(f'{self.data_dir}/{self.prefix}')
            item = self.indexed_bs[index]
        return item

    def __getitem__(self, index):
        hparams = self.hparams
        index = self.avail_idxs[index]
        key = self.idx2key[index]
        item = self._get_item(index)
        
        # spec = torch.LongTensor(item['mel'])
        # energy = (spec.exp() ** 2).sum(-1).sqrt()[:hparams['max_frames']]
        #print('mel2ph', item['mel2ph'].shape, 'code', item['code'].shape)

        mel2ph = torch.LongTensor(item['mel2ph'])[:hparams['max_frames']]
        if (self.hparams['use_pitch_embed'] or self.hparams["apply_pitch_on_x0"]):
            try:
                assert sum(item["f0"]) >  0
                f0, uv = process_f0(item["f0"], hparams)
            except:
                return None
        else:
            f0 = None
            uv = None
            
        max_phone = max(mel2ph.max().item(), hparams['max_input_tokens'])
        phone = torch.LongTensor(item['phone'][:max_phone])

        if hparams["remove_bos"]:
            phone = phone[1:]
        
        code = torch.Tensor(item['code'][:hparams['max_frames']].astype(np.int32))
        
        # product data not match
        if code.shape[0] > mel2ph.shape[0 ]:
            code = code[:mel2ph.shape[0 ], ...]

        mel2ph[mel2ph > phone.shape[0]] = 0
        
        f0 = f0[:hparams['max_frames']] if (self.hparams['use_pitch_embed'] or self.hparams["apply_pitch_on_x0"]) else None
        uv = uv[:hparams['max_frames']] if (self.hparams['use_pitch_embed'] or self.hparams["apply_pitch_on_x0"]) else None
        
        ret_info = extact_random_ref_from_item_separately(phone_ids=phone, mel2ph=mel2ph, code=code, pitch=f0, uv=uv)
        
        phone_res = ret_info['phone_ids_new']
        if phone_res.shape[0] == 0:
            # print(ret_info)
            return None
        code_res = ret_info['code_new']
        if f0 is not None:
            f0_res = ret_info['pitch_new']
        else:
            f0_res = None
        
        if uv is not None:
            uv_res = ret_info['uv_new']
        else:
            uv_res = None
        
        code_ref = ret_info['code_ref']
        # if code_ref.shape[0] <= 10:
        #     return None
        
        ref_mask = torch.zeros(code_ref.shape[0])
        
        mel2ph_ref = ret_info["mel2ph_new"]
        
        refs = item.get('refs', [])
        # print('refs', refs)
        
        
        sample = {
            "id": index,
            "utt_id": key,
            "text": item.get('txt', ''),
            "source": phone_res,
            "target": code_res,
            "ref_code": code_ref,
            "ref_code_mask": ref_mask,
            # "pitch": torch.LongTensor(item.get("pitch"))[:hparams['max_frames']],
            # "energy": energy,
            "f0": f0_res,
            "uv": uv_res,
            "mel2ph": mel2ph_ref,
            "refs": refs
        }
        if self.num_spk > 1:
            sample["spk_id"] = item['spk_id']
            sample["spk_embed"] = item['spk_embed']
        del item
        return sample

    def collater(self, samples):
        samples = [i for i in samples if i is not None]
        if len(samples) == 0:
            return {}
        pad_idx = self.phone_encoder.pad()
        id = torch.LongTensor([s['id'] for s in samples])
        utt_ids = [s['utt_id'] for s in samples]
        refs = [s['refs'] for s in samples]
        text = [s['text'] for s in samples]
        
        src_tokens = utils.collate_1d([s['source'] for s in samples], pad_idx)
        f0 = utils.collate_1d([s['f0'] for s in samples], -200) if (self.hparams['use_pitch_embed'] or self.hparams["apply_pitch_on_x0"]) else None
        uv = utils.collate_1d([s['uv'] for s in samples]) if (self.hparams['use_pitch_embed'] or self.hparams["apply_pitch_on_x0"]) else None
        # energy = utils.collate_1d([s['energy'] for s in samples], pad_idx) if self.hparams['use_energy_embed'] else None
        mel2ph = utils.collate_1d([s['mel2ph'] for s in samples], pad_idx)
        target = utils.collate_2d([s['target'] for s in samples], pad_idx)
        
        ref_left_pad = self.hparams['ref_left_pad']
        ref_codes = utils.collate_2d([s['ref_code'] for s in samples], pad_idx, left_pad=ref_left_pad)
        ref_codes_mask = utils.collate_1d([s['ref_code_mask'] for s in samples], 1, left_pad=ref_left_pad)
            
        prev_output_mels = utils.collate_2d([s['target'] for s in samples], pad_idx, shift_right=True)

        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        target_lengths = torch.LongTensor([s['target'].shape[0] for s in samples])
    
        ntokens = sum(len(s['source']) for s in samples)
        nmels = sum(len(s['target']) for s in samples)

        batch = {
            'id': id,
            'utt_id': utt_ids,
            'nsamples': len(samples),
            'ntokens': ntokens,
            'nmels': nmels,
            'text': text,
            'src_tokens': src_tokens,
            'mel2ph': mel2ph,
            'src_lengths': src_lengths,
            'targets': target.long(),
            "ref_codes": ref_codes.long(),
            "ref_codes_mask": ref_codes_mask,
            # 'energy': energy,
            'target_lengths': target_lengths,
            'prev_output_mels': prev_output_mels,
            'pitch': f0,
            'uv': uv,
            'refs': refs,
        }

        if self.num_spk > 1:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            spk_embed = torch.FloatTensor([s['spk_embed'] for s in samples])
            batch['spk_ids'] = spk_ids
            batch['spk_embed'] = spk_embed
        return batch
