
from tasks.base_task import BaseDataset
from utils.indexed_datasets import IndexedDataset
from utils.world_utils import process_f0
import utils


import numpy as np
import torch
import pickle

from utils.chanpin_utils import chanpin_phone_dict

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
            
        self.sep_token_id = len(phone_encoder) + 500 # special token
        
        
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
        if hparams['remove_eos']:
            phone = phone[:-1]
            
        T_t = phone.shape[0]
        dur_gt = mel2ph.new_zeros(T_t + 1).scatter_add(0, mel2ph, torch.ones_like(mel2ph))
        dur_gt = dur_gt[1:]
        # import json
        # phone_set_fn = '/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_full_discrete_res16hop200_vqemb_44w_fixcode_fixbug_wmeta_wof0s/phone_set.json'
        # with open(phone_set_fn, 'r') as f:
        #     phone_set = json.load(f)
        # print('before--------')
        # for idx, (phone_, dur) in enumerate(zip(phone, dur_gt)):
        #     print(idx, phone_set[phone_.item()-3], dur.item())
            
        if hparams['online_prepocess']:
            new_phone =[]
            new_dur = []
            # delete_set = ['br0', '-']
            # delete_set = set(phone_set.index(i) + 3 for i in delete_set)
            # print('delete_set', delete_set)
            delete_set = set([633, 4147])
            # replace_set = ['br1', 'br2', 'br3', 'br4', '-sil-']
            # replace_set = set(phone_set.index(i)+3 for i in replace_set)
            # print('replace_set', replace_set)
            replace_set = set([4148, 4149, 4150, 4151, 4153])
            process_set = delete_set | replace_set
            # br1 = phone_set.index('br1') + 3
            # print('br1', br1)
            br1 = 4148
            
            for idx, (phone_, dur) in enumerate(zip(phone, dur_gt)):
                phone_ = phone_.item()
                dur = dur.item()
                # print(phone_, dur)
                if phone_ not in process_set:
                    new_phone.append(phone_)
                    new_dur.append(dur)
                else:
                    if phone_ in delete_set:
                        if dur != 0: 
                            # invalid data
                            print('invalid data', phone, dur)
                            return None
                        continue
                    if phone_ in replace_set:
                        if idx == 0 or phone[idx-1].item() not in process_set:
                            new_phone.append(br1)
                            new_dur.append(dur)
                        else:
                            new_dur[-1] += dur
            if new_phone[-2] == br1: 
                # merge br1 to punc. in the end  punc. br1 ~ --> punc. ~
                # new_phone.insert(-2 + 1, br1)
                new_phone.pop(-2)
                dur = new_dur.pop(-2)
                new_dur[-2] += dur
            # print('after--------')
            # for idx, (phone_, dur) in enumerate(zip(new_phone, new_dur)):
            #     print(idx, phone_set[phone_-3], dur)
            
            phone = torch.Tensor(new_phone)
            dur = torch.LongTensor(new_dur)
            mel2ph = dur.cumsum(dim=-1)
            mel2ph = torch.nn.functional.one_hot(mel2ph, num_classes=int(mel2ph.max().item()) + 1)[:-1, :-1].sum(-2).cumsum(dim=-1)
            mel2ph = mel2ph + 1
            # print('mel2ph', mel2ph.shape,   'new_phone', phone.shape)
            # print(mel2ph)
        # exit()
        
        if hparams.get("append_sep", False):
            phone[-1].fill_(self.sep_token_id)
                    
        
        code = torch.Tensor(item['code'][:hparams['max_frames']].astype(np.int32))
        
        # product data not match
        if code.shape[0] > mel2ph.shape[0]:
            code = code[:mel2ph.shape[0], ...]

        # mel2ph[mel2ph > phone.shape[0]] = 0
        
        refs = item.get('refs', [])
        # print('refs', refs)
        
        
        sample = {
            "id": index,
            "utt_id": key,
            "text": item.get('txt', ''),
            "source": phone,
            "target": code,
            # "pitch": torch.LongTensor(item.get("pitch"))[:hparams['max_frames']],
            # "energy": energy,
            "f0": f0[:hparams['max_frames']] if (self.hparams['use_pitch_embed'] or self.hparams["apply_pitch_on_x0"]) else None,
            "uv": uv[:hparams['max_frames']] if (self.hparams['use_pitch_embed'] or self.hparams["apply_pitch_on_x0"]) else None,
            "mel2ph": mel2ph,
            "refs": refs
        }
        # print(sample)
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
