import pytorch_lightning as pl
from utils.hparams import hparams
# if hparams['use_random_segment_as_ref']:
#     from usr_dir.datasets.latent_dataset_separate_ref import FastSpeechDataset
# else:
#     from usr_dir.datasets.latent_dataset import FastSpeechDataset
from usr_dir.datasets.latent_dataset_separate_ref import FastSpeechDataset
import torch
import torch.distributed as dist
import numpy as np
import utils
import os, json
from utils.text_encoder import TokenTextEncoder
from torch.utils.data.sampler import BatchSampler, SequentialSampler


class VariableSampler(BatchSampler):
    def __init__(self, sampler, drop_last: bool):
        
        self.data_list = sampler
        self.sampler = SequentialSampler(sampler)
        
        super().__init__(self.sampler, 1, drop_last)
        
    def __iter__(self):
        
        for batch_ids in self.data_list:
            yield batch_ids
        
    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
    

class GPTTtsDataModule(pl.LightningDataModule):
    def __init__(self, use_ddp=True) -> None:
        super().__init__()
        self.max_tokens = hparams['max_tokens']
        self.max_sentences = hparams['max_sentences']
        self.max_eval_tokens = hparams['max_eval_tokens']
        if self.max_eval_tokens == -1:
            hparams['max_eval_tokens'] = self.max_eval_tokens = self.max_tokens
        self.max_eval_sentences = hparams['max_eval_sentences']
        if self.max_eval_sentences == -1:
            hparams['max_eval_sentences'] = self.max_eval_sentences = self.max_sentences
        
        self.use_ddp = use_ddp
        
    def setup(self, stage: str) -> None:
        self.phone_encoder = self.build_phone_encoder(hparams['data_dir'])
    
    def build_phone_encoder(self, data_dir):
        phone_list_file = os.path.join(data_dir, 'phone_set.json')
        phone_list = json.load(open(phone_list_file))
        return TokenTextEncoder(None, vocab_list=phone_list)

    def train_dataloader(self):
        train_dataset = FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                          hparams['train_set_name'], hparams, shuffle=True)
        return self.build_dataloader(train_dataset, True, self.max_tokens, self.max_sentences,
                                     endless=hparams['endless_ds'])
        
    def val_dataloader(self):
        valid_dataset = FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                          hparams['valid_set_name'], hparams,
                                          shuffle=False)
        return self.build_dataloader(valid_dataset, False, self.max_eval_tokens, self.max_eval_sentences)

    def test_dataloader(self):
        test_dataset = FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                         hparams['test_set_name'], hparams, shuffle=False)
        return self.build_dataloader(test_dataset, False, self.max_eval_tokens, self.max_eval_sentences)
    
    def build_dataloader(self, dataset, shuffle, max_tokens=None, max_sentences=None,
                         required_batch_size_multiple=-1, endless=False):
        if required_batch_size_multiple == -1:
            required_batch_size_multiple = torch.cuda.device_count()

        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches

        if max_tokens is not None:
            max_tokens *= torch.cuda.device_count()
        if max_sentences is not None:
            max_sentences *= torch.cuda.device_count()
        indices = dataset.ordered_indices()
        batch_sampler = utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
            if endless:
                batches = [b for _ in range(1000) for b in shuffle_batches(list(batch_sampler))]
        else:
            batches = batch_sampler
            if endless:
                batches = [b for _ in range(1000) for b in batches]
        num_workers = dataset.num_workers
        if self.use_ddp:
            num_replicas = dist.get_world_size()
            
            rank = dist.get_rank()
            print("DDP, .....", num_replicas, rank, flush=True)
            batches = [x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0]
        
        return torch.utils.data.DataLoader(dataset,
                                           collate_fn=dataset.collater,
                                           batch_sampler=VariableSampler(batches, drop_last=False),
                                           num_workers=num_workers,
                                           persistent_workers=True,
                                           pin_memory=False)