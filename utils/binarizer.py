import torch
from abc import ABC, abstractmethod
from collections import Counter
import logging
import os
import typing as tp
from dataclasses import dataclass
from multiprocessing import Pool

from utils.file_binarize_utils import find_tar_offsets
from utils.indexed_datasets import FullIndexedDatasetBuilder, FullIndexedDataset
from utils.text_encoder import NUM_RESERVED_TOKENS, UNK_ID, EOS_ID
from utils.preprocessor import process_utterance, get_pitch, get_mel2ph, f0_to_coarse

import tqdm
import tarfile, hashlib, pickle
import numpy as np

logger = logging.getLogger("binarizer")


@dataclass
class BinarizeSummary:
    """
    Keep track of what's going on in the binarizer
    """

    num_seq: int = 0
    replaced: tp.Optional[Counter] = None
    num_tok: int = 0

    @property
    def num_replaced(self) -> int:
        if self.replaced is None:
            return 0
        return sum(self.replaced.values())

    @property
    def replaced_percent(self) -> float:
        return 100 * self.num_replaced / self.num_tok

    def __str__(self) -> str:
        base = f"{self.num_seq} sents, {self.num_tok} tokens"
        if self.replaced is None:
            return base

        return f"{base}, {self.replaced_percent:.3}% replaced"

    def merge(self, other: "BinarizeSummary"):
        replaced = None
        if self.replaced is not None:
            replaced = self.replaced
        if other.replaced is not None:
            if replaced is None:
                replaced = other.replaced
            else:
                replaced += other.replaced
        self.replaced = replaced
        self.num_seq += other.num_seq
        self.num_tok += other.num_tok
        
        
def _worker_prefix(output_prefix: str, worker_id: int):
    return f"{output_prefix}_pt{worker_id}"

class Binarizer(ABC):
    """
    a binarizer describes how to take a string and build a tensor out of it
    """

    @abstractmethod
    def binarize_item(
        self,
        line: str,
        summary: BinarizeSummary,
    ) -> torch.IntTensor:
        ...
        
        
def dur_to_mel2ph(dur):
    dur = torch.from_numpy(dur)
    mel2ph = dur.cumsum(dim=-1)
    mel2ph = torch.nn.functional.one_hot(mel2ph, num_classes=mel2ph.max() + 1)[:-1, :-1].sum(-2).cumsum(dim=-1)
    mel2ph = mel2ph + 1
    return mel2ph.numpy()

class LatentDiffusionBinarizer(Binarizer):
    """
    Takes a Dictionary/Vocabulary, assign ids to each
    token using the dictionary encode_line function.
    """

    def __init__(
        self,
        code_data_dir,
        emb_data_dir,
        hparams
    ) -> None:
        super().__init__()
        self.code_data_dir = code_data_dir
        self.emb_data_dir = emb_data_dir
        self.hparams = hparams

    def binarize_item(
        self,
        item_name, file_info,
        summary: BinarizeSummary,
    ):
        if summary.replaced is None:
            summary.replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == self.dict.unk_index and word != self.dict.unk_word:
                summary.replaced.update([word])
        phone_encoded = []
        phone_encoded.append(UNK_ID)
        for i in file_info['phone_id']:
            phone_encoded.append(NUM_RESERVED_TOKENS + i)
        phone_encoded.append(EOS_ID)

        token = torch.load(os.path.join(self.code_data_dir, f'{item_name}.code'), map_location='cpu')
        token = token.squeeze(1).transpose(0, 1).numpy()

        spk_id = file_info['speaker_id']
        dur = file_info['duration']
        mel2ph = dur_to_mel2ph(dur)
        
        if self.hparams["f0_use_product"]:
            f0 = file_info['f0']
            pitch_coarse = f0_to_coarse(f0) + 1
        else:
            f0, pitch_coarse = get_pitch(file_info["speech"], file_info["mel"], self.hparams)
            
        ret_item = {
            'item_name': item_name,
            'phone': phone_encoded,
            'code': token.astype(np.uint16),
            'mel2ph': mel2ph,
            'spk_id': spk_id,
            'f0': f0,
            'refs': [],  # empty ref for training
        }

        return ret_item


class FileBinarizer:
    """
    An file binarizer can take a file, tokenize it, and binarize each line to a tensor
    """

    @classmethod
    def multiprocess_dataset(
        cls,
        binarizer,
        tar_list: str,
        dataset_impl: str,
        output_prefix: str,
        tar_keys,
        num_workers=1,
        utt_ids=set(),
    ) -> BinarizeSummary:
        final_summary = BinarizeSummary()
        
        os.makedirs(os.path.split(output_prefix)[0], exist_ok=True)
        
        tar_list_offsets, num_workers = find_tar_offsets(tar_list, num_workers)
        
        logger.info("Use {} workers to binarize the {} tars".format(num_workers, len(tar_list)))
        
        first_chunk = tar_list_offsets[0]
        more_chunks = tar_list_offsets[1:]

        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            worker_results = [
                pool.apply_async(
                    cls._binarize_chunk_and_finalize,
                    args=(
                        binarizer,
                        tar_list_chunk,
                        _worker_prefix(
                            output_prefix,
                            worker_id,
                        ),
                        dataset_impl,
                        tar_keys,
                        utt_ids,
                    )
                )
                for worker_id, (tar_list_chunk) in enumerate(
                    more_chunks, start=1
                )
            ]

            pool.close()
            pool.join()
            for r in worker_results:
                summ = r.get()
                final_summary.merge(summ)
        
        final_ds, summ = cls._binarize_file_chunk(
            binarizer=binarizer,
            tar_list_chunk=first_chunk,
            output_prefix=output_prefix,
            dataset_impl=dataset_impl,
            tar_keys=tar_keys,
            utt_ids=utt_ids,
        )
        
        if num_workers > 1:
            for worker_id in range(1, num_workers):
                # merge the worker outputs
                worker_output_prefix = _worker_prefix(
                    output_prefix,
                    worker_id,
                )
                final_ds.merge_file_(worker_output_prefix)
                try:
                    os.remove(FullIndexedDataset.data_file_path(worker_output_prefix))
                    os.remove(FullIndexedDataset.index_file_path(worker_output_prefix))
                    os.remove(FullIndexedDataset.lengths_file_path(worker_output_prefix))
                    os.remove(FullIndexedDataset.f0s_file_path(worker_output_prefix))
                    os.remove(FullIndexedDataset.all_keys_file_path(worker_output_prefix))
                except Exception as e:
                    logger.error(
                        f"couldn't remove {worker_output_prefix}.*", exc_info=e
                    )
        
        
        final_ds.finalize()
        return final_summary

    @staticmethod
    def _binarize_file_chunk(
        binarizer,
        tar_list_chunk: list,
        output_prefix: str,
        dataset_impl: str,
        tar_keys,
        utt_ids: set,
    ) -> tp.Tuple[tp.Any, BinarizeSummary]:  # (dataset builder, BinarizeSummary)
        """
        creates a dataset builder and append binarized items to it. This function does not
        finalize the builder, this is useful if you want to do other things with your bin file
        like appending/merging other files
        """
        ds = FullIndexedDatasetBuilder(output_prefix)
        
        summary = BinarizeSummary()
        
        for tar in tqdm.tqdm(tar_list_chunk):
            tar_item = dict()
            tar_obj = tarfile.open(tar, mode='r')
            for info in tqdm.tqdm(tar_obj):
                try:
                    if not info.isfile():
                        continue
                    for key in tar_keys:
                        if info.name.endswith(f'.{key}'):
                            hash_value = hashlib.sha256(
                                os.path.join(tar, info.name.replace(f'.{key}', '')).encode('utf-8')).hexdigest()
                            if key == 'speaker_id':
                                # spk_id not from file, but from filename
                                # e.g. mls_0.{spk_id}_{}_{}.speaker_id
                                value = int(info.name.split('.')[1].split('_')[0])
                            else:
                                cur_f = tar_obj.extractfile(info)
                                value = pickle.load(cur_f)
                            if hash_value not in tar_item:
                                tar_item[hash_value] = dict() 
                            tar_item[hash_value][key] = value
                            break
                except:
                    continue
            for hash_value, file_info in tqdm.tqdm(tar_item.items()):
                if len(file_info) != len(tar_keys):
                    print(file_info.keys())
                    continue
                if hash_value not in utt_ids:
                    continue
                data_item = binarizer.binarize_item(hash_value, file_info, summary)
                ds.add_item(data_item)
                
        return ds, summary
        

    @classmethod
    def _binarize_chunk_and_finalize(
        cls,
        binarizer,
        tar_list_chunk: list,
        output_prefix: str,
        dataset_impl: str,
        tar_keys,
        utt_ids: set
    ):
        """
        same as above, but also finalizes the builder
        """
        ds, summ = cls._binarize_file_chunk(
            binarizer=binarizer,
            tar_list_chunk=tar_list_chunk,
            output_prefix=output_prefix,
            dataset_impl=dataset_impl,
            tar_keys=tar_keys,
            utt_ids=utt_ids,
        )
        
        ds.finalize()

        return summ