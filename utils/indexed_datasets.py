import pickle
import numpy as np
import os
import pickle

class IndexedDataset:
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data_file = None
        self.data_offsets = np.load(self.index_file_path(self.path), allow_pickle=True).item()['offsets']
        self.data_file = open(self.data_file_path(self.path), 'rb', buffering=0)

    def check_index(self, i):
        if i < 0 or i >= len(self.data_offsets) - 1:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, i):
        self.check_index(i)
        self.data_file.seek(self.data_offsets[i])
        b = self.data_file.read(self.data_offsets[i + 1] - self.data_offsets[i])
        item = pickle.loads(b)
        return item

    def __len__(self):
        return len(self.data_offsets) - 1
    
    @classmethod
    def index_file_path(cls, path):
        return f"{path}.idx"

    @classmethod
    def data_file_path(cls, path):
        return f"{path}.data"
    
class IndexedDatasetBuilder:
    def __init__(self, path):
        self.path = path
        self.out_file = open(f"{path}.data", 'wb')
        self.byte_offsets = [0]

    def add_item(self, item):
        s = pickle.dumps(item)
        bytes = self.out_file.write(s)
        self.byte_offsets.append(self.byte_offsets[-1] + bytes)
        
    def finalize(self):
        self.out_file.close()
        np.save(open(f"{self.path}.idx", 'wb'), {'offsets': self.byte_offsets})
        
    
class FullIndexedDataset(IndexedDataset):
    def __init__(self, path):
        super().__init__(path=path)
        
        self.lengths = np.load(self.lengths_file_path(self.path), allow_pickle=True)
        # try:
        #     self.f0s = np.load(self.f0s_file_path(self.path), allow_pickle=True)
        # except:
        #     with open(self.f0s_file_path(self.path), 'rb') as f:
        #         self.f0s = pickle.load(f)
        # self.f0s = np.load(self.f0s_file_path(self.path), allow_pickle=True)
        self.all_keys = np.load(self.all_keys_file_path(self.path), allow_pickle=True)

    @classmethod
    def lengths_file_path(cls, path):
        return f"{path}_lengths.npy"
    
    @classmethod
    def f0s_file_path(cls, path):
        return f"{path}_f0s.npy"
    
    @classmethod
    def all_keys_file_path(cls, path):
        return f"{path}_all_keys.npy"
    

class FullIndexedDatasetBuilder:
    def __init__(self, path):
        self.path = path
        self.out_file = open(FullIndexedDataset.data_file_path(self.path), 'wb')
        self.byte_offsets = [0]
        self.all_keys = []
        self.lengths = []
        # self.f0s = []

    def add_item(self, item):
        s = pickle.dumps(item)
        bytes = self.out_file.write(s)
        self.byte_offsets.append(self.byte_offsets[-1] + bytes)
        
        self.all_keys.append(item["item_name"])
        self.lengths.append(item["code"].shape[0])
        # self.f0s.append(item["f0"])

    def finalize(self):
        self.out_file.close()
        np.save(open(FullIndexedDataset.index_file_path(self.path), 'wb'), {'offsets': self.byte_offsets})
        np.save(open(FullIndexedDataset.all_keys_file_path(self.path), 'wb'), self.all_keys)
        np.save(open(FullIndexedDataset.lengths_file_path(self.path), 'wb'), self.lengths)
        # np.save(open(FullIndexedDataset.f0s_file_path(self.path), 'wb'), self.f0s)
        
    def merge_file_(self, another_file):
        index = FullIndexedDataset(another_file)

        begin = self.byte_offsets[-1]
        for offset in index.data_offsets[1:]:
            self.byte_offsets.append(begin + offset)
            
        self.all_keys.extend(index.all_keys)
        self.lengths.extend(index.lengths)
        # self.f0s.extend(index.f0s)
        

        with open(FullIndexedDataset.data_file_path(another_file), "rb") as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break

import os

class FullIndexedDatasetBuilderForMerge:
    def __init__(self, path):
        self.path = path
        self.out_file = open(FullIndexedDataset.data_file_path(self.path), 'ab+')
        self.byte_offsets = [0]
        
        raise NotImplementedError("")
        # TODO: load all_keys, lengths, f0s from existing files
        self.all_keys = []
        self.lengths = []
        # self.f0s = []

    def finalize(self):
        self.out_file.close()
        np.save(open(FullIndexedDataset.index_file_path(self.path), 'wb'), {'offsets': self.byte_offsets})
        np.save(open(FullIndexedDataset.all_keys_file_path(self.path), 'wb'), self.all_keys)
        np.save(open(FullIndexedDataset.lengths_file_path(self.path), 'wb'), self.lengths)
        np.save(open(FullIndexedDataset.f0s_file_path(self.path), 'wb'), self.f0s)
        
    def merge_file_(self, another_file):
        index = FullIndexedDataset(another_file)

        begin = self.byte_offsets[-1]
        for offset in index.data_offsets[1:]:
            self.byte_offsets.append(begin + offset)
            
        self.all_keys.extend(index.all_keys)
        self.lengths.extend(index.lengths)
        self.f0s.extend(index.f0s)
        
        self.out_file.seek(0, os.SEEK_END)
        
        assert self.out_file.read() == b''
        
        with open(FullIndexedDataset.data_file_path(another_file), "rb") as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break



class FullIndexedDatasetBuilderForMerge(FullIndexedDatasetBuilder):
    def __init__(self, path):
        self.path = path
        self.out_file = open(FullIndexedDataset.data_file_path(self.path), 'ab+')
        idx = FullIndexedDataset(self.path)
        self.byte_offsets = [0]
        
        # raise NotImplementedError("")
        # TODO: load all_keys, lengths, f0s from existing files
        index = FullIndexedDataset(self.path)
        begin = self.byte_offsets[-1]
        self.all_keys = []
        self.lengths = []
        for offset in index.data_offsets[1:]:
            self.byte_offsets.append(begin + offset)
        self.all_keys.extend(index.all_keys)
        self.lengths.extend(index.lengths)
        # self.f0s.extend(index.f0s)
         
    def finalize(self):
        self.out_file.close()
        np.save(open(FullIndexedDataset.index_file_path(self.path), 'wb'), {'offsets': self.byte_offsets})
        np.save(open(FullIndexedDataset.all_keys_file_path(self.path), 'wb'), self.all_keys)
        np.save(open(FullIndexedDataset.lengths_file_path(self.path), 'wb'), self.lengths)
        # try:
        #     np.save(open(FullIndexedDataset.f0s_file_path(self.path), 'wb'), self.f0s)
        # except:
        #     with open(FullIndexedDataset.f0s_file_path(self.path), 'wb') as f:
        #         pickle.dump(self.f0s, f)
                
    def merge_file_(self, another_file):
        index = FullIndexedDataset(another_file)
        begin = self.byte_offsets[-1]
        for offset in index.data_offsets[1:]:
            self.byte_offsets.append(begin + offset)
        self.all_keys.extend(index.all_keys)
        self.lengths.extend(index.lengths)
        # self.f0s.extend(index.f0s)
        self.out_file.seek(0, os.SEEK_END)
        assert self.out_file.read() == b''
        with open(FullIndexedDataset.data_file_path(another_file), "rb") as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break



if __name__ == "__main__":
    import random
    from tqdm import tqdm
    ds_path = '/tmp/indexed_ds_example'
    size = 100
    items = [{"a": np.random.normal(size=[10000, 10]),
              "b": np.random.normal(size=[10000, 10])} for i in range(size)]
    builder = IndexedDatasetBuilder(ds_path)
    for i in tqdm(range(size)):
        builder.add_item(items[i])
    builder.finalize()
    ds = IndexedDataset(ds_path)
    for i in tqdm(range(10000)):
        idx = random.randint(0, size - 1)
        assert (ds[idx]['a'] == items[idx]['a']).all()


