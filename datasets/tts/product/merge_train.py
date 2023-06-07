import sys
from utils.indexed_datasets import FullIndexedDatasetBuilderForMerge, FullIndexedDatasetBuilder, FullIndexedDataset
from tqdm import tqdm

data_dir=sys.argv[1]
if data_dir[-1] == '/':
    data_dir = data_dir[:-1]
prefix = sys.argv[2] if len(sys.argv) > 2 else 'train'
split_num = int(sys.argv[3]) if len(sys.argv) > 3 else 8

print(data_dir, prefix, split_num)

for split in tqdm(range(split_num)):
    if split == 0:
        ds = FullIndexedDatasetBuilder(f'{data_dir}/{prefix}')
    else:
        ds = FullIndexedDatasetBuilderForMerge(f'{data_dir}/{prefix}')
    ds.merge_file_(f'{data_dir}_{split}/{prefix}')
    ds.finalize()

print('Merging done. Now testing merged datasets...')
begin = 0
split = 0
merged = FullIndexedDataset(f'{data_dir}/{prefix}')
cur = FullIndexedDataset(f'{data_dir}_{split}/{prefix}')
cur_l = len(cur)
tot_len = len(merged)
for i in tqdm(range(tot_len)):
    if i < cur_l + begin:
        assert merged[i] == cur[i - begin], f"{merged[i]}, {cur[i - begin]}"
    else:
        split += 1
        begin = cur_l + begin
        cur = FullIndexedDataset(f'{data_dir}_{split}/{prefix}')
        cur_l = len(cur)
    