import sys
from utils.indexed_datasets import FullIndexedDatasetBuilderForMerge, FullIndexedDatasetBuilder, FullIndexedDataset
from tqdm import tqdm




singing_repeat = 100
prefix = 'train'
tts = FullIndexedDatasetBuilderForMerge(f'/home/aiscuser/tts_singing_merge/{prefix}')
for split in tqdm(range(singing_repeat)):
    tts.merge_file_(f'/home/aiscuser/singing_20k_full/{prefix}')
tts.finalize()

print('Merging done, singing repeat {singing_repeat}')
    