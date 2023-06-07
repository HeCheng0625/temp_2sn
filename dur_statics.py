import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#LJ
fn = '/blob/v-shenkai/code/tts/fs2_fromyc/mm_lm_nat/data-bin/ljspeech_discrete_res16hop200_vqemb_44w/train_durs.npy'
# product_1000h
# fn = '/blob/v-zeqianju/code/tts/fs2_fromyc/mm_lm_nat/data-bin/product_1000_discrete_res16hop200_vqemb_44w_fixcode_fixbug/train_durs.npy'
with open(fn, 'rb') as f:
    durs = np.load(f, allow_pickle=True)

print(len(durs)) # 242127
print(durs[0].shape) # (128,)
durs = durs
max_dur = 100
freq = np.zeros(max_dur + 1)

for dur in tqdm(durs):
    # find the unique elements and their counts
    values, counts = np.unique(dur, return_counts=True)
    # print(values, counts)
    # filter out the values that are not between 0 and 200
    for value, count in zip(values, counts):
        # print(value, count)
        if value > max_dur:
            value = max_dur
        freq[value] += count
    
   

freq = freq / freq.sum()
print(freq)
plt.plot([i for i in range(max_dur + 1)], freq)
plt.savefig('dur.png')
    

