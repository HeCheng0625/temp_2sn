import glob
import torch
name = 'zero-shot'
mean = 0
std = 0
num =0 
for fn in glob.glob(f'spk_emb/{name}/*.pt'):
    num +=1
    cur = torch.load(fn)
    mean += cur.mean()
    std += cur.std()

print(mean / num, std / num)    