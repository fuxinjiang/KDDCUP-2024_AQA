
from unittest import result
import torch
import os.path as osp
import numpy as np
gpu_num = 4
model_name =  "nvidia/NV-Embed-v1"
mnt_root = '../output'
sentences = []
emb = []
metadatas = []
for gpu in range(gpu_num):
    result_part = torch.load(osp.join(mnt_root, f'{model_name}_gpu_{gpu}_emb.bin'))
    sentences = result_part[0]
    emb_part = result_part[1]
    emb.append(emb_part)
    metadatas = result_part[2]

all_emb = np.concatenate(emb, axis=0)
print(f"embedding: {len(all_emb)}")

torch.save([sentences, all_emb, metadatas], osp.join(mnt_root, f'{model_name}_all_emb.bin'), pickle_protocol=5)