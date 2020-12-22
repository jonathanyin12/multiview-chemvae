import numpy as np
from models.utils import many_one_hot
import h5py
from tqdm import tqdm

f = open('250k_rndm_zinc_drugs_clean.smi', 'r')

L = []
chars = ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[', '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+',
         's', 'B', 'r', '/', '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']
DIM = len(chars)
for line in f:
    line = line.strip()
    L.append(line)
f.close()

count = 0
MAX_LEN = 120
OH = np.zeros((len(L), MAX_LEN, DIM), dtype=np.int8)
for chem in tqdm(L):
    indices = []
    for c in chem:
        indices.append(chars.index(c))
    if len(indices) < MAX_LEN:
        indices.extend((MAX_LEN - len(indices)) * [DIM - 1])
    OH[count, :, :] = many_one_hot(np.array(indices), DIM)
    count = count + 1
f.close()
h5f = h5py.File('char_dataset.h5', 'w')
h5f.create_dataset('data', data=OH)
h5f.create_dataset('chr', data=np.string_(chars))
h5f.close()
