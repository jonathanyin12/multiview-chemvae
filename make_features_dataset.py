from descriptastorus.descriptors import rdNormalizedDescriptors
import h5py
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed


generator = rdNormalizedDescriptors.RDKit2DNormalized()

def rdkit_2d_normalized_features(smiles: str):
    results = generator.process(smiles)
    processed, features = results[0], results[1:]
    if processed is None:
        print("Unable to process smiles %s", smiles)
    return features

f = open('250k_rndm_zinc_drugs_clean.smi', 'r')

L = []
for line in f:
    line = line.strip()
    L.append(line)
f.close()


def main():
    features = Parallel(n_jobs=-1, backend='multiprocessing', verbose=0)(delayed(rdkit_2d_normalized_features)(compound) for compound in tqdm(L))
    features=np.expand_dims(np.array(features), axis=2)
    h5f = h5py.File('features_dataset.h5', 'w')
    h5f.create_dataset('data', data=features)
    h5f.close()

if __name__ == '__main__':
    main()