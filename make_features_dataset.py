from descriptastorus.descriptors import rdNormalizedDescriptors
import h5py
from tqdm import tqdm

f = open('data/250k_rndm_zinc_drugs_clean.smi', 'r')

L = []

for line in f:
    line = line.strip()
    L.append(line)
f.close()

generator = rdNormalizedDescriptors.RDKit2DNormalized()


def rdkit_2d_normalized_features(smiles: str):
    results = generator.process(smiles)
    processed, features = results[0], results[1:]
    if processed is None:
        print("Unable to process smiles %s", smiles)
    return features


features = []
for compound in tqdm(L):
    features.append(rdkit_2d_normalized_features(compound))

h5f = h5py.File('features_dataset.h5', 'w')
h5f.create_dataset('data', data=features)
h5f.close()
