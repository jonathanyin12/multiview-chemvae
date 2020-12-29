from rdkit.Chem import Descriptors, MolFromSmiles, MolToSmiles, rdmolops
import networkx as nx
import sascorer
import numpy as np
import h5py
from tqdm import tqdm


def calculate_cycle_score(mol):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return cycle_length


# We load the smiles data

fname = '../250k_rndm_zinc_drugs_clean.smi'

with open(fname) as f:
    smiles = f.readlines()

logP_values = []
SA_scores = []
cycle_scores = []
for s in tqdm(smiles):
    mol = MolFromSmiles(s.strip())
    logP_values.append(Descriptors.MolLogP(mol))
    SA_scores.append(sascorer.calculateScore(mol))
    cycle_scores.append(calculate_cycle_score(mol))

SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

targets = logP_values_normalized - SA_scores_normalized - cycle_scores_normalized

# We store the results

h5f = h5py.File('targets.h5', 'w')

h5f.create_dataset('logP_values', data=logP_values)
h5f.create_dataset('SA_scores', data=SA_scores)
h5f.create_dataset('cycle_scores', data=cycle_scores)
h5f.create_dataset('targets', data=targets)
h5f.close()
