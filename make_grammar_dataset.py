
# ==================================================================================================
#
# This code is modified from code written for "Grammar Variational Autoencoder":
# https://arxiv.org/abs/1703.01925
# 
# Original reference code: 
# https://github.com/mkusner/grammarVAE/blob/master/make_zinc_dataset_grammar.py
#
# ==================================================================================================

import nltk
import zinc_grammar
import numpy as np
import h5py
import molecule_vae
from tqdm import tqdm
from joblib import Parallel, delayed

f = open('250k_rndm_zinc_drugs_clean.smi', 'r')
L = []

for line in f:
    line = line.strip()
    L.append(line)
f.close()

MAX_LEN = 277
NCHARS = len(zinc_grammar.GCFG.productions())


def to_one_hot(smiles):
    """ Encode a list of smiles strings to one-hot vectors """
    assert type(smiles) == list
    prod_map = {}
    for ix, prod in enumerate(zinc_grammar.GCFG.productions()):
        prod_map[prod] = ix
    tokenize = molecule_vae.get_zinc_tokenizer(zinc_grammar.GCFG)
    tokens = list(map(tokenize, smiles))
    parser = nltk.ChartParser(zinc_grammar.GCFG)
    parse_trees = [next(parser.parse(t)) for t in tokens]
    productions_seq = [tree.productions() for tree in parse_trees]
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    one_hot = np.zeros((len(indices), MAX_LEN, NCHARS), dtype=np.int8)
    for i in range(len(indices)):
        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions), indices[i]] = 1.
        one_hot[i][np.arange(num_productions, MAX_LEN), -1] = 1.

    if len(one_hot)==1:
        return one_hot[0]
    else:
        return one_hot


def main():
    grammars = Parallel(n_jobs=-1, backend='multiprocessing', verbose=5)(delayed(to_one_hot)([i]) for i in tqdm(L))
    grammars = np.array(grammars, dtype=np.int8)
    h5f = h5py.File('grammar_dataset.h5', 'w')
    h5f.create_dataset('data', data=grammars)
    h5f.close()

if __name__ == '__main__':
    main()
    