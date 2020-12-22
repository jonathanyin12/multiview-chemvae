import nltk
import numpy as np

import zinc_grammar
import models.grammar_vae
import models.char_vae
import models.two_tower_grammar_vae
import models.two_tower_char_vae


def get_zinc_tokenizer(cfg):
    long_tokens = [a for a in list(cfg._lexical_index.keys()) if len(a) > 1]
    replacements = ['$', '%', '^']  # ,'&']
    assert len(long_tokens) == len(replacements)
    for token in replacements:
        assert token not in cfg._lexical_index

    def tokenize(smiles):
        for i, token in enumerate(long_tokens):
            smiles = smiles.replace(token, replacements[i])
        tokens = []
        for token in smiles:
            try:
                ix = replacements.index(token)
                tokens.append(long_tokens[ix])
            except:
                tokens.append(token)
        return tokens

    return tokenize


def pop_or_nothing(S):
    try:
        return S.pop()
    except:
        return 'Nothing'


def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == 'Nothing':
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix + 1:]
                break
    try:
        return ''.join(seq)
    except:
        return ''


class GrammarModel(object):

    def __init__(self, weights_file, latent_rep_size=56, two_tower=False):
        """ Load the (trained) zinc encoder/decoder, grammar model. """
        self._grammar = zinc_grammar
        self._two_tower = two_tower
        self._model = models.grammar_vae if not self._two_tower else models.two_tower_grammar_vae
        self.MAX_LEN = self._model.MAX_LEN
        self._productions = self._grammar.GCFG.productions()
        self._prod_map = {}
        for ix, prod in enumerate(self._productions):
            self._prod_map[prod] = ix
        self._parser = nltk.ChartParser(self._grammar.GCFG)
        self._tokenize = get_zinc_tokenizer(self._grammar.GCFG)
        self._n_chars = len(self._productions)
        self._lhs_map = {}
        for ix, lhs in enumerate(self._grammar.lhs_list):
            self._lhs_map[lhs] = ix
        self.vae = self._model.MoleculeVAE()
        self.vae.load(self._productions, weights_file, latent_rep_size=latent_rep_size)

    def encode(self, smiles, features=None):
        """ Encode a list of smiles strings into the latent space """
        assert type(smiles) == list
        tokens = list(map(self._tokenize, smiles))
        parse_trees = [next(self._parser.parse(t)) for t in tokens]
        productions_seq = [tree.productions() for tree in parse_trees]
        indices = [np.array([self._prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
        one_hot = np.zeros((len(indices), self.MAX_LEN, self._n_chars), dtype=np.float32)
        for i in range(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions), indices[i]] = 1.
            one_hot[i][np.arange(num_productions, self.MAX_LEN), -1] = 1.
        if self._two_tower:
            return self.vae.encoderMV.predict(one_hot)[0]
        else:
            return self.vae.encoderMV.predict([one_hot, features])[0]

    def _sample_using_masks(self, unmasked):
        """ Samples a one-hot vector, masking at each timestep.
            This is an implementation of Algorithm ? in the paper. """
        eps = 1e-100
        X_hat = np.zeros_like(unmasked)

        # Create a stack for each input in the batch
        S = np.empty((unmasked.shape[0],), dtype=object)
        for ix in range(S.shape[0]):
            S[ix] = [str(self._grammar.start_index)]

        # Loop over time axis, sampling values and updating masks
        for t in range(unmasked.shape[1]):
            next_nonterminal = [self._lhs_map[pop_or_nothing(a)] for a in S]
            mask = self._grammar.masks[next_nonterminal]
            masked_output = np.exp(unmasked[:, t, :]) * mask + eps
            sampled_output = np.argmax(np.random.gumbel(size=masked_output.shape) + np.log(masked_output), axis=-1)
            X_hat[np.arange(unmasked.shape[0]), t, sampled_output] = 1.0

            # Identify non-terminals in RHS of selected production, and
            # push them onto the stack in reverse order
            rhs = [
                [a for a in self._productions[i].rhs() if (type(a) == nltk.grammar.Nonterminal) and (str(a) != 'None')]
                for i in sampled_output]
            for ix in range(S.shape[0]):
                S[ix].extend(list(map(str, rhs[ix]))[::-1])
        return X_hat  # , ln_p

    def decode(self, z):
        """ Sample from the grammar decoder """
        assert z.ndim == 2
        if self._two_tower:
            unmasked, features = self.vae.decoder.predict(z)
        else:
            unmasked = self.vae.decoder.predict(z)
        X_hat = self._sample_using_masks(unmasked)
        # Convert from one-hot to sequence of production rules
        prod_seq = [[self._productions[X_hat[index, t].argmax()]
                     for t in range(X_hat.shape[1])]
                    for index in range(X_hat.shape[0])]
        return [prods_to_eq(prods) for prods in prod_seq]


class CharacterModel(object):

    def __init__(self, weights_file, latent_rep_size=56, two_tower=False):
        self._two_tower = two_tower
        self._model = models.char_vae if not self._two_tower else models.two_tower_char_vae
        self.MAX_LEN = 120
        self.vae = self._model.MoleculeVAE()
        self.charlist = ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[',
                         '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+', 's', 'B', 'r', '/',
                         '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']
        self._char_index = {}
        for ix, char in enumerate(self.charlist):
            self._char_index[char] = ix
        self.vae.load(self.charlist, weights_file, latent_rep_size=latent_rep_size)

    def encode(self, smiles, features=None):
        """ Encode a list of smiles strings into the latent space """
        indices = [np.array([self._char_index[c] for c in entry], dtype=int) for entry in smiles]
        one_hot = np.zeros((len(indices), self.MAX_LEN, len(self.charlist)), dtype=np.float32)
        for i in range(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions), indices[i]] = 1.
            one_hot[i][np.arange(num_productions, self.MAX_LEN), -1] = 1.
        if self._two_tower:
            return self.vae.encoderMV.predict(one_hot)[0]
        else:
            return self.vae.encoderMV.predict([one_hot, features])[0]

    def decode(self, z):
        """ Sample from the character decoder """
        assert z.ndim == 2
        if self._two_tower:
            out, features = self.vae.decoder.predict(z)
        else:
            out = self.vae.decoder.predict(z)
        noise = np.random.gumbel(size=out.shape)
        sampled_chars = np.argmax(np.log(out) + noise, axis=-1)
        char_matrix = np.array(self.charlist)[np.array(sampled_chars, dtype=int)]
        return [''.join(ch).strip() for ch in char_matrix]
