#!/usr/bin/env python
# coding=utf-8


from collections import Counter, defaultdict
import os
import itertools
import pickle
import numpy as np

def unk_idx():
    return 1

class Vocab:

    """
    Vocabulary class
    """
    def __init__(self, 
            path, 
            max_vocab, 
            min_freq, 
            pad_token='<PAD>',
            unk_token='<UNK>',
            sos_token='<SOS>',
            eos_token='<EOS>',
            prefix='all',
            ):
        """
        Args:
        """
        self.path = path
        self.max_vocab = max_vocab
        self.min_freq = min_freq

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.pad_idx, self.unk_idx = 0, 1
        self.sos_idx, self.eos_idx = 2, 3

        self.prefix = prefix

        self.tokens = self.freqs = self.itos = self.stoi = None
        
        self.keys = [
                'path', 'max_vocab', 'min_freq',
                'pad_idx', 'unk_idx', 'sos_idx', 'eos_idx', 
                'unk_token', 'pad_token', 'sos_token', 'eos_token', 
                'tokens', 'freqs', 'itos', 'stoi',
                'prefix']

    def build(self, rebuild=True, data=None):
        """
        Args:
            data (list of token lists, or None)
            rebuild (boolean): Rebuild vocabulary
        """

        if not rebuild and os.path.exists(os.path.join(self.path, '{0}_vocab.pkl'.format(self.prefix))):
            print("Loading vocab from {0}".format(self.path))
            self._load()

        elif data is not None:
            self._init_vocab()
            print("Building vocab to {}".format(self.path))
            self.update(data)

            # Reference: torchtext
            min_freq = max(self.min_freq, 1)
            max_vocab = None if self.max_vocab is None else self.max_vocab + len(self.itos)
            words_and_frequencies = list(self.freqs.items())
            words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True) # sort by order

            for word, freq in words_and_frequencies:
                if not (freq < min_freq or len(self.itos) == max_vocab) and word not in self.itos:
                    self.itos.append(word)
 
            self.stoi.update({tok: i for i, tok in enumerate(self.itos)})
            self._dump()

        else:
            raise ValueError('Please provide data to build vocabulary')


    def _init_vocab(self):
        """
        Initialize vocabulary
        """
        specials = [self.pad_token, self.unk_token, self.sos_token, self.eos_token]
        self.itos = list(specials)
        self.stoi = defaultdict(unk_idx)
        self.freqs = Counter([])

    def update(self, data):
        """
        Update current word frequency
        """
        allwords = list(itertools.chain.from_iterable(data))
        self.freqs.update(allwords)

    
    def _finish(self):
        """
        Finish updating and rebuild vocabulary
        """
        min_freq = max(self.min_freq, 1)
        max_vocab = None if self.max_vocab is None else self.max_vocab + len(self.itos)
        words_and_frequencies = list(self.freqs.items())
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True) # sort by order

        for word, freq in words_and_frequencies:
            if not (freq < min_freq or len(self.itos) == max_vocab) and word not in self.itos:
                self.itos.append(word)
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})
        self._dump()


    def build_bpe(self, corpus, n_merges):

        def get_pairs(freqs):
            pairs = collections.defaultdict(int)
            for word, freq in freqs.items():
                symbols = word.split()
                for i in range(len(symbols)-1):
                    pairs[symbols[i], symbols[i+1]] += freq
            return pairs

        def merge_vocab(pair, v_in):
            v_out = {}
            bigram = re.escape(' '.join(pair))
            p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            for word in v_in:
                w_out = p.sub(''.join(pair), word)
                v_out[w_out] = v_in[word]
            return v_out

        self.merges = []
        freqs = self.freqs
        for i in range(num_meregs):
            pairs = get_pairs(freqs)
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            freqs = merge_vocab(best, freqs)
            print(best)



    def summary(self):
        s = "Vocabulary summary:\n"
        s += '  Vocab size:\t{0}\n'.format(len(self.itos))
        s += '  Min freq:\t{0}\n'.format(self.min_freq)
        s += '  Max vocab:\t{0}\n'.format(self.max_vocab)
        print(s)
        with open(os.path.join(self.path, '{0}_vocab_summary.txt'.format(self.prefix)), 'w') as f:
            f.write(s)

        with open(os.path.join(self.path, '{0}_vocab.txt'.format(self.prefix)), 'w') as f:
            for w in self.itos:
                f.write(w + '\n')


    def __len__(self):
        return len(self.itos)


    def _dump(self):
        d = dict()
        for k in self.keys:
            d[k] = getattr(self, k)
        with open(os.path.join(self.path, '{0}_vocab.pkl'.format(self.prefix)), 'wb') as f:
            pickle.dump(d, f)


    def _load(self):
        with open(os.path.join(self.path, '{0}_vocab.pkl'.format(self.prefix)), 'rb') as f:
            d = pickle.load(f)
        for k in self.keys:
            setattr(self, k, d[k])


    def toi(self, seq, ignore_specials=True):
        specials = [self.unk_token, self.pad_token, \
                self.sos_token, self.eos_token]
        return [self.stoi[word] for word in seq \
                if not (ignore_specials and word in specials)]


    def tos(self, seq, ignore_specials=True):
        specials = [self.unk_idx, self.pad_idx, \
                self.sos_idx, self.eos_idx]
        return [self.itos[idx] for idx in seq \
                if not (ignore_specials and idx in specials)]

    
    def to_full_idx(self, ):
        # TODO
        pass




