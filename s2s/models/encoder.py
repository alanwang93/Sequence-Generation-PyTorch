#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Yifan WANG <yifanwang1993@gmail.com>
#
# Distributed under terms of the MIT license.

import os
import time


import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

from s2s.models.common import Feedforward, Embedding



"""
Encoder
"""

class Encoder(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, inputs, lengths, hidden=None):
        raise NotImplementedError()

    def init_weights(self):
        raise NotImplementedError()

    def load_embedding(self, embed_file):
        # TODO
        pass


class RNNEncoder(Encoder):

    def __init__(self, 
            hidden_size,
            num_layers,
            embed_size,
            vocab,
            bidirectional=True,
            dropout=0.2,
            pretrained=None,
            pretrained_size=None,
            projection=False):

        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.pad_idx = vocab.pad_idx
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dropout = dropout
        # embedding
        self.pretrained = pretrained
        self.pretrained_size = pretrained_size
        self.projection = projection

        self.embedding = Embedding(
                self.embed_size,
                self.vocab,
                self.pretrained,
                self.pretrained_size,
                self.projection)


        # TODO
        self.rnn = nn.GRU(
                input_size = self.embed_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional)

    def forward(self, inputs, lengths, hidden=None, concat_output=True):
        """
        Return:
            last (batch_size, hidden_size)
            output (batch_size, seq_len, hidden_size)
        """
        concat_output = False
        inputs = self.embedding(inputs)
        lens, indices = torch.sort(lengths, 0, True)
        inputs = inputs[indices]
        outputs, h_n = self.rnn(pack(inputs, lens, 
                batch_first=True), hidden)
        outputs = unpack(outputs, batch_first=True)[0]
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices]
        if not concat_output:
            h_n = h_n.view(self.num_layers, self.num_directions, inputs.size(0), self.hidden_size)
            h_n = torch.mean(h_n, 1)
            outputs = torch.mean(outputs.view(outputs.size(0), outputs.size(1), self.num_directions, self.hidden_size), 2)
        # print(h_n.size())
        # print(outputs.size*)
        # masks = (lengths-1).view(-1, 1, 1).expand(batch_size, 1, output.size(2))
        # last = torch.gather(output, 1, masks)
        return h_n, outputs



class CNNEncoder(Encoder):

    def __init__(self, config):
        pass
    
    def forward(self, inputs, lengths, hidden=None):
        pass


class TransformerEncoder(Encoder):

    def __init__(self, config):
        pass

    def forward(self, inputs, lengths, hidden=None):
        pass





