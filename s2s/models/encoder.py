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
            vocab_size,
            bidirectional=True,
            pad_idx=0,
            dropout=0.2,
            embed_file=None):

        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.pad_idx = pad_idx
        self.bidirectional = bidirectional
        self.num_directions = 1 if bidirectional else 2
        self.dropout = dropout
        self.embed_file = embed_file

        # TODO
        self.embedding = nn.Embedding(
                self.vocab_size,
                self.embed_size,
                padding_idx = self.pad_idx)

        # TODO
        self.rnn = nn.GRU(
                input_size = self.embed_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True)

    def forward(self, inputs, lengths, hidden=None, concat_output=True):
        """
        Return:
            last (batch_size, hidden_size)
            output (batch_size, seq_len, hidden_size)
        """
        inputs = self.embedding(inputs)
        batch_size, seq_len, _ = inputs.size()
        output, h_n = self.rnn(inputs, hidden)
        if not concat_output:
            output = output.view(seq_len, batch_size, self.num_directions, self.hidden_size)
            output = torch.mean(output, 2)
        masks = (lengths-1).view(-1, 1, 1).expand(batch_size, 1, output.size(2))
        last = torch.gather(output, 1, masks)
        return last, output


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





