#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright Â© 2018 Yifan WANG <yifanwang1993@gmail.com>
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

class Decoder(nn.Module):
    
    def __init__(self):
        super().__init__()


    def step(self, inputs, lengths, hidden=None, context=None, context_lengths=None, tf_ratio=1.0):
        """ One-step computation
        
        Args:
            inputs: batch x 1 x dim

        Returns:
            output: hidden state of last layer
            hidden: hidden states of all layers
        """

    def forward(self, inputs, lengths, hidden=None, context=None, context_lengths=None, tf_ratio=1.0):
        raise NotImplementedError()

    def init_weights(self):
        pass

    def load_embedding(self, embed_file):
        pass


    def greedy_decode(self, inputs, lengths, hidden=None, context=None, context_lengths=None):
        
    
    def beam_search(self, inputs, lengths, hidden=None, context=None, context_lengths=None):
        pass

    def sample(self, inputs, lengths, hidden=None, context=None, context_lengths=None, temperature=None):
        pass
        

class RNNDecoder(Decoder):

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

        # TODO: flexible hidden_size
        self.linear_out = nn.Linear(
                self.hidden_size*2 if self.bidirectional else self.hidden_size,
                self.vocab_size)

    def step(self, inputs, hidden=None, context=None, context_lengths=None):
        inputs = self.embedding(inputs)
        batch_size, _, _ = inputs.size()
        # output: (batch, 1, hidden_size)
        output, h_n = self.rnn(inputs, hidden)
        logit = self.linear_out(output)# (batch, 1, vocab_size)
        return logit, output, h_n

    def forward(self, inputs, lengths, hidden=None, context=None, context_lengths=None, tf_ratio=1.0):
        
        batch_size, seq_len = inputs.size()
        inputs = inputs.split(seq_len, 1)
        logits = []
        for inp in inputs:
            logit, output, hidden = self.step(inp, hidden)
            logits.append(logit)
        logits = torch.cat(logits, dim=1)
        return logits

