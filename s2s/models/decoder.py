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
import torch.nn.functional as F


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


    def greedy_decode(self, hidden, sos_idx, eos_idx, context=None, context_lengths=None, max_length=10):
        """
        inputs (batch, 1, dim)
        """
        batch_size = hidden.size(1)
        logits = []
        preds = []
        inp = sos_idx * torch.ones((batch_size, 1), dtype=torch.long)
        while True:
            # print('hidden', hidden[:4])
            # print('inp', inp[:4])

            logit, output, hidden = self.step(inp, hidden)
            # print('logit1', logit)
            # logit2 = self.forward(inp, )
            # if len(preds) == 0:
                # print('decode', logit)
            # print('logit', logit[:4])
            logp = F.log_softmax(logit, dim=2)
            maxv, maxidx = torch.max(logp, dim=2)
            if ((maxidx == eos_idx).all() and len(preds) > 0)  or len(preds) >= max_length:
                break
            logits.append(logit)
            preds.append(maxidx.numpy())
            inp = maxidx
        logits = torch.cat(logits, dim=1)
        preds = np.concatenate(preds, 1)
        # print(preds.shape)
        return preds
        
    
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

    def forward(self, inputs, lengths, hidden, context=None, context_lengths=None, tf_ratio=1.0):
        batch_size, seq_len = inputs.size()
        inputs = inputs.split(1, 1)
        logits = []
        for inp in inputs:
            # print('inp', inp[:3])
            logit, output, hidden = self.step(inp, hidden)
            logits.append(logit)
        logits = torch.cat(logits, dim=1)
        # print('train', logits)
        return logits

