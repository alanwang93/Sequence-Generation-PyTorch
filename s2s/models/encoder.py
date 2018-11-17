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
import torch.nn.functional as F

from s2s.models.common import Feedforward, Embedding, sequence_mask


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


class PosGatedEncoder(Encoder):

    def __init__(self, 
            hidden_size,
            num_layers,
            embed,
            bidirectional=True,
            rnn_dropout=0.2,
            pos_size=120,
            cell='gru'):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed.embed_size
        self.vocab_size = embed.vocab_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        #self.rnn_dropout = rnn_dropout
        self.pos_size = pos_size
        self.cell = cell
        # embedding
        self.embedding = embed
        self.pos_embedding = nn.Embedding(pos_size, self.embed_size)
        self.len_embedding = nn.Embedding(pos_size, self.embed_size)


        # TODO
        self.rnn = nn.GRU(
                input_size = self.embed_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=True)

        self.gate = nn.Linear(self.embed_size*2+hidden_size*2, hidden_size*2)

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
        h_n = h_n[:,_indices,:]

        len_embed = self.len_embedding(lengths).unsqueeze(1).expand(-1, outputs.size(1), -1)
        pos = torch.arange(0, outputs.size(1)).unsqueeze(0).to(inputs.device)
        pos_embed = self.pos_embedding(pos).expand(outputs.size(0), -1, -1) # batch, len, dim
        #tmp = pos_embed + len_embed
        tmp = torch.cat((pos_embed, len_embed, outputs), -1)
        gate = F.sigmoid(self.gate(tmp))
        outputs = outputs * gate

        # from SEASS
        #forward_last = h_n[0] # of last or 1st layer?
        #backward_last = h_n[1]

        #h_n = h_n.view(self.num_layers, self.num_directions, inputs.size(0), self.hidden_size)
        #outputs = outputs.view(outputs.size(0), outputs.size(1), self.num_directions, self.hidden_size)
        #if not concat_output:
        #    h_n = torch.mean(h_n, 1)
        #    outputs = torch.mean(outputs, 2)
        return outputs, h_n

class SelectiveEncoder(Encoder):

    def __init__(self, 
            hidden_size,
            num_layers,
            embed,
            bidirectional=True,
            rnn_dropout=0.2,
            cell='gru'):

        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed.embed_size
        self.vocab_size = embed.vocab_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        #self.rnn_dropout = rnn_dropout
        self.cell = cell
        # embedding
        self.embedding = embed


        # TODO
        self.rnn = nn.GRU(
                input_size = self.embed_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=True)

        self.selective_gate = nn.Linear(hidden_size*4, hidden_size*2)

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
        h_n = h_n[:,_indices,:]
        # from SEASS
        forward_last = h_n[0] # of last or 1st layer?
        backward_last = h_n[1]
        sentence_vector = torch.cat((forward_last, backward_last), -1)
        tmp = torch.cat((sentence_vector.unsqueeze(1).expand_as(outputs), outputs), 2)
        gate = F.sigmoid(self.selective_gate(tmp))
        outputs = outputs * gate
        #h_n = h_n.view(self.num_layers, self.num_directions, inputs.size(0), self.hidden_size)
        #outputs = outputs.view(outputs.size(0), outputs.size(1), self.num_directions, self.hidden_size)
        #if not concat_output:
        #    h_n = torch.mean(h_n, 1)
        #    outputs = torch.mean(outputs, 2)
        return outputs, h_n


class RNNEncoder(Encoder):

    def __init__(self, 
            hidden_size,
            num_layers,
            embed,
            bidirectional=True,
            rnn_dropout=0.2,
            cell='gru'):

        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed.embed_size
        self.vocab_size = embed.vocab_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        #self.rnn_dropout = rnn_dropout
        self.cell = cell
        # embedding
        self.embedding = embed


        # TODO
        self.rnn = nn.GRU(
                input_size = self.embed_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=True)

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
        h_n = h_n[:,_indices,:]
        # from SEASS
        #forward_last = h_n[0] # of last or 1st layer?
        #backward_last = h_n[1]

        #h_n = h_n.view(self.num_layers, self.num_directions, inputs.size(0), self.hidden_size)
        #outputs = outputs.view(outputs.size(0), outputs.size(1), self.num_directions, self.hidden_size)
        #if not concat_output:
        #    h_n = torch.mean(h_n, 1)
        #    outputs = torch.mean(outputs, 2)
        return outputs, h_n


class SelfFusionEncoder(Encoder):

    def __init__(self, 
            hidden_size,
            num_layers,
            embed,
            bidirectional=True,
            rnn_dropout=0.2,
            cell='gru'):

        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed.embed_size
        self.vocab_size = embed.vocab_size
        self.bidirectional = True #bidirectional
        self.num_directions = 2 if bidirectional else 1
        #self.dropout = dropout
        self.cell = cell
        # embedding
        self.embedding = embed


        # TODO
        self.rnn = nn.GRU(
                input_size = self.embed_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=self.bidirectional)

        self.attn_type = 'symmetric'
        if self.attn_type == 'symmetric':
            self.inter_dim = hidden_size
            self.D = nn.Linear(hidden_size, self.inter_dim, bias=False)
            self.U = nn.Linear(self.inter_dim, self.inter_dim, bias=False)
            #self.attn_out = nn.Linear(hidden_size*2, hidden_size)


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
        h_n = h_n[:, _indices, :]
        B, L, D = outputs.size()
        outputs = outputs.view(B, L, 2, D//2)
        forward = outputs[:,:,0]
        backward = outputs[:,:,1]

        mask = sequence_mask(lengths, L)# B x L
        if self.attn_type == 'symmetric':
            forward_ = F.relu(self.D(forward))
            backward_ = F.relu(self.D(backward))
            scores = torch.matmul(forward_, backward_.transpose(1,2)) # B, Forw, Back
            scores_b = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
            p_b = F.softmax(scores_b, -1)
            scores_f = scores.masked_fill(mask.unsqueeze(2) == 0, -1e9).transpose(1,2)
            p_f = F.softmax(scores_f, -1)
            weighted_b = torch.matmul(p_b, backward)
            weighted_f = torch.matmul(p_f, forward)
            # new_outputs = F.tanh(self.attn_out(torch.cat([outputs, outputs*weighted], -1)))
            forward = forward + weighted_b
            backward = backward + weighted_f
            outputs = torch.cat((forward, backward), -1)
        return outputs, h_n


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





