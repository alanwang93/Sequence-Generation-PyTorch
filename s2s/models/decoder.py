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

from s2s.models.common import Feedforward, Embedding, sequence_mask


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
        pass

    def forward(self, inputs, lengths, hidden=None, context=None, context_lengths=None, tf_ratio=1.0):
        raise NotImplementedError()

    def init_weights(self):
        pass

    def greedy_decode(self, hidden, sos_idx, eos_idx, context=None, context_lengths=None, max_length=10):
        """
        inputs (batch, 1, dim)
        """
        batch_size = hidden.size(1)
        logits = []
        preds = []
        inp = sos_idx * torch.ones((batch_size, 1), dtype=torch.long).to(hidden.device)
        while True:
            logit, output, hidden = self.step(inp, hidden, context, context_lengths)
            logp = F.log_softmax(logit, dim=2)
            maxv, maxidx = torch.max(logp, dim=2)
            if ((maxidx == eos_idx).all() and len(preds) > 0)  or len(preds) >= max_length:
                break
            logits.append(logit)
            preds.append(maxidx.cpu().numpy())
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
            embed,
            rnn_dropout=0.,
            mlp_dropout=0.,
            attn_type=None):

        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = embed
        self.vocab_size = embed.vocab_size
        self.embed_size = embed.embed_size

        #self.dropout = dropout
        self.embedding = embed

        self.cell = 'gru'
        # TODO
        self.rnn = nn.GRU(
                input_size = self.embed_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first=True,
                dropout=rnn_dropout)

        # TODO: flexible hidden_size
        self.mlp1 = nn.Linear(
                self.hidden_size,
                self.vocab_size)

        self.init_weights()

    def init_weights(self):
        # RNN
        for i in range(self.num_layers):
            if self.cell == 'gru':
                K = 3
            elif self.cell == 'lstm':
                K = 4
            for k in range(K):
                s = k*self.hidden_size
                e = (k+1)*self.hidden_size
                w = getattr(self.rnn, 'weight_ih_l{0}'.format(i))[s:e]
                nn.init.orthogonal_(w)
                w = getattr(self.rnn, 'weight_hh_l{0}'.format(i))[s:e]
                nn.init.orthogonal_(w)


    def step(self, inputs, hidden=None, context=None, context_lengths=None):
        inputs = self.embedding(inputs)
        batch_size, _, _ = inputs.size()
        # output: (batch, 1, hidden_size)
        output, h_n = self.rnn(inputs, hidden)
        # (batch, 1, vocab_size)
        logit = self.mlp1(output.squeeze(1))
        return logit.unsqueeze(1), output, h_n


    def forward(self, inputs, lengths, hidden, context=None, context_lengths=None, tf_ratio=1.0):
        """
        
        """
        #batch_size, seq_len = inputs.size()
        embedded = self.embedding(inputs)
        outputs, h_n = self.rnn(embedded, hidden)
        logits = self.mlp1(outputs)
        return logits



class AttnRNNDecoder(Decoder):

    def __init__(self,
            hidden_size,
            num_layers,
            embed,
            rnn_dropout=0., 
            mlp_dropout=0.,
            attn_type='bilinear'):

        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed.embed_size
        self.vocab_size = embed.vocab_size
        self.attn_type = attn_type

        self.embedding = embed
        self.cell = 'gru'

        # TODO
        self.rnn = nn.GRU(
                input_size = self.embed_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                batch_first=True,
                dropout=rnn_dropout)

        # TODO: flexible hidden_size
        self.mlp1 = nn.Linear(
                self.hidden_size,
                self.vocab_size)

        self.dropout = nn.Dropout(mlp_dropout)

        if self.attn_type == 'symmetric':
            self.inter_dim = 200
            self.D = nn.Linear(hidden_size, self.inter_dim, bias=False)
            self.U = nn.Linear(self.inter_dim, self.inter_dim, bias=False)
            self.attn_out = nn.Linear(hidden_size*2, hidden_size)
        elif self.attn_type == 'add':
            self.inter_dim = hidden_size
            self.Wh = nn.Linear(hidden_size, self.inter_dim, bias=False) 
            self.Ws = nn.Linear(hidden_size, self.inter_dim) 
            self.v = nn.Linear(self.inter_dim, 1, bias=False)
            self.attn_out = nn.Linear(hidden_size*2, hidden_size)
        elif self.attn_type == 'bilinear':
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
            self.attn_out = nn.Linear(hidden_size*2, hidden_size)

        self.score_layer = '2linear'

        self.init_weights()

    def init_weights(self):
        # RNN
        for i in range(self.num_layers):
            if self.cell == 'gru':
                K = 3
            elif self.cell == 'lstm':
                K = 4
            for k in range(K):
                s = k*self.hidden_size
                e = (k+1)*self.hidden_size
                w = getattr(self.rnn, 'weight_ih_l{0}'.format(i))[s:e]
                nn.init.orthogonal_(w)
                w = getattr(self.rnn, 'weight_hh_l{0}'.format(i))[s:e]
                nn.init.orthogonal_(w)


    def step(self, inputs, hidden=None, context=None, context_lengths=None):
        inputs = self.embedding(inputs)
        batch_size, _, _ = inputs.size()
        # output: (batch, 1, hidden_size)
        output, h_n = self.rnn(inputs, hidden)
        # (batch, 1, vocab_size)

        mask = sequence_mask(context_lengths, context.size(1)).unsqueeze(1)

        # batch_size, len, dim
        if self.attn_type == 'symmetric':
            output_ = F.relu(self.D(output))
            context_ = F.relu(self.D(context))
            scores = torch.matmul(output_, context_.transpose(1,2)) # batch_size, tgt_len, src_len
            scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, -1)
            weighted = torch.matmul(p_attn, context)
            new_output = F.tanh(self.attn_out(torch.cat([output, weighted], -1)))
        elif self.attn_type == 'add':
            scores = torch.tanh(self.Wh(context) + self.Ws(output))
            scores = self.v(scores).transpose(1,2)
            scores = scores.masked_fill(mask == 0, -1e9)

            p_attn = F.softmax(scores, -1)
            weighted = torch.matmul(p_attn, context)
            new_output = F.tanh(self.attn_out(torch.cat([output, weighted], -1)))
        elif self.attn_type == 'bilinear':
            scores = self.W(context)
            scores = torch.matmul(output, scores.transpose(1,2))
            scores = scores.masked_fill(mask == 0, -1e9)
            # batch, tgt_len, src_len
            p_attn = F.softmax(scores, -1)
            weighted = torch.matmul(p_attn, context)
            new_output = F.tanh(self.attn_out(torch.cat([output, weighted], -1)))


        new_output = self.dropout(new_output)
        logit = self.mlp1(new_output.squeeze(1))
        logit = self.dropout(logit).unsqueeze(1)
        return logit, new_output, h_n

    def forward(self, inputs, lengths, hidden, context=None, context_lengths=None, tf_ratio=1.0):
        """
        
        """
        batch_size, seq_len = inputs.size()
        #embedded = self.embedding(inputs)
        #outputs = []
        logits = []
        for inp_t in inputs.split(1, dim=1):
            logit, output, hidden = self.step(inp_t, hidden, context, context_lengths)
            logits.append(logit)
            #outputs.append(output)
        #outputs = torch.stack(outputs, 1)
        logits = torch.stack(logits, 1).squeeze(2)
        return logits
"""
        outputs, h_n = self.rnn(embedded, hidden)

        mask = sequence_mask(context_lengths, context.size(1)).unsqueeze(1)

        # batch_size, seq_len, dim
        if self.attn_type == 'symmetric':
            outputs_ = F.relu(self.D(outputs))
            context_ = F.relu(self.D(context))
            scores = torch.matmul(outputs_, context_.transpose(1,2)) # batch_size, tgt_len, src_len
            scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, -1)
            weighted = torch.matmul(p_attn, context)
            new_outputs = F.tanh(self.attn_out(torch.cat([outputs, weighted], -1)))
        elif self.attn_type == 'add':
            scores = torch.tanh(self.Wh(context).unsqueeze(1).expand(-1, outputs.size(1),-1,-1)\
                    +  self.Ws(outputs).unsqueeze(2).expand(-1, -1, context.size(1), -1))
            scores = self.v(scores).squeeze(-1)
            scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, -1)
            weighted = torch.matmul(p_attn, context)
            new_outputs = F.tanh(self.attn_out(torch.cat([outputs, weighted], -1)))
        elif self.attn_type == 'bilinear':
            scores = self.W(context) # batch, src_len, dim
            scores = torch.matmul(outputs, scores.transpose(1,2))
            scores = scores.masked_fill(mask == 0, -1e9)
            # batch, tgt_len, src_len
            p_attn = F.softmax(scores, -1)
            weighted = torch.matmul(p_attn, context)
            
        if self.score_layer == '2linear':
            new_outputs = self.attn_out(torch.cat([outputs, weighted], -1))
            new_outputs = self.dropout(new_outputs)
            logits = self.mlp1(new_outputs)
            logits = self.dropout(logits)
"""


