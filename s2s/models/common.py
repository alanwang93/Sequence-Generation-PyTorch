#!/usr/bin/env python
# coding=utf-8

# Reference: https://github.com/salesforce/decaNLP

import math
import os
import sys

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


INF = 1e10
EPS = 1e-10


class Feedforward(nn.Module):

    def __init__(self, d_in, d_out, activation=None, bias=True):
        super().__init__()
        if activation is not None:
            self.activation = getattr(F, activation)
        else:
            self.activation = lambda x: x
        self.linear = Linear(d_in, d_out, bias=bias)

    def forward(self, x):
        return self.activation(self.linear(x))


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    B X L
    """
    device = lengths.device
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .to(device)
            .lt(lengths.unsqueeze(1)))


class MaxOut(nn.Module):
    def __init__(self, pool_size):
        super(MaxOut, self).__init__()
        self.pool_size = pool_size
        
    def forward(self, input):
        input_size = list(input.size())
        assert input_size[-1] % self.pool_size == 0
        output_size = [d for d in input_size]
        output_size[-1] = output_size[-1] // self.pool_size
        output_size.append(self.pool_size)
        last_dim = len(output_size) - 1
        input = input.view(*output_size)
        input, idx = input.max(last_dim, keepdim=True)
        output = input.squeeze(last_dim)

        return output


class ScoreLayer(nn.Module):
    def __init__(self, 
            vocab_size,
            enc_size,
            dec_size,
            embed_size,
            score_type='readout',
            dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.enc_size = enc_size # or context size
        self.dec_size = dec_size
        self.embed_size = embed_size
        self.score_type = score_type

        if self.score_type == 'readout':
            self.readout = nn.Linear(enc_size+dec_size+embed_size, dec_size)
            self.maxout = MaxOut(pool_size=2)
            self.W = nn.Linear(dec_size//2, vocab_size)
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, output, context, embed):
        if self.score_type == 'readout':
            readout = self.readout(torch.cat((output, context, embed), -1))
            maxout = self.maxout(readout)
            maxout = self.dropout(maxout)
            logit = self.W(maxout)

        return logit

class AttentiveGate(nn.Module):
    
    def __init__(self,
            value_dim,
            query_dim,
            attn_type,
            attn_dim):
        super().__init__()
        
        self.attn_type = attn_type
        if self.attn_type == 'concat':
            self.Wh = nn.Linear(value_dim, attn_dim, bias=True)
            self.Ws = nn.Linear(query_dim, attn_dim, bias=True)
            self.a2ctx = nn.Linear(attn_dim, value_dim, bias=True)
        elif self.attn_type == 'linear':
            self.W = nn.Linear(value_dim+query_dim, value_dim, bias=True)
            #self.Ws = nn.Linear(query_dim, value_dim, bias=False)

    def forward(self, output, context, mask):
        if self.attn_type == 'concat':
            context_ = self.Wh(context)
            output_ = self.Ws(output).unsqueeze(1).expand_as(context_)
            tmp = torch.tanh(context_ + output_) # batch x src_len x d
            gate = torch.sigmoid(self.a2ctx(tmp)) # B x L x VD
            #scores = scores.masked_fill(mask == 0, -1e9)
            gate = gate * mask.transpose(1,2).float()
            weighted = torch.mean(gate * context, 1)
        elif self.attn_type == 'linear':
            tmp = torch.cat((context, output.unsqueeze(1).expand(-1, context.size(1), -1)), -1)
            gate = torch.sigmoid(self.W(tmp))
            gate = gate * mask.transpose(1,2).float()
            weighted = torch.mean(gate*context, 1)

        return weighted, gate



class Attention(nn.Module):
    def __init__(self, 
            value_dim, 
            query_dim, 
            attn_type, 
            attn_dim):
        super().__init__()
        self.value_dim = value_dim
        self.query_dim = query_dim
        self.attn_type = attn_type
        self.attn_dim = attn_dim

        if self.attn_type == 'symmetric':
            #assert query_dim == value_dim
            self.q2v = nn.Linear(query_dim, value_dim)
            self.D = nn.Linear(value_dim, attn_dim, bias=False)
            self.U = nn.Linear(attn_dim, attn_dim, bias=False)
        elif self.attn_type == 'concat':
            self.Wh = nn.Linear(value_dim, attn_dim, bias=True) 
            self.Ws = nn.Linear(query_dim, attn_dim, bias=False) 
            self.v = nn.Linear(attn_dim, 1, bias=False)
        elif self.attn_type == 'bilinear':
            self.W = nn.Linear(query_dim, value_dim, bias=False)


    def forward(self, output, context, mask):
        """
        output: batch x dim
        context: batch x src_len x dim
        lengths: batch x src_len
        """

        if self.attn_type == 'symmetric':
            output = self.q2v(output)
            output_ = F.relu(self.D(output)).unsqueeze(1)
            context_ = F.relu(self.D(context))
            scores = torch.bmm(output_, context_.transpose(1,2)) # batch_size, 1, src_len
            scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, -1) # batch x 1 x src_len
            weighted = torch.bmm(p_attn, context).squeeze(1)
        elif self.attn_type == 'concat':
            context_ = self.Wh(context)
            output_ = self.Ws(output).unsqueeze(1).expand_as(context_)
            tmp = torch.tanh(context_ + output_) # batch x src_len x d
            scores = self.v(tmp).transpose(1,2)  # batch x 1 x src_len
            scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, -1)
            weighted = torch.bmm(p_attn, context).squeeze(1)
        elif self.attn_type == 'bilinear':
            tmp = self.W(context)
            scores = torch.matmul(output.unsqueeze(1), tmp.transpose(1,2))
            scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, -1)
            weighted = torch.bmm(p_attn, context).suqeeze(1)
        return weighted, p_attn



class StackedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        """
        input: batch x input_size
        hidden: batch x hidden_size
        """
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            if h_0 is None:
                h_1_i = layer(input, h_0)
            else:
                # last hidden: num_layer*num_directions x batch x dim
                h_1_i = layer(input, h_0[i]) # assuming uni-directional
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1


class DecInit(nn.Module):
    """
    Convert encoder hidden states into decoder initial 
    hidden states
    """
    def __init__(self, enc_hidden_size, dec_hidden_size, bidirectional=False):
        super(DecInit, self).__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.initer = nn.Linear(self.enc_hidden_size*self.num_directions, self.dec_hidden_size)
        #self.initer = nn.Linear(self.enc_hidden_size, self.dec_hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, enc_hidden):
        """
        enc_hidden: num_layers*num_directions x  batch_size x enc_hidden_size

        Return:
            num_layers x batch_size x dec_hidden_size
        """
        if self.bidirectional:
            _, batch, dim = enc_hidden.size()
            enc_hidden = enc_hidden.view(-1, 2, batch, dim).contiguous()
            enc_hidden = enc_hidden.transpose(1,2).contiguous()
            enc_hidden = enc_hidden.view(-1, batch, 2*dim).contiguous()

        return self.tanh(self.initer(enc_hidden))


class Embedding(nn.Module):

    def __init__(self, embed_size, vocab, pretrained=None, \
            pretrained_size=None, projection=False, embed_dropout=0.2):
        """
        1. use pretrained embedding
            1.1 with projection => embed_size 
            1.2 without projection
        2. no pretrained embedding
        """
        super().__init__()
        self.embed_size =pretrained_size or embed_size
        self.vocab = vocab
        self.projection = projection
        self.vocab_size = len(vocab)
        self.pretrained = pretrained
        self.pretrained_size = pretrained_size

        if pretrained is not None:
            assert pretrained_size is not None
            if not projection:
                assert pretrained_size == embed_size

        self.out_size = pretrained_size or embed_size
        self.embeddings = nn.Embedding(self.vocab_size, self.embed_size, \
                padding_idx=vocab.pad_idx)
        self.load_embeddings()

        if projection:
            self.embeddings.weight.requires_grad = False
            self.projection_layer = Feedforward(d_in=pretrained_size, d_out=embed_size)
            self.embed_size = embed_size
        self.dropout = nn.Dropout(embed_dropout)


    def forward(self, x, lengths=None):
        embeddings = self.dropout(self.embeddings(x))
        if self.projection: 
            return self.projection_layer(embeddings)
        else:
            return embeddings
    
    def load_embeddings(self):
        weights = 0.02 * np.random.randn(self.vocab_size, self.pretrained_size or self.embed_size)
        weights[self.vocab.pad_idx] = 0.
        # Glove
        if self.pretrained is not None:
            num_found = 0
            with open(self.pretrained, 'r') as f:
                print('Loading pretrained embeddings from {0}'.format(self.pretrained))
                for line in f:
                    s = line.strip().split(' ')
                    word = s[0]
                    if word.lower() in self.vocab.stoi:
                        num_found += 1
                        vector = np.asarray([float(s[i]) for i in range(1, len(s))])
                        idx = self.vocab.stoi[word]
                        if idx != self.vocab.pad_idx:
                            weights[idx] = vector
            print('Found: {0}, Not found: {1}'.format(num_found, self.vocab_size-num_found))
        self.embeddings.weight.data = torch.from_numpy(weights.astype(np.float32))





class LSTMDecoder(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(LSTMDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers 
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            input = self.dropout(input)
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


def positional_encodings_like(x, t=None):
    if t is None:
        positions = torch.arange(0, x.size(1))
        if x.is_cuda:
            positions = positions.cuda(x.get_device())
    else:
        positions = t
    encodings = torch.zeros(*x.size()[1:])
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())
    for channel in range(x.size(-1)):
        if channel % 2 == 0:
            encodings[:, channel] = torch.sin(
                positions / 10000 ** (channel / x.size(2)))
        else:
            encodings[:, channel] = torch.cos(
                positions / 10000 ** ((channel - 1) / x.size(2)))
    return Variable(encodings)


# torch.matmul can't do (4, 3, 2) @ (4, 2) -> (4, 3)
def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-2)).squeeze(-2)


def pad_to_match(x, y):
    x_len, y_len = x.size(1), y.size(1)
    if x_len == y_len:
        return x, y
    extra = x.data.new(x.size(0), abs(y_len - x_len)).fill_(1)
    if x_len < y_len:
        return torch.cat((x, extra), 1), y
    return x, torch.cat((y, extra), 1)


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualBlock(nn.Module):

    def __init__(self, layer, dim, dropout):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNorm(dim)

    def forward(self, *x, padding=None):
        return self.layernorm(x[0] + self.dropout(self.layer(*x, padding=padding)))


class Attention2(nn.Module):

    def __init__(self, d_key, dropout_ratio, causal):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(dropout_ratio)
        self.causal = causal

    def forward(self, query, key, value, padding=None):
        dot_products = matmul(query, key.transpose(1, 2))
        if query.dim() == 3 and self.causal:
            tri = key.data.new(key.size(1), key.size(1)).fill_(1).triu(1) * INF
            dot_products.data.sub_(tri.unsqueeze(0))
        if not padding is None:
            dot_products.data.masked_fill_(padding.unsqueeze(1).expand_as(dot_products), -INF)
        return matmul(self.dropout(F.softmax(dot_products / self.scale, dim=-1)), value)


class MultiHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, dropout_ratio, causal=False):
        super().__init__()
        self.attention = Attention2(d_key, dropout_ratio, causal=causal)
        self.wq = Linear(d_key, d_key, bias=False)
        self.wk = Linear(d_key, d_key, bias=False)
        self.wv = Linear(d_value, d_value, bias=False)
        self.n_heads = n_heads

    def forward(self, query, key, value, padding=None):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (
            x.chunk(self.n_heads, -1) for x in (query, key, value))
        return torch.cat([self.attention(q, k, v, padding=padding)
                          for q, k, v in zip(query, key, value)], -1)


class LinearReLU(nn.Module):

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.feedforward = Feedforward(d_model, d_hidden, activation='relu')
        self.linear = Linear(d_hidden, d_model)

    def forward(self, x, padding=None):
        return self.linear(self.feedforward(x))


class TransformerEncoderLayer(nn.Module):

    def __init__(self, dimension, n_heads, hidden, dropout):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead(
                dimension, dimension, n_heads, dropout),
            dimension, dropout)
        self.feedforward = ResidualBlock(
            LinearReLU(dimension, hidden),
            dimension, dropout)

    def forward(self, x, padding=None):
        return self.feedforward(self.selfattn(x, x, x, padding=padding))


class TransformerEncoder(nn.Module):

    def __init__(self, dimension, n_heads, hidden, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(dimension, n_heads, hidden, dropout) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding=None):
        x = self.dropout(x)
        encoding = [x]
        for layer in self.layers:
            x = layer(x, padding=padding)
            encoding.append(x)
        return encoding


class TransformerDecoderLayer(nn.Module):

    def __init__(self, dimension, n_heads, hidden, dropout, causal=True):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead(dimension, dimension, n_heads,
                      dropout, causal),
            dimension, dropout)
        self.attention = ResidualBlock(
            MultiHead(dimension, dimension, n_heads,
                      dropout),
            dimension, dropout)
        self.feedforward = ResidualBlock(
            LinearReLU(dimension, hidden),
            dimension, dropout)

    def forward(self, x, encoding, context_padding=None, answer_padding=None):
        x = self.selfattn(x, x, x, padding=answer_padding)
        return self.feedforward(self.attention(x, encoding, encoding, padding=context_padding))


class TransformerDecoder(nn.Module):

    def __init__(self, dimension, n_heads, hidden, num_layers, dropout, causal=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(dimension, n_heads, hidden, dropout, causal=causal) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.d_model = dimension

    def forward(self, x, encoding, context_padding=None, positional_encodings=True, answer_padding=None):
        if positional_encodings:
            x = x + positional_encodings_like(x)
        x = self.dropout(x)
        for layer, enc in zip(self.layers, encoding[1:]):
            x = layer(x, enc, context_padding=context_padding, answer_padding=answer_padding)
        return x


def mask(targets, out, squash=True, pad_idx=1):
    mask = (targets != pad_idx)
    out_mask = mask.unsqueeze(-1).expand_as(out).contiguous()
    out_after = out[out_mask].contiguous().view(-1, out.size(-1))
    targets_after = targets[mask]
    return out_after, targets_after


class Highway(torch.nn.Module):
    def __init__(self, d_in, activation='relu', n_layers=1):
        super(Highway, self).__init__()
        self.d_in = d_in
        self._layers = torch.nn.ModuleList([Linear(d_in, 2 * d_in) for _ in range(n_layers)])
        for layer in self._layers:
            layer.bias[d_in:].data.fill_(1)
        self.activation = getattr(F, activation)

    def forward(self, inputs):
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part = projected_input[:, :self.d_in] if projected_input.dim() == 2 else projected_input[:, :, :self.d_in]
            nonlinear_part = self.activation(nonlinear_part)
            gate = projected_input[:, self.d_in:(2 * self.d_in)] if projected_input.dim() == 2 else projected_input[:, :, self.d_in:(2 * self.d_in)]
            gate = F.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class LinearFeedforward(nn.Module):

    def __init__(self, d_in, d_hid, d_out, activation='relu'):
        super().__init__()
        self.feedforward = Feedforward(d_in, d_hid, activation=activation)
        self.linear = Linear(d_hid, d_out)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.dropout(self.linear(self.feedforward(x)))


class PackedLSTM(nn.Module):

    def __init__(self, d_in, d_out, bidirectional=False, num_layers=1, 
        dropout=0.0, batch_first=True):
        """A wrapper class that packs input sequences and unpacks output sequences"""
        super().__init__()
        if bidirectional:
            d_out = d_out // 2
        self.rnn = nn.LSTM(d_in, d_out,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        self.batch_first = batch_first

    def forward(self, inputs, lengths, hidden=None):
        lens, indices = torch.sort(inputs.data.new(lengths).long(), 0, True)
        inputs = inputs[indices] if self.batch_first else inputs[:, indices] 
        outputs, (h, c) = self.rnn(pack(inputs, lens.tolist(), 
            batch_first=self.batch_first), hidden)
        outputs = unpack(outputs, batch_first=self.batch_first)[0]
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices] if self.batch_first else outputs[:, _indices]
        h, c = h[:, _indices, :], c[:, _indices, :]
        return outputs, (h, c)


class Linear(nn.Linear):

    def forward(self, x):
        size = x.size()
        return super().forward(
            x.contiguous().view(-1, size[-1])).view(*size[:-1], -1)


class Feedforward(nn.Module):

    def __init__(self, d_in, d_out, activation=None, bias=True, dropout=0.0):
        super().__init__()
        if activation is not None:
            self.activation = getattr(F, activation)
        else:
            self.activation = lambda x: x
        self.linear = Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.activation(self.linear(self.dropout(x)))





class SemanticFusionUnit(nn.Module):
 
    def __init__(self, d, l):
        super().__init__()
        self.r_hat = Feedforward(d*l, d, 'tanh')
        self.g = Feedforward(d*l, d, 'sigmoid')
        self.dropout = nn.Dropout(0.2)
 
    def forward(self, x):
        c = self.dropout(torch.cat(x, -1))
        r_hat = self.r_hat(c)
        g = self.g(c)
        o = g * r_hat + (1 - g) * x[0]
        return o


class GlobalAttention(nn.Module):
    def __init__(self, dim, dot=False):
        super().__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(2 * dim, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None
        self.dot = dot

    def applyMasks(self, context_mask):
        self.context_mask = context_mask

    def forward(self, input, context):
        """
        input: batch x dim
        context: batch x len x dim
        """
        if not self.dot:
            targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1
        else:
            targetT = input.unsqueeze(2)

        context_scores = torch.bmm(context, targetT).squeeze(2)
        context_scores.data.masked_fill_(self.context_mask, -float('inf'))
        context_attention = F.softmax(context_scores, dim=-1) + EPS
        context_alignment = torch.bmm(context_attention.unsqueeze(1), context).squeeze(1)

        combined_representation = torch.cat([input, context_alignment], 1)
        output = self.tanh(self.linear_out(combined_representation))

        return output, context_attention, context_alignment
