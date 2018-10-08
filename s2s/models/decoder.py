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
    
    def __init__(self, config):
        nn.Module.__init__()


    def step(self, inputs, lengths, hidden=None, context=None, context_lengths=None, tf_ratio=1.0):
        """ One-step computation
        
        Args:
            inputs:

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
        pass
    
    def beam_search(self, inputs, lengths, hidden=None, context=None, context_lengths=None):
        pass

    def sample(self, inputs, lengths, hidden=None, context=None, context_lengths=None, temperature=None):
        pass
        

class RNNDecoder(Decoder):

    def __init__(self):
        super().__init__()