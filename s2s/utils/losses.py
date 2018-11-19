#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn

class WeightedCE(nn.Module):
    def __init__(self, ce):
        super().__init__()
        self.loss = ce
        
    #def weight(self, k):
    #    return 1 + 2/k
    
    def forward(self, input, target):
        """
        """
        i = 0
        loss = 0
        for t, l in zip(target.split(1,1), input.split(1,1)):
            i += 1
            weight = 5 if i == 1 else 1.
            loss += self.loss(l.squeeze(1), t.squeeze(1)) * weight
        return loss



class DecayCE(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss
        self.ls = nn.LogSoftmax(-1)
        
    def forward(self, input, target):
        """
        input: B x L x C
        target: B x L
        """
        i = 0
        loss = 0
        for t, l in zip(target.split(1,1), input.split(1,1)):
            logp = self.ls(l)
            i += 1
            loss += self.loss(l.squeeze(1), t.squeeze(1))*self.weight(i)
        return loss


