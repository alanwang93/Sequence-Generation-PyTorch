#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import numpy as np

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
            #weight = 5 if i == 1 else 1.
            weight = 1+1/i
            loss += self.loss(l.squeeze(1), t.squeeze(1)) * weight
        return loss



class DecayCE(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss
        self.step = 0

    def get_weights(self):
        weights = np.zeros(50)
        for i in range(min(50, 1+(self.step)//500)):
            weights[i] = 1.
        #for i in range(50):
        #    weights[i] += (1/(2+i)) #** float(np.log(self.step/1000+1))
        #if self.step % 50 == 0:
            #print('weights', weights)
        #if self.step < 1000:
        #    for i in range(50):
        #        weights[i] += 2./(i+1)
        #elif self.step < 2000:
        #    weights[:6] += 1.
        #elif self.step < 3000:
        #    weights[:12] += 1.
        #elif self.step < 5000:
        #    weights[:20] += 1.
        #else:
        #    weights[:] = 1.
        return weights
        
    def forward(self, input, target):
        """
        input: B x L x C
        target: B x L
        """
        loss = 0
        i = 0
        for t, l in zip(target.split(1,1), input.split(1,1)):
            i += 1
            weight = self.get_weights()[i]
            loss += self.loss(l.squeeze(1), t.squeeze(1)) * weight
        return loss


