#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 22:55:21 2023

@author: nabil
"""


import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(MultiTaskLoss, self).__init__()
        self.csloss= nn.CrossEntropyLoss()
        self.bceloss = nn.BCELoss()


    def forward(self, predictions, targets):

        loss1 = self.csloss(predictions[:,0,:] , targets[:,0,:])
        loss2 = self.csloss(predictions[:,1,:] , targets[:,1,:])
        
        loss =0.5*loss1 + 0.5*loss2

        return loss
