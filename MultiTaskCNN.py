#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:29:44 2023

@author: nabil
"""

import torch
import torch.nn as nn
class MultiTaskCNN(nn.Module):
    def __init__(self,filters):
        super(MultiTaskCNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=filters,kernel_size=(5,5)) #output form = 60*60 pixels
        self.pool1=nn.AvgPool2d(kernel_size=(5,5)) # #output form = 12*12 pixels
        self.relu=nn.LeakyReLU()
        dims =  filters*12**2
        self.fc1=nn.Linear(in_features=dims,out_features=dims//2)
        self.dropout= nn.Dropout()
        self.fc21=nn.Linear(in_features=dims//2,out_features=10)
        self.fc22=nn.Linear(in_features=dims//2,out_features=10)
     

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = torch.flatten(x,start_dim=1)
        x=self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x1 = self.fc21(x)
        x2 = self.fc22(x)
        #returned value size : (batch_size,2,10)
        return torch.cat((x1.unsqueeze(1) , x2.unsqueeze(1)),dim=1)