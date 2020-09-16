# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 01:09:46 2020

@author: Conglei Tan
"""
import torch 
import torch.nn as nn
# This file is to construct the Convolutional Auto-Encoder Model
class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder,self).__init__()
        self.encoder = torch.nn.Sequential(
           torch.nn.Conv2d(1,4,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2),
            torch.nn.Conv2d(4,8,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2))

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2,mode="nearest"),
            torch.nn.Conv2d(8,4,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2,mode="nearest"),
            torch.nn.Conv2d(4,1,kernel_size=3,stride=1,padding=1))
        
    def forward(self,input):
        encode = self.encoder(input)
        decode = self.decoder(encode)
        return decode

