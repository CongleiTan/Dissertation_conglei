# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 18:01:44 2020

@author: lenovo
"""

import torch
import get_data
from torchvision import transforms
import numpy as np 


root_dir = 'Train_data_12'
transform = transforms.Compose([transforms.ToTensor()])
dataset = get_data.climate_data(root_dir,transform=transform)
arima = torch.zeros(len(dataset),2400)
for i in range(len(dataset)):
    output = torch.flatten(dataset[i].unsqueeze(0))
    arima[i,:] = output
arima = arima.detach().numpy()
np.savetxt('Dataset_2.csv',arima,delimiter=',')


