# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:42:58 2020

@author: Conglei Tan
"""
# This file is to train Convolutional Auto-Encoder
import get_data
import torch.utils.data
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
import CNNEncoder 
import torch.nn as nn
from torch.nn import functional as F
import scipy as sp
import scipy.sparse as sparse
import numpy as np
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(20)
transform = transforms.Compose([transforms.ToTensor()])
root_dir = 'ALL_DATASET_RESIZED'
dataset = get_data.climate_data(root_dir,transform=transform)
batch_size = 20
train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size,drop_last=True)
model = CNNEncoder.CNNEncoder()
gpu_available = torch.cuda.is_available()
lr = 1e-5
optimizer=torch.optim.Adam(model.parameters(),lr =lr,weight_decay=0.0001)
to_pil_image = transforms.ToPILImage()
criterion = torch.nn.MSELoss()
model = model.cuda()
epoches = 3000
error = []
std = []
mean_error = []
criterion_L = nn.MSELoss()
for epoch in range(epoches):
    running_loss = 0.0
    error_epoch = []
    for data in train_loader:
        data = data.cuda()
        train_pre = model(data)
        loss = criterion(train_pre, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        for i in range(len(train_pre)):
            error_epoch.append(criterion_L(train_pre[i],data[i]))
    if (epoch)%50 == 0:
        print("Epoch {}/{}".format(epoch+1, epoches))
        print("Loss is:{:.7f}".format(running_loss/len(train_loader)))
    error.append(running_loss/len(train_loader))
    mean_error.append(torch.mean(torch.tensor(error_epoch)))
    std.append(torch.std(torch.tensor(error_epoch)))
error = np.asarray(error)
std = np.asarray(std)
x_axis = np.asarray([i+1 for i in range(epoches)])
file_name='error_for_convlution_training.csv'
np.savetxt(file_name,error.T,delimiter=',')
np.savetxt('error_for_convlution_training.csv',std.T,delimiter=',')
torch.save(model,'CNNEncoder.pk1')
plt.figure()
plt.xlabel("epoch")
plt.ylabel("error")
plt.yscale("log")
plt.plot(x_axis,error,color='yellow', label='Error')
plt.legend()
plt.savefig("error_for_convlution_training.jpg")
 
plt.figure()
plt.xlabel("epoch")
plt.ylabel("error")
plt.yscale("log")
plt.plot(x_axis,mean_error,color='yellow', label='Error')
plt.fill_between(x_axis,mean_error-std, mean_error+std,color='green', )
plt.legend()
plt.savefig("std_for_convlution_training.jpg")

