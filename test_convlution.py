# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:17:10 2020

@author: lenovo
"""

import numpy as np
import torch
import CNNEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
import get_data
import matplotlib.pyplot as plt

Coder = CNNEncoder.CNNEncoder()
Coder = torch.load('CNNEncoder.pk1')
root_dir = 'ALL_DATASET_RESIZED'
transform = transforms.Compose([transforms.ToTensor()])
dataset = get_data.climate_data(root_dir,transform=transform)
batch_size=1
train_loader = DataLoader(dataset,shuffle=False,batch_size=batch_size)
i = 1
to_pil_image = transforms.ToPILImage()
error =[]
Coder = Coder.cpu()
for data in train_loader:
    output1 = Coder(data)
    save_file_second = "CONVLUTION_PREDICTION/"+str(i)+".jpg"
    img = to_pil_image(output1.squeeze(0))
    img.save(save_file_second)
    i =i+1
    loss = torch.nn.MSELoss()
    output = loss(data,output1)
    print('output is', output)
    error.append(output.detach().numpy())
error = np.asarray(error)
file_name='error_for_convlution_prediction.csv'
np.savetxt(file_name,error.T,delimiter=',')
plt.figure()
plt.plot(error, color='yellow', label='Error')
plt.legend()
plt.savefig("error_convlution_preddiction.jpg")