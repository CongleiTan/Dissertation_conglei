# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 20:37:09 2020

@author: lenovo
"""

import numpy as np
from torch.nn import functional as F
import torch.sparse
import scipy as sp
import scipy.sparse as sparse
import torch
import CNNEncoder
import get_data
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import ConvESNModel

    


    
epoches = 3000         
lr = 1e-5
root_dir = 'ALL_DATASET_RESIZED'
transform = transforms.Compose([transforms.ToTensor()])
dataset = get_data.climate_data(root_dir,transform=transform)
Coder = CNNEncoder.CNNEncoder()
Coder = torch.load('CNNEncoder.pk1')
model = ConvESNModel.ConvESN(Coder)
optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr =lr)
criterion = torch.nn.MSELoss()
model_params = ConvESNModel.model_params(0.25,10,1200,22)
res_params = ConvESNModel.res_params(model_params,2400,0.1,3,0.5,1200)
print("Start")
A = ConvESNModel.generate_reservoir(res_params.N, res_params.radius, res_params.degree)
print("A")
Win = ConvESNModel.Win(res_params.N,res_params.num_inputs,res_params.sigma)
print("Win")
states = ConvESNModel.reservoir_layer(A,Coder.encoder,Win, dataset, res_params)
print("states")
states= Variable(states)
error = []
image_set = torch.zeros(1200,1,40,60)
for epoch in range(epoches):
    running_loss = 0.0
    output = model(states)
    for i in range(1200):
        image_set[i,:,:,:] = dataset[i+1]
    image_set = Variable(image_set)
    loss = criterion(output, image_set)
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss = running_loss+loss.item()
    if (epoch+1)%50 ==0:
        print("Epoch {}/{}".format(epoch+1, epoches))
        print("Loss is:{:.7f}".format(running_loss))
    error.append(running_loss)  
    if epoch==2999:
        to_pil_image = transforms.ToPILImage()
        for i in range(1200):
            save_file_second = "CONVESN_TRAINING_PICTURE/"+str(i+1)+".jpg"
            img = to_pil_image(output[i].squeeze(0))
            img.save(save_file_second)
error = np.asarray(error)
file_name='error_for_convESN_training.csv'
np.savetxt(file_name,error.T,delimiter=',')
A = A.detach().numpy()
np.savetxt('A.csv',A,delimiter=',')
Win = Win.detach().numpy()
np.savetxt('Win.csv',Win,delimiter=',')
states = states.detach().numpy()
np.savetxt('states.csv',states,delimiter=',')
plt.figure()
plt.xlabel("epoch")
plt.ylabel("error")
plt.yscale("log")
plt.plot(error, color='yellow', label='Error')
plt.legend()
plt.savefig("error_conESN_Training.jpg")
torch.save(model,'CONVESN.pk1')