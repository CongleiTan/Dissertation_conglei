# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 16:42:07 2020

@author: lenovo
"""

import numpy as np
import torch
import torch.nn as nn
import get_data
import CNNEncoder
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
shift_k = 0
#shift_k=int(shift_k)
res_params = {
              'train_length': 1799,
              'predict_length': 10
              }
class ANN_res_params(object):
    def __init__(self,train_length,predict_length):
        self.train_length=train_length
        self.predict_length=predict_length

class ANNNet(nn.Module):
    def __init__(self,coder,input_dim,hidden_dim,output_dim):
        super(ANNNet,self).__init__()
        self.Coder = Coder
        for p in self.parameters():
            p.requires_grad=False
        self.ANN =torch.nn.Sequential(
           torch.nn.Linear(input_dim, hidden_dim),
           torch.nn.ReLU(True),
           torch.nn.Linear(hidden_dim, hidden_dim),
           torch.nn.ReLU(True),
           torch.nn.Linear(hidden_dim, hidden_dim),
           torch.nn.ReLU(True),
           torch.nn.Linear(hidden_dim, output_dim))
    
    def forward(self,data):
        out = self.ANN(data)
        image_pre_set = torch.zeros(1200,1,40,60)
        for i in range(1200):
            data = torch.reshape(out[i,:],(1,8,10,15))
            image_pre_set[i] = self.Coder.decoder(data)
        return image_pre_set

shift_k = 0
root_dir = 'ALL_DATASET_RESIZED'
transform = transforms.Compose([transforms.ToTensor()])
dataset = get_data.climate_data(root_dir,transform=transform)
Coder = CNNEncoder.CNNEncoder()
Coder = torch.load('CNNEncoder.pk1')    
ANN_res_params=ANN_res_params(1200,12)   
all_data = torch.tensor([])
Coder = Coder.cpu()
for i in range(len(dataset)):
        output1 = torch.flatten(Coder.encoder(dataset[i].unsqueeze(0))).unsqueeze(0)
        all_data = torch.cat([all_data,output1],0)
train = all_data[shift_k:shift_k+ANN_res_params.train_length,:]
input_dim = 1200
hidden_dim = 1200
output_dim = 1200
model = ANNNet(Coder,input_dim,hidden_dim,output_dim)
print('# Coder parameters:', sum(param.numel() for param in Coder.parameters()))
print('# ANN parameters:', sum(param.numel() for param in model.parameters()))
model = model.cuda()
train = train.cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=0.000005) 
criterion = torch.nn.MSELoss() 
epoches = 3000
error = []
image_set = torch.zeros(1200,1,40,60)
for i in range(1200):
    image_set[i] = dataset[i+1]
image_set = Variable(image_set)
for epoch in range(epoches):
    np.random.seed(epoch)
    running_loss = 0.0
    output = model(train)
    loss = criterion(output, image_set)
    
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    running_loss = running_loss+loss.item()
    if epoch%50 == 0:
        print("Epoch {}/{}".format(epoch+1, epoches))
        print("Loss is:{:.7f}".format(running_loss))
    error.append(running_loss)  
    if epoch==2999:
        to_pil_image = transforms.ToPILImage()
        for i in range(1200):
            save_file_second = "CONVANN_TRAINING/"+str(i+1)+".jpg"
            img = to_pil_image(output[i].squeeze(0))
            img.save(save_file_second)

plt.figure()
plt.xlabel("epoch")
plt.ylabel("error")
plt.yscale("log")
plt.plot(error, color='yellow', label='Error')
plt.legend()
plt.savefig("error_train_ANN.jpg")
torch.save(model,'ConvANN.pk1')


