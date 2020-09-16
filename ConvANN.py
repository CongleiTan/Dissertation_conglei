# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 16:42:07 2020

@author: lenovo
"""
#This file build the ConvANN model and train it.
import os
import numpy as np
import torch
import torch.nn as nn
import get_data
import CNNEncoder
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True
     

class ANNNet(nn.Module):
    def __init__(self,Coder,input_dim,hidden_dim,output_dim):
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
        image_pre_set = torch.zeros(len(out),1,40,60)
        for i in range(len(out)):
            data = torch.reshape(out[i,:],(1,8,10,15))
            image_pre_set[i] = self.Coder.decoder(data)
        return image_pre_set
def train_and_test(root,seed):
    
    setup_seed(seed)
    shift_k = 0
    root_dir = 'Train_Data_'+str(root)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = get_data.climate_data(root_dir,transform=transform)
    Coder = CNNEncoder.CNNEncoder()
    Coder = torch.load('CNNEncoder.pk1')       
    all_data = torch.tensor([])
    Coder = Coder.cpu()
    train_length = len(dataset)-1
    for i in range(len(dataset)-1):
        output = torch.flatten(Coder.encoder(dataset[i].unsqueeze(0))).unsqueeze(0)
        all_data = torch.cat([all_data,output],0)
    train = all_data
    input_dim = 1200
    hidden_dim = 1200
    output_dim = 1200
    model = ANNNet(Coder,input_dim,hidden_dim,output_dim)
    print('# Coder parameters:', sum(param.numel() for param in Coder.parameters()))
    print('# ANN parameters:', sum(param.numel() for param in model.parameters()))
    model = model.cuda()
    train = train.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=0.0001) 
    criterion = torch.nn.MSELoss() 
    epoches = 1000
    std = []
    mean_error = []
    criterion_L = nn.MSELoss()
    image_set = torch.zeros(len(dataset)-1,1,40,60)
    for i in range(len(dataset)-1):
        image_set[i] = dataset[i+1]
    image_set = Variable(image_set)
    for epoch in range(epoches):
        running_loss = 0.0
        output = model(train)
        loss = criterion(output, image_set)
    
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss = running_loss+loss.item()
        error_epoch = [criterion_L(output[i],image_set[i]) for i in range(len(output))]
        if epoch%50 == 0:
           print("Epoch {}/{}".format(epoch+1, epoches))
           print("Loss is:{:.7f}".format(running_loss)) 
        mean_error.append(torch.mean(torch.tensor(error_epoch)))
        std.append(torch.std(torch.tensor(error_epoch)))

    mean_error = np.asarray(mean_error)
    std = np.asarray(std)
    x_axis = np.asarray([i+1 for i in range(epoches)])
    mean_error_file_name = 'mean_error_for_convANN_seed_'+str(seed)+'_slice_'+str(root)+'.csv'
    std_file_name = 'std_for_convANN_seed_'+str(seed)+'_slice_'+str(root)+'.csv'
    np.savetxt(mean_error_file_name,mean_error.T,delimiter=',')
    np.savetxt(std_file_name,std.T,delimiter=',')
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.yscale("log")
    plt.plot(x_axis,mean_error,color='yellow', label='Error')
    plt.fill_between(x_axis,mean_error-std, mean_error+std,color='green' )
    plt.legend()
    jpg_name = 'covANN_seed_'+str(seed)+'_slice_'+str(root)+'.jpg'
    plt.savefig(jpg_name)
    
    
    print("Test---------------------------------------Test")
    test_dir = 'Test_Data'
    test_error = []
    test_dataset = get_data.climate_data(test_dir,transform=transform)
    test_data = torch.tensor([])
    Coder = Coder.cpu()
    for i in range(len(test_dataset)-1):
        output1 = torch.flatten(Coder.encoder(test_dataset[i].unsqueeze(0))).unsqueeze(0)
        test_data = torch.cat([test_data,output1],0)
    test_image_set = torch.zeros(len(test_dataset)-1,1,40,60)
    for i in range(len(test_dataset)-1):
        test_image_set[i] = test_dataset[i+1]
    test_image_set = Variable(test_image_set)
    test_data = test_data.cuda()
    model = model.cuda()
    test_output = model(test_data)
    to_pil_image = transforms.ToPILImage()
    for i in range(len(test_output)):
        img = to_pil_image(test_output[i].squeeze(0))
        save_file_second = "result/"+str(i)+".jpg"
        img.save(save_file_second)
    test_error = [criterion_L(test_output[i],test_image_set[i]) for i in range(len(test_dataset)-1)]
    mean_test = torch.mean(torch.tensor(test_error,dtype=torch.float32))
    std_test = torch.std(torch.tensor(test_error,dtype=torch.float32))
    mean_error_test_file_name = 'mean_error_for_test_convANN_seed_'+str(seed)+'_slice_'+str(root)
    std_test_file_name = 'std_for_test_convANN_seed_'+str(seed)+'_slice_'+str(root)
    mean_test_list = [mean_test]
    std_test_list = [std_test]
    np.savetxt(mean_error_test_file_name,mean_test_list)
    np.savetxt(std_test_file_name,std_test_list)
    
    model_name = 'ConvANN_seed_'+str(seed)+'_slice_'+str(root)+'.pk1'
    torch.save(model,model_name)
if __name__=="__main__":
    train_and_test(12,600)