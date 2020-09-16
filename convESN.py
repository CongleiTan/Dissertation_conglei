# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 20:37:09 2020

@author: lenovo
"""
 # This file is to train the ConvESN
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

    
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True

def train_and_test(root,seed):
    setup_seed(seed)
    epoches = 1000
    lr = 1e-4
    root_dir = 'Train_Data_'+str(root)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = get_data.climate_data(root_dir,transform=transform)
    Coder = CNNEncoder.CNNEncoder()
    Coder = torch.load('CNNEncoder.pk1')
    model = ConvESNModel.ConvESN(Coder)
    optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr =lr)
    criterion = torch.nn.MSELoss()
    model_params = ConvESNModel.model_params(0.25,10,1200,22)
    res_params = ConvESNModel.res_params(model_params,7200,0.1,3,0.5,len(dataset)-1)
    multi_reservoir_parameter = ConvESNModel.multi_reservoir_parameter(7200,899,0.3)
    Coder = Coder.cpu()
    print("Start")
    A = ConvESNModel.generate_reservoir(res_params.N, res_params.radius, res_params.degree)
    print("A")
    print(A.size())
    Win = ConvESNModel.Win(res_params.N,res_params.num_inputs,res_params.sigma)
    print("Win")
    print(Win.size())
    states = ConvESNModel.reservoir_layer(A,Coder.encoder,Win,ConvESNModel.Wres,multi_reservoir_parameter, dataset, res_params,1)
    print("states")
    states= Variable(states)
    image_set = torch.zeros(len(dataset)-1,1,40,60)
    states = states.cuda()
    model = model.cuda()
    criterion_L = nn.MSELoss()
    std = []
    mean_error = []
    for epoch in range(epoches):
        running_loss = 0.0
        output = model(states)
        error_epoch = []
        for i in range(len(dataset)-1):
            image_set[i,:,:,:] = dataset[i+1]
            image_set = Variable(image_set)
        loss = criterion(output, image_set)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss = running_loss+loss.item()
        if  epoch%50 == 0:
            print("Epoch {}/{}".format(epoch+1, epoches))
            print("Loss is:{:.7f}".format(running_loss))
        error_epoch = [criterion_L(output[i],image_set[i]) for i in range(len(output))]
        mean_error.append(torch.mean(torch.tensor(error_epoch)))
        std.append(torch.std(torch.tensor(error_epoch)))
        
    mean_error = np.asarray(mean_error)
    std = np.asarray(std)
    x_axis = np.asarray([i+1 for i in range(epoches)])
    mean_error_file_name = 'mean_error_for_convESN_seed_'+str(seed)+'_slice_'+str(root)+'.csv'
    std_file_name = 'std_for_convESN_seed_'+str(seed)+'_slice_'+str(root)+'.csv'
    np.savetxt(mean_error_file_name,mean_error.T,delimiter=',')
    np.savetxt(std_file_name,std.T,delimiter=',')
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.yscale("log")
    plt.plot(x_axis,mean_error,color='yellow', label='Error')
    plt.fill_between(x_axis,mean_error-std, mean_error+std,color='green' )
    plt.legend()
    jpg_name = 'covESN_seed_'+str(seed)+'_slice_'+str(root)+'.jpg'
    plt.savefig(jpg_name)
    model_name = 'ConvESN_seed_'+str(seed)+'_slice_'+str(root)+'.pk1'
    torch.save(model,model_name)
    
    print("Test---------------------------------------Test")
    test_dir = 'Test_Data'
    test_error = []
    test_dataset = get_data.climate_data(test_dir,transform=transform)
    test_multi_reservoir_parameter = ConvESNModel.multi_reservoir_parameter(7200,200,0.3)
    Coder = Coder.cpu()
    test_res_params = ConvESNModel.res_params(model_params,7200,0.1,3,0.5,len(test_dataset)-1)
    states = ConvESNModel.reservoir_layer(A,Coder.encoder,Win, ConvESNModel.Wres,test_multi_reservoir_parameter,test_dataset, test_res_params,1)
    test_image_set = torch.zeros(len(dataset)-1,1,40,60)
    for i in range(len(dataset)-1):
        test_image_set[i,:,:,:] = dataset[i+1]
        test_image_set = Variable(test_image_set)
    states = states.cuda()
    model = model.cuda()
    test_output = model(states)
    test_error = [criterion_L(test_output[i],test_image_set[i]) for i in range(len(test_dataset)-1)]
    mean_test = torch.mean(torch.tensor(test_error,dtype=torch.float32))
    std_test = torch.std(torch.tensor(test_error,dtype=torch.float32))
    mean_error_test_file_name = 'mean_error_for_test_convESN_seed_'+str(seed)+'_slice_'+str(root)
    std_test_file_name = 'std_for_test_convESN_seed_'+str(seed)+'_slice_'+str(root)
    mean_test_list = [mean_test]
    std_test_list = [std_test]
    np.savetxt(mean_error_test_file_name,mean_test_list)
    np.savetxt(std_test_file_name,std_test_list)

if __name__=="__main__":
    train_and_test(9,12)
