# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 18:49:47 2020

@author: lenovo
"""
# This file is to train the ConvLSTM
from torch.utils.data import DataLoader
import numpy as np 
import get_data
from torchvision import transforms
import torch
import ConvLSTMModel
import matplotlib.pyplot as plt
def Make_ConvLSTM_Dataset(root,lookback):
    root_dir = 'Train_Data_'+str(root)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = get_data.climate_data(root_dir,transform=transform)
    train_input = torch.zeros(1,len(dataset)-lookback,1,40,60)
    train_target = torch.zeros(1,len(dataset)-lookback,1,40,60)
    for i in range(len(dataset)-lookback):
            train_input[0,i,:,:,:] = dataset[i]
            train_target[0,i,:,:,:] = dataset[i+lookback]

    return train_input,train_target

def Make_Test_Dataset(lookback):
    root_dir = 'Test_Data'
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = get_data.climate_data(root_dir,transform=transform)
    train_input = torch.zeros(1,len(dataset)-lookback,1,40,60)
    train_target = torch.zeros(1,len(dataset)-lookback,1,40,60)
    for i in range(len(dataset)-lookback):
            train_input[0,i,:,:,:] = dataset[i]
            train_target[0,i,:,:,:] = dataset[i+lookback]

    return train_input,train_target
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True
     
def train_and_test(root,seed):
    lookback = 3
    train_x,train_y = Make_ConvLSTM_Dataset(root,lookback)
    setup_seed(seed)
    print("DONE")
    epoches = 1000
    lr = 1e-3
    channels = 1
    model = ConvLSTMModel.ConvLSTM(input_dim=channels,hidden_dim=[4,16,1],kernel_size=(5, 5),num_layers=3,batch_first=True,bias=True,return_all_layers=False)
    model = model.cuda()
    print('# ConvLSTM parameters:', sum(param.numel() for param in model.parameters()))
    optimizer=torch.optim.Adam(model.parameters(),lr =lr)
    criterion = torch.nn.MSELoss()
    criterion_L = torch.nn.MSELoss()
    mean_error = []
    std = []
    error_epoch = []
    train_x,train_y =  Make_ConvLSTM_Dataset(root,lookback)
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    for epoch in range(epoches):
        running_loss = 0
        layer_output_list,last_state_list= model(train_x,None)
        loss = criterion(layer_output_list[0][0], train_y[0])
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss = running_loss+loss.item()
        #data = data.cuda()
        #out = out.cuda()
        #layer_output_list,last_state_list= model(data,None)
        #loss = criterion(layer_output_list[0], out)
         
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        #running_loss = running_loss+loss.item()
        error_epoch = [criterion_L(layer_output_list[0][0][i],train_y[0][i]) for i in range(train_y[0].shape[0])]
        if epoch%50 == 0:
            print("Epoch {}/{}".format(epoch+1, epoches))
            print("Loss is:{:.7f}".format(running_loss))
        mean_error.append(torch.mean(torch.tensor(error_epoch)))
        std.append(torch.std(torch.tensor(error_epoch)))

    mean_error = np.asarray(mean_error)
    std = np.asarray(std)
    x_axis = np.asarray([i+1 for i in range(epoches)])
    mean_error_file_name = 'mean_error_for_convLSTM_seed_'+str(seed)+'_slice_'+str(root)+'.csv'
    std_file_name = 'std_for_convLSTM_seed_'+str(seed)+'_slice_'+str(root)+'.csv'
    np.savetxt(mean_error_file_name,mean_error.T,delimiter=',')
    np.savetxt(std_file_name,std.T,delimiter=',')
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.yscale("log")
    plt.plot(x_axis,mean_error,color='yellow', label='Error')
    plt.fill_between(x_axis,mean_error-std, mean_error+std,color='green' )
    plt.legend()
    jpg_name = 'covLSTM_seed_'+str(seed)+'_slice_'+str(root)+'.jpg'
    plt.savefig(jpg_name)
    
    print("Test---------------------------------------Test")
    test_x,test_y = Make_Test_Dataset(lookback)
    test_x = test_x.cuda()
    test_y = test_y.cuda()
    model = model.cuda()
    test_layer_output_list,test_last_state_list = model(test_x)
    test_error = [criterion_L(test_layer_output_list[0][0][i],test_y[0][i]) for i in range(test_y[0].shape[0])]
    mean_test = torch.mean(torch.tensor(test_error,dtype=torch.float32))
    std_test = torch.std(torch.tensor(test_error,dtype=torch.float32))
    mean_error_test_file_name = 'mean_error_for_test_convLSTM_seed_'+str(seed)+'_slice_'+str(root)
    std_test_file_name = 'std_for_test_convLSTM_seed_'+str(seed)+'_slice_'+str(root)
    mean_test_list = [mean_test]
    std_test_list = [std_test]
    np.savetxt(mean_error_test_file_name,mean_test_list)
    np.savetxt(std_test_file_name,std_test_list)
    
    model_name = 'ConvLSTM_seed_'+str(seed)+'_slice_'+str(root)+'.pk1'
    torch.save(model,model_name)
if __name__=="__main__":
    train_and_test(9,20)