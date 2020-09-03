# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 18:49:47 2020

@author: lenovo
"""
import numpy as np 
import get_data
from torchvision import transforms
import torch
import ConvLSTM
import matplotlib.pyplot as plt
def Make_ConvLSTM_Dataset(Sample,Frame):
    root_dir = 'ALL_DATASET_RESIZED'
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = get_data.climate_data(root_dir,transform=transform)
    input = torch.zeros(Sample,Frame,1,40,60)
    target = torch.zeros(Sample,Frame,1,40,60)
    for i in range(Sample):
        np.random.seed(i)
        start_number = np.random.randint(0,1200)
        for j in range(Frame):
            input[i,j,:,:,:] = dataset[j+start_number]
            target[i,j,:,:,:] = dataset[j+start_number+1]
    return input,target
input, target = Make_ConvLSTM_Dataset(2000,6)
print("DONE")
epoches = 3000
lr = 0.00003
channels = 1
model = ConvLSTM.ConvLSTM(input_dim=channels,hidden_dim=[64,96,112,64,32,1],kernel_size=(5, 5),num_layers=6,batch_first=True,bias=True,return_all_layers=False)
model = model.cuda()
print('# ConvLSTM parameters:', sum(param.numel() for param in model.parameters()))
optimizer=torch.optim.Adam(model.parameters(),lr =lr)
criterion = torch.nn.MSELoss()
error = []
for epoch in range(epoches):
    running_loss = 0
    np.random.seed(epoch)
    start_number = np.random.randint(0,1980)
    data = input[start_number:start_number+20,:,:,:,:]
    out = target[start_number:start_number+20,:,:,:,:]
    data = data.cuda()
    out = out.cuda()
    layer_output_list,last_state_list= model(data,None)
    loss = criterion(layer_output_list[0], out)
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    running_loss = running_loss+loss.item()
    if epoch%50 == 0:
       print("Epoch {}/{}".format(epoch+1, epoches))
       print("Loss is:{:.7f}".format(running_loss))
    to_pil_image = transforms.ToPILImage()
    error.append(running_loss)
    if epoch == 2999:
       for i in range(20):
           save_file_second = "CONVLSTM_TRAINING/"+str(i+1)+".jpg"
           img = to_pil_image(layer_output_list[0][i][2].cpu())
           img.save(save_file_second)
error = np.asarray(error)
file_name='error_for_convlstm_prediction.csv'
np.savetxt(file_name,error.T,delimiter=',')
plt.figure()
plt.xlabel("epoch")
plt.ylabel("error")
plt.yscale("log")
plt.plot(error, color='yellow', label='Error')
plt.legend()
plt.savefig("error_training_convlstm.jpg")
torch.save(model,'CONVLSTM.pk1')