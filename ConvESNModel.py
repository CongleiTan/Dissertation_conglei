# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:32:15 2020

@author: lenovo
"""
#This file is to build the model of ConvESN
import torch
import scipy as sp
import scipy.sparse as sparse
class model_params(object):
    def __init__(self, tau, nstep, N, d):
        self.tau = tau
        self.nstep = nstep
        self.N = N
        self.d = d
        
class res_params(object):
    def __init__(self,model_params,approx_res_size,radius,degree,sigma,train_length):
        self.model_params = model_params
        self.approx_res_size = approx_res_size
        self.radius = radius
        self.degree = degree
        self.sigma = sigma
        self.train_length = train_length
        self.N = int(torch.floor(torch.tensor(approx_res_size/model_params.N)))*model_params.N
        self.num_inputs = model_params.N
        
class multi_reservoir_parameter(object):
    def __init__(self,N,number_inputs,sigma):
        self.N = N
        self.number_inputs = number_inputs
        self.sigma = sigma
        
def generate_reservoir(size,radius,degree):
        sparsity = degree/float(size)
        A = torch.tensor(sparse.rand(size, size, density=sparsity).todense(),dtype=torch.float32)
        e = torch.max(torch.eig(A,eigenvectors=False)[0])
        A = (A/e)*radius
        return A

def Win(N,num_inputs,sigma):
    q = int(N/num_inputs)
    Win = torch.zeros((N,num_inputs))
    for i in range(num_inputs):
        torch.manual_seed(seed=i)
        Win[i*q: (i+1)*q,i] = sigma * (-1 + 2 * torch.rand(1,q)[0])
    return Win
    
def Wres(N,num_inputs,sigma):
    q = int(N/num_inputs)
    Wres = torch.zeros((N,num_inputs))
    for i in range(num_inputs):
        torch.manual_seed(seed=i)
        Wres[i*q: (i+1)*q,i] = sigma * (-1 + 2 * torch.rand(1,q)[0])
    return Wres

def reservoir_layer(A,encoder,Win,Wres,multi_reservoir_parameter, data, res_params,number_reservoir_layer):
        states = torch.zeros((res_params.N, res_params.train_length))
        for i in range(res_params.train_length-1):
            output = torch.flatten(encoder(data[i+1].unsqueeze(0)))
            states[:,i+1] = torch.tanh((torch.mm(A,states[:,i].unsqueeze(-1)) + torch.mm(Win,output.unsqueeze(0).t())).squeeze(-1))
        for i in range(1,number_reservoir_layer):
            last_states = states
            states = torch.zeros((res_params.N, res_params.train_length))
            Wres = Wres(multi_reservoir_parameter.N,multi_reservoir_parameter.number_inputs,multi_reservoir_parameter.sigma)
            for j in range(res_params.train_length-1):
                states[:,j+1] = torch.tanh((torch.mm(A,states[:,j].unsqueeze(-1)) + torch.mm(Wres,last_states[j+1].unsqueeze(0).t())).squeeze(-1))
        return states
class ConvESN(torch.nn.Module):
    def __init__(self,Coder):
        super(ConvESN, self).__init__()
        self.Coder = Coder
        for p in self.parameters():
            p.requires_grad=False 
        self.m = torch.nn.Linear(7200,1200)
    def forward(self,input):
        output = self.m(input.t())
        image_pre_set = torch.zeros(len(output),1,40,60)
        for i in range(len(output)):
            data = torch.reshape(output[i,:],(1,8,10,15))
            image_pre_set[i] = self.Coder.decoder(data)
        return image_pre_set