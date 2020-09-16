# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:41:57 2020

@author: Conglei tan
"""
# This file is to make the climate_data dataset
import netCDF4 as nc
import numpy as np
import xarray
import os
import torch.utils.data
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
class climate_data(Dataset):
    
    def __init__(self, root_dir,transform=None):
        super(climate_data, self).__init__()
        self.root_dir = root_dir
        self.dataset = os.listdir(self.root_dir)
        self.transform = transform
    def __getitem__(self, index):
        data_index = self.dataset[index]
        data_path = os.path.join(self.root_dir, data_index)
        image = Image.open(data_path)
        if self.transform:
            image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.dataset)

