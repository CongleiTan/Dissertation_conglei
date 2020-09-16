# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 18:49:57 2020

@author: lenovo
"""
#This file is to convert the netcdf to image
import numpy as np
import xarray
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image
def nc_to_picture(root,number):
    for i in range(13,number):
        filename = root+str(i)+'.nc'
        data=xarray.open_dataset(filename)
        temperature=data['temp']
        # Define the map projection
        #print(np.min(np.array(temperature.isel(time=0,pfull=12))))
        #print(np.max(np.array(temperature.isel(time=0,pfull=12))))
        fig = plt.figure()
        proj = ccrs.PlateCarree()
        ax = fig.add_subplot(121, projection=proj)
        ax.set_global()
        ccrs.PlateCarree()
        #ax.add_feature(cfeature.LAND)
        #ax.add_feature(cfeature.OCEAN)
        #ax.add_feature(cfeature.COASTLINE)
        orig_cmap = matplotlib.cm.coolwarm
        cmap = matplotlib.cm.jet
        norm = matplotlib.colors.Normalize(vmin=210, vmax=275) 
        temperature.isel(time=0,pfull=12).plot.pcolormesh(ax=ax,cmap='Greys', transform=ccrs.PlateCarree(), norm=norm,add_colorbar=False, add_labels=False)
        save_file = 'ALL_DATASET/'+str(i-12)+'.jpg'
        plt.savefig(save_file,bbox_inches='tight', pad_inches = 0)

def Resize_Picture(number):
    for i in range(1000,number):
        filename = 'ALL_DATASET/'+str(i)+'.jpg'
        img=Image.open(filename)
        #img = img.crop((115, 20, 135, 40))
        img = img.resize((60,40),Image.ANTIALIAS)
        #img = img.crop((25, 15, 35, 25))
        img = img.convert("L")
        print(len(img.split()))
        save_file = 'ALL_DATASET_RESIZED/00'+str(i)+'.jpg'
        img.save(save_file)
if __name__=="__main__":
    root = 'data/'
    nc_to_picture(root,1813)
    Resize_Picture(1801)