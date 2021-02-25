"""
Generates masks from the training data available
"""


from PIL import Image
import numpy as np 
import xarray as xr
import os
import matplotlib.pyplot as plt
import cv2

dataset_path = '../../datasets/'

ref_path = dataset_path + 'data.nc'
xdata = xr.open_dataset(ref_path)


masks_sttn = np.isnan(xdata['flag'].values)*255
masks_sttn = masks_sttn[:,5:197,5:197]
print(masks_sttn.shape)
for i in range(len(masks_sttn)): 
    cv2.imwrite(f'../examples/ssh_examples/mask_{i}.png',masks_sttn[i])
