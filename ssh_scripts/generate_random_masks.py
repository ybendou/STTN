import sys
sys.path.append("../STTN/core/")

from utils import create_random_shape_with_random_motion
import numpy as np 
import xarray as xr
import os 

dataset_path = '../../datasets/ssh/'

ref_path = dataset_path + 'ref.nc'
xdata = xr.open_dataset(ref_path)
data = xdata['ssh'].values


masks = create_random_shape_with_random_motion(*data.shape)

for i in range(len(masks)):
    masks[i].save(f'./random_masks/mask_{i}.png')


os.system('cp -r random_masks ../STTN/examples/')