from PIL import Image
import numpy as np 
import xarray as xr
import os
import cv2 

from torchvision import transforms
import torch

dataset_path = '../../datasets/'

ref_path = dataset_path + 'ref.nc'
xgt = xr.open_dataset(ref_path)
gt = xgt['ssh'].values
maxx = gt.max()
minn = gt.min()


transformed_ssh = 255*(gt-minn)/(maxx-minn) # à revoir
# transformed_ssh = 255*data # à revoir
# transformed_ssh =  transformed_ssh[..., np.newaxis]
# transformed_ssh = torch.from_numpy(transformed_ssh).permute(2, 3, 0, 1).contiguous()



SAVEMODE=True

for i in range(len(transformed_ssh)):
    if SAVEMODE : 
        image =  cv2.imwrite(f'./ssh_obs/image_{i}.png', transformed_ssh[i]) 


def save():
    os.system("ffmpeg -r 10 -i ssh_obs/image_%01d.png -vcodec mpeg4 -y ssh_train_video.mp4")

if SAVEMODE : 
    print('yes')
    save()
    os.system("mv ssh_train_video.mp4 ../examples/")
    os.system('rm ssh_obs/*')