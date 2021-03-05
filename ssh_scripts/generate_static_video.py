from PIL import Image
import numpy as np 
import xarray as xr
import os
import cv2 

from torchvision import transforms
import torch

image = cv2.imread('../cat.png')

frames = [image]*365

SAVEMODE=True

for i in range(len(frames)):
    if SAVEMODE : 
        image =  cv2.imwrite(f'./static_cat/image_{i}.png', transformed_ssh[i]) 


def save():
    os.system("ffmpeg -r 10 -i static_cat/image_%01d.png -vcodec mpeg4 -y static_video.mp4")

if SAVEMODE : 
    print('yes')
    save()
