from PIL import Image
import cv2
import torchvision
import torch
image = cv2.imread('ssh_obs/image_0.png')


pic = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

img = torch.ByteTensor(
            torch.ByteStorage.from_buffer(pic.tobytes()))
img = img.view(pic.size[1], pic.size[0], len(pic.mode))
# put it from HWC to CHW format
# yikes, this transpose takes 80% of the loading time/CPU
img = img.transpose(0, 1).transpose(0, 2).contiguous()
img = img.float().div(255)


print(img.reshape(201,201, 3))