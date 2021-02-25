import os


import argparse


parser = argparse.ArgumentParser(description="test")
parser.add_argument("-m", "--mask", type=str, required=True)

args = parser.parse_args()

os.system('python ../STTN/test.py --video examples/ssh_video.mp4 --mask examples/ssh_masks --ckpt checkpoints/sttn.pth ')