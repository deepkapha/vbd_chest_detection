"""
torch_utils.py - module for image transform
"""

# import depedencies
import oos
import torch

from torchvision import transforms

def get_tarnsform():
	return transforms.Compose([
		transforms.RandomHorizontalFlip(0.5),
		transforms.RandomRotationo(degress = 45),
		transforms.Resize((800, 800)),
		transforms.ToTensor(),
		transforms.Normalize(mean = (0, 0, 0), std = (1,1,1))]) 
