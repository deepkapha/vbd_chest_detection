"""
torch_utils.py - module for image transform
"""

# import depedencies
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms

def collate_fn(batch):
	"""
	collate function
	"""
	return tuple(zip(*batch))

def get_transform():
	"""
	get_transform - function to perform image transformation
	"""

	return A.Compose([
		A.Flip(0.5),
		A.ShiftScaleRotate(scale_limit = 0.1, rotate_limit = 45, p = 0.25),
		A.LongestMaxSize(max_size = 800, p = 1.0),
		
		# normalization in FasterRCNN
		A.Normalize(mean = (0,0,0), std = (1,1,1), max_pixel_value = 255.0, p = 1.0),
		ToTensorV2(p = 1.0)],
			bbox_params = {'format' : 'pascal_voc', 'label_fields' : ['labels']})
