"""
train.py - module for training VinBigData Chest X-Ray detection
"""

# import dependencies
import os
import argparse

def tensorflow_train(args):
	return None

def torch_train(args):

	# import dependencies
	import torch
	from torch.utils.data import Dataloader
	from torch.utils.data.sampler import SequentialSampler

	from core.torch_utils import get_transform, collate_fn
	from core.torch_data import VBD_Dataset

	# initialize datsets
	train_dataset = VBD_Dataset(data = args.df, image_dir = args.image_dir,
		transforms = get_transform())

	# initialize data loader
	train_data_loader = DataLoader(
		train_dataset,
		batch_size = args.batch_szie,
		shuffle = args.shuffle,
		num_workers = args.workers,
		collate_fn = collate_fn)

	# select device
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	return None

def main(args):
	if args.type == 'torch':
		torch_train(args)
	elese:
		tensorflow_train(args)

if __name__ == '__main__':
	# create argparse instance
	parser = argparse.ArgumentParser('Argument Parser')

	# add arguments
	parser.add_argument('--type', type = str, default = 'torch')
	parser.add_argument('--df', type = str, default = './vbg_chest_xrays/train.csv')
	parser.add_argument('--image-dir', type = str, default = './vbg_chest_xrays/train')
	parser.add_argument('--batch-size', type = int, default = '8')
	parser.add_argument('--shuffle', type = boolean, default = True)
	parser.add_argument('--workers', type = int, default = 4)

	main(parser.parse_args())
