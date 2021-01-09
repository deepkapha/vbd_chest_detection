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
	from torch.utils.data import DataLoader
	from torch.utils.data.sampler import SequentialSampler

	from core.torch_utils import get_transform, collate_fn
	from core.torch_data import VBD_Dataset
	from core import models, losses

	# initialize datsets
	train_dataset = VBD_Dataset(data = args.df, image_dir = args.image_dir,
		transforms = get_transform())

	# initialize data loader
	train_data_loader = DataLoader(
		train_dataset,
		batch_size = args.batch_size,
		shuffle = args.shuffle,
		num_workers = args.workers,
		collate_fn = collate_fn)

	# select device
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	# initialize model
	model = models.FasterRCNN()
	model.to(device) # move model to suitable device for accelerated learning

	# initialize hyperameters
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr = args.lr, momentum = 0.9, weight_decay = 0.0005)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma = 0.1)

	# training pipeeline
	itr = 1
	loss_hist = losses.Averager()
	for epoch in range(args.epochs):
		loss_hist.reset()
		
		for images, targets in train_data_loader:
			images = list(image.to(device) for image in images)
			targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

			loss_dict = model(images, targets)
			
			# compute total loss
			losses = sum(loss for loss in loss_dict.values())
			loss_value = losses.item()

			# update loss_value
			loss_hist.send(loss_value)

			# backpropagation
			optimizer.zero_grad()
			losses.backward()
			optimizer.step()
			

			if itr % 100 == 0:
				print(f"Iteration #{itr} lloss: {loss_hist.value}")
			itr += 1
		# learning-rate scheduler
		if lr_scheduler is not None:
			lr_scheduler.step()

		print(f"Epoch #{epoch} loss: {loss_hisgt.value}")

	return None

def main(args):
	if args.type == 'torch':
		torch_train(args)
	else:
		tensorflow_train(args)

if __name__ == '__main__':
	# create argparse instance
	parser = argparse.ArgumentParser('Argument Parser')

	# add arguments
	parser.add_argument('--type', type = str, default = 'torch')
	parser.add_argument('--df', type = str, default = './vbd_chest_xrays/train.csv')
	parser.add_argument('--image-dir', type = str, default = './vbd_chest_xrays/train')
	parser.add_argument('--num-class', type = int, default = 15)
	parser.add_argument('--batch-size', type = int, default = '8')
	parser.add_argument('--shuffle', default = False, action = 'store_true')
	parser.add_argument('--workers', type = int, default = 4)
	parser.add_argument('--epochs', type = int, default = 50)
	parser.add_argument('--lr', type = float, default = 0.005)

	main(parser.parse_args())
