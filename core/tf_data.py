"""
data.py - module for data loading
"""

# import dependencies
import os
import pydicom
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# set random seeed
tf.random.set_seed(2021)

class VBD_Dataset:
	""" VBD_Dataset - class for loading VinBigDataset """

	def __init__(self, data, image_dir, seed = 2021):
		"""
		VBD_Dataset class constructor
		Inputs:
			- data : str or pandas.DataFrame
				Dataframe (or path to dataframe) of image and labels
			- image_dir : str
				Path to image directory
		"""

		self.data = pd.read_csv(data) if isinstance(data, str) else data
		self.image_ids = self.data['image_id'].unique()
		self.image_dir = image_dir
		self.seed = seed

		# re-set random seed if different from 2021
		tf.random.set_seed(self.seed)

	def process(self, image_ids, image_dir, lables = None):

		# process images
		# convert to Tensorflow Dataset slices
		image_ids = tf.data.Dataset.from_tensor_slices(image_ids)

		return None, None if labels else None
	def __cal__(self, test_size):
		"""
		__call__ - calling function to generate TF dataset
		"""
		if test_size > 0.0:
			# split dataset into training and validation sets
			train_image_ids, val_image_ids = train_test_split(self.image_ids,
				test_size = test_size, seed = self.seed)

			return self.process(train_image_ids), self.process(val_image_ids) 
		else:
			return self.process(self.image_ids)
