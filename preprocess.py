"""
preprocess.py - module for preprocessing data into Tensorflow Records formats
"""

# import dependencies
import os
import argparse
from multiprocessing import Pool
import pandas as pd
import tensorflow as tf

def write_tfrecords(inputs):
	"""
	write_tfrecords - function to write sample to tfrecords
	Inputs:
		- inputs : str
			String sample
	Outputs:
		- _ : tf.train.Example instance
	"""
	output = {
		'sample' : tf.train.Feature(bytes_list = tf.train.Byteslist(value = [inputs]))}

	return tf.train.Example(features = tf.train.Features(feature = output)).SerializeToString()

def concat(image_id, data, image_dir, writer):
	"""
	concat - function to concat image and labels together and writer to tfrecoords
	Inputs:
		- image_id : str
		- data : pandas.DataFrame
		- image_dir : str
		- writer : tf.io.TFRecordWriter instance
	"""

	# retrieve labels
	labels = data[data['image_id'] == image_id]

	# generate path to xray
	img_path = os.path.join(image_dir, image_dir + '.dicom')
	
	# generate sample as below
	# sample (separated by space) : image label1;label2;label3;label...
	# labels(separated by comma) : class,x1,y1,x2,y2

	# concat image and labels
	labels = ';'.join([str(x['class_id']) + ',' + str(x['x_min']) + ',' + str(x['y_min']) + ',' + str(x['x_max']) + ',' + str(x['y_max']) for x in labels])
	sample = img_path + ' ' + labels

	# write tfrecords
	writer.write(write_tf_records(sample))

def preprocess(args):
	"""
	preprocess - function to concat image and labels for efficient loading
	"""

	# read training dataframe
	df = pd.read_csv(args.df)

	# initialize multiprocessing instance
	pool = Pool(processes = 10)

	# extract unique image ids and corresponding labels
	image_ids = df['image_id'].unique()

	# initialize TFRecordWriter
	writer = tf.io.TFReccordWriter(args.output)

	pools = [pool.apply(concat, args = (id, df, args.image_dir, writer)) for id in image_ids]
	return None

if __name__ == '__main__':
	# create argument parser
	parser = argparse.ArgumentParser('Argument parser for preprocess module')

	# add arguments
	parser.add_argument('--df', type = str, './vbg_chest_xrays/train.csv')
	parser.add_argument('--image-dir', type = str, './vbg_chest_xrays/train')
	parser.add_argument('--output', type = str, './vbg_chest_xrays/chest_xrays.tfrecords')
	preprocess(parser.parse_args())
