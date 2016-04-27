# get_h5data.py 
# Copyright 2016, Akash Maharaj, All rights reserved.
# 
# This is a lightly modified version of Google's inumpyut_data.py 
# provided in the TensorFlow MNIST tutorial. Changes are made
# to handle the loading of hdf5 data as output by matlab code
#
# Provides functions used by TensorFlow 
# ==================================================

"""Functions for loading and reading CRFS data from hdf5 data format """

from __future__ import division
from __future__ import print_function
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy
import h5py
import os

def extract_images(filename):
 	"""Extract images from hdf5 format into a 4D uint8 numpy array [index, y, x, depth]."""
  
 	print('Extracting', filename)

 	with h5py.File(filename,'r') as hf:
		data = hf.get('data')
		labels = hf.get('label')
		np_data = numpy.array(data, dtype=numpy.uint8).transpose(0,3,2,1)
		
	return np_data


def dense_to_one_hot(labels_dense, num_classes=3):
	"""Convert class labels from scalars to one-hot vectors."""
	num_labels = labels_dense.shape[0]
	index_offset = numpy.arange(num_labels) * num_classes
	labels_one_hot = numpy.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot

def extract_labels(filename, one_hot=False):
 	"""Extract the labels into a 1D uint8 numpy array [index]."""
 	print('Extracting', filename)
 
 	with h5py.File(filename,'r') as hf:
		labels = hf.get('label')
		np_labels = numpy.array(labels, dtype=numpy.uint8)
		

	if one_hot:
		return dense_to_one_hot(np_labels)

	return np_labels


class DataSet(object):

	def __init__(self, images, labels, fake_data=False, one_hot=False,
	           dtype=tf.float32):
		"""Construct a DataSet.

		one_hot arg is used only if fake_data is true.  `dtype` can be either
		`uint8` to leave the inumpyut as `[0, 255]`, or `float32` to rescale into
		`[0, 1]`.
		"""
		dtype = tf.as_dtype(dtype).base_dtype
		if dtype not in (tf.uint8, tf.float32):
		  raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
		                  dtype)
		if fake_data:
		  self._num_examples = 10000
		  self.one_hot = one_hot
		else:
		  assert images.shape[0] == labels.shape[0], (
		      'images.shape: %s labels.shape: %s' % (images.shape,
		                                             labels.shape))
		  self._num_examples = images.shape[0]

		  # Convert shape from [num examples, rows, columns, depth]
		  # to [num examples, rows*columns] (assuming depth == 1)
		  assert images.shape[3] == 1
		  images = images.reshape(images.shape[0],
		                          images.shape[1] * images.shape[2])
		  if dtype == tf.float32:
		    # Convert from [0, 255] -> [0.0, 1.0].
		    images = images.astype(numpy.float32)
		    images = numpy.multiply(images, 1.0 / 255.0)

		self._images = images
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size, fake_data=False):
		"""Return the next `batch_size` examples from this data set."""
		if fake_data:
			fake_image = [1] * 784
			if self.one_hot:
			 	fake_label = [1] + [0] * 9
			else:
				fake_label = 0
			return [fake_image for _ in xrange(batch_size)], [
			    fake_label for _ in xrange(batch_size)]

		start = self._index_in_epoch
		self._index_in_epoch += batch_size

		if self._index_in_epoch > self._num_examples:
		  # Finished epoch
		  self._epochs_completed += 1
		  # Shuffle the data
		  perm = numpy.arange(self._num_examples)
		  numpy.random.shuffle(perm)
		  self._images = self._images[perm]
		  self._labels = self._labels[perm]
		  # Start next epoch
		  start = 0
		  self._index_in_epoch = batch_size
		  assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, fake_data=False, one_hot=False, dtype=tf.float32):
	class DataSets(object):
		pass

  	data_sets = DataSets()

	if fake_data:
		def fake():
	  		return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)
		data_sets.train = fake()
		data_sets.validation = fake()
		data_sets.test = fake()
		return data_sets

	TRAIN = os.path.join(train_dir, 'sample_train_gzip.h5') 
	TEST = os.path.join(train_dir, 'sample_test_gzip.h5') 

	VALIDATION_SIZE = 500

	train_images = extract_images(TRAIN)
	train_labels = extract_labels(TRAIN, one_hot=one_hot)



	test_images = extract_images(TEST)
	test_labels = extract_labels(TEST, one_hot=one_hot)

	#shuffle the datasets
 	perm = numpy.arange(train_images.shape[0])
	numpy.random.shuffle(perm)
	train_images = train_images[perm]
	train_labels = train_labels[perm]

	perm2 = numpy.arange(test_images.shape[0])
	numpy.random.shuffle(perm2)
	test_images = test_images[perm2]
	test_labels = test_labels[perm2]


	validation_images = train_images[:VALIDATION_SIZE]
	validation_labels = train_labels[:VALIDATION_SIZE]
	train_images = train_images[VALIDATION_SIZE:]
	train_labels = train_labels[VALIDATION_SIZE:]

	data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
	data_sets.validation = DataSet(validation_images, validation_labels,
	                             dtype=dtype)
	data_sets.test = DataSet(test_images, test_labels, dtype=dtype)

 	return data_sets
