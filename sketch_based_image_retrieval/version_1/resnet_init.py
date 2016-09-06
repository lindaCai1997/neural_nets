import skimage.io 
import skimage.transform
import tensorflow as tf 
import numpy as np
from numpy import array
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

import datetime
import os
import time

''' This pretraining network is a reconstruction based on memory of
	https://github.com/ry/tensorlow-resnet/blob/master/resnet.py

	It's only meant as an exercise for the author and by no means replaces the orginial resnet

	function order(up to down): define variable, single layer, 
	block of layers, stack of blocks, full network( without loss function ), training function
'''

MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 16, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 500000, "max steps")
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op

activation = tf.nn.relu

# ==========wrapper function to initialize a variable================
def _get_variable(name, shape, initializer, weight_decay = 0.0, 
	dtype = 'float', trainable=True):
	'''A little wrapper around tf.get_variable to do weight decay and
	   add to resnet collection
	'''
	if weight_decay > 0:
		regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
	else:
		regularizer = None 
	collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
	return tf.get_variable(name, shape=shape, initializer=initializer, 
		dtype=dtype, regularizer=regularizer, collections=collections, trainable=trainable)

# ===================================================================
# ===========wrapper functions to define network layers==============
def conv(x, size, num_units_out):
	num_units_in = x.get_shape()[3]
	shape = [size, size, num_units_in, num_units_out]
	initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
	w = _get_variable('weights', shape=shape, dtype='float', 
		initializer=initializer, weight_decay=CONV_WEIGHT_DECAY)
	return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding = 'SAME')

def fc(x, num_units_out):
	num_units_in = x.get_shape()[1]
	weight_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
	weights = _get_variable('weights', shape=[num_units_in, num_units_out], 
		initializer=weight_initializer, weight_decay=FC_WEIGHT_STDDEV)
	biases = _get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer)
	return tf.nn.xw_plus_b(x, weights, biases)

def _max_pool(x, size, step):
	return tf.nn.max_pool(x, ksize = [1, size, size, 1], strides = [1, step, step, 1], padding = 'SAME')

def bn(x, use_bias=False, is_training=True):
	is_training = tf.convert_to_tensor(is_training, dtype='bool', name='is_training')
	x_shape = x.get_shape()
	params_shape = x_shape[-1:]
	if use_bias:
		bias = _get_variable('bias', params_shape, initializer=tf.zeros_initializer)
		return x + bias

	axis = list(range(len(x_shape) - 1))
	beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer)
	gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer)
	moving_mean = _get_variable('moving_mean', params_shape, 
		initializer=tf.zeros_initializer, trainable=False)
	moving_variance = _get_variable('moving_variance', params_shape, 
		initializer=tf.ones_initializer, trainable=False)
	mean, variance = tf.nn.moments(x, axis)
	update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
	update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
	tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
	tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
	mean, variance = control_flow_ops.cond(is_training, lambda: (mean, variance), lambda: (moving_mean, moving_variance))
	x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
	return x


def loss(logits, labels):
	weird = logits.get_shape()
	two = labels.get_shape()
	print weird, two
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
	tf.scalar_summary('loss', loss_)
	return loss_

#================================================================================
#========helper function to define a block of resnet layers======================
def block(x, size, bottleneck, block_filter_internal, is_training, use_bias):
	filters_in = x.get_shape()[-1]
	# print filters_in
	m = 4 if bottleneck else 1
	filters_out = m*block_filter_internal
	shortcut = x
	if bottleneck:
		num_units_out = block_filter_internal
		with tf.varable_scope('a'):
			size = 1
			x = conv(x, size, num_units_out)
			x = bn(x, use_bias, is_training)
			x = activation(x)

		with tf.variable_scope('b'):
			x = conv(x, size, num_units_out)
			x = bn(x, use_bias, is_training)
			x = activation(x)

		with tf.variable_scope('c'):
			size = 1
			num_units_out = filters_out
			x = conv(x, size, num_units_out)
			x = bn(x, use_bias, is_training)

	else:
		size = 3
		num_units_out = block_filter_internal
		with tf.variable_scope('A'):
			x = conv(x, size, num_units_out)
			x = bn(x, use_bias, is_training)
			x = activation(x)

		with tf.variable_scope('B'):
			x = conv(x, size, num_units_out)
			x = bn(x, use_bias, is_training)

	with tf.variable_scope('shortcut'):
		if filters_out != filters_in:
			size = 1
			shortcut = conv(shortcut, size, filters_out)
			shortcut = bn(shortcut, use_bias=use_bias, is_training=is_training)

	return activation(x + shortcut)		
# ===============================================================================
# ======== a stack of full 6 block===============================================
def stack(x, size, num_blocks, bottleneck, block_filter_internal, is_training, use_bias):
	for n in range(num_blocks):
		with tf.variable_scope('block%d' % (n + 1)):
			x = block(x, size, bottleneck, block_filter_internal, is_training, use_bias)
	return x
#================================================================================
#=========full network========================================
def network_init(x, is_training, num_classes=125, num_blocks=[3, 4, 6 ,3], bottleneck=False, 
	use_bias=False):
	fc_units_out = num_classes
	pool_step = 2

	with tf.variable_scope('scale1'):
		size = 7
		num_units_out = 64
		x = conv(x, size, num_units_out)
		x = bn(x, use_bias, is_training)
		x = activation(x)

	step = 2
	with tf.variable_scope('scale2'):
		size = 3
		step = 2
		block_filter_internal = 64
		x = _max_pool(x, size, step)
		x = stack(x, size, num_blocks[0], bottleneck, block_filter_internal, is_training, use_bias)

	with tf.variable_scope('scale3'):
		size = 3
		block_filter_internal = 128
		x = _max_pool(x, size, step)
		x = stack(x, size, num_blocks[1], bottleneck, block_filter_internal, is_training, use_bias)

	with tf.variable_scope('scale4'):
		size = 3
		block_filter_internal = 256
		x = _max_pool(x, size, step)
		x = stack(x, size, num_blocks[2], bottleneck, block_filter_internal, is_training, use_bias)

	with tf.variable_scope('scale5'):
		size = 3
		block_filter_internal = 512
		x = _max_pool(x, size, step)
		x = stack(x, size, num_blocks[3], bottleneck, block_filter_internal, is_training, use_bias)

	x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

	if num_classes != None:
		with tf.variable_scope('fc'):
			x = fc(x, num_classes)
	return x
# =============================================================================
def pre_train_photo(photos):
	batch_size = 100
	images = tf.placeholder(tf.float32, shape = [None, None, None, 3])
	labels = tf.placeholder(tf.int64, shape = [None])

	sess = tf.InteractiveSession()
	# batch.images = tf.reshape(batch.images, [-1, 332, 500, 3])
	logits = network_init(images, is_training=True, num_classes=125, num_blocks=[2, 2, 2, 2], 
		bottleneck=False)
	global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
		trainable=False)
	val_step = tf.get_variable('val_step', [], initializer=tf.constant_initializer(0), 
		trainable=False)
	loss_ = loss(logits, labels)
	predictions = tf.nn.softmax(logits)
	in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
	num_correct = tf.reduce_sum(in_top1)
	top_1_error = (batch_size - num_correct) / batch_size
	train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_)
	# loss_avg
	sess.run(tf.initialize_all_variables())
	for i in range(20000):
		batch = photos.next_batch(batch_size, True)
		train_step.run(feed_dict={images: batch.images, labels: batch.labels})

		error = top_1_error.eval(feed_dict={images: batch.images, labels: batch.labels})
		print 'Step', i, 'Training error:', error

		if (i % 100 == 0 and i != 0):
			batch = photos.next_batch(50, False)
			error = top_1_error.eval(feed_dict={images: batch.images, labels: batch.labels})


