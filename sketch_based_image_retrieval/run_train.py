import tensorflow as tf 
import numpy as np
import os

from PIL import Image
from scipy import ndimage
from skimage import feature
from resnet_init import pre_train_edgemaps, pre_train_sketch
from retrieve_data import get_sketchy_data



edgemaps = get_sketchy_data("edgemap")
sketches = get_sketchy_data("sketch")
'''
pre_train_sketches(sketches, True)
'''
pre_train_edgemaps(edgemaps, True)



def cross_training():
		sketch_net = tf.train.latest_checkpoint(FLAGS.sketch_train_dir)
		saver.restore(sess, sketch_net)




