import tensorflow as tf 
import numpy as np
import os
import random
import matplotlib.image as mpimg
import PIL
from PIL import Image

# specify import paths
photo_paths = '/Users/linda/Desktop/sketchy/photos'
sketch_paths = '/Users/linda/Desktop/sketchy/sketches'
# ==============structs for data sets============
class data():
	names = []
	images = []
	labels = []

class data_set():
	train = data()
	teste = data()
	def __init__(self):
		self.train = data()
		self.train.names = []
		self.train.images = []
		self.train.labels = []
		self.test = data()
		self.train.names = []
		self.test.images = []
		self.test.labels = []
	def next_batch(self, batch_size, for_train):
		batch = data()
		batch.names = []
		batch.images = []
		batch.labels = []
		sample = random.sample(range(len(self.train.names)), batch_size);
		for i in sample:
			if(for_train == True):
				batch.names.append(self.train.names[i])
				batch.images.append(self.train.images[i])
				batch.labels.append(self.train.labels[i])
			else:
				batch.names.append(self.test.names[i])
				batch.images.append(self.test.images[i])
				batch.labels.append(self.test.labels[i])

		return batch

# ========function to load images==============
def process_dir(paths):
	dir_list = os.listdir(paths)
	img_list = []
	label = []

	for x in dir_list:
		if (x.count('.') == 0):
			img_list.append(os.listdir(paths + "/" + x))
			label.append(x)

	return dir_list, img_list, label
# =======helper function to generate ======================================
def change_to_absolute_path(paths, dir_list, img_list):
	abs_img_list = []
	i = 0
	for x in img_list:
		temp = []
		for y in x:
			if (y.count('.txt') == 0 and y.count('.DS_Store') == 0):
				y = paths + "/" + dir_list[i + 1] + "/" + y
				temp.append(y)
	
		abs_img_list.append(temp)	
		N = np.shape(abs_img_list[i])[0]
		print 'Retrieving from', N , 'images'
		i = i + 1

	return abs_img_list

# =======helper function to generate file name without file type( e.g. .jpg)======
def cut_file_type(img_list, option):
	refined_list = []
	for x in img_list:
		temp = []
		for y in x:
			if (y.count('.txt') == 0 and y.count('.DS_Store') == 0):
				if (option == 'photo'):
					y = y.split('.jpg')
					temp.append(y[0])
				else:
					y = y.split('-')
					temp.append(y[0])

		refined_list.append(temp)

	return refined_list

#========helper function to load training and testing data==================
def load_data(img_list, name_list, label, train, test):
	for i in range(len(img_list)):
		'''
		curr_label = []
		for j in range(len(img_list)):
			if (i == j):
				curr_label.append(1)
			else:
				curr_label.append(0)

		tf.to_int64(curr_label)
		'''
		for j in range(len(img_list[i])):
			image = Image.open(img_list[i][j])
			resize_image = image.resize((100, 66))
			resize_image_arr = np.asarray(resize_image)

			'''
			curr_image = mpimg.imread(imlist[i][j])

			# read image
			key, value = reader.read(filename_queue)
			
			# decode image 
			curr_image = tf.image.decode_jpeg(value, channels=3);
			
			# resize images so that they are of of the same size
			curr_image = tf.image.resize_images(curr_image, 332, 500)

			resize_image = np.asarray(curr_image)
			print resize_image.get_shape()
			'''
			if (j < (len(img_list[i])*0.9)):
				train.names.append(name_list[i][j])
				train.images.append(resize_image_arr)
				train.labels.append(i)
			
			else:
				test.names.append(name_list[i][j])
				test.images.append(resize_image_arr)
				test.labels.append(i)

		N1 = len(train.images)
		N2 = len(train.labels)
		N3 = len(test.images)
		N4 = len(test.labels)
		print 'processed: ', N1, ' training images, ', N2, ' training labels, ', N3,' testing images, ', N4, 'testing labels'
	
	return
#=============================================================

#=======load up sketches and images===========================
def get_data():
	p_dir_list, photo_list, labels = process_dir(photo_paths)
	s_dir_list, sketch_list, labels = process_dir(sketch_paths)
	abs_photo_list = change_to_absolute_path(photo_paths, p_dir_list, photo_list)
	abs_sketch_list = change_to_absolute_path(sketch_paths, s_dir_list, sketch_list)
	photo_list = cut_file_type(photo_list, 'photo')
	sketch_list = cut_file_type(sketch_list, 'sketch')

	photos = data_set()
	sketches = data_set()
	load_data(abs_photo_list, photo_list, labels, photos.train, photos.test)
#	load_data(abs_sketch_list, sketch_list, labels, sketches.train, sketches.test)

	return photos
#	return photos, sketches
#=============================================================

