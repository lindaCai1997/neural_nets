import tensorflow as tf 
import numpy as np
import os
import random
from random import shuffle
import matplotlib.image as mpimg
import PIL
from PIL import Image

# specify import paths
sketchy_photo_paths = '/Users/linda/Desktop/sketchy/photos'
sketchy_sketch_paths = '/Users/linda/Desktop/sketchy/sketches'
sketchy_edgemap_paths = '/Users/linda/Desktop/sketchy/edge_maps'
train_image_file = '/train_images.npy'
train_name_file = '/train_names.npy'
train_label_file = '/train_labels.npy'
test_image_file = '/test_images.npy'
test_name_file = '/test_names.npy'
test_label_file = '/test_labels.npy'

# ==============structs for data sets============
class data():
	names = []
	images = []
	labels = []

class data_set():
	train = data()
	test = data()
	curr_location = 0

	def __init__(self):
		self.train = data()
		self.train.names = []
		self.train.images = []
		self.train.labels = []
		self.test = data()
		self.test.names = []
		self.test.images = []
		self.test.labels = []
		curr_location = 0

	def next_batch(self, batch_size, for_train, shuffle):
		
		batch = data()
		batch.names = []
		batch.images = []
		batch.labels = []

		if (for_train == True):
			length = len(self.train.images)
			print length
			if (shuffle == True):
				sample = random.sample(range(length), batch_size)
			else:
				start = self.curr_location
				if (self.curr_location + batch_size >= length):
					end = length
				else:
					end = self.curr_location + batch_size
				sample = list(range(start, end, 1))
				if (end >= length):
					self.curr_location = 0
				else:
					self.curr_location = end

			for i in sample:
				batch.names.append(self.train.names[i])
				batch.images.append(self.train.images[i])
				batch.labels.append(self.train.labels[i])
			
		else:
			length = len(self.test.images)
			sample = random.sample(range(length), batch_size)
			for i in sample:
				batch.names.append(self.test.names[i])
				batch.images.append(self.test.images[i])
				batch.labels.append(self.test.labels[i])
			
		return batch

# ========function to load images==============
def process_dir(paths):
	dir_list = os.listdir(paths)
	img_list = []
	dir_name_list = []

	for x in dir_list:
		if (x.count('.') == 0):
			img_list.append(os.listdir(paths + "/" + x))
			dir_name_list.append(x)

	return img_list, dir_name_list
# =======helper function to generate ======================================
def change_to_absolute_path(paths, dir_list, img_list):
	abs_img_list = []
	i = 0
	for x in img_list:
		temp = []
		for y in x:
			if (y.count('.txt') == 0 and y.count('.DS_Store') == 0):
				y = paths + "/" + dir_list[i] + "/" + y
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
				elif (option == 'edgemap'):
					y = y.split('.png')
					temp.append(y[0])
				else:
					y = y.split('-')
					temp.append(y[0])

		refined_list.append(temp)

	return refined_list

#========helper function to load training and testing data==================
def load_data(datatype, img_list, name_list, train, test):
	# make data directory if necessary
	if datatype == "photo":
		directory = "photo"
		image_shape = np.zeros((66, 100, 3))
	elif datatype == "sketch":
		directory = "sketch"
		image_shape = np.zeros((66, 100))
	elif datatype == "edgemap":
		directory = "edgemap"
		image_shape = np.zeros((66, 100))

	if not os.path.exists(directory):
		os.makedirs(directory)

	# process data, separate into training and test data
	images = []
	labels = []
	names = []
	for i in range(len(img_list)):

		for j in range(len(img_list[i])):
			# discard corrupted files
			valid_pic = True
			try:
				if (datatype == "photo"):
					image = Image.open(img_list[i][j])
				else:
					image = Image.open(img_list[i][j])
			except IOError:
				valid_pic = False
				print "invalid:", img_list[i][j]

			if (valid_pic == True):
				resize_image = image.resize((100, 66))
				resize_image_arr = np.asarray(resize_image)

				if (datatype != "photo"):
					temp = np.zeros([66, 100, 1])
					for ii in range(len(resize_image_arr)):
						for jj in range(len(resize_image_arr[0])):
							temp[ii][jj] = np.array(resize_image_arr[ii][jj])
					resize_image_arr = temp


				if (j < (len(img_list[i])*0.9)):
					train.names.append(name_list[i][j])
					train.images.append(resize_image_arr)
					train.labels.append(i)
			
				else:
					test.names.append(name_list[i][j])
					test.images.append(resize_image_arr)
					test.labels.append(i)

		N1 = len(images)
		N2 = len(labels)
		N3 = len(test.images)
		N4 = len(test.labels)
		print 'processed: ', N1, ' training images, ', N2, ' training labels, ', N3,' testing images, ', N4, 'testing labels'

	# shuffle training data
	'''
	x = list(range(len(images)))
	shuffle(x)
	for i in x:
		if images[i].shape == image_shape.shape:
			train.images.append(images[i])
			train.names.append(names[i])
			train.labels.append(labels[i])
	'''

	# convert lists to np arrays
	train.images = np.asarray(train.images)
	train.names = np.asarray(train.names)
	train.labels = np.asarray(train.labels)
	test.images = np.asarray(test.images)
	test.names = np.asarray(test.names)
	test.labels = np.asarray(test.labels)

	# save data
	open(directory + train_image_file, "w+")
	open(directory + train_name_file, "w+")
	open(directory + train_label_file, "w+")
	open(directory + test_image_file, "w+")
	open(directory + test_name_file, "w+")
	open(directory + test_label_file, "w+")

	np.save(directory + train_image_file, train.images)
	np.save(directory + train_name_file, train.names)
	np.save(directory + train_label_file, train.labels)
	np.save(directory + test_image_file, test.images)
	np.save(directory + test_name_file, test.names)
	np.save(directory + test_label_file, test.labels)

	return
#=============================================================

#=======load up sketches and images===========================
def get_sketchy_data(datatype):
	data = data_set()
	if (datatype == "photo"):
		path = sketchy_photo_paths
	elif (datatype == "sketch"):
		path = sketchy_sketch_paths
	elif (datatype == "edgemap"):
		path = sketchy_edgemap_paths
	else:
		print "Datatype can only be 'photo', 'sketch', or 'edgemap'"
		return
 	# if data has already been processed, directly load them as np arrays
	if (os.path.isfile(datatype + train_image_file)):
			data.train.images =  np.load(datatype + train_image_file)
			data.train.names =  np.load(datatype + train_name_file)
			data.train.labels =  np.load(datatype + train_label_file)
			data.test.images =  np.load(datatype + test_image_file)
			data.test.names =  np.load(datatype + test_name_file)
			data.test.labels =  np.load(datatype + test_label_file)

	# otherwise, process and load data
	else:
		image_list, name_list = process_dir(path)	
		abs_list = change_to_absolute_path(path, name_list, image_list)
		image_list = cut_file_type(image_list, datatype)
		load_data(datatype, abs_list, image_list, data.train, data.test)


	return data
#=============================================================



