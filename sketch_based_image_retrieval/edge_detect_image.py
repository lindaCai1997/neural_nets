import tensorflow as tf 
import numpy as np
import os

from PIL import Image
from scipy import ndimage
from skimage import feature
from resnet_init import pre_train_photo

sketchy_photo_paths = '/Users/linda/Desktop/sketchy/photos'
def process_dir(paths):
	dir_list = os.listdir(paths)
	img_list = []
	dir_name_list = []

	for x in dir_list:
		if (x.count('.') == 0):
			img_list.append(os.listdir(paths + "/" + x))
			dir_name_list.append(x)

	return img_list, dir_name_list

def change_to_absolute_path(paths, dir_list, img_list):
	abs_img_list = []
	name_list = []
	i = 0
	for x in img_list:
		temp = []
		temp2 = []
		for y in x:
			if (y.count('.txt') == 0 and y.count('.DS_Store') == 0):
				temp2.append(y)
				y = paths + "/" + dir_list[i] + "/" + y
				temp.append(y)
	
		abs_img_list.append(temp)	
		name_list.append(temp2)
		N = np.shape(abs_img_list[i])[0]
		print 'Retrieving from', N , 'images'
		i = i + 1

	return abs_img_list, name_list

def create_edge_map(file_name_list, abs_list, dir_name_list, start_class):
	if not os.path.exists("edge_maps"):
		os.makedirs("edge_maps")
	begin = False
	for i in range(len(abs_list)):
		if (dir_name_list[i] == start_class):
			begin = True
		if (begin == False):
			continue

		if not os.path.exists("edge_maps/" + dir_name_list[i]):
			os.makedirs("edge_maps/" + dir_name_list[i])

		for j in range(len(abs_list[i])):

			# convert image to grayscale
			image = Image.open(abs_list[i][j])
			image = np.asarray(image)
			image_shape = np.shape(image)
			if(len(image_shape) == 3):
				image_gray = np.zeros([image_shape[0], image_shape[1]])

				for ii in range(image_shape[0]):
					for jj in range(image_shape[1]):
						image_gray[ii][jj] = (int)(0.2989*image[ii][jj][0] + 0.5870*image[ii][jj][1] 
							+ 0.1140*image[ii][jj][2])
			else:
				image_gray = image
			image_gray = np.asarray(image_gray, dtype=np.uint8)

			#generate edges
			edges = feature.canny(image_gray, sigma=2)
			edges = np.uint8(edges*255)
			edge_im = Image.fromarray(edges, mode='L')
			edge_im.save("edge_maps/" + dir_name_list[i] + "/" + file_name_list[i][j])
        

# photos, sketches = get_sketchy_data()
def create_edge_maps(start_class):
		photo_list, p_name_list = process_dir(sketchy_photo_paths)
		abs_photo_list, name_list = change_to_absolute_path(sketchy_photo_paths, p_name_list, photo_list)
		if (start_class == None):
			create_edge_map(name_list, abs_photo_list, p_name_list, 'airplane')
		else:
			create_edge_map(name_list, abs_photo_list, p_name_list, start_class)

create_edge_maps(None)

