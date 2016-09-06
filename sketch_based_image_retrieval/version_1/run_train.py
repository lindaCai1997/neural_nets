import tensorflow as tf 
import numpy as np
import os

from retrieve_data import get_data
from resnet_init import pre_train_photo

# images, sketches = get_data()
images = get_data()
pre_train_photo(images)



