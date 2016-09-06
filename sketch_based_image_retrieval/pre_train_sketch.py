import tensorflow as tf 
import numpy as np
import os

from retrieve_data import get_data
from resnet_init import pre_train_sketch

photos, sketches = get_sketchy_data()
pre_train_sketch(sketches, True)
