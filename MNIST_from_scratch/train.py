import cPickle
import gzip
import single_layer_fully_connected
from single_layer_fully_connected import fully_connected_net
import numpy as np

dest = '/Users/linda/Desktop/conv_network/mnist.pkl.gz'
learning_rate = [0.5, 0.5]

def vectorized_result(j):
	e = np.zeros((10,1))
	e[j] = 1.0
	return e

def load_data():
	# open the dataset at downloaded position
	f = gzip.open(dest, 'rb')
	training_data, validation_data, test_data = cPickle.load(f)
	f.close()
	training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
	training_results = [vectorized_result(y) for y in training_data[1]]
	training_data = zip(training_inputs, training_results)
	validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
	validation_data = zip(validation_inputs, validation_data[1])
	test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
	test_data = zip(test_inputs, test_data[1])
	return (training_data, validation_data, test_data)

training_data, validation_data, test_data = load_data()
net = fully_connected_net([784, 200, 10])
net.sgd(training_data, 10, 400, learning_rate, test_data=test_data)