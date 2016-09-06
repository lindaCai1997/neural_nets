import numpy as np
import random
'''
This network is a reimplementation of the full connected network describe in 
"Neural Network and Deep Learning"
Link: http://neuralnetworksanddeeplearning.com/chap1.html#implementing_our_network_to_classify_digits
The purpose of this work is for the author to understand back propagation
and underlying structure of neural network. It could also serve as a baseline 
for our study on neural network's performance on different ML tasks in the future 
'''
class fully_connected_net:
	def __init__(self, size):
		# size: an array with the number of neurons we should have
		# for each layer
		self.num_layers = len(size)
		self.size = size
		self.biases = []
		self.weights = []
		for i in xrange(len(size)):
			if (i > 0):
				self.biases.append(np.random.randn(size[i], 1))
				self.weights.append(np.random.randn(size[i], size[i - 1]))

	def feed_forward(self, a):
		for i in xrange(len(self.weights) - 1):
			a = sigma(np.dot(self.weights[i], a) + self.biases[i])
		a = softmax(np.dot(self.weights[-1], a) + self.biases[-1])
		return a
	
	def back_propagation(self, x, y): 
		# back propagate using 4 conditions:
		# 1) dc/dL = Delta(aL) * softmax_prime(zL)
		# 2) dc/dl = delta(l + 1) * w(l+1) * sigma_prime(zl)
		# 3) dc/dwij = dc/dl*a(zl-1)
		# 4) dc/db = dc/dl
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(b.shape) for b in self.weights]
		a = x
		activations = [x]
		zs = []
		i = 0
		layers = len(self.biases) + 1
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, a) + b
			zs.append(z)
			if (i < len(self.biases) - 1):
				a = sigma(z)
			else:
				a = softmax(z)
			activations.append(a)
			i += 1
		
		# delta_L
		delta_curr = cost_function_prime(activations[-1], y)
		nabla_b[-1] = delta_curr
		nabla_w[-1] = np.dot(delta_curr, activations[-2].transpose())
		for i in xrange(2, layers):
			z = zs[-i]
			delta_curr = np.dot(self.weights[-i + 1].transpose(), delta_curr) * sigma_prime(z)
			nabla_b[-i] = delta_curr
			nabla_w[-i] = np.dot(delta_curr, activations[-i - 1].transpose())

		return (nabla_b, nabla_w)

	def sgd(self, training_data, mini_batch_size, num_of_epoches, learning_rate, test_data=None):
		# define mini_batches
		position = 0
		n = len(training_data)
		mini_batches = []
		if test_data:
			n_test = len(test_data)
		for i in xrange(num_of_epoches):
			# shuffle and sample data
			random.shuffle(training_data)
			mini_batches = [training_data[k : k + mini_batch_size]
			for k in xrange(0, n, mini_batch_size)]
			# update weight and biases for each mini_batch
			for mini_batch in mini_batches:
				if (i < 15):
					self.update_mini_batch(mini_batch, learning_rate[0])
				else:
					self.update_mini_batch(mini_batch, learning_rate[1])
			# after the epoch is done training, test on the validation set
			if (test_data != None):
				print "Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), n_test)
			else:
				print "Epoch {0} complete".format(i)


	def update_mini_batch(self, mini_batch, learning_rate):
		# find all the gradients from back prop
		size = len(mini_batch)
		nabla_b = [np.zeros(nb.shape) for nb in self.biases]
		nabla_w = [np.zeros(nw.shape) for nw in self.weights]
		for x, y in mini_batch:
			delta_b, delta_w = self.back_propagation(x, y)
			nabla_b = [nb + db for  nb, db in zip(nabla_b, delta_b)]
			nabla_w = [nw + dw for nw, dw in zip(nabla_w, delta_w)]
		
		# update weights and biases
		self.biases = [cb - (learning_rate/size)*db for cb, db in zip(self.biases, nabla_b)]
		self.weights = [cw - (learning_rate/size)*dw for cw, dw in zip(self.weights, nabla_w)]

	def evaluate(self, test_data):
		test_results = [(np.argmax(self.feed_forward(x)), y)
						for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)



def cost_function_prime(a, y):
	# something
	return a - y
	# return a - z

'''
def softmax_prime(z):
	sumz = np.sum(np.exp(z))
	results = np.zeros(z.shape)
	for i in xrange(len(z)):
			results[i] += (np.exp(z[i])/sumz - (np.exp(z[i])**2)/(sumz**2))
			for j in xrange(len(z)):
				if (j != i):
					results[j] -= np.exp(z[i])*np.exp(z[j])/(sumz**2)
	return results
'''

def softmax(z):
	sumz = np.sum(np.exp(z))
	activations = np.zeros(z.shape)
	for i in xrange(len(z)):
			activations[i] = np.exp(z[i])/sumz
	return activations

def sigma(z):
	return sigmoid(z)

def sigma_prime(z):
	return sigmoid_prime(z)


def sigmoid(z):
	"""The sigmoid function."""
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	"""Derivative of the sigmoid function."""
	return sigmoid(z)*(1-sigmoid(z))


def relu(z):
	result = np.zeros(z.shape)
	for i in xrange(len(z)):
		if (z[i] > 0):
			result[i] = z[i]
		else:
			result[i] = 0
	return result

def relu_prime(z):
	result = np.zeros(z.shape)
	for i in xrange(len(z)):
		if (z[i] > 0):
			result[i] = 1
		else:
			result[i] = 0
	return result












