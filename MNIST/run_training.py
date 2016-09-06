from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
"""
NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/Users/linda/Desktop/minst_tes/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/Users/linda/Desktop/minst_tes/mnist_logs', 'Summaries directory')


batch_size = 100

images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
"""
# download data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# initialize variables x, w, b
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# y = wx + b
y = tf.nn.softmax(tf.matmul(x, w) + b)

# correct answer y_
y_ = tf.placeholder(tf.float32, [None, 10])

# train to reduce cross_entropy = -sum(yi_*loy(yi))/10
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# run initialization
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# training
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# testing
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))






