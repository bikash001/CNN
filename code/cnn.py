"""
	Code to train, test and deploy a CNN using tensorflow

	Acknowledgements: 
		Tutorials: https://www.tensorflow.org/tutorials/layers

	Rahul Kejriwal, CS14B023
	Bikash Gogoi, CS14B039
"""

# Python Lib Imports
import numpy as np
from argparse import ArgumentParser

# Custom Imports
import tensorflow as tf


"""
	Function to check if string is 1 or multiple of 5
"""
def mul_5(string):
	val = int(string)
	if val == 1 or val % 5 == 0:	return val
	else:	raise ArgumentTypeError("%r is not a multiple of 5" % string)


"""
	Parses cmd line args
"""
def parse_cmd_args():

	# Create Parser
	parser = ArgumentParser()

	# Add arguments
	parser.add_argument("--lr", help="initial learning rate for gradient descent based algorithms", type=float, required=True)
	parser.add_argument("--batch_size", help="batch size to be used - valid values are 1 and multiples of 5", type=mul_5, required=True)
	parser.add_argument("--init", help="initialization method to be used - 1 for Xavier, 2 for He", type=int, choices=[1,2], required=True)
	parser.add_argument("--save_dir", help="directory where pickled model should be stored", type=str, required=True)

	# Parse args
	args = parser.parse_args()
	return args


"""
	CNN Model
"""
class CNN:

	def __init__(self, lr, batch_size, init_method):
		self.lr = lr
		self.batch_size = batch_size
		self.init_method = init_method


	"""
		Raw CNN API for Training, Validating and Testing 
	"""
	def model_fn(features, labels, mode):

		# Input Layer
		input_layer = tf.reshape(features, [-1, 28, 28, 1])

		# CONV1 Layer
		conv1 = tf.layers.conv2d(
			inputs= input_layer,
			filters=64,
			kernel_size=[3,3],
			padding="same",
			activation=tf.nn.relu)

		# POOL1 Layer
		pool1 = tf.layers.max_pooling2d(
			inputs=conv1,
			pool_size=[2,2],
			strides=2)

		# CONV2 Layer
		conv2 = tf.layers.conv2d(
			inputs= pool1,
			filters=128,
			kernel_size=[3,3],
			padding="same",
			activation=tf.nn.relu)

		# POOL2 Layer
		pool2 = tf.layers.max_pooling2d(
			inputs=conv2,
			pool_size=[2,2],
			strides=2)

		# CONV3 Layer
		conv3 = tf.layers.conv2d(
			inputs= pool2,
			filters=256,
			kernel_size=[3,3],
			padding="same",
			activation=tf.nn.relu)

		# CONV4 Layer
		conv4 = tf.layers.conv2d(
			inputs= conv3,
			filters=256,
			kernel_size=[3,3],
			padding="same",
			activation=tf.nn.relu)

		# POOL3 Layer
		pool3 = tf.layers.max_pooling2d(
			inputs=conv4,
			pool_size=[2,2],
			strides=2)

		# Flatten Output
		pool3_flat = tf.reshape(pool3, [-1, 4*4*256])

		# FC1 Layer
		fc1 = tf.layers.dense(
			inputs=pool3_flat,
			units=1024,
			activation=tf.nn.relu)

		# FC2 Layer
		fc2 = tf.layers.dense(
			inputs=fc1,
			units=1024,
			activation=tf.nn.relu)

		# Batch Normalization Layer
		bn = fc2

		# SOFTMAX Layer
		logits = tf.layers.dense(
			inputs=bn,
			units=10)

		# Make predictions from model
		predictions = {
			"classes": tf.argmax(input=logits, axis=1)
			"probabilities": tf.nn.softmax(logits, name='softmax_tensor')
		}

		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

		# Find loss for model
		onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
		loss = tf.losses.softmax_cross_entropy(
			onehot_labels=onehot_labels,
			logits=logits)

		# Train model
		if mode == tf.estimator.ModeKeys.TRAIN:
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=)
			train_op = optimizer.minimize(
				loss=loss,
				global_step=tf.train.get_global_step())
			return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

		# Validate Model
		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(
				labels=labels,
				predictions=predictions["classes"])
		}
		return tf.estimator.EstimatorSpec(mode=mode, eval_metric_ops=eval_metric_ops)


	def train():
		pass

if __name__ == '__main__':

	args = parse_cmd_args()

