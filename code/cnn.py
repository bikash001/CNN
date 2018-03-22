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
from util import normalize_data

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

	def __init__(self, lr=0.001, batch_size=20, init_method=1, save_dir=None, steps=1000, dropout=True):
		self.lr = lr
		self.batch_size = batch_size
		self.init_method = init_method
		self.save_dir = save_dir
		self.max_steps = steps
		self.apply_dropout = dropout
		if self.init_method == 1:
			self.kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
		else:
			self.kernel_initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")



	"""
		Raw CNN API for Training, Validating and Testing 
	"""
	def __model_fn(self, features, labels, mode):
		# print "inside model function"

		# Input Layer
		input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

		# CONV1 Layer
		conv1 = tf.layers.conv2d(
			inputs= input_layer,
			filters=64,
			kernel_size=[3,3],
			padding="same",
			activation=tf.nn.relu,
			kernel_initializer=self.kernel_initializer)

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
			padding="valid",
			activation=tf.nn.relu,
			kernel_initializer=self.kernel_initializer)

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
			activation=tf.nn.relu,
			kernel_initializer=self.kernel_initializer)

		# CONV4 Layer
		conv4 = tf.layers.conv2d(
			inputs= conv3,
			filters=256,
			kernel_size=[3,3],
			padding="same",
			activation=tf.nn.relu,
			kernel_initializer=self.kernel_initializer)

		# POOL3 Layer
		pool3 = tf.layers.max_pooling2d(
			inputs=conv4,
			pool_size=[2,2],
			strides=2)

		# Flatten Output
		pool3_flat = tf.reshape(pool3, [-1, 3*3*256])

		# FC1 Layer
		fc1 = tf.layers.dense(
			inputs=pool3_flat,
			units=1024,
			activation=tf.nn.relu,
			kernel_initializer=self.kernel_initializer)

		do1 = tf.layers.dropout(
			inputs=fc1,
			training=self.apply_dropout)

		# FC2 Layer
		fc2 = tf.layers.dense(
			inputs=do1,
			units=1024,
			activation=tf.nn.relu,
			kernel_initializer=self.kernel_initializer)

		do2 = tf.layers.dropout(
			inputs=fc2,
			training=self.apply_dropout)

		# Batch Normalization Layer
		# bn = fc2
		bn = tf.layers.batch_normalization(
			inputs=do2,
			training=(mode == tf.estimator.ModeKeys.TRAIN))

		# SOFTMAX Layer
		logits = tf.layers.dense(
			inputs=bn,
			units=10,
			kernel_initializer=self.kernel_initializer)

		# Make predictions from model
		predictions = {
			"classes": tf.argmax(input=logits, axis=1),
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
			optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
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
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


	def fit(self, X, Y):
		classifier = tf.estimator.Estimator(model_fn=self.__model_fn,
			model_dir=self.save_dir)

		#logging
		tensors_to_log = {"probabilities": "softmax_tensor"}
		logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
			every_n_iter=50)

		#train model
		train_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={'x': X}, y=Y, batch_size=self.batch_size,
			num_epochs=None, shuffle=True)

		classifier.train(input_fn=train_input_fn,
			steps=self.max_steps, hooks=[logging_hook])

		self.__classifier = classifier
		return self


	def run_save(self):
		classifier = tf.estimator.Estimator(model_fn=self.__model_fn,
			model_dir=self.save_dir, warm_start_from='../model/')

		self.__classifier = classifier
		return self

	def predict(self, X):
		self.apply_dropout = False
		pr_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={'x': X},
			shuffle=False)

		eval_results = self.__classifier.predict(input_fn=pr_input_fn)
		return [x['classes'] for x in eval_results]

	def accuracy(self, X, Y):
		self.apply_dropout = False
		eval_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={'x': X},
			y=Y,
			num_epochs=1,
			shuffle=False)

		eval_results = self.__classifier.evaluate(input_fn=eval_input_fn)
		return eval_results


if __name__ == '__main__':

	args = parse_cmd_args()

	model = CNN(args.lr, args.batch_size, args.init, args.save_dir,  30000)
	
	train, val, test = normalize_data('../data/', scaled=True)

	# model.run_save()
	# val = normalize_data('../data/', scaled=True, file='../data/val.csv')
	print 'train:', model.fit(*train).accuracy(*train)
	print 'val:', model.accuracy(*val)
	# test = normalize_data('../data/', True, '../data/test.csv', True)
	rs = model.predict(test)
	with open('../data/result.csv', 'w') as fp:
		print >> fp, 'id,label'
		for i, x in enumerate(rs):
			print >> fp, '%d,%d' %(i,x)
