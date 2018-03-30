import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time

class CNN:
	def __init__(self, lr=0.001, batch_size=20, init_method=1, save_dir=None, steps=1000, dropout=True):
		self.lr = lr
		self.batch_size = batch_size
		self.init_method = init_method
		self.save_dir = save_dir
		self.max_steps = steps
		self.dropout = dropout
		if self.init_method == 1:
			self.kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
		else:
			self.kernel_initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

		self.test_logits = None
		self.train_logits = None
		self.session = None
		# self.__initialize__()

	def __del__(self):
		if self.session is not None:
			if self.save_dir is not None:
				saver = tf.train.Saver()
				saver.save(self.session, 'model', self.steps)
			self.session.close()

	def __initialize__(self, is_training, dropout):
		self.x = tf.placeholder(tf.float32, shape=[None, 784], name='input')
		x_ = tf.reshape(self.x, [-1, 28, 28, 1])
		# x = tf.placeholder(tf.float32, shape=[None, 28,28,1])
		self.y = tf.placeholder(tf.int32, shape=[None,], name='output')
		self.apply_dropout = tf.Variable(dropout)
		self.is_training = tf.Variable(is_training)

		# CONV1 Layer
		conv1 = tf.layers.conv2d(
			inputs= x_,
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
			training=self.is_training)

		# SOFTMAX Layer
		logits = tf.layers.dense(
			inputs=bn,
			units=10,
			kernel_initializer=self.kernel_initializer)

		return logits

	def fit(self, X, Y, val_x, val_y):
		logits = self.__initialize__(True, self.dropout)
		cross_entropy = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.y,depth=10) , logits=logits))

		train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cross_entropy)
		
		self.prediction = tf.argmax(logits,1, output_type=tf.int32)
		self.predictor = tf.equal(self.prediction, self.y, name='predictor')
		self.accuracy = tf.reduce_mean(tf.cast(self.predictor, tf.float32), name='accuracy')

		sess = tf.Session()	
		self.session = sess
		sess.run(tf.global_variables_initializer())
		
		# training
		dataset = tf.data.Dataset.from_tensor_slices((X, Y))
		dataset = dataset.shuffle(1000).batch(self.batch_size)
			
		with sess.as_default():
			self.step = 0
			epoch = 0
			max_epoch = 10
			while self.step < self.max_steps:
				epoch += 1
				iterator = dataset.make_one_shot_iterator()
				next_el = iterator.get_next()
				while True:
					self.step += 1
					try:
						train_x, train_y = sess.run(next_el)
						train_step.run(feed_dict={self.x: train_x, self.y:train_y, self.apply_dropout: self.dropout, self.is_training: True})
					except tf.errors.OutOfRangeError:
						break
					# if self.step > self.max_steps:
					# 	break
				print 'epoch: %d --> %f' %(epoch, self.predict_score(val_x, val_y))
				if epoch >= max_epoch:
					break
		


	def predict_score(self, X, Y):
		#testing
		dataset = tf.data.Dataset.from_tensor_slices((X,Y))
		dataset = dataset.batch(100)
		iterator = dataset.make_one_shot_iterator()
		next_el = iterator.get_next()
		
		ac = 0.
		result = np.empty((0,), dtype=np.int)
		# sess = tf.Session()	
		if self.session is not None:
			sess = self.session
		elif self.save_dir is not None:
			sess = tf.Session()
			saver = tf.train.import_meta_graph(self.save_dir+'model-'+str(self.step)+'.meta')
			saver.restore(sess,tf.train.latest_checkpoint(self.save_dir))
			graph = tf.get_default
		else:
			raise ValueError('run fit or provide directory for saved model')
		
		with sess.as_default():
			while True:
				try:
					test_x, test_y = sess.run(next_el)
					a = self.predictor.eval(feed_dict={self.x: test_x, self.y: test_y, self.apply_dropout: False, self.is_training: False})
					# print(a.shape)
					result = np.concatenate((result,a))
				except tf.errors.OutOfRangeError:
					break
				
			ac =  self.accuracy.eval(feed_dict={self.predictor: np.array(result)})
		return ac

	def predict(self, X):
		dataset = tf.data.Dataset.from_tensor_slices(X)
		dataset = dataset.batch(100)
		iterator = dataset.make_one_shot_iterator()
		next_el = iterator.get_next()
		
		ac = 0.
		result = np.empty((0,), dtype=np.int)
		# sess = tf.Session()	
		sess = self.session
		with sess.as_default():
			while True:
				try:
					test_x = sess.run(next_el)
					a = self.prediction.eval(feed_dict={self.x: test_x, self.apply_dropout: False, self.is_training: False})
					# print(a.shape)
					result = np.concatenate((result,a))
				except tf.errors.OutOfRangeError:
					break
				
		return result


def cnn(train, test, init_method=2, batch_size=10, lr=0.001, max_step=100, dropout=False):
	apply_dropout = False
	x_ = tf.placeholder(tf.float32, shape=[None, 784])
	x = tf.reshape(x_, [-1, 28, 28, 1])
	# x = tf.placeholder(tf.float32, shape=[None, 28,28,1])
	y_ = tf.placeholder(tf.int32, shape=[None,])
	
	if init_method == 1:
		kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
	else:
		kernel_initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

	# CONV1 Layer
	conv1 = tf.layers.conv2d(
		inputs= x,
		filters=64,
		kernel_size=[3,3],
		padding="same",
		activation=tf.nn.relu,
		kernel_initializer=kernel_initializer)

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
		kernel_initializer=kernel_initializer)

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
		kernel_initializer=kernel_initializer)

	# CONV4 Layer
	conv4 = tf.layers.conv2d(
		inputs= conv3,
		filters=256,
		kernel_size=[3,3],
		padding="same",
		activation=tf.nn.relu,
		kernel_initializer=kernel_initializer)

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
		kernel_initializer=kernel_initializer)

	do1 = tf.layers.dropout(
		inputs=fc1,
		training=apply_dropout)

	# FC2 Layer
	fc2 = tf.layers.dense(
		inputs=do1,
		units=1024,
		activation=tf.nn.relu,
		kernel_initializer=kernel_initializer)

	do2 = tf.layers.dropout(
		inputs=fc2,
		training=apply_dropout)

	# Batch Normalization Layer
	# bn = fc2
	bn = tf.layers.batch_normalization(
		inputs=do2,
		training=True)

	# SOFTMAX Layer
	logits = tf.layers.dense(
		inputs=bn,
		units=10,
		kernel_initializer=kernel_initializer)

	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y_,depth=10) , logits=logits))

	train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)
	
	correct_prediction = tf.equal(tf.argmax(logits,1, output_type=tf.int32), y_)

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	
	sess = tf.Session()	
	sess.run(tf.global_variables_initializer())
	
	# training
	dataset = tf.data.Dataset.from_tensor_slices(train)
	dataset = dataset.shuffle(1000).repeat().batch(20)
	iterator = dataset.make_one_shot_iterator()
	next_el = iterator.get_next()
	
	with sess.as_default():
		for _ in range(max_step):
			train_x, train_y = sess.run(next_el)
			# train_x = np.reshape(train_x, [-1,28,28,1])
			train_step.run(feed_dict={x_: train_x, y_:train_y})

	#testing
	dataset = tf.data.Dataset.from_tensor_slices(val)
	dataset = dataset.batch(20)
	iterator = dataset.make_one_shot_iterator()
	next_el = iterator.get_next()
	
	result = np.empty((0,), dtype=np.int)
	i = 0
	with sess.as_default():
		# print(accuracy.eval(feed_dict={x_: val[0][:100], y_:val[1][:100]}))
		while True:
			i += 1
			try:
				test_x, test_y = sess.run(next_el)
				a = correct_prediction.eval(feed_dict={x_: test_x, y_: test_y})
				# print(a.shape)
				result = np.concatenate((result,a))
			except tf.errors.OutOfRangeError:
				break
			
		print(accuracy.eval(feed_dict={correct_prediction: np.array(result)}))
	# sess.run(tf.global_variables_initializer())
	sess.close()


if __name__ == '__main__':
	# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	# print mnist.train.labels.shape
	# base = '../data/'
	# train = (np.load(base+'train_x.npy'), np.load(base+'train_y.npy'))
	# val = (np.load(base+'val_x.npy'), np.load(base+'val_y.npy'))
	# # cnn(train, val, max_step=10000)
	# net = CNN(steps=100000)
	# net.fit(train[0], train[1], val[0], val[1])
	# # print net.predict_score(val[0], val[1])
	# print 'testing......'
	# ar = net.predict(np.load(base+'test_x.npy'))
	# np.save('test_result', ar)
	# print time.clock()

	# ar = np.load('test_result.npy')
	# with open('../data/result.csv', 'w') as f:
	# 	print >> f, 'id,label'
	# 	for i, val in enumerate(ar):
	# 		print >> f, '%d,%d' %(i, val)
