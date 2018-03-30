"""
	Code to train, test and deploy a CNN using tensorflow

	Rahul Kejriwal, CS14B023
	Bikash Gogoi, CS14B039
"""

# Python Lib Imports
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import os

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

# Custom Imports
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
	CNN Network Class
"""
class Net(nn.Module):


	"""
		Declare Parameters		
	"""
	def __init__(self, initializer):
		super(Net, self).__init__()

		self.fsize = 5
		self.padding = 2

		self.conv1 = nn.Conv2d(1,64,self.fsize,padding=self.padding)
		self.conv2 = nn.Conv2d(64,128,self.fsize,padding=self.padding)
		self.conv3 = nn.Conv2d(128,256,self.fsize,padding=self.padding)
		self.conv4 = nn.Conv2d(256,256,self.fsize,padding=self.padding)
		self.fc1 = nn.Linear(256*4*4, 1024)
		self.drop1 = nn.Dropout(p=0.7)
		self.fc2 = nn.Linear(1024, 1024)
		self.drop2 = nn.Dropout(p=0.7)
		self.bn = nn.BatchNorm1d(1024)
		self.fc3 = nn.Linear(1024, 10)

		for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.fc1, self.fc2, self.fc3]:
			initializer(layer.weight)
			layer.bias.data.fill_(0.01)

	"""
		Forward Propagate Input
	"""
	def forward(self, x):
		x = F.max_pool2d(F.rrelu(self.conv1(x)), (2,2))
		x = F.max_pool2d(F.rrelu(self.conv2(x)), (2,2))
		x = F.rrelu(self.conv3(x))
		x = F.max_pool2d(F.rrelu(self.conv4(x)), (2,2), padding=1)
		x = x.view(-1, self.num_flat_features(x))
		x = F.rrelu(self.fc1(x))
		x = self.drop1(x)
		x = F.rrelu(self.fc2(x))
		x = self.drop2(x)
		x = self.bn(x)
		x = F.softmax(self.fc3(x))
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]
		nf = 1
		for s in size:
			nf *= s
		return nf


"""
	Main 
"""
if __name__ == '__main__':

	"""
		CONFIG vars
	"""
	FIRST = True
	USE_CUDA = True
	TRAIN =  True
	NUM_EPOCHS = 15

	# Parse cmdline args
	args = parse_cmd_args()

	# Get train, val and test data
	if FIRST:
		train, val, test = normalize_data('../data/')
		np.save('train_x', train[0])
		np.save('train_y', train[1])
		np.save('val_x', val[0])
		np.save('val_y', val[1])
		np.save('test_x', test)
	else:
		train = (np.load('train_x.npy'), np.load('train_y.npy'))
		val = (np.load('val_x.npy'), np.load('val_y.npy'))
		test = np.load('test_x.npy')

	print("Data Normalization Complete!")


	# Find Initializer
	if args.init == 1:
		initializer = nn.init.xavier_normal
	elif args.init == 2:
		initializer = nn.init.kaiming_normal
	else:
		raise NotImplementedError


	# Build objects
	net = Net(initializer)
	if USE_CUDA:
		net.cuda()
	optimizer = optim.SGD(net.parameters(), lr=args.lr)
	criterion = nn.CrossEntropyLoss()

	if TRAIN:
		x_train = train[0]
		y_train = train[1]

		# Training Constants
		num_epochs = NUM_EPOCHS
		bs = args.batch_size
		xs = x_train.shape[0]
		num_batches = int(np.ceil(float(xs)/bs))

		"""
			Training Phase
		"""
		loss_data = []
		for epoch in tqdm(range(num_epochs)):
			for i in tqdm(range(num_batches)):
				x = np.reshape(x_train[i*bs:(i+1)*bs], (-1,1,28,28))
				x = Variable(torch.from_numpy(x))
				y = Variable(torch.from_numpy(np.array([y_train[i*bs:(i+1)*bs]]).reshape((-1,))))

				if USE_CUDA:
					x = x.cuda()
					y = y.cuda()

				optimizer.zero_grad()
				output = net(x)
				loss = criterion(output, y)
				loss.backward()
				optimizer.step()

		# Save Model
		torch.save(net.state_dict(), args.save_dir+'/model')

	# Load Model
	net.load_state_dict(torch.load(args.save_dir+'/model'))


	"""
		Eval Phase
	"""
	
	# Switch Model to Eval Mode
	net.eval()

	correct = 0
	x_val = val[0]
	y_val = val[1]
	for i in tqdm(range(y_val.shape[0])):
		x = np.reshape(x_val[i], (1,1,28,28))
		x = Variable(torch.from_numpy(x), volatile=True)
		
		if USE_CUDA:
			x = x.cuda()

		pred = net(x)
		pred = pred.data.cpu().numpy()
		if np.argmax(pred) == y_val[i]:
			correct += 1

	net.train()
	print (float(correct) / y_val.shape[0])


	"""
		Test Phase
	"""
	results = [['id', 'label']]
	for i, x in tqdm(enumerate(test)):
		x = np.reshape(x, (1,1,28,28))
		x = Variable(torch.from_numpy(x), volatile=True)
		if USE_CUDA:
			x = x.cuda()

		pred = net(x)
		pred = pred.data.cpu().numpy()
		results.append([i, int(np.argmax(pred))])

	dir_name = args.save_dir
	if not os.path.isdir(dir_name):
		os.makedirs(dir_name)
	if dir_name[-1] != '/':
		dir_name = dir_name+'/'
	np.savetxt(dir_name+'test_submission.csv', np.array(results, np.str), '%s', ',')