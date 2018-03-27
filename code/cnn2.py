"""
	Code to train, test and deploy a CNN using tensorflow

	Rahul Kejriwal, CS14B023
	Bikash Gogoi, CS14B039
"""

# Python Lib Imports
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

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


class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1,64,3,padding=1)
		self.conv2 = nn.Conv2d(64,128,3,padding=1)
		self.conv3 = nn.Conv2d(128,256,3,padding=1)
		self.conv4 = nn.Conv2d(256,256,3,padding=1)
		self.fc1 = nn.Linear(256*3*3, 1024)
		self.fc2 = nn.Linear(1024, 1024)
		self.bn = nn.BatchNorm1d(1024)
		self.fc3 = nn.Linear(1024, 10)

	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
		x = F.relu(self.conv3(x))
		x = F.max_pool2d(F.relu(self.conv4(x)), (2,2))
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		# Batch Normalization Layer
		x = self.bn(x)
		x = F.softmax(self.fc3(x))
		
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]
		nf = 1
		for s in size:
			nf *= s
		return nf

def xavier_init_weights(m):
	nn.init.xavier_normal(m.weight) 
	m.bias.data.fill_(0.01)

def he_init_weights(m):
	nn.init.kaiming_normal(m.weight) 
	m.bias.data.fill_(0.01)	

"""
	Main 
"""
if __name__ == '__main__':
	# torch.set_default_tensor_type('torch.cuda.FloatTensor')

	# Parse cmdline args
	args = parse_cmd_args()

	# Get train, val and test data
	# train, val, test = normalize_data('../data/', scaled=True)
	# np.save('train_x', train[0])
	# np.save('train_y', train[1])
	# np.save('val_x', val[0])
	# np.save('val_y', val[1])
	# np.save('test_x', test)
	train = (np.load('train_x.npy'), np.load('train_y.npy'))
	val = (np.load('val_x.npy'), np.load('val_y.npy'))
	test = np.load('test_x.npy')

	print("Data Normalization Complete!")

	# Build objects
	net = Net()
	# net.cuda()
	optimizer = optim.Adam(net.parameters(), lr=args.lr)
	criterion = nn.CrossEntropyLoss()

	# Network Initialization
	# if args.init == 1:
	# 	net.apply(xavier_init_weights)
	# elif args.init == 2:
	# 	net.apply(he_init_weights)
	# else:
	# 	raise NotImplementedError

	# Training Constants
	num_epochs = 1
	bs = args.batch_size

	x_train = train[0]
	y_train = train[1]
	# shuffle data

	xs = x_train.shape[0]
	num_batches = int(np.ceil(float(xs)/bs))


	"""
		Training Phase
	"""
	for epoch in tqdm(range(num_epochs)):
		for i in tqdm(range(2)):
			x = np.reshape(x_train[i*bs:(i+1)*bs], (bs,1,28,28))
			x = torch.from_numpy(x)
			y = Variable(torch.from_numpy(np.array([y_train[i*bs:(i+1)*bs]]).reshape((20,))))

			optimizer.zero_grad()
			output = net(Variable(x))
			loss = criterion(output, y)
			loss.backward()
			optimizer.step()


	"""
		Test Phase
	"""
	correct = 0
	x_val = val[0]
	y_val = val[1]
	for i in range(y_val.shape[0]):
		x = np.reshape(x_val[i], (1,1,28,28))
		x = torch.from_numpy(x)
		pred = net(Variable(x))
		pred = pred.data.cpu().numpy()
		if np.argmax(pred) == y_val[i]:
			correct += 1

	print float(correct) / y_val.shape[0]