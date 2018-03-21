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
		x = F.softmax(self.fc3(x))
		
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]
		nf = 1
		for s in size:
			nf *= s
		return nf


if __name__ == '__main__':

	args = parse_cmd_args()

	train, val, test = normalize_data('../data/')
	print("Data Normalization Complete!")

	net = Net()
	optimizer = optim.SGD(net.parameters(), lr=args.lr)
	criterion = nn.CrossEntropyLoss()
	num_epochs = 1

	x_train = train[0][:500,:]
	y_train = train[1][:500]
	for epoch in tqdm(range(num_epochs)):
		for i in tqdm(range(y_train.shape[0])):
			x = np.reshape(x_train[i], (1,1,28,28))
			x = torch.from_numpy(x)
			y = Variable(torch.from_numpy(np.array([y_train[i]])))

			optimizer.zero_grad()
			output = net(Variable(x))
			loss = criterion(output, y)
			loss.backward()
			optimizer.step()

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