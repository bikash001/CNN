"""
	Program to plot CONV1 filters of CNN

	Rahul Kejriwal, CS14B023
	Bikash Gogoi, CS14B039
"""

# Python Imports
from cnn2 import Net
import numpy as np
from matplotlib import pyplot as plt

# PyTorch Imports
import torch
import torch.nn as nn


"""
	Tensor is 64x3x3
"""
def plot_filters(tensor, grid_cols=8):
	# print tensor.shape

	num_filters = tensor.shape[0]
	num_rows = np.ceil(num_filters / grid_cols)

	fig = plt.figure(figsize=(num_rows, grid_cols))
	for i in range(num_filters):
		subfig = fig.add_subplot(num_rows, grid_cols, i+1)
		subfig.imshow(tensor[i].reshape((3,3)), cmap='Greys', interpolation='none')
		subfig.axis('off')
		subfig.set_xticklabels([])
		subfig.set_yticklabels([])

	plt.subplots_adjust(wspace=0.3, hspace=0.3)
	plt.show()


if __name__ == "__main__":

	# Model save directory
	# MODEL_LOCATION = '../subs/3/net.pth'
	MODEL_LOCATION = './net.pth'

	# Load Model
	model = Net(nn.init.kaiming_normal)
	model.load_state_dict(torch.load(MODEL_LOCATION))

	# Get CONV1 weights
	layer1_filters = model.conv1.weight.data.numpy()

	# Normalize the weights to [0,255]
	max_val = np.argmax(layer1_filters)
	layer1_filters = layer1_filters / max_val * 127 + 127

	# Plot each filter
	plot_filters(layer1_filters)