"""
	Augment data using: 
		fliplr
		pad
		gaussian blur
		edge detect
		additive Gaussian noise
		coarse dropout
		affine scale
		affine translate 
"""

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

import matplotlib.pyplot as plt
import csv

def show_img(img):
	plt.imshow(np.reshape(img, (28,28)).astype(dtype=float), cmap='gray')
	plt.show()

def generate_augs(imgs, labels):
	# Flips
	aug = iaa.Fliplr(1.0)
	flipped = aug.augment_images(lib_friendly_x)

	# Pad
	aug = iaa.CropAndPad(percent=(-0.25, 0.25), pad_mode='edge')
	cped = aug.augment_images(lib_friendly_x)

	# Gaussian Blur
	aug = iaa.GaussianBlur(sigma=(0.0,3.0))
	blurred = aug.augment_images(lib_friendly_x)

	# edge detect
	# aug  = iaa.EdgeDetect(alpha=1.0)
	# edged = aug.augment_images(lib_friendly_x)

	# Additive Gaussian Noise
	aug = iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))
	noised = aug.augment_images(lib_friendly_x)

	# Coarse Dropout
	aug = iaa.CoarseDropout(0.1, size_percent=0.5)
	dropped = aug.augment_images(lib_friendly_x)

	# affine scale
	# aug = iaa.Affine(scale=(0.5,1.2))
	# scaled = aug.augment_images(lib_friendly_x)

	# affine translate
	# aug = iaa.Affine(translate_percent={"x":(-0.2,0.2), "y": (-0.2, 0.2)})
	# translated = aug.augment_images(lib_friendly_x)

	all_imgs = flipped + cped + blurred + edged + noised + dropped + scaled + translated
	flattened_imgs = [np.reshape(img, (28,28)) for img in all_imgs]
	labels_imgs    = [label for i in range(8) for label in labels]

	return flattened_imgs, labels_imgs

# data = np.loadtxt('../data/train.csv', np.str, delimiter=',')[1:, 1:].astype(np.float32)
with open('../data/train.csv') as fp:
	csv_reader = csv.reader(fp)
	csv_reader.next()
	data = np.array([csv_reader.next()[1:]])

data_x = data[:,:-1]
data_y = data[:,-1]

lib_friendly_x = [np.reshape(x, (28,28,1)).astype(dtype=float) for x in data_x]

aug_imgs, labels = generate_augs(lib_friendly_x, data_y)

data = np.array([np.append(img, label) for img, label in zip(aug_imgs, labels)]).astype(dtype=float)

np.savetxt('aug.csv', data, delimiter=',', fmt="%.3f")