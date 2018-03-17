import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def normalize_data(base):
	if base[-1] != '/':
		base += '/'

	data = np.loadtxt(base+'train.csv', np.str, delimiter=',')[1:,1:].astype(np.float32)
	scaler = StandardScaler()
	scaler.fit(data[:, :-1])
	x_train = scaler.transform(data[:, :-1])
	y_train = data[:, -1]

	data = np.loadtxt(base+'val.csv', np.str, delimiter=',')[1:, 1:].astype(np.float32)
	x_val = scaler.transform(data[:, :-1])
	y_val = data[:, -1]

	data = np.loadtxt(base+'test.csv', np.str, delimiter=',')[1:, 1:].astype(np.float32)
	x_test = scaler.transform(data)

	return (x_train, y_train), (x_val, y_val), x_test

if __name__ == '__main__':
	normalize_data('../data/')