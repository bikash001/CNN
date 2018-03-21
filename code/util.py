import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle


def normalize_data(base, scaled=False, file=None):
	if base[-1] != '/':
		base += '/'

	if file is not None:
		with open('scaler.pkl', 'rb') as fp:
			scaler = pickle.load(fp)

		data = np.loadtxt(file, np.str, delimiter=',')[1:, 1:].astype(np.float32)
		x_val = scaler.transform(data[:, :-1])
		y_val = data[:, -1].astype(np.int)
		return x_val, y_val

	else:
		if scaled:
			with open('scaler.pkl', 'rb') as fp:
				scaler = pickle.load(fp)

			data = np.loadtxt(base+'train.csv', np.str, delimiter=',')[1:,1:].astype(np.float32)
			x_train = scaler.transform(data[:, :-1])
			y_train = data[:, -1].astype(np.int)

			data = np.loadtxt(base+'val.csv', np.str, delimiter=',')[1:, 1:].astype(np.float32)
			x_val = scaler.transform(data[:, :-1])
			y_val = data[:, -1].astype(np.int)

			data = np.loadtxt(base+'test.csv', np.str, delimiter=',')[1:, 1:].astype(np.float32)
			x_test = scaler.transform(data)
			
			return (x_train, y_train), (x_val, y_val), x_test
		else:
			data = np.loadtxt(base+'train.csv', np.str, delimiter=',')[1:,1:].astype(np.float32)
			scaler = StandardScaler()
			scaler.fit(data[:, :-1])
			x_train = scaler.transform(data[:, :-1])
			y_train = data[:, -1].astype(np.int)

			data = np.loadtxt(base+'val.csv', np.str, delimiter=',')[1:, 1:].astype(np.float32)
			x_val = scaler.transform(data[:, :-1])
			y_val = data[:, -1].astype(np.int)

			data = np.loadtxt(base+'test.csv', np.str, delimiter=',')[1:, 1:].astype(np.float32)
			x_test = scaler.transform(data)

			with open('scaler.pkl', 'wb') as fp:
				pickle.dump(scaler, fp)

			return (x_train, y_train), (x_val, y_val), x_test

if __name__ == '__main__':
	normalize_data('../data/')