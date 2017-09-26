# from __future__ import absolute_import, division, print_function
# import tensorflow as tf
# import numpy as np
# from sklearn.model_selection import train_test_split

# import tflearn

# RANDOM_SEED = 42
# tf.set_random_seed(RANDOM_SEED)

# def get_data():

# 	ethDataFromCSV = np.genfromtxt("ethereum.csv", delimiter=",")
# 	ethData = np.array(ethDataFromCSV[:,2])

# 	btcDataFromCSV = np.genfromtxt("bitcoin.csv", delimiter=",")
# 	btcData = np.array(btcDataFromCSV[:,1])

# 	n = btcData.size
# 	X = np.ones((n-10,20))
# 	Y = np.array(btcData[10:n])
# 	Y.reshape(-1, 1)

# 	for i in range(0,n-10):
# 		X[i,0:10] = np.array(btcData[i:i+10])
# 		X[i,10:20] = np.array(ethData[i:i+10])
		
# 	X_train, X_test, y_train, y_test = train_test_split(X, Y , test_size=0.1, random_state=42)
# 	return X_train, X_test, y_train.reshape(-1, 1), y_test.reshape(-1, 1)

# def main():
# 	data, data_test, labels, labels_test = get_data()
# 	net = tflearn.input_data(shape=[None, 20])
# 	net = tflearn.fully_connected(net, 32)
# 	net = tflearn.fully_connected(net, 32)
# 	net = tflearn.fully_connected(net, 1, activation='softmax')
# 	net = tflearn.regression(net)

# 	# Define model
# 	model = tflearn.DNN(net)
# 	# Start training (apply gradient descent algorithm)
# 	model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=False)

# 	prediction = model.predict(data_test);
# 	print(prediction)


# if __name__ == '__main__':
#     main()


from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

import tflearn

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def get_data():

	ethDataFromCSV = np.genfromtxt("ethereum.csv", delimiter=",")
	ethData = np.array(ethDataFromCSV[:,2])

	btcDataFromCSV = np.genfromtxt("bitcoin.csv", delimiter=",")
	btcData = np.array(btcDataFromCSV[:,1])

	n = btcData.size
	X = np.ones((n-10,20))
	Y = np.array(btcData[10:n])
	Y.reshape(-1, 1)

	for i in range(0,n-10):
		X[i,0:10] = np.array(btcData[i:i+10])
		X[i,10:20] = np.array(ethData[i:i+10])
		
	X_train, X_test, y_train, y_test = train_test_split(X, Y , test_size=0.1, random_state=42)
	return X_train, X_test, y_train.reshape(-1, 1), y_test.reshape(-1, 1)

def main():
	data, data_test, labels, labels_test = get_data()
	n_epochs = 10
	learning_rate = 0.00001

	X = tf.constant(data, dtype=tf.float32, name="X")
	y = tf.constant(labels, dtype=tf.float32, name="y")
	theta = tf.Variable(tf.random_uniform([20, 1], -1.0, 1.0, seed=42), name="theta")
	y_pred = tf.matmul(X, theta, name="predictions")
	error = y_pred - y
	mse = tf.reduce_mean(tf.square(error), name="mse")

	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	training_op = optimizer.minimize(mse)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(n_epochs):
			# print(error.eval())
			print(mse.eval())
			if epoch % 10 == 0:
			    print("Epoch", epoch, "MSE =", mse.eval())
			sess.run(training_op)
	    
		best_theta = theta.eval()

	print("Best theta:")
	print(best_theta)


if __name__ == '__main__':
    main()

