import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def get_data():

	ethDataFromCSV = np.genfromtxt("ethereum.csv", delimiter=",")
	ethData = np.array(ethDataFromCSV[:,2])

	btcDataFromCSV = np.genfromtxt("bitcoin.csv", delimiter=",")
	btcData = np.array(btcDataFromCSV[:,1])

	n = btcData.size
	X = np.ones((n-10,20))
	Y = np.array(ethData[10:n])
	Y.reshape(-1, 1)

	for i in range(0,n-10):
		X[i,0:10] = np.array(btcData[i:i+10])
		X[i,10:20] = np.array(ethData[i:i+10])
		
	X_train, X_test, y_train, y_test = train_test_split(X, Y , test_size=0.05, random_state=42)
	return X_train, X_test, y_train, y_test

def main():

	X_train, X_test, y_train, y_test = get_data()
	regr = RandomForestRegressor(max_depth=20, random_state=42)
	regr.fit(X_train, y_train)
	RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
	           max_features='auto', max_leaf_nodes=None,
	           min_impurity_decrease=0.0, min_impurity_split=None,
	           min_samples_leaf=1, min_samples_split=2,
	           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
	           oob_score=False, random_state=42, verbose=0, warm_start=False)
	prediction = regr.predict(X_test) - y_test
	print(prediction.sum())
	print(regr.predict(X_test))
	print(y_test)

if __name__ == '__main__':
    main()