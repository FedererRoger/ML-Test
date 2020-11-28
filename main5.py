import numpy as np
import pandas
import sklearn
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

clf = Perceptron()
scaler = StandardScaler()

data_train = pandas.read_csv('perceptron-train.csv', names=['class', 'p1', 'p2'])
data_test = pandas.read_csv('perceptron-test.csv', names=['class', 'p1', 'p2'])
X_train = data_train[['p1', 'p2']]
X_test = data_test[['p1', 'p2']]
Y_train = data_train['class']
Y_test = data_test['class']

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#clf.fit(X_train, Y_train)
#prediction = clf.predict(X_test)
clf.fit(X_train_scaled, Y_train)
prediction = clf.predict(X_test_scaled)
quality = sklearn.metrics.accuracy_score(Y_test, prediction)

print(quality)