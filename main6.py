import numpy as np
import pandas
from sklearn.svm import SVC

clf = SVC(C=100000, kernel='linear', random_state=241)

data = pandas.read_csv('svm-data.csv', names=['class', 'p1', 'p2'])
X = data[['p1', 'p2']]
Y = data['class']

clf.fit(X, Y)
Index = clf.support_
print(Index)