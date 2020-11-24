import pandas
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection
from sklearn.preprocessing import scale

data = pandas.read_csv('wine.data', names=['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
                                         'magnesium', 'total_phenols','flavanoids', 'nonflavanoid_phenols',
                                         'proanthocyanins', 'color_intensity', 'hue', 'of_diluted_wines', 'proline'])
Y = data['class']
X = data.drop(['class'], axis=1)
X = sklearn.preprocessing.scale(X)
CV = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
A = np.empty(50)
B = np.empty(50)
for i in range(0, 50):
    clf = KNeighborsClassifier(n_neighbors=i+1)
    quality = sklearn.model_selection.cross_val_score(clf, X, Y, cv=CV,scoring='accuracy')
    q_mean = np.mean(quality)
    A[i] = q_mean
    B[i] = i
Min = A.max()
Index_max = np.argmax(A)
k_max = B[Index_max]
print(A, k_max)