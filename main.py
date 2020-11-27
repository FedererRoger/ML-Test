import numpy as np
import sklearn.datasets
import sklearn.model_selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale

Z = sklearn.datasets.load_boston()
X = Z['data']
Y = Z['target']
X = sklearn.preprocessing.scale(X)
CV = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
A = np.empty(200)
B = np.empty(200)
p_metric = np.linspace(1, 20, 200)
for i in range(len(p_metric)):
    clf = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p_metric[i])
    quality = sklearn.model_selection.cross_val_score(clf, X, Y, scoring='neg_mean_squared_error')
    q_mean = np.mean(quality)
    A[i] = q_mean
    B[i] = i
Max = A.max()
Index_max = np.argmax(A)
print(Max, p_metric[Index_max])