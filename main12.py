import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
data = pd.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
Y = data['Rings']
X = data.drop(['Rings'], axis=1)

CV = KFold(n_splits=5, shuffle=True, random_state=1)
A = np.empty(50)
B = np.empty(50)
for i in range(0, 50):
    enc = RandomForestRegressor(n_estimators=i+1, random_state=1)
    enc.fit(X,Y)
    Y_predictions = enc.predict(X)
    quality = cross_val_score(enc, X, Y, cv=CV, scoring='r2')
    q_mean = np.mean(quality)
    if q_mean > 0.52:
        break
print(i+1)
#Min = A.max()
#Index_max = np.argmax(A)
#k_max = B[Index_max]