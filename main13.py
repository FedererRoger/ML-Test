import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
#import math

X_origin = pd.read_csv('gbm-data.csv')
y_origin = X_origin['Activity']
X_origin = X_origin.drop(['Activity'], axis=1)

X = X_origin.values
y = y_origin.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

#l_rates = [1.0, 0.5, 0.3, 0.2, 0.1]
l_rates = [0.2]
for l_rate in l_rates:
    GBC = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate = l_rate)
    GBC.fit(X_train, y_train)
    train_loss = []
    test_loss = []
    for y_pred in GBC.staged_predict_proba(X_train):
        train_loss.append(log_loss(y_train, y_pred[:,1]))
    for y_pred in GBC.staged_predict_proba(X_test):
        test_loss.append(log_loss(y_test, y_pred[:,1]))
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()

min_i = 0
min_loss = 100.0
for i, val in enumerate(test_loss):
    if val < min_loss:
        min_loss = val
        min_i = i+1

print(str(round(min_loss, 2)) + ' ' + str(min_i))

RF = RandomForestClassifier(n_estimators=min_i, random_state=241)
RF.fit(X_train, y_train)
pred = RF.predict_proba(X_test)
y_pred_rf = [y_p[1] for y_p in pred]
rf_loss = log_loss(y_test, y_pred_rf)

print(str(round(rf_loss, 2)))
