import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier
#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
X = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
X = X.dropna(axis=0)
Y = X['Survived']
X = X.drop(['Survived'], axis=1)
Sex=X['Sex']
X = X.drop(['Sex'], axis=1)
#OHE = OneHotEncoder()
LHE = LabelEncoder()
Sex = LHE.fit_transform(Sex)
X['Sex'] = Sex
#Sex = Sex.reshape(len(Sex), 1)
#Encoded = OHE.fit_transform(Sex)
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, Y)
importances =clf.feature_importances_
print(importances)
print(X)