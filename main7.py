from sklearn.svm import SVC
import numpy as np
import pandas
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
X = newsgroups.data
Y = newsgroups.target
Vectorizer = TfidfVectorizer()
X_vect = Vectorizer.fit_transform(X)

#grid = {'C': np.power(10.0, np.arange(-5, 6))}
#cv = KFold(n_splits=5, shuffle=True, random_state=241)
#clf = SVC(kernel='linear', random_state=241)
#gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
#gs.fit(X_vect, Y)
#C_best = gs.best_params_

cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(C=1.0, kernel='linear', random_state=241)
clf.fit(X_vect, Y)

result_test = pandas.DataFrame(clf.coef_.todense())
result_test = result_test.abs()
result_test = result_test.sort_values(0, axis=1)
N_index = result_test.iloc[:, -10:].columns
A = []

for i in range (len(N_index)):
    num = N_index[i]
    feature_mapping = Vectorizer.get_feature_names()
    A.append(feature_mapping[num])
result = pandas.DataFrame(A,columns=["coca-cola"])
result = result.sort_values(by='coca-cola')
print(result)