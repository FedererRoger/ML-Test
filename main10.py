import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

data = pd.read_csv('salary-train.csv')
data_test = pd.read_csv('salary-test-mini.csv')

lower = []
lower_test = []
Y_train = data['SalaryNormalized']
Y_test = data_test['SalaryNormalized']
text = data['FullDescription']
text_test = data_test['FullDescription']
for i in range (len(text)):
    lower.append(text[i].lower())
for j in range(len(text_test)):
    lower_test.append(text_test[j].lower())
text = pd.DataFrame(lower, columns=['FullDescription'])
text_test = pd.DataFrame(lower_test, columns=['FullDescription'])
text = text.replace('[^a-zA-Z0-9]', ' ', regex = True)
text_test = text_test.replace('[^a-zA-Z0-9]', ' ', regex = True)

Vectorizer = TfidfVectorizer(min_df=5)
text_vect = Vectorizer.fit_transform(text['FullDescription'])
text_test_vect = Vectorizer.transform(text_test['FullDescription'])

data['LocationNormalized'].fillna('nan', inplace=True)
data_test['LocationNormalized'].fillna('nan', inplace=True)
data['ContractTime'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_categ = enc.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_for_train = hstack([text_vect, X_train_categ])
X_for_test = hstack([text_test_vect, X_test_categ])

clf = Ridge(alpha=1.0, random_state=241)
clf.fit(X_for_train,Y_train)
Result = clf.predict(X_for_test)

print(Result)


