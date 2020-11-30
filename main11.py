import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

data = pd.read_csv('сlose_prices.csv')
djia_index = pd.read_csv('djia_index.csv')
data = data.drop(['date'], axis=1)
djia_index= djia_index.drop(['date'], axis=1)
pca = PCA(n_components=10)
pca.fit(data)
sigma = pca.explained_variance_ratio_
sum = np.sum(sigma)
#print(sum)
c1 = pca.transform(data)[:, 0]
Pirson = np.corrcoef(c1.T, djia_index.T)[1, 0]
#Print Pirson
c2 = np.argmax (pca.components_[0])
print(data.columns[c2])


