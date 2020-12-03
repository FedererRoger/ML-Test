import pandas as pd
import numpy as np
import time
import datetime
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import  LogisticRegression
###################################################################################
####################ПОДГОТОВКА ДАННЫХ##############################################
###################################################################################
#Считываем признаки
features = pd.read_csv('features.csv', index_col='match_id')
features = features.drop(['duration', 'tower_status_radiant', 'tower_status_dire',
                          'barracks_status_radiant', 'barracks_status_dire'], axis=1)
#Проверка на пропуски
row_count = features.shape[0]
for columns in features.columns.values:
    N_nan = features[columns].count()
    Complete = round(N_nan/row_count, 2)
    #if Complete < 1:
        #Выводим названия признаков и "заполненность столбца"
        #print(columns, Complete)
#Замена NaN на нули
features = features.fillna(value=0)
#Целевая переменная
Y = features['radiant_win']
X = features.drop(['radiant_win'], axis=1)
###################################################################################
####################ГРАДИЕНТНЫЙ БУСТИНГ############################################
###################################################################################
#Обучение классификатора (градиентный бустинг)
#CV = KFold(n_splits=5, shuffle=True)
#n_estimate_list = [10, 20, 30, 50, 100]
#for n_estimate in n_estimate_list:
#    start_time = datetime.datetime.now()
#    clf = GradientBoostingClassifier(n_estimators=n_estimate)
#    quality = np.mean(cross_val_score(clf, X, Y, cv=CV, scoring='roc_auc'))
#    print(quality)
#    print('Time elapsed:', datetime.datetime.now() - start_time)
###################################################################################
####################ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ########################################
###################################################################################
# масштабируем признаки
#Scaler = StandardScaler()
#X_scaled = Scaler.fit_transform(X)
#Обучение классификатора (логистическая регрессия) c некорректными категориальными признаками
C_list = [1e-6, 1e-3, 1e-2, 1e-1, 1.0, 1e2, 1e6]
CV = KFold(n_splits=5, shuffle=True)
#for C_value in C_list:
#    start_time = datetime.datetime.now()
#    clf = LogisticRegression(C = C_value)
#    quality = np.mean(cross_val_score(clf, X_scaled, Y, cv=CV, scoring='roc_auc'))
#    print(quality)
#    print('Time elapsed:', datetime.datetime.now() - start_time)
#Исключаем категориальные признаки
X_short = X.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r2_hero', 'r3_hero', 'r4_hero'
                  , 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'], axis=1)
# масштабируем признаки
Scaler = StandardScaler()
X_scaled = Scaler.fit_transform(X_short)
#Обучение классификатора (логистическая регрессия) без категориальных признаков
#for C_value in C_list:
#    start_time = datetime.datetime.now()
#    clf = LogisticRegression(C = C_value)
#    quality = np.mean(cross_val_score(clf, X_scaled, Y, cv=CV, scoring='roc_auc'))
#    print(quality)
#    print('Time elapsed:', datetime.datetime.now() - start_time)
#Количество встречающихся в играх идентификаторов героев
#unique = [len(X['r1_hero'].unique()), len(X['d1_hero'].unique()),
#          len(X['r2_hero'].unique()), len(X['d2_hero'].unique()),
#          len(X['r3_hero'].unique()), len(X['d3_hero'].unique()),
#          len(X['r4_hero'].unique()), len(X['d4_hero'].unique()),
#          len(X['r5_hero'].unique()), len(X['d5_hero'].unique())]
#unique = max(unique)
#print(unique)
#Максимальный номер героя
N_max = [X['r1_hero'].max(), X['d1_hero'].max(),
         X['r2_hero'].max(), X['d2_hero'].max(),
         X['r3_hero'].max(), X['d3_hero'].max(),
         X['r4_hero'].max(), X['d4_hero'].max(),
         X['r5_hero'].max(), X['d5_hero'].max()]
N = max(N_max)
print(N)
#Формирование мешка слов
X_pick = np.zeros((X.shape[0], N))
for i, match_id in enumerate(X.index):
    for p in range(5):
        X_pick[i, X.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, X.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
#Формирование новой матрицы объекты-признаки
X_scaled = np.hstack((X_scaled, X_pick))
#Обучение классификатора (логистическая регрессия) с преобразованными категориальными признаками
#for C_value in C_list:
#   start_time = datetime.datetime.now()
#   clf = LogisticRegression(C = C_value)
#   quality = np.mean(cross_val_score(clf, X_scaled, Y, cv=CV, scoring='roc_auc'))
#   print(quality)
#   print('Time elapsed:', datetime.datetime.now() - start_time)
#Подготовка тестовых данных
X_test = pd.read_csv('features_test.csv', index_col='match_id')
X_test = X_test.fillna(value=0)

X_test_pick = np.zeros((X_test.shape[0], N))
for i, match_id in enumerate(X_test.index):
    for p in range(5):
        X_test_pick[i, X_test.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_test_pick[i, X_test.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1

X_test_short = X_test.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r2_hero', 'r3_hero', 'r4_hero'
                  , 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'], axis=1)

X_test_scaled = Scaler.transform(X_test_short)

X_test_scaled = np.hstack((X_test_scaled, X_test_pick))

#Прогноз для тестовой выборки
clf = LogisticRegression(C=1.0)
clf.fit(X_scaled, Y)
pred = clf.predict_proba(X_test_scaled)
min_pred = min(pred[:, 1])
max_pred = max(pred[:, 1])
print(min_pred, max_pred)
