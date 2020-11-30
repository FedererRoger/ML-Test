import pandas as pd
import math
from sklearn.metrics import roc_auc_score

def gradient_method(data, C, w0):
    k = 0.1 
    max_iter_count = 10000
    eps = 1e-5
    iter_count = 0
    w = []
    w.append(w0[0])
    w.append(w0[1])
    for it in range(max_iter_count):
        sum_term_1 = 0.0
        sum_term_2 = 0.0
        for (i, row) in data.iterrows():
            yi = row[0]
            xi1 = row[1]
            xi2 = row[2]
            temp = 1.0 - 1.0/(1.0 + math.exp(-yi*(w[0]*xi1 + w[1]*xi2)))
            sum_term_1 += yi*xi1*temp
            sum_term_2 += yi*xi2*temp
        l = len(data)
        dw0 = k/l*sum_term_1 - k*C*w[0]
        dw1 = k/l*sum_term_2 - k*C*w[1]
        w[0] += dw0
        w[1] += dw1
        iter_count = it+1
        if math.sqrt(dw0*dw0 + dw1*dw1) <= eps:                
            break         
    return w, iter_count

def calc_score(data, w):
    y_target = data[0]
    y_score = []
    for (i, row) in data.iterrows():
        y_score.append(1.0/(1.0 + math.exp(- w[0]*row[1] - w[1]*row[2])))
    return roc_auc_score(y_target, y_score)

#считываем данные
data = pd.read_csv('data-logistic.csv')

#считаем без регуляризации
w0 = (0.0, 0.0) 
C_without = 0.0
w_without, it_without = gradient_method(data, C_without, w0)
print(it_without)
print(w_without)
score_without = calc_score(data, w_without)

#считаем с регуляризацией 
C_with = 10.0
w_with, it_with = gradient_method(data, C_with, w0)
print(it_with)
print(w_with)
score_with = calc_score(data, w_with)

#выводим результат в файл
print(score_without, score_with)
f = open('LogisticRegression.txt', 'w')
f.write(str(round(score_without, 3))+' '+str(round(score_with, 3)))  
f.close()
