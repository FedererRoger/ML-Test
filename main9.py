import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

data = pd.read_csv('classification.csv')
Pred_col = data['pred']
True_col = data['true']
TP = 0
FP = 0
TN = 0
FN = 0
for i in range (len(data)):
    if Pred_col[i] > 0.5:
        if True_col[i] > 0.5:
            TP +=1
        else:
            FP +=1
    else:
        if True_col[i] > 0.5:
            FN += 1
        else:
            TN += 1
quality_acc = accuracy_score(True_col, Pred_col)
quality_pr = precision_score(True_col, Pred_col)
quality_re = recall_score(True_col, Pred_col)
quality_f1 = f1_score(True_col, Pred_col)

print(TP, FP, FN, TN)
print(quality_acc, quality_pr, quality_re, quality_f1)

data2 = pd.read_csv('scores.csv')
true_col = data2['true']
score_logreg = data2['score_logreg']
score_svm = data2['score_svm']
score_knn = data2['score_knn']
score_tree = data2['score_tree']

Max = np.empty(4)
Max[0] = roc_auc_score(true_col, score_logreg)
Max[1] = roc_auc_score(true_col, score_svm)
Max[2] = roc_auc_score(true_col, score_knn)
Max[3] = roc_auc_score(true_col, score_tree)
print(np.argmax(Max))

pr_max = np.empty(4)
pr = precision_recall_curve(true_col, score_logreg)
pr_max[0] = pr[0][pr[1] >= 0.7].max()
pr = precision_recall_curve(true_col, score_svm)
pr_max[1] = pr[0][pr[1] >= 0.7].max()
pr = precision_recall_curve(true_col, score_knn)
pr_max[2] = pr[0][pr[1] >= 0.7].max()
pr = precision_recall_curve(true_col, score_tree)
pr_max[3] = pr[0][pr[1] >= 0.7].max()
print(data2)