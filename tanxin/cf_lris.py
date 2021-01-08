# -*- coding: utf-8 -*-
# @Time : 2020/12/16 16:57
# @Author : BigZhuang
# @Site : 
# @File : cf_lris.py
# @Software: PyCharm
# @version:
# @alert:
import xgboost as xgb
import numpy as np
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
y = iris.target
x = iris.data

train_X,test_X,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=0)

kf = KFold(n_splits=5,shuffle=True,random_state=0)

xgb_model_list = []
xgb_accuracy_list = []
for train_index , test_index in kf.split(train_X):
    xgb_model = xgb.XGBClassifier().fit(x[train_index],y[train_index])
    xgb_model_list.append(xgb_model)
    prediction = xgb_model.predict(x[test_index])
    accu = accuracy_score(y[test_index],prediction)
    xgb_accuracy_list.append(accu)

print('xgb_accuracy_list:{}'.format(xgb_accuracy_list))
print('accu 的均值为:{}'.format(np.mean(xgb_accuracy_list)))

rf_model_list = []
rf_accuracy_list = []
for train_index,test_index in kf.split(train_X):
    rf_model = RandomForestClassifier().fit(x[train_index],y[train_index])
    rf_model_list.append(rf_model)
    prediction = rf_model.predict(x[test_index])
    accu = accuracy_score(y[test_index],prediction)
    rf_accuracy_list.append(accu)

print('rf_accuracy_list:{}'.format(rf_accuracy_list))
print('accu 的均值为:{}'.format(np.mean(rf_accuracy_list)))

if np.mean(rf_accuracy_list) <= np.mean(xgb_accuracy_list):
    max_accu = max(xgb_accuracy_list)
    ind = xgb_accuracy_list.index(max_accu)
    best_model = xgb_model_list[ind]
    print('best model is xgb')
    print('best model is random forest {},accu is {}'.format(ind,max_accu))
else:
    max_accu = max(rf_accuracy_list)
    ind = rf_accuracy_list.index(max_accu)
    best_model = rf_model_list[ind]
    print('best model is rf')
    print('best model is random forest {},accu is {}'.format(ind, max_accu))

pred = best_model.predict(test_X)
accu = accuracy_score(pred,test_y)
print(classification_report(pred,test_y))
print(confusion_matrix(pred,test_y))

print(classification_report(test_y,pred))
print(confusion_matrix(test_y,pred))
