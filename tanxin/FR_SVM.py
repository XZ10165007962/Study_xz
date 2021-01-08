# -*- coding: utf-8 -*-
# @Time : 2020/12/8 22:30
# @Author : BigZhuang
# @Site : 
# @File : FR_SVM.py
# @Software: PyCharm
# @version:
# @alert:

import time
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix

def evaluate_cross_validation(clf,X,y,k):
    cv = KFold(k,shuffle=True,random_state=0)

    scores = cross_val_score(clf,X,y,cv=cv)
    print(scores)
    print('Mean score: {0:.3f})'.format(
        np.mean(scores)
    ))

def train_cross_validation(clf,X_train,X_test,y_train,y_test):
    clf.fit(X_train,y_train)

    print('Accuracy on training set')
    print(clf.score(X_train,y_train))
    print('Accuracy on testing set')
    print(clf.score(X_test,y_test))

    y_pred = clf.predict(X_test)

    print('Classification Report:')
    print(classification_report(y_test,y_pred))
    print('confuseion Matrix:')
    print(confusion_matrix(y_test,y_pred))

def create_taret(num_samples,segments):

    y = np.zeros(num_samples)

    for (start,end) in segments:
        y[start:end + 1] = 1
    return y


def main():
    faces = fetch_olivetti_faces()
    print(faces.DESCR)
    print(faces.keys())
    print(faces.images.shape)
    print(faces.data.shape)
    print(faces.target.shape)
    print(np.max(faces.data))
    print(np.min(faces.data))
    print(np.mean(faces.data))

    svc_1 = SVC(kernel='linear')
    print(svc_1)

    X_train,X_test,y_train,y_test = train_test_split(faces.data,faces.target,test_size=0.25,random_state=0)
    evaluate_cross_validation(svc_1,X_train,y_train,5)
    train_cross_validation(svc_1,X_train,X_test,y_train,y_test)

    glasses = [
        (10,19),(30,32),(37,38),(50,59),(63,64),(69,69),
        (120,121),(124,129),(130,139),(160,161),(164,169),(180,182),(185,185),
        (189,189),(190,192),(194,194),(196,199),(260,269),(270,279),(300,309),
        (330,339),(358,359),(360,369)
    ]

    num_samples = faces.target.shape[0]
    target_glasses = create_taret(num_samples,glasses)

    svc_2 = SVC(kernel='linear')
    X_train,X_test,y_train,y_test = train_test_split(faces.data,target_glasses,test_size=0.25,random_state=0)
    evaluate_cross_validation(svc_2,X_train,y_train,5)
    train_cross_validation(svc_2,X_train,X_test,y_train,y_test)

if __name__ == '__main__':
    t_start = time.time()
    main()
    t_end = time.time()
    t_all = t_end - t_start
    print('FR_SVM.py: whole time: {:.2f} min'.format(t_all) )