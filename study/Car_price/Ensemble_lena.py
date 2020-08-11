# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import mean_absolute_error
def load_data():
    boston = load_boston()
    data = boston.data
    label = boston.target
    return data, label
def gradient_descent(train_x, train_y, maxCycle, alpha):
    numSamples, numFeatures = np.shape(train_x)
    #weights = np.zeros((numFeatures, 1))
    weights = np.random.random((numFeatures, 1))
    print(weights)
    errors = []
    for i in range(maxCycle):
        h = train_x * weights
        err = h - train_y
        errors.extend(abs(err.A[0]))
        weights = weights - (alpha * err.T * train_x).T
        print(i,weights)
    #print(errors[:100])
    #show_error(errors)
    return weights
def stochastic_gradient_descent(train_x, train_y, maxCycle, alpha):
    numSamples, numFeatures = np.shape(train_x)
    #weights = np.zeros((numFeatures, 1))
    weights = np.random.random((numFeatures, 1))
    errors = []
    #for i in range(maxCycle):
    for j in range(numSamples):
        h = train_x[j, :] * weights
        err = h - train_y[j, 0]
        errors.extend(abs(err.A[0]))
        weights = weights - (alpha * err.T * train_x[j, :]).T
    print(errors[:100])
    print(len(errors))
    show_error(errors)
    return weights

def stochastic_gradient_descent1(train_x, train_y, maxCycle, alpha):
    numSamples, numFeatures = np.shape(train_x)
    weights = np.zeros((numFeatures, 1))
    #weights = np.random.random((numFeatures, 1))
    errors = []

    for i in range(maxCycle):
        dataIndex = list(range(numSamples))
        randIndex = int(random.uniform(0, len(dataIndex)))
        h = train_x[randIndex, :] * weights
        err = h - train_y[randIndex, 0]
        print(randIndex,err)
        errors.extend(abs(err.A[0]))
        weights = weights - (alpha * err.T * train_x[randIndex, :]).T
    #print(len(errors))
    #show_error(errors)
    return weights

def show_error(errors):
    plt.figure()
    plt.plot(np.arange(0,len(errors)), np.array(errors))
    plt.show()

if __name__ == "__main__":
    data, label = load_data()
    data = preprocessing.normalize(data.T).T

    train_x, test_x, train_y, test_y = train_test_split(data, label, train_size=0.75, random_state=33)
    train_x = np.mat(train_x)
    test_x = np.mat(test_x)
    train_y = np.mat(train_y).T  # (3,)转为矩阵变为行向量了,需要转置
    test_y = np.mat(test_y).T

    weights = gradient_descent(train_x, train_y, 1000, 0.001)
    predict_y = test_x * weights
    '''[[ 0.88763807]
 [35.2229548 ]
 [30.40132559]
 [18.29572575]
 [45.91446369]
 [54.60204654]
 [40.9222278 ]
 [49.69732714]
 [23.64315696]
 [37.40528302]
 [48.22180872]
 [53.64086804]
 [27.50797253]]'''
    print(weights)
    err = mean_absolute_error(test_y,predict_y)
    print(err)#8.002023472897626
