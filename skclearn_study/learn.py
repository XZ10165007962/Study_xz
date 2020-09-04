import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

X_train = np.mat([[6],[8],[10],[14],[18]])
y_train = np.mat([[7],[9],[13],[17.5],[18]])
alpha = 0.001
weights = 1
for i in range(10):
    h = X_train * weights
    err = y_train - h
    print(err)
    weights += alpha * X_train.transpose() * err
    print(weights)









