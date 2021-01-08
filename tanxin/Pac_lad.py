# -*- coding: utf-8 -*-
# @Time : 2021/1/4 11:29
# @Author : BigZhuang
# @Site : 
# @File : Pac_lad.py.py
# @Software: PyCharm
# @version:
# @alert:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
pd.options.display.max_columns = None

#df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
df_wine = pd.read_csv('wine.csv',header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']
# print(df_wine.head())
# print(df_wine['Class label'].unique())

X,y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train ,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)
# print(X_train_std.shape)
# print(np.mean(X_train_std,axis=0))
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
# print('\nEigenvalues \n%s' % eigen_vals)
# print('\nEigenvecs \n%s' % eigen_vecs)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals,reverse=True)]
cum_var_exp = np.cumsum(var_exp)
# print(var_exp)
# print(cum_var_exp)

# plt.bar(range(1,14),var_exp,alpha=0.5,align='center',
#         label='individual explained variance')
# plt.step(range(1, 14), cum_var_exp, where='mid',
#          label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal component index')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()

eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i] )for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k : k[0],reverse=True)
# print(eigen_pairs)

w = np.hstack((eigen_pairs[0][1][:,np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
# print(w)

X_train_pca = X_train_std.dot(w)
colors = ['r','b','g']
markers = ['s','x','o']

for l ,c ,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(X_train_pca[y_train == l,0],
                X_train_pca[y_train == l,1],
                c=c,label=l,marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
