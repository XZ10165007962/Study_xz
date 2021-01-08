'''
使用scikit-learn解决监督学习任务代码示例
'''

import numpy as np
from sklearn import datasets

'''iris_x , iris_y = datasets.load_iris(return_X_y=True)
print(np.unique(iris_y))
np.random.seed(0)
indices = np.random.permutation(len(iris_x))
iris_x_train = iris_x[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_x_test = iris_x[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(iris_x_train,iris_y_train)

print(knn.predict(iris_x_test))'''

'''diabetes_X,diabetes_y = datasets.load_digits(return_X_y=True)
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(diabetes_X_train,diabetes_y_train)
print(regr.coef_)

print(np.mean((regr.predict(diabetes_X_test) - diabetes_y_test)**2))

print( regr.score(diabetes_X_test,diabetes_y_test))
'''


'''X = np.c_[.5,1].T
y = [.5,1]
test = np.c_[0,2].T

from sklearn import linear_model
regr = linear_model.Ridge(alpha=0.1)

import matplotlib.pyplot as plt
plt.figure()

np.random.seed(0)
for _ in range(6):
    this_X = .1 * np.random.normal(size=(2,1)) + X
    regr.fit(this_X,y)
    plt.plot(test,regr.predict(test))
    plt.scatter(this_X,y,s=3)
plt.show()'''

'''from sklearn.model_selection import KFold,cross_val_score
X = ['a','a','a','b','b','c','c','c','c','c']
k_fold = KFold(n_splits=5)
for train_indices,test_indices in k_fold.split(X):
    print('Train: %s | test: %s' % (train_indices, test_indices))'''

'''from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
parameters = {'kernel':('rbf','linear'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)
print(clf.cv_results_)
print('-----')
print(clf.best_score_)
print('----------')
print(clf.best_estimator_)
print('----------')
print(clf.best_params_)
print('----------')
print(clf.best_index_)
print('----------')
print(clf.scorer_)'''

'''import hdfs

client = hdfs.Client("http://hadoop:50070")

client.write(hdfs_path, dataframe,overwrite=True,append=False)

client.write(hdfs_path, dataframe.to_csv(header=False,index=False,sep="\t"), encoding='utf-8',overwrite=True)'''

