import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

'''X_train = np.array([[158,64],[170,86],[183,84],
                    [191,80],[155,49],[163,59],
                    [180,67],[158,54],[170,67]])

y_train =['male','male','male','male','female','female','female','female','female']

plt.figure()
plt.title('Human heights and weights by sex')
plt.xlabel('height in cm')
plt.ylabel('weight in kg')


#for i ,x in enumerate(X_train):
#    plt.scatter(x[0],x[1],c='k',marker='x' if y_train[i] == 'male' else 'D')
#plt.grid(True)
#plt.show()


x = np.array([[155,70]])
distances = np.sqrt(np.sum((X_train - x) ** 2 , axis=1))

nearest_neighbor_indices = distances.argsort()[:3]
nearest_neighbor_indices = np.take(y_train,nearest_neighbor_indices)
b = Counter(np.take(y_train,distances.argsort()[:3]))

from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier

lb = LabelBinarizer()
y_train_binarized = lb.fit_transform(y_train)
k = 3
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X_train,y_train_binarized.reshape(-1))
prediction_binarized = clf.predict(np.array([155,70]).reshape(1,-1))[0]
prediction_label = lb.inverse_transform(prediction_binarized)


from sklearn.metrics import accuracy_score,precision_score,f1_score

X_test = np.array([[168,65],[180,96],[160,52],[169,67]])
y_test = ['male','male','female','female']
y_test_binarized = lb.transform(y_test)
predictions_binarized = clf.predict(X_test)
print('Accuracy: %s' %accuracy_score(y_test_binarized,predictions_binarized))
print('precision: %s' %precision_score(y_test_binarized,predictions_binarized))
print('F1: %s' %f1_score(y_test_binarized,predictions_binarized))

from sklearn.metrics import classification_report
print(classification_report(y_test_binarized,predictions_binarized,
                            target_names=['male'],labels=[1]))'''


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

X_train = np.array([
    [158,1],
    [170,1],
    [183,1],
    [191,1],
    [155,0],
    [163,0],
    [180,0],
    [158,0],
    [170,0]
])
y_train = [64,86,84,80,49,59,67,54,67]

X_test = np.array([
    [168,1],
    [180,1],
    [160,0],
    [169,0]
])
y_test = [65,96,52,67]

k = 3
clf = KNeighborsRegressor(n_neighbors=k)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print('Coefficient of determination : %s ' % r2_score (y_test ,predictions))
print('Mean absolute error : %s ' % mean_absolute_error (y_test ,predictions))
print('Mean squared error : %s ' % mean_squared_error (y_test ,predictions))

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)

X_test_scaled = ss.transform(X_test)
clf.fit(X_train_scaled,y_train)
predictions = clf.predict(X_test_scaled)
print(predictions)
print('Coefficient of determination : %s ' % r2_score (y_test ,predictions))
print('Mean absolute error : %s ' % mean_absolute_error (y_test ,predictions))
print('Mean squared error : %s ' % mean_squared_error (y_test ,predictions))

