import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
X_train = np.array([[158,64],[170,86],[183,84],
                    [191,80],[155,49],[163,59],
                    [180,67],[158,54],[170,67]])

y_train =['male','male','male','male','female','female','female','female','female']

plt.figure()
plt.title('Human heights and weights by sex')
plt.xlabel('height in cm')
plt.ylabel('weight in kg')

'''
for i ,x in enumerate(X_train):
    plt.scatter(x[0],x[1],c='k',marker='x' if y_train[i] == 'male' else 'D')
plt.grid(True)
plt.show()
'''

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