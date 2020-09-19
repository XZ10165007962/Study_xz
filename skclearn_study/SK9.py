from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

X,y = make_classification(
    n_samples=1000,n_features=100,n_informative=20,
    n_clusters_per_class=2,
    random_state=11
)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=11)

'''clf = DecisionTreeClassifier(random_state=11)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print(classification_report(y_test,predictions))

clf = RandomForestClassifier(n_estimators=10,random_state=11)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print(classification_report(y_test,predictions))'''


'''clf = DecisionTreeClassifier(random_state=11)
clf.fit(X_train,y_train)
print('Decision tree accuracy : %s' % clf.score(X_test,y_test))

clf = AdaBoostClassifier(n_estimators=50,random_state=11)
clf.fit(X_train,y_train)
accuracies = []
accuracies.append(clf.score(X_test,y_test))

plt.title('Ensemble Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of base estimators in ensemble')
plt.plot(range(1,51),[accuracy for accuracy in clf.staged_score(X_test,y_test)])
plt.show()'''

