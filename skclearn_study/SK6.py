import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

pipline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])
prarmeters = {
    'vect__max_df':(0.25,0.5,.75),
    'vect__stop_words':('english',None),
    'vect__max_features':(2500,500,1000,None),
    'vect__ngram_range':((1,1),(1,2)),
    'vect__use_idf':(True,False),
    'vect__norm':('l1','l2'),
    'clf__penalty':('l1','l2'),
    'clf__C':(0.01,0.1,1,10)
}
df = pd.read_csv('SMSSpamCollection',delimiter='\t',header=None)

x = df[1].values
y = df[0].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

x_train_raw ,x_test_raw,y_train,y_test = train_test_split(x,y)

grid_search = GridSearchCV(pipline,prarmeters,n_jobs=-1,
                           verbose=1,scoring='accuracy',cv=3)
grid_search.fit(x_train_raw,y_train)
print('best score:%0.3f'%grid_search.best_score_)
print('best parameters set:')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(best_parameters.keys()):
    print('t%s:%r'%(param_name,best_parameters[param_name]))
    predictions = grid_search.predict(x_test_raw)
    print(' Accuracy :' , accuracy_score(y_test, predictions))
    print(' Precision :', precision_score(y_test, predictions))
    print(' Recal1 :', recall_score(y_test, predictions ) )
'''vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train_raw)
x_test = vectorizer.transform(x_test_raw)
classifier = LogisticRegression()
classifier.fit(x_train,y_train)
scores = cross_val_score(classifier,x_train,y_train,cv=5)
print('Accuracies:%s'%scores)
print('Mean accuracy:%s'%np.mean(scores))
scores1 = cross_val_score(classifier,x_train,y_train,cv=5,scoring='precision')
print('Precision:%s'%scores1)
print('Mean precision:%s'%np.mean(scores1))
scores2 = cross_val_score(classifier,x_train,y_train,cv=5,scoring='recall')
print('Recall:%s'%scores2)
print('Mean recall:%s'%np.mean(scores2))
scores2 = cross_val_score(classifier,x_train,y_train,cv=5,scoring='f1')
print('F1:%s'%scores2)
print('Mean f1:%s'%np.mean(scores2))'''

'''from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

y_test = [0,0,0,0,0,1,1,1,1,1]
y_pred = [0,1,0,0,0,0,0,1,1,1]
confusion_matrix = confusion_matrix(y_test,y_pred)
print(confusion_matrix)
plt.matshow(confusion_matrix)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True lable')
plt.xlabel('predicted label')
plt.show()'''



