from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import datasets

mnist = datasets.load_digits()

if __name__ == '__main__':
    X,y = mnist.data,mnist.target
    X = X/255.0*2-1
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=11)

    pipline = Pipeline([('clf',SVC(kernel='rbf',gamma=0.01,C=1))])

    parameters = {
        'clf__gamma' : (0.01,0.03,0.1,1),
        'clf__C': (0.1,0.3,1,3,10,30)
    }

    grid_search = GridSearchCV(pipline,parameters,n_jobs=-1,verbose=1,scoring='accuracy')
    grid_search.fit(X_train,y_train)
    print('best score %0.3f' % grid_search.best_score_)
    print('best parameters set:')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s : %r' %(param_name,best_parameters[param_name]))

    predictions = grid_search.predict(X_test)
    print(classification_report(y_test,predictions))