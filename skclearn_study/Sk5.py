import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
'''X_train = [[6],[8],[10],[14],[18]]
y_train = [[7],[9],[13],[17.5],[18]]
X_test = [[6],[8],[11],[16]]
y_test = [[8],[12],[15],[18]]

regressor = LinearRegression()
regressor.fit(X_train,y_train)
xx = np.linspace(0,26,100)
yy = regressor.predict(xx.reshape(xx.shape[0],1))
plt.plot(xx,yy)
quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
print(X_train_quadratic)
X_test_quadratic = quadratic_featurizer.transform(X_test)

regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic,y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0],1))
plt.plot(xx,regressor_quadratic.predict(xx_quadratic),c='r',linestyle='--')
plt.scatter(X_train,y_train)
plt.show()'''

datas = datasets.load_wine()

df = datas.data
y = datas.target
name = datas.feature_names
df = pd.DataFrame(df,columns=name)
df['quailty'] = pd.Series(y)

'''plt.scatter(df['alcohol'],df['quailty'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol Against Quality')
plt.show()'''

X = df[list(df.columns)[:-1]]
y = df['quailty']

'''X_train,X_test,y_train,y_test = train_test_split(X,y)
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_predictions = regressor.predict(X_test)
print('R-squared: %s' % regressor.score(X_test,y_test))'''

'''regressor = LinearRegression()
scores = cross_val_score(regressor,X,y,cv=5)
print(scores.mean())
print(scores)'''

X_train,X_test,y_train,y_test = train_test_split(datas.data,datas.target)
X_scatter =StandardScaler()
y_scatter = StandardScaler()

X_train = X_scatter.fit_transform(X_train)

X_test = X_scatter.transform(X_test)


regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor,X_train,y_train,cv=5)
print('cross validation r-squared scores : %s'%scores)
print('average cross validation r-squared score: %s'%np.mean(scores))
regressor.fit(X_train,y_train)
print('test set r-squared score %s'%regressor.score(X_test,y_test))
