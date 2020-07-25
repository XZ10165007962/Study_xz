import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X_train = [[6],[8],[10],[14],[18]]
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
#plt.plot(X_test_quadratic,regressor_quadratic.predict(X_test_quadratic),c='g')
plt.scatter(X_train,y_train)
plt.show()