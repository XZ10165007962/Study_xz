import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,make_scorer

sample_feature = pd.read_csv(r'F:\data\二手车\used_car_train_20200313.csv',sep=' ')
test = pd.read_csv(r'F:\data\二手车\used_car_testB_20200421.csv',sep=' ')

for i in sample_feature.columns:
    dtype = sample_feature[i].dtype
    if dtype == 'float64':
        sample_feature[i] = sample_feature[i].astype(np.float32)
    elif dtype == 'int64':
        sample_feature[i] = sample_feature[i].astype(np.int32)

continuous_feature_names = [x for x in sample_feature.columns if x not in ['price','brand','model','brand']]
sample_feature = sample_feature.dropna().replace('-',0).reset_index(drop=True)
train = sample_feature[continuous_feature_names + ['price']]

train_X = train[continuous_feature_names]
train_y = train['price']


model = LinearRegression(normalize=True)
model = model.fit(train_X,train_y)

#print(sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x:x[1], reverse=True))

subsample_index = np.random.randint(low=0,high=len(train_y),size=50)

'''plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
plt.scatter(train_X['v_9'][subsample_index], model.predict(train_X.loc[subsample_index]), color='blue')
plt.xlabel('v_9')
plt.ylabel('price')
plt.legend(['True Price','Predicted Price'],loc='upper right')
print('The predicted price is obvious different from true price')
plt.show()'''

'''print('It is clear to see the price shows a typical exponential distribution')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(train_y)
plt.subplot(1,2,2)
sns.distplot(train_y[train_y < np.quantile(train_y, 0.9)])
plt.show()'''

train_y_ln = np.log(train_y)
'''train_y_ln = np.log(train_y+1)
train_y_ln1 = np.log(train_y)

print('The transformed price seems like normal distribution')
plt.figure(figsize=(15,5))
plt.subplot(2,2,1)
sns.distplot(train_y_ln)
plt.subplot(2,2,2)
sns.distplot(train_y_ln1[train_y_ln < np.quantile(train_y_ln1, 0.9)])
plt.subplot(2,2,3)
sns.distplot(train_y_ln1)
plt.subplot(2,2,4)
sns.distplot(train_y_ln1[train_y_ln1 < np.quantile(train_y_ln1, 0.9)])
plt.show()'''

model = model.fit(train_X,train_y_ln)
'''plt.scatter(train_X['v_9'][subsample_index], train_y[subsample_index], color='black')
plt.scatter(train_X['v_9'][subsample_index], np.exp(model.predict(train_X.loc[subsample_index])), color='blue')
plt.xlabel('v_9')
plt.ylabel('price')
plt.legend(['True Price','Predicted Price'],loc='upper right')
print('The predicted price seems normal after np.log transforming')
plt.show()'''

def log_transfer(func):
    def wrapper(y,yhat):
        result = func(np.log(y),np.nan_to_num(yhat))
        return result
    return wrapper
#[7030.95519974 6947.23957542 6940.89808749 6926.92147724 6957.5210726 ]
scores = cross_val_score(model, X=train_X, y=train_y, verbose=1, cv = 5,scoring=make_scorer(mean_absolute_error))
#print('AVG:', np.mean(scores))
scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=1, cv = 5, scoring=make_scorer(mean_absolute_error))
#print('AVG:', np.mean(scores))

scores = pd.DataFrame(scores.reshape(1,-1))
scores.columns = ['cv' + str(x) for x in range(1, 6)]
scores.index = ['MAE']
#print(scores)

train = sample_feature[continuous_feature_names + ['price']].dropna()

train_X = train[continuous_feature_names]
train_y = train['price']
train_y_ln = np.log(train_y + 1)

models = [LinearRegression(),Ridge(),Lasso()]
result = dict()
for model in models:
    model_name = str(model).split('(')[0]
    scores = cross_val_score(model,X = train_X,y = train_y_ln,verbose=0,cv=5,scoring=make_scorer(mean_absolute_error))
    result[model_name] = scores
    print(model_name + 'is finished')

result = pd.DataFrame(result)
result.index = ['cv' + str(x) for x in range(1, 6)]
print(result)

'''model = LinearRegression().fit(train_X, train_y_ln)
print('intercept:'+ str(model.intercept_))
sns.barplot(abs(model.coef_), continuous_feature_names)
plt.show()'''

'''model = Lasso().fit(train_X,train_y_ln)
print('intercept:'+ str(model.intercept_))
sns.barplot(abs(model.coef_),continuous_feature_names)
plt.show()'''

objective = ['regression', 'regression_l1', 'mape', 'huber', 'fair']
num_leaves = [3,5,10,15,20,40, 55]
max_depth = [3,5,10,15,20,40, 55]
bagging_fraction = []
feature_fraction = []
drop_rate = []

best_obj = dict()
for obj in objective:
    model = LGBMRegressor

