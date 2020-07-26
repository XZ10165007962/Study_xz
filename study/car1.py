#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


train = pd.read_csv(r'E:\dataset\car\used_car_train_20200313.csv',sep=' ')
test = pd.read_csv(r'E:\dataset\car\used_car_testB_20200421.csv',sep=' ')

'''print('train:',train.shape) #train: (150000, 31)
print('test:',test.shape) #test: (50000, 30)
pd.set_option('display.max_columns',None)
print(train.head())'''
train['notRepairedDamage'] = train['notRepairedDamage'].replace('-',np.nan).astype(np.float32)
test['notRepairedDamage'] = test['notRepairedDamage'].replace('-',np.nan).astype(np.float32)
for i in train.columns:
    dtype = train[i].dtype
    if dtype == 'float64':
        train[i] = train[i].astype(np.float32)
    elif dtype == 'int64':
        train[i] = train[i].astype(np.int32)
for i in test.columns:
    dtype = test[i].dtype
    if dtype == 'float64':
        test[i] = test[i].astype(np.float32)
    elif dtype == 'int64':
        test[i] = test[i].astype(np.int32)
'''print(train.info())
print(train.describe())'''
train.drop(['SaleID','name','offerType'],axis=1,inplace=True)
train = train[train['power'] <= 600]
test.drop(['SaleID','name','offerType'],axis=1,inplace=True)
'''f,ax = plt.subplots(1,2,figsize=(13,6))
ax[0].set_title('价格分布图')
sns.distplot(train.price,bins=100,ax=ax[0])
stats.probplot(train.price,plot=ax[1])
plt.show()'''

train['price'] = np.log(train['price'])
'''f,ax = plt.subplots(1,2,figsize=(13,6))
ax[0].set_title('价格分布图')
sns.distplot(train.price,bins=100,ax=ax[0])
stats.probplot(train.price,plot=ax[1])
plt.show()'''

train_na_num = train.isnull().sum()
train_na_rate = (train_na_num / len(train)) * 100
#print(pd.concat([train_na_num,train_na_rate],axis=1,keys=['number','rate']))

train = train[train.model.notnull()]
train['bodyType'] = train['bodyType'].fillna(train['bodyType'].mode()[0])
train['fuelType'] = train['fuelType'].fillna(train['fuelType'].mode()[0])
train['gearbox'] = train['gearbox'].fillna(train['gearbox'].mode()[0])

test['bodyType'] = test['bodyType'].fillna(test['bodyType'].mode()[0])
test['fuelType'] = test['fuelType'].fillna(test['fuelType'].mode()[0])
test['gearbox'] = test['gearbox'].fillna(test['gearbox'].mode()[0])
corr_p = train.corrwith(train['price'])
corr_p = corr_p.reindex(corr_p.abs().sort_values().index)
'''plt.figure(figsize=(15,5))
plt.title('各变量与price的相关系数')
corr_p.plot.bar(rot=30)
plt.show()'''
train = train[corr_p[corr_p.abs() > 0.1].index]
lis = list(corr_p[corr_p.abs() > 0.1].index)[:-1]
test = test[lis]

def as_str(n):
    train[n] = train[n].astype(np.str)
    test[n] = test[n].astype(np.str)
cate_cols = ['model','bodyType','fuelType','gearbox','notRepairedDamage']
for col in cate_cols:
    as_str(col)

#train = pd.get_dummies(train)
#test = pd.get_dummies(test)
#train.drop([ 'model_1.0', 'bodyType_1.0', 'fuelType_1.0', 'gearbox_1.0', 'notRepairedDamage_1.0'],axis=1,inplace=True)
#test.drop([ 'model_1.0', 'bodyType_1.0', 'fuelType_1.0', 'gearbox_1.0', 'notRepairedDamage_1.0'],axis=1,inplace=True)
train['notRepairedDamage'].replace('-',0,inplace=True)
test['notRepairedDamage'].replace('-',1,inplace=True)

X = np.asarray(train.drop('price',axis=1))
y = np.asarray(train['price'])
test_x = np.asarray(test)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

model = LinearRegression()
results = model.fit(X,y)
scores = cross_val_score(model,X,y,cv=4)
print(scores)
print(results.predict(test_x))


