import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

train_data = pd.read_csv(r'E:\dataset\car\used_car_train_20200313.csv',sep=' ')
test_data = pd.read_csv(r'E:\dataset\car\used_car_testB_20200421.csv',sep=' ')
pd.set_option('display.max_columns',None)
for i in train_data.columns:
    dtype = train_data[i].dtype
    if dtype == 'float64':
        train_data[i] = train_data[i].astype(np.float32)
    elif dtype == 'int64':
        train_data[i] = train_data[i].astype(np.int32)

print(train_data.describe())

train_data['notRepairedDamage'].replace('-',0,inplace=True)
test_data['notRepairedDamage'].replace('-',1,inplace=True)
#train_data['notRepairedDamage'] = train_data['notRepairedDamage'].astype(np.float32)
train_data.drop(['SaleID','seller','offerType','name'],axis=1,inplace=True)
test_data.drop(['SaleID','seller','offerType','name'],axis=1,inplace=True)


'''f,ax = plt.subplots(1,2,figsize=(13,6))
ax[0].set_title('价格分布图')
sns.distplot(train_data.price,bins=100,ax=ax[0])
stats.probplot(train_data.price,plot=ax[1])
plt.show()'''

train_data['price'] = np.log(train_data['price'])
'''f,ax = plt.subplots(1,2,figsize=(13,6))
ax[0].set_title('价格分布图')
sns.distplot(train_data.price,bins=100,ax=ax[0])
stats.probplot(train_data.price,plot=ax[1])
plt.show()'''

#train_data = train_data[train_data['power'] > 600]

train_data = train_data[train_data.model.notnull()]
train_data['bodyType'] = train_data['bodyType'].fillna(train_data['bodyType'].mode()[0])
train_data['fuelType'] = train_data['fuelType'].fillna(train_data['fuelType'].mode()[0])
train_data['gearbox'] = train_data['gearbox'].fillna(train_data['gearbox'].mode()[0])


test_data['bodyType'] = test_data['bodyType'].fillna(test_data['bodyType'].mode()[0])
test_data['fuelType'] = test_data['fuelType'].fillna(test_data['fuelType'].mode()[0])
test_data['gearbox'] = test_data['gearbox'].fillna(test_data['gearbox'].mode()[0])


corr_p = train_data.corrwith(train_data['price'])
corr_p = corr_p.reindex(corr_p.abs().sort_values().index)
'''plt.figure(figsize=(15,5))
plt.title('各变量与price的相关系数')
corr_p.plot.bar(rot=30)
plt.show()'''

'''#处理notRepairedDamage缺失值
xx = test_data[test_data['notRepairedDamage'].notna()]
train_xx = test_data[test_data['notRepairedDamage'].notna()].drop(['notRepairedDamage'],axis=1)
yy = xx['notRepairedDamage']

test_xx = test_data[test_data['notRepairedDamage'].isna()].drop(['notRepairedDamage'],axis=1)

from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import cross_val_score

model = LogisticRegression()
result = model.fit(train_xx,yy)
finall = result.predict(test_xx)
num0 = 0
num1 = 0
for i in finall:
    if i == 0:
        num0+= 1
    else:
        num1 += 1
print(num1,num0)'''



X = np.asarray(train_data.drop('price',axis=1))
print('---X---')
print(X)
y = np.asarray(train_data['price'])
test_X = np.asarray(test_data)
print('---y---')
print(test_X)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

model = LinearRegression()
results = model.fit(X,y)
finall = results.predict(test_X)
finall = np.power(np.e,finall)
print(finall)
sub = pd.DataFrame()
sub['SaleID'] = test_data.index
sub['price'] = finall
sub.to_csv('./sub_Stacking.csv',index=False)
