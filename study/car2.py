import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

train_data = pd.read_csv(r'E:\dataset\car\used_car_train_20200313.csv',sep=' ')
test_data = pd.read_csv(r'E:\dataset\car\used_car_testB_20200421.csv',sep=' ')
pd.set_option('display.max_columns',None)
'''
#查看数据基本信息
print(train_data.head().append(train_data.tail()))
print(train_data.shape)
print(test_data.head().append(test_data.tail()))
print(test_data.shape)
print(train_data.describe())
print(test_data.describe())
print(train_data.info())
print(test_data.info())'''

#查看缺失值
'''print(train_data.isnull().sum())
print(test_data.isnull().sum())

missing = train_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()
plt.show()'''

'''msno.matrix(train_data)
plt.show()

msno.bar(train_data)
plt.show()

msno.matrix(test_data)
plt.show()

msno.bar(test_data)
plt.show()'''

print(train_data['notRepairedDamage'].value_counts())
#将列中的-替换成nan
train_data['notRepairedDamage'].replace('-',np.nan,inplace=True)
print(train_data['notRepairedDamage'].value_counts())

test_data['notRepairedDamage'].replace('-',np.nan,inplace=True)

train_data.drop(['seller','offerType'],axis=1,inplace=True)
test_data.drop(['seller','offerType'],axis=1,inplace=True)

import scipy.stats as st
'''f,ax = plt.subplots(1,3,figsize=(13,6))
y = train_data['price']
#plt.figure(1)
ax[0].set_title('Johnson SU')
sns.distplot(y,kde=False,fit=st.johnsonsu,ax=ax[0])
#plt.figure(2)
ax[1].set_title('Normal')
sns.distplot(y,kde=False,fit=st.norm,ax=ax[1])
#plt.figure(3)
ax[2].set_title('Log Normal')
sns.distplot(y,kde=False,fit=st.lognorm,ax=ax[2])
plt.show()'''

train_data['price'].skew()




