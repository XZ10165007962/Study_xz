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

#查看数据基本信息
print(train_data.head().append(train_data.tail()))
print(train_data.shape)
print(test_data.head().append(test_data.tail()))
print(test_data.shape)
print(train_data.describe())
print(test_data.describe())
print(train_data.info())
print(test_data.info())

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

'''sns.distplot(train_data['price'])
print('Skewness: %f'% train_data['price'].skew())
print('Kurtosis: %f'% train_data['price'].kurt())'''

y_train =train_data['price']

# 数字特征
# numeric_features = Train_data.select_dtypes(include=[np.number])
# numeric_features.columns
# # 类型特征
# categorical_features = Train_data.select_dtypes(include=[np.object])
# categorical_features.columns

numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14' ]

categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode',]

for cat_fea in categorical_features:
    print(cat_fea + '的特征分布如以下：')
    print('{}特征有{}个不同的值'.format(cat_fea,train_data[cat_fea].unique()))
    print(train_data[cat_fea].value_counts())

    # 特征nunique分布
for cat_fea in categorical_features:
    print(cat_fea + "的特征分布如下：")
    print("{}特征有个{}不同的值".format(cat_fea, test_data[cat_fea].nunique()))
    print(test_data[cat_fea].value_counts())

numeric_features.append('price')
price_numeric = train_data[numeric_features]
correlation = price_numeric.corr()
print(correlation['price'].sort_values(ascending=False),'\n')

'''plt.figure(figsize=(7,7))
plt.title('Correlation of Numeric Features with Price',y=1,size=16)
sns.heatmap(correlation,square = True,  vmax=0.8)
plt.show()'''

price_numeric.drop('price',axis=1,inplace=True)

## 4) 数字特征相互之间的关系可视化
'''sns.set()
columns = ['price', 'v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
sns.pairplot(train_data[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()'''






