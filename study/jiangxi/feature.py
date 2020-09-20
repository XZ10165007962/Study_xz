import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFECV
pd.set_option('display.max_columns',None)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#shape:(605864, 102)
train_data = pd.read_excel(r'E:\dataset\jiangxi\deluxe_train_data_0914.xlsx')

#shape:(157834, 102)
test_data = pd.read_excel(r'E:\dataset\jiangxi\deluxe_test_data_0914.xlsx')

'''print(train_data.head())
print(train_data.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True))
print(train_data.describe())
for cat_fea in train_data.columns:
    print(cat_fea + '的特征分布如以下：')
    print('{}特征有{}个不同的值'.format(cat_fea,train_data[cat_fea].unique()))
    print(train_data[cat_fea].value_counts())'''

'''print(test_data.head())
print(test_data.info(verbose=True, max_cols=True, memory_usage=True, null_counts=True))
print(test_data.describe())
for cat_fea in test_data.columns:
    print(cat_fea + '的特征分布如以下：')
    print('{}特征有{}个不同的值'.format(cat_fea,test_data[cat_fea].unique()))
    print(test_data[cat_fea].value_counts())'''

'''train_flag = train_data['IS_DELUXE']
test_flag = test_data['IS_DELUXE']

f,ax = plt.subplots(1,2,figsize=(13,6))
ax[0].set_title('训练集豪华会员')
sns.distplot(train_flag,bins=100,ax=ax[0])
ax[1].set_title('测试集豪华会员')
sns.distplot(test_flag,bins=100,ax=ax[1])
plt.show()'''

'''for i in train_data.columns:
    print(i)
    plt.title(i)
    sns.distplot(train_data.loc[:,[i]],bins=100)
    plt.savefig(i+'png')
    plt.show()
    print(i+'保存成功')'''
for i in train_data.columns:
    plt.title(i)
    sns.jointplot(x=i,y='IS_DELUXE',data=train_data)
    plt.savefig(i)
    plt.show()
    print(i+'保存成功')
