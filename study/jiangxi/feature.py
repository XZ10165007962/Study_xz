import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import PolynomialFeatures

pd.set_option('display.max_columns',None)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#shape:(605864, 102)
#train_data = pd.read_excel(r'E:\dataset\jiangxi\deluxe_train_data_0914.xlsx')
train_data = pd.read_excel(r'E:\jiangxi\deluxe_train_data_0914.xlsx')

#shape:(157834, 102)
#test_data = pd.read_excel(r'E:\dataset\jiangxi\deluxe_test_data_0914.xlsx')

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
'''for i in train_data.columns:
    plt.title(i)
    sns.jointplot(x=i,y='IS_DELUXE',data=train_data)
    plt.savefig(i)
    plt.show()
    print(i+'保存成功')'''

'''train_data.fillna(0,inplace=True)


corr_p = train_data.corrwith(train_data['IS_DELUXE'])
corr_p = corr_p.reindex(corr_p.abs().sort_values().index)'''

'''plt.figure(figsize=(15,5))
plt.title('各变量与目标变量的相关系数')
corr_p.plot.bar(rot=30)
plt.show()
#['VAS_FEE', 'BAOTUO_FLAG', 'CHARGE_AMT', 'FLAG_1', 'MEMBER_NUM', 'USER_ONLINE', 'AIQIYI_SCORE', 'QQYINYUE_SCORE', 'MOU_TC', 'FLAG_10', 'IS_ZERO_HEYUE', 'VOICEFLUX_SHARE_FLAG', 'QQYINYUE_SCORE_RK', 'FLOW', 'FLOW_TY', 'GROUP_FLAG', 'IS_5G', 'YOUKUSHIPIN_SCORE_RK', 'YOUKUSHIPIN_SCORE', 'XIMALAYA_SCORE_RK', 'AIQIYI_SCORE_RK', 'BILIBILI_SCORE_RK', 'MANGGUOTV_SCORE_RK', 'MAX_PAY_FEE', 'AVG_MOU', 'MIGUSHIPIN_SCORE_RK', 'TENGXUNSHIPIN_SCORE_RK', 'SHUQIXIAOSHUO_SCORE_RK', 'CALL_DUR', 'DOU', 'DX_FEE', 'QQYUEDU_SCORE_RK', 'AVG_DOU', 'MIGUYUEDU_SCORE_RK', 'TC_ARPU', 'ZHANGYUE_SCORE_RK', 'USER_CLASS', 'BRDBAND_FLAG', 'HITV_FLAG', 'PLAN_ZIFEI', 'MANYBU_CNT', 'IF_BXL', 'TENGXUNSHIPIN_SCORE', 'CHARGE_CNT', 'FLAG_8', 'PLAN_FEE', 'DM_ARPU', 'AVG_ARPU']
train = train_data[corr_p[corr_p.abs() > 0.1].index]
lis = list(corr_p[corr_p.abs() > 0.1].index)[:-1]
print(lis)'''


'''train_X = train_data.iloc[:,0:-1]
train_y = train_data.iloc[:,-1]
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(train_X.values)

X_df = pd.DataFrame(X)
corr_p = X_df.corrwith(train_y)
corr_p = corr_p.reindex(corr_p.abs().sort_values().index)

plt.figure(figsize=(15,5))
plt.title('各变量与目标变量的相关系数')
corr_p.plot.bar(rot=30)
plt.show()
#['VAS_FEE', 'BAOTUO_FLAG', 'CHARGE_AMT', 'FLAG_1', 'MEMBER_NUM', 'USER_ONLINE', 'AIQIYI_SCORE', 'QQYINYUE_SCORE', 'MOU_TC', 'FLAG_10', 'IS_ZERO_HEYUE', 'VOICEFLUX_SHARE_FLAG', 'QQYINYUE_SCORE_RK', 'FLOW', 'FLOW_TY', 'GROUP_FLAG', 'IS_5G', 'YOUKUSHIPIN_SCORE_RK', 'YOUKUSHIPIN_SCORE', 'XIMALAYA_SCORE_RK', 'AIQIYI_SCORE_RK', 'BILIBILI_SCORE_RK', 'MANGGUOTV_SCORE_RK', 'MAX_PAY_FEE', 'AVG_MOU', 'MIGUSHIPIN_SCORE_RK', 'TENGXUNSHIPIN_SCORE_RK', 'SHUQIXIAOSHUO_SCORE_RK', 'CALL_DUR', 'DOU', 'DX_FEE', 'QQYUEDU_SCORE_RK', 'AVG_DOU', 'MIGUYUEDU_SCORE_RK', 'TC_ARPU', 'ZHANGYUE_SCORE_RK', 'USER_CLASS', 'BRDBAND_FLAG', 'HITV_FLAG', 'PLAN_ZIFEI', 'MANYBU_CNT', 'IF_BXL', 'TENGXUNSHIPIN_SCORE', 'CHARGE_CNT', 'FLAG_8', 'PLAN_FEE', 'DM_ARPU', 'AVG_ARPU']
train = X_df[corr_p[corr_p.abs() > 0.1].index]
lis = list(corr_p[corr_p.abs() > 0.1].index)
print(lis)'''



