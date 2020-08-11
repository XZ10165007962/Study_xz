import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
pd.set_option('display.max_columns',None)

train_data = pd.read_csv(r'E:\dataset\car\used_car_train_20200313.csv',sep=' ')
test_data = pd.read_csv(r'E:\dataset\car\used_car_testB_20200421.csv',sep=' ')
pd.set_option('display.max_columns',None)

#Train data shape: (150000, 31)
#TestA data shape: (50000, 30)
print('删除多余特征')
train_data.drop(['SaleID','name','offerType','seller'],axis=1,inplace=True)
test_data.drop(['SaleID','name','offerType','seller'],axis=1,inplace=True)
numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14' ]
categorical_features = [ 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode']
train_data['notRepairedDamage'].replace('-',1,inplace=True)
train_data['notRepairedDamage'] = train_data['notRepairedDamage'].astype(np.float32)
train_data['power'] = train_data['power'].map(lambda x: 600 if x>600 else x)
test_data['notRepairedDamage'].replace('-',1,inplace=True)
test_data['notRepairedDamage'] = test_data['notRepairedDamage'].astype(np.float32)
test_data['power'] = test_data['power'].map(lambda x: 600 if x>600 else x)

#用众数填充缺失值
print('众数填充')
train_data['bodyType'] = train_data['bodyType'].fillna(train_data['bodyType'].mode()[0])
train_data['fuelType'] = train_data['fuelType'].fillna(train_data['fuelType'].mode()[0])
train_data['gearbox'] = train_data['gearbox'].fillna(train_data['gearbox'].mode()[0])
train_data['model'] = train_data['model'].fillna(train_data['model'].mode()[0])
test_data['bodyType'] = test_data['bodyType'].fillna(test_data['bodyType'].mode()[0])
test_data['fuelType'] = test_data['fuelType'].fillna(test_data['fuelType'].mode()[0])
test_data['gearbox'] = test_data['gearbox'].fillna(test_data['gearbox'].mode()[0])
test_data['model'] = test_data['model'].fillna(test_data['model'].mode()[0])

# 对可分类的连续特征进行分桶，kilometer是已经分桶了
print('特征分桶')
bin = [i*10 for i in range(31)]
train_data['power_bin'] = pd.cut(train_data['power'], bin, labels=False)
test_data['power_bin'] = pd.cut(test_data['power'], bin, labels=False)
bin = [i*10 for i in range(24)]
train_data['model_bin'] = pd.cut(train_data['model'], bin, labels=False)
test_data['model_bin'] = pd.cut(test_data['model'], bin, labels=False)

#处理时间列
print('处理时间特征')
train_data['used_time'] = (pd.to_datetime(train_data['creatDate'],format='%Y%m%d',errors='coerce')-
                     pd.to_datetime(train_data['regDate'],format='%Y%m%d',errors='coerce')).dt.days
train_data['used_time'].fillna(0,inplace=True)

test_data['used_time'] = (pd.to_datetime(test_data['creatDate'],format='%Y%m%d',errors='coerce')-
                     pd.to_datetime(test_data['regDate'],format='%Y%m%d',errors='coerce')).dt.days
test_data['used_time'].fillna(0,inplace=True)

#时间提取出年，月，日和使用时间
from datetime import datetime
def date_process(x):
    year = int(str(x)[:4])
    month = int(str(x)[4:6])
    day = int(str(x)[6:8])

    if month < 1:
        month = 1

    date = datetime(year, month, day)
    return date
train_data['regDate'] = train_data['regDate'].apply(date_process)
train_data['creatDate'] = train_data['creatDate'].apply(date_process)
train_data['regDate_year'] = train_data['regDate'].dt.year
train_data['regDate_month'] = train_data['regDate'].dt.month
train_data['regDate_day'] = train_data['regDate'].dt.day
train_data['creatDate_year'] = train_data['creatDate'].dt.year
train_data['creatDate_month'] = train_data['creatDate'].dt.month
train_data['creatDate_day'] = train_data['creatDate'].dt.day
train_data['car_age_year'] = round(train_data['used_time'] / 365, 1)#二手车使用年数

test_data['regDate'] = test_data['regDate'].apply(date_process)
test_data['creatDate'] = test_data['creatDate'].apply(date_process)
test_data['regDate_year'] = test_data['regDate'].dt.year
test_data['regDate_month'] = test_data['regDate'].dt.month
test_data['regDate_day'] = test_data['regDate'].dt.day
test_data['creatDate_year'] = test_data['creatDate'].dt.year
test_data['creatDate_month'] = test_data['creatDate'].dt.month
test_data['creatDate_day'] = test_data['creatDate'].dt.day
test_data['car_age_year'] = round(test_data['used_time'] / 365, 1)#二手车使用年数


#隐藏数据特征组合，由于0，3，8，12与价格的相关性较高
print('特征组合')
num_cols = [0, 3, 8, 12]
for i in num_cols:
    for j in num_cols:
        train_data['new' + str(i) + '*' + str(j)] = train_data['v_' + str(i)] * train_data['v_' + str(j)]
for i in num_cols:
    for j in num_cols:
        train_data['new' + str(i) + '+' + str(j)] = train_data['v_' + str(i)] + train_data['v_' + str(j)]
for i in num_cols:
    for j in num_cols:
        train_data['new' + str(i) + '-' + str(j)] = train_data['v_' + str(i)] - train_data['v_' + str(j)]
for i in range(15):
    train_data['new' + str(i) + '*year'] = train_data['v_' + str(i)] * train_data['car_age_year']

for i in num_cols:
    for j in num_cols:
        test_data['new' + str(i) + '*' + str(j)] = test_data['v_' + str(i)] * test_data['v_' + str(j)]
for i in num_cols:
    for j in num_cols:
        test_data['new' + str(i) + '+' + str(j)] = test_data['v_' + str(i)] + test_data['v_' + str(j)]
for i in num_cols:
    for j in num_cols:
        test_data['new' + str(i) + '-' + str(j)] = test_data['v_' + str(i)] - test_data['v_' + str(j)]
for i in range(15):
    test_data['new' + str(i) + '*year'] = test_data['v_' + str(i)] * test_data['car_age_year']


print('删除相关性较低的特征')
corr_p = train_data.corrwith(train_data['price'])
corr_p = corr_p.reindex(corr_p.abs().sort_values().index)
train = train_data[corr_p[corr_p.abs() > 0.002].index]
lis = list(corr_p[corr_p.abs() > 0.002].index)[:-1]
test_data = test_data[lis]
print(test_data.columns)
print(train.columns)
print(test_data.shape)
print(train.shape)

train['price'] = np.log(train['price'])
X = np.asarray(train.drop('price',axis=1))
y = np.asarray(train['price'])
test_X = np.asarray(test_data)

model =  LGBMRegressor(
    n_estimators=10000,
    learning_rate=0.02,
    boosting_type= 'gbdt',
    objective = 'regression_l1',
    max_depth = -1,
    num_leaves=31,
    min_child_samples = 20,
    feature_fraction = 0.8,
    bagging_freq = 1,
    bagging_fraction = 0.8,
    lambda_l2 = 2,
    random_state=2020,
    metric='mae'
)
model1 = RandomForestRegressor(n_estimators=1000,verbose = 1,random_state=2020)

results = model.fit(X,y)
print('完成GBDT模型构建')
reg_X = np.asarray(train.drop(['price','power_bin','model_bin'],axis=1))
results1 = model1.fit(reg_X,y)
print('完成随机森林模型构建')

finall = results.predict(test_X)
finall = np.power(np.e,finall)

reg_test_X = np.asarray(test_data.drop(['power_bin','model_bin'],axis=1))
finall1 = results1.predict(reg_test_X)
finall1 = np.power(np.e,finall1)

print('GBDT',finall)
print('随机森林',finall1)

result_finall = (finall1 * 0.8 + finall * 0.2)
#result_finall = (finall1 + finall)*0.5
print(result_finall)
sub = pd.DataFrame()
sub['SaleID'] = test_data.index
sub['price'] = result_finall
sub.to_csv('./sub_Stacking.csv',index=False)