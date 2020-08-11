import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.regressor import StackingRegressor
from scipy import stats

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

train_data = pd.read_csv(r'E:\dataset\car\used_car_train_20200313.csv',sep=' ')
test_data = pd.read_csv(r'E:\dataset\car\used_car_testB_20200421.csv',sep=' ')
pd.set_option('display.max_columns',None)

#查看数据基本信息
#print(train_data.head())
#print(train_data.describe())
#print(train_data.info())
#print(test_data.info())

train_data = train_data[train_data.model.notnull()]

#处理时间列
train_data['used_time'] = (pd.to_datetime(train_data['creatDate'],format='%Y%m%d',errors='coerce')-
                     pd.to_datetime(train_data['regDate'],format='%Y%m%d',errors='coerce')).dt.days
train_data['time1'] = 19940101
train_data['time'] = (pd.to_datetime(train_data['regDate'],format='%Y%m%d',errors='coerce')-
                     pd.to_datetime(train_data['time1'],format='%Y%m%d',errors='coerce')).dt.days
train_data.drop(['time1','regDate','creatDate'],axis=1,inplace=True)
train_data['used_time'].fillna(0,inplace=True)
train_data['time'].fillna(0,inplace=True)

test_data['used_time'] = (pd.to_datetime(test_data['creatDate'],format='%Y%m%d',errors='coerce')-
                     pd.to_datetime(test_data['regDate'],format='%Y%m%d',errors='coerce')).dt.days
test_data['time1'] = 19940101
test_data['time'] = (pd.to_datetime(test_data['regDate'],format='%Y%m%d',errors='coerce')-
                     pd.to_datetime(test_data['time1'],format='%Y%m%d',errors='coerce')).dt.days
test_data.drop(['time1','regDate','creatDate'],axis=1,inplace=True)
test_data['used_time'].fillna(0,inplace=True)
test_data['time'].fillna(0,inplace=True)

#处理无用数据
train_data.drop(['SaleID','name','offerType'],axis=1,inplace=True)
test_data.drop(['SaleID','name','offerType'],axis=1,inplace=True)
#缺失值处理
'''
bodyType              4506
fuelType              8680
gearbox               5981
离散数据类型
'''
train_data['notRepairedDamage'].replace('-',np.nan,inplace=True)
from sklearn.neighbors import KNeighborsClassifier
KnnColumns = ['bodyType','fuelType','gearbox','notRepairedDamage']
data_copy = train_data.copy()
for i in KnnColumns:
    Knn_x = data_copy[data_copy[i].notnull()].drop(KnnColumns,axis=1)
    Knn_y = data_copy[i]
    train_x = Knn_x
    train_y = Knn_y[data_copy[i].notnull()]
    test_x = data_copy[data_copy[i].isnull()].drop(KnnColumns,axis=1)
    Knn_model = KNeighborsClassifier()
    Knn_model.fit(train_x, train_y)
    result = Knn_model.predict(test_x)
    data_copy[i][data_copy[i].isnull()] = list(result)
    print('完成 %s 的缺失值填充' % i )
train_data = data_copy

test_data['notRepairedDamage'].replace('-',np.nan,inplace=True)
data_copy = test_data.copy()
for i in KnnColumns:
    Knn_x = data_copy[data_copy[i].notnull()].drop(KnnColumns,axis=1)
    Knn_y = data_copy[i]
    train_x = Knn_x
    train_y = Knn_y[data_copy[i].notnull()]
    test_x = data_copy[data_copy[i].isnull()].drop(KnnColumns,axis=1)
    Knn_model = KNeighborsClassifier()
    Knn_model.fit(train_x, train_y)
    result = Knn_model.predict(test_x)
    data_copy[i][data_copy[i].isnull()] = list(result)
    print('完成 %s 的缺失值填充' % i )
test_data = data_copy

#相关性计算
corr_p = train_data.corrwith(train_data['price'])
corr_p = corr_p.reindex(corr_p.abs().sort_values().index)
# plt.figure(figsize=(15,5))
# plt.title('各变量与price的相关系数')
# corr_p.plot.bar(rot=30)
# plt.show()
train_data = train_data[corr_p[corr_p.abs() > 0.1].index]
lis = list(corr_p[corr_p.abs() > 0.1].index)[:-1]
test_data = test_data.loc[:,lis]


from sklearn.linear_model import LinearRegression,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
train_data['price'] = np.log(train_data['price'])
X = np.asarray(train_data.drop('price',axis=1))
y = np.asarray(train_data['price'])
test_X = np.asarray(test_data)


#scaler = StandardScaler()
#X = scaler.fit_transform(X)
#test_X = scaler.transform(test_X)


poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(X)
test_X = poly.transform(test_X)
print('完成特征多项式组合')
'''model = LinearRegression()
results = model.fit(X,y)
scores = cross_val_score(model,X,y)
print(scores)94'''

clf1 = LinearRegression()
clf2 = KNeighborsRegressor()
clf3 = SVR()
clf4 = DecisionTreeRegressor()

'''sclf = StackingRegressor(regressors=[clf1,clf2,clf3],meta_regressor=clf4)

results = sclf.fit(X,y)
scores = cross_val_score(sclf,X,y)
print(scores)'''
results = GradientBoostingRegressor(random_state=0).fit(X,y)
scores = cross_val_score(GradientBoostingRegressor(random_state=0),X,y)
print(scores)
finall = results.predict(test_X)
finall = np.power(np.e,finall)
print(finall)
sub = pd.DataFrame()
sub['SaleID'] = test_data.index
sub['price'] = finall
sub.to_csv('./sub_Stacking.csv',index=False)

