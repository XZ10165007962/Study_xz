import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

pd.set_option('display.max_columns',None)
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def load_data(data_path):
    data = pd.read_csv(data_path,sep='\t')
    return data


if __name__ == '__main__':
    train_data = load_data(r'E:\dataset\Industrial_steam_forecast\zhengqi_train.txt')
    test_data = load_data(r'E:\dataset\Industrial_steam_forecast\zhengqi_test.txt')

    #查看蒸汽数据分布，较符合正态分布
    '''f, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].set_title('蒸汽分布图')
    sns.distplot(train_data.target, bins=100, ax=ax[0])
    stats.probplot(train_data.target, plot=ax[1])
    plt.show()'''

    #print(train_data.info())#2888没有空值
    #print(test_data.info())#1925没有空值

    #查看数据分布
    '''sns.set()
    sns.pairplot(train_data,  kind='scatter', diag_kind='kde')
    plt.show()'''

    #查看数据训练、测试数据是否分布均匀
    #"V5","V9","V11","V17","V22","V28"不是同分布
    '''for column in train_data.columns[0:-1]:
        g = sns.distplot(train_data[column],color='Red',hist=False)
        g = sns.distplot(test_data[column],  color="Blue",hist=False)
        g.set_xlabel(column)
        g.set_ylabel("Frequency")
        g = g.legend(["train", "test"])
        plt.show()'''

    #train_data.drop(["V5", "V9", "V11", "V17", "V22", "V28"], axis=1, inplace=True)
    #test_data.drop(["V5", "V9", "V11", "V17", "V22", "V28"], axis=1, inplace=True)
    '''corr = train_data.corrwith(train_data['target'])
    corr = corr.reindex(corr.abs().sort_values().index)
    train_data_new = train_data[corr[corr.abs() > 0.1].index]
    lis = list(corr[corr.abs() > 0.1].index)[:-1]
    test_data = test_data.loc[:, lis]

    train_data_x = train_data_new.iloc[:,:-1]
    train_data_y = train_data_new.iloc[:,-1]'''
    train_data_x = train_data.iloc[:, :-1]
    train_data_y = train_data.iloc[:, -1]
    X,X_test,y,y_test = train_test_split(train_data_x,train_data_y,test_size=0.4,random_state=2020)

    line_model = Lasso(alpha=0.1)
    model = LGBMRegressor(
        n_estimators=10000,
        learning_rate=0.02,
        boosting_type='gbdt',
        objective='regression_l1',
        max_depth=-1,
        num_leaves=31,
        min_child_samples=20,
        feature_fraction=0.8,
        bagging_freq=1,
        bagging_fraction=0.8,
        lambda_l2=2,
        random_state=2020,
        metric='mae'
    )
    model.fit(X, y)
    print(mean_squared_error(model.predict(X_test), y_test))
    target = model.predict(test_data)
    print(target)
    line_model.fit(X, y)
    print(mean_squared_error(line_model.predict(X_test), y_test))
    line_target = line_model.predict(test_data)
    print(line_target)

    print(mean_squared_error((0.7*model.predict(X_test)+0.3*line_model.predict(X_test)),y_test))

    finall = 0.3*line_target+0.7*target

    sub = pd.DataFrame()
    sub['target'] = finall
    sub.to_csv('./sub_Stacking.txt', index=False,header=False)


