import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
pd.set_option('display.max_columns',None)

Train_data = pd.read_csv(r'F:\data\二手车\used_car_train_20200313.csv',sep=' ')
Test_data = pd.read_csv(r'F:\data\二手车\used_car_testB_20200421.csv',sep=' ')
'''print(Train_data.shape)
print(Test_data.shape)
print(Train_data.head())
print(Train_data.columns)'''

Train_data['train'] = 1
Test_data['train'] = 0
data = pd.concat([Train_data,Test_data],ignore_index=True)
data['used_time'] = (pd.to_datetime(data['creatDate'],format='%Y%m%d',errors='coerce')-
                     pd.to_datetime(data['regDate'],format='%Y%m%d',errors='coerce')).dt.days

#print(data['used_time'].isnull().sum())
data['city'] = data['regionCode'].apply(lambda x : str(x)[:-3])
data = data

Train_gb = Train_data.groupby('brand')
all_info = {}
for kind,kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['brand_amount'] = len(kind_data)
    info['brand_price_max'] = kind_data.price.max()
    info['brand_price_median'] = kind_data.price.median()
    info['brand_price_min'] = kind_data.price.min()
    info['brand_price_sum'] = kind_data.price.sum()
    info['brand_price_std'] = kind_data.price.std()
    info['brand_price_average'] = round(kind_data.price.sum() / (len(kind_data) + 1),2)
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={'index':'brand'})
data = data.merge(brand_fe,how='left',on = 'brand')

bin = [i * 10 for i in range(31)]
data['power_bin'] = pd.cut(data['power'],bin,labels=False)

data = data.drop(['creatDate','regDate','regionCode'],axis=1)
'''print(data.shape)
print(data.columns)'''

#data.to_csv('data_for_tree.csv',index=0)
'''data['power'].plot.hist()
plt.show()'''


'''Test_data['power'].plot.hist()
plt.show()'''

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
data['power'] = np.log(data['power'] + 1 )
data['power'] = ((data['power'] - np.min(data['power'])) / (np.max(data['power']) - np.min(data['power'])))
'''data['power'].plot.hist()
plt.show()'''

# 当然也可以直接看图
data_numeric = data[['power', 'kilometer', 'brand_amount', 'brand_price_average',
                     'brand_price_max', 'brand_price_median']]
correlation = data_numeric.corr()

f , ax = plt.subplots(figsize = (7, 7))
plt.title('Correlation of Numeric Features with Price',y=1,size=16)
sns.heatmap(correlation,square = True,  vmax=0.8)
plt.show()