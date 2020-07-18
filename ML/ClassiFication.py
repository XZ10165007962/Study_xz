
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing


pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.width",1000)


def classify0(inX,dataSet,k,j):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    '''sqDistances = dataDistance(diffMat)
    distances = sqDistances**0.5'''

    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    #获得距离索引的排序
    sortedDistIndicies = distances.argsort()
    print('---------')
    print('距离最近的j个点')
    print(distances[sortedDistIndicies[:j]])
    bestIndex = sortedDistIndicies[:j]
    bestDistances = np.mean(distances[sortedDistIndicies[:j]])
    print('平均值')
    print(bestDistances)
    '''diffMat1 = np.tile(dataSet[bestIndex], (dataSetSize, 1)) - dataSet
    sqDistances1 = dataDistance(diffMat1)
    distances1 = sqDistances1 ** 0.5'''
    bests = []
    for i in bestIndex:
        diffMat1 = np.tile(dataSet[i], (dataSetSize, 1)) - dataSet
        sqDiffMat1 = diffMat1 ** 2
        sqDistances1 = sqDiffMat1.sum(axis=1)
        distances1 = sqDistances1 ** 0.5
        # 获得距离索引的排序
        sortedDistIndicies1 = distances1.argsort()
        bestIndex1 = sortedDistIndicies1[1:k+1]
        print('距离最近的k个点')
        print(distances1[bestIndex1])
        meanDis = np.mean(distances1[bestIndex1])
        bests.append(meanDis)
    print('平均值')
    print(bests)
    meanBest = np.mean(bests)
    if meanBest > bestDistances:
        return 1
    else:
        return 0

if __name__ == '__main__':
    leables = []
    dataYang = pd.read_csv(r'F:\data\63\yang1.csv',encoding='utf-8').dropna(axis=1) # 806x200
    dataTest = pd.read_csv(r'F:\data\63\testnew.csv',encoding='utf-8')
    print(dataYang.head())
    print(dataTest.head())
    print(0/1)
    # 归一化数据
    dataYang.drop_duplicates(inplace=True )
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(dataYang)
    X_test = min_max_scaler.fit_transform(dataTest)
    #X_train = preprocessing.scale(dataYang)
    #X_test = preprocessing.scale(dataTest)
    for i in X_test:
        leables.append(classify0(i,X_train,10,2))
    num = 0
    for i in leables:
        if i == 1:
            num += 1
    err = float(num / len(X_test) * 100)
    print(err)
    print(leables)
    dataTest['label'] = list(leables)
    #dataTest.to_csv(r'F:\data\63\datatest1.csv')
    #print('生成预测文件，算法运行结束')










