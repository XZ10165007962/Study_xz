
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,f1_score,recall_score


pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.width",1000)


def classify0(inX,dataSet,k,j):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet

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
        np.row_stack((dataSet,inX))
        return 1
    else:
        return 0


if __name__ == '__main__':
    # shape:(605864, 102)
    # train_data = pd.read_excel(r'E:\dataset\jiangxi\deluxe_train_data_0914.xlsx')
    train_data = pd.read_excel(r'E:\jiangxi\deluxe_train_data_0914.xlsx').fillna(0)

    # shape:(157834, 102)
    test_data = pd.read_excel(r'E:\jiangxi\deluxe_test_data_0914.xlsx').fillna(0)
    #test_data = pd.shuffle(test_data)

    #获取正样本
    train_data = train_data[train_data['IS_DELUXE'] != 0]
    train_data = train_data.iloc[:,:-1]
    test_data = test_data.iloc[:,:-1]
    #打乱数据
    #train_data = pd.shuffle(train_data)

    min_max_scatter = MinMaxScaler()
    train_data = min_max_scatter.fit_transform(train_data)
    test_data = min_max_scatter.fit_transform(test_data)

    #存储预测标签
    leable = []
    k = 5
    j = 5
    number = 0
    for i in test_data:
        print('行数',number)
        leable.append(classify0(i,train_data,k,j))
        print(leable)
        number += 1
    print('完成模型预测')
    print('模型标签')
    print(leable)
    accuracy = accuracy_score(leable,test_data['IS_DELUXE'])
    print('accuracy %0.5f' % accuracy)

    f1 = f1_score(leable,test_data['IS_DELUXE'])
    print('f1 %0.5f' % f1)

    recall = recall_score(leable,test_data['IS_DELUXE'])
    print('recall %0.5f' % recall)

