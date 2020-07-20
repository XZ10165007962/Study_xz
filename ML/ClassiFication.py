
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


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

def classify1(inX,dataSet):

    cent = np.mean(dataSet,axis=0)
    dataSetSize = np.shape(cent)
    diffMat = inX - cent
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum()
    print(sqDistances)
    distances = sqDistances ** 0.5
    print(distances)

def classify2(dataYang,dataTest):

    Y = dataYang['label']
    dataYang.drop('label',axis=1,inplace=True)
    X = min_max_scaler.transform(dataYang)
    X_train1,X_test,Y_train1,Y_test = train_test_split(dataYang,Y)
    lin_model = LogisticRegression().fit(X_train1,Y_train1)
    lin_model = SVC(C=1000, kernel="rbf", gamma=0.1)
    scores = cross_val_score(lin_model,X_test,Y_test,cv=5)
    print(scores)



if __name__ == '__main__':
    '''da = pd.DataFrame([[1,2,3],[3,4,5]])
    print(np.mean(da,axis=0))
    print(ai)'''
    leables = []
    dataYang = pd.read_csv(r'E:\dataset\718_rdkit\rdkit_train.csv',encoding='utf-8') # (779, 354)
    dataTest = pd.read_csv(r'E:\dataset\718_rdkit\rdkit_predict.csv',encoding='utf-8')#(4624, 354)
    #dataYang = dataYang[dataYang['label'] > 0]
    #dataYang.drop('label',axis=1,inplace=True)
    #dataYang.dropna(inplace=True)
    #dataTest.dropna(inplace=True)
    # 归一化数据
    dataYang.drop_duplicates(inplace=True )
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(dataYang)
    X_test = min_max_scaler.fit_transform(dataTest)
    #X_train = preprocessing.scale(dataYang)
    #X_test = preprocessing.scale(dataTest)
    data_std = X_train.std(ddof=0)
    classify2(dataYang,dataTest)
    #for i in X_test:
        #leables.append(classify0(i,X_train,10,1))
        #leables.append(classify1(i, X_train))
    num = 0
    for i in leables:
        if i == 1:
            num += 1
    err = float(num / len(X_test) * 100)
    print(err)
    print(leables)
    #dataTest['label'] = list(leables)
    #dataTest.to_csv(r'F:\data\63\datatest1.csv')
    #print('生成预测文件，算法运行结束')










