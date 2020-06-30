import numpy as np


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print('This matrix is singular,cannot do inverse')
        return
    #ws为列向量
    ws = xTx.I * (xMat.T*yMat)
    return ws

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = xMat.shape[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = np.exp(np.power(diffMat*diffMat.T,0.5)/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0:
        print('This matrix is singular,cannot do inverse')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = testArr.shape[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def rigdeRegres(xMat,yMat,lam = 0.2):
    xTx = xMat.T*xMat
    denom = xTx + np.eye(xMat.shape[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

def rigdeTest(xArr,yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat,axis=0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat,axis=0)
    xVar = np.var(xMat,axis=0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts,xMat.shape[1]))
    for i in range(numTestPts):
        ws = rigdeRegres(xMat,yMat,np.exp(i-10))
        wMat[i,:] = ws.T
    return wMat

def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, axis=0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, axis=0)
    xVar = np.var(xMat, axis=0)
    xMat = (xMat - xMeans) / xVar
    m,n = np.shape(xMat)
    returnMat = np.zeros((numIt,n))
    ws = np.zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1,1]:
                #首先将全部特征权重置为0
                wsTest = ws.copy()
                #通过循环将各个权重进行初始化
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                #计算损失大小，如果此次改变的权重使得模型损失减小，则将损失大小变为较小的损失
                #并且将此次权重改变记录下来
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    #观察最后输出的权重矩阵，如果某个特征权重为0，则说明该权重不重要，可以将其删除
    return returnMat

xArr,yArr = loadDataSet(r'F:\电子书\machinelearninginaction-master\machinelearninginaction-master\Ch08\abalone.txt')
xMat = np.mat(xArr)
yMat = np.mat(yArr)
ws = stageWise(xMat,yMat)

'''xArr,yArr = loadDataSet(r'F:\电子书\machinelearninginaction-master\machinelearninginaction-master\Ch08\ex0.txt')
xMat = np.mat(xArr)
yMat = np.mat(yArr)
yHat = lwlrTest(xMat,xMat,yMat,0.09)

srtInd = xMat[:,1].argsort(0)
xSort = xMat[srtInd][:,0,:]

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:,1],yHat[srtInd])
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0],s=2,c='red')
plt.show()'''
'''import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy*ws
ax.plot(xCopy[:,1],yHat)
plt.show()'''

