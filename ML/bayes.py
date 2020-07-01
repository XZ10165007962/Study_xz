import numpy as np

def loadDataSet():
    postingList=[['my','dog','has','flea',
                  'problems','help','please'],
                 ['maybe','not','take','him',
                  'to','dog','park','stupid'],
                 ['my','dalmation','is','so','coute',
                  'I','love','him'],
                 ['stop','posting','stupid','worthless',
                  'garbage'],
                 ['mr','licks','ate','my','steak','how',
                  'to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]

    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word : %s is not in my Vocabularry' % word)
    return returnVec

def trainNB0(trainMartix,trainCategory):
    numTrainDocs = len(trainMartix)
    numWords = len(trainMartix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMartix[i]
            p1Denom += sum(trainMartix[i])
        else:
            p0Num += trainMartix[i]
            p0Denom += sum(trainMartix[i])

    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

listOposts , listClasses = loadDataSet()
myVocabList = createVocabList(listOposts)
trainMat = [setOfWords2Vec(myVocabList,listOposts[0])]
print(trainMat)
p0,p1,pab = trainNB0(trainMat,listClasses)
print(p0)
print(p1)
print(pab)

