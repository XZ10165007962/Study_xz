import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

import random
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem.EState import Fingerprinter
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import ShuffleSplit
import joblib


x_train = pd.read_csv(r'E:\dataset\smile\smile_train.csv', encoding='utf-8')
x_test = pd.read_csv(r'E:\dataset\smile\smile_predict.csv', encoding='utf-8')

y = x_train['label']
x_train = x_train['smile']
x_test = x_test['smile']

smi_data = []
smi_data_test = []
activity_data = []

for m in x_train:
    mol = Chem.MolFromSmiles(m)
    smi_data.append(mol)

for m in x_test:
    mol = Chem.MolFromSmiles(m)
    smi_data_test.append(mol)

for active in y:
    activity_data.append(active)


def calcMorganFingerprint(mols):
    fps = []
    for mol in mols:
        arr = np.zeros((0,))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    fps = np.array(fps, dtype=np.float)
    return fps


validation_size = 0.2
seed = 10


fp_data = calcMorganFingerprint(smi_data)

X_train,X_test,y_train,y_test = train_test_split(fp_data,
                                                 activity_data,
                                                 test_size=validation_size,
                                                 random_state=seed)

model = LogisticRegression(penalty='l1',max_iter=10,tol=0.0001).fit(X_train,y_train)
#model = SVC(C=1000, kernel="rbf", gamma=0.1).fit(X_train, y_train)
value = model.predict(X_test)
finall = model.predict(calcMorganFingerprint(smi_data_test))
for i in finall:
    print(i,end="")
print(np.array(value))
print(np.array(y_test))
print(metrics.accuracy_score(value, y_test))
