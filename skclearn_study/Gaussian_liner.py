
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns

class GPR:
    def __init__(self, h):
        self.is_fit = False
        self.train_x, self.train_y = None, None
        self.h = h

    def fit(self, x, y):
        self.train_x = np.asarray(x)
        self.train_y = np.asarray(y)
        self.is_fit = True

    def predict(self, x):
        if not self.is_fit:
            print("Sorry! GPR Model can't fit!")
            return

        x = np.asarray(x)
        kff = self.kernel(x, x)
        kyy = self.kernel(self.train_x, self.train_x)
        kfy = self.kernel(x, self.train_x)
        kyy_inv = np.linalg.inv(kyy + 1e-8 * np.eye(len(self.train_x)))

        mu = kfy.dot(kyy_inv).dot(self.train_y)
        return mu

    # 定义核函数
    def kernel(self, x1, x2):
        m, n = x1.shape[0], x2.shape[0]
        dist_matrix = np.zeros((m, n), dtype=float)
        for i in range(m):
            for j in range(n):
                dist_matrix[i][j] = np.sum((x1[i] - x2[j]) ** 2)
        return np.exp(-0.5 / self.h ** 2 * dist_matrix)


# 创造训练集
'''train_x = np.arange(0, 10).reshape(-1, 1)
train_y = np.cos(train_x) + train_x
# 制造槽点
train_y = train_y + np.random.normal(0, 0.01, size=train_x.shape)'''

boston = datasets.load_boston()
data = boston.data
test = boston.target

train_x ,test_x, train_y, test_y = train_test_split(data,test,test_size=0.3,random_state=2020)
# 显示训练集的分布
plt.figure()
plt.scatter(train_x[:,12], train_y, label="train", c="red", marker="x")
plt.legend()


# 创建训练集
#test_x = np.arange(0, 10, 0.1).reshape(-1,1)

# 针对不同h值得到拟合图像
h=0.1
for i in range(10):
    gpr = GPR(h)
    gpr.fit(train_x, train_y)
    mu = gpr.predict(test_x)
    #test_y = mu.ravel()
    plt.figure()
    plt.title("h=%.2f"%(h))
    plt.plot(mu.ravel(), test_y, label="predict")
    plt.scatter(train_x[:,12], train_y, label="train", c="red", marker="x")
    plt.legend()
    plt.show()
    h += 0.1
