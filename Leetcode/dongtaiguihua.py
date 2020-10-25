'''
1.动态规划就是根据状态转移方程计算出求解结果例如f(n) = f(n+1)+2
2.保存并复用以往结果：
动态规划的主要思想：一个大问题往往可以分解成许多小问题进行解答，并且小问题的解可以重复利用
3.按顺序从小往大计算
'''
import numpy as np
import time
#斐波那契数列 递归
def fib(n):
    if n < 2:
        return n
    else:
        return fib(n-1) + fib(n-2)
#斐波那契数列 动态规划
def fib1(n):
    result = list(range(n + 1))
    print(result)
    for i in range(n + 1):
        if i < 2:
            result[i] = i
        else:
            result[i] = result[i-1] + result[i-2]

    return result[-1]
#不同路径问题  动态规划
'''
一个机器人位于m*n网格的左上角，机器人每次只能向下跟向右移动，机器人想要到达右下角，总共有多少不同的路径
机器人到达f(m,n)地，只与f(m-1,n),f(m,n-1)两个点相关
如果用f(m,n)存储到达该点的路径的条数，则状态转移方程为：f(m,n) = f(m-1,n) + f(m,n-1)
'''
def count_paths(m,n):
    result = np.ones((m,n))
    print(result)
    # 数组最外圈都是1，只有一条路径
    for i in range(1,m):
        for j in range(1,n):
            result[i,j] = result[i-1,j] + result[i,j-1]

    return result[m-1,n-1]


#最长连续递增子序列
def MaxChildArray(data):
    dp = np.ones((len(data),1))
    for i in range(1,len(data)):
        if data[i] > data[i-1]:
            dp[i] = dp[i-1] + 1

    return max(dp)

#最长不连续递增子序列
def MaxChildArrayOrder(data):
    dp = np.ones((len(data),1))
    for i in range(1,len(data)):
        for j in range(i):
            if data[i] > data[j]:
                dp[i] = dp[j]+1
    print(dp)
    return max(dp)

#数组最大连续子序列和
def maxSubArray(data):
    cursum = 0
    max =  data[0]
    for i in range(len(data)):
        if (cursum <= 0):
            cursum = data[i]
        else:
            cursum += data[i]

        if cursum > max:
            max = cursum
        print('max',max)
        print('cursum',cursum)
    return max

def maxSubArray1(data):

    datamax = data[0]
    sum = data[0]
    for i in range(1,len(data)):
        sum = max(sum+data[i] , data[i])
        if sum >= datamax:
            datamax = sum
        print('datamax',datamax)
        print('sum',sum)
    return datamax

#数字塔从上到下所有路径中和最大的路径
def minNumberInRotateArray():
    data = [[3],[1,5],[8,4,3],[2,6,7,9],[6,2,3,5,1]]
    dp = [0] * len(data)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if j == 0:
                dp[j] = dp[j] + data[i][j]
            elif i == j:
                dp[j] = dp[j-1] + data[i][j]

    return max(dp)

#最长公共子串
def MaxTwoArraySameOrder():
    str1 = "BDCABA"
    str2 = "ABCBDAB"
    m = len(str1)
    n = len(str2)

    dp = [[0 for i in range(len(str2)+1)] for j in range(len(str1)+1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[ i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
            print(dp)
            print('-----')
        print('===')
    return dp[-1][-1]

#背包问题
def bag_0_1(weight, value, weight_most):  # return max value
    num = len(weight)
    weight.insert(0, 0)  # 前0件要用
    value.insert(0, 0)  # 前0件要用
    bag = np.zeros((num + 1, weight_most + 1), dtype=np.int32)  # 下标从零开始
    for i in range(1, num + 1):
        for j in range(1, weight_most + 1):
            if weight[i] <= j:
                bag[i][j] = max(bag[i - 1][j - weight[i]] + value[i], bag[i - 1][j])
            else:
                bag[i][j] = bag[i - 1][j]
    print(bag)
    return bag[-1, -1]

if __name__ == '__main__':
    weight = [2, 2, 6, 5, 4,1]
    value = [3, 6, 5, 4, 6,7]
    weight_most = 10
    result = bag_0_1(weight, value, weight_most)
    print(result)

