

'''
给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种？

输入: 3
输出: 5
解释:
给定 n = 3, 一共有 5 种不同结构的二叉搜索树:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3

'''

'''
考虑每个树都是有左右子树的，二叉搜索树是左孩子比根节点小，右孩子比根节点大
每个数字都有做根节点的机会
因此考虑只有一个数时，只有根节点
有两个数时，都是左子树，都是右子树，左右子树都有，因此是f(2)=f(0)f(1)+f(1)f(1)+f(1)f(0)
......
有多个数时都是一样
考虑递归实现
'''
import math
def numTrees1(n):
    i = 1
    result = 0
    if n == 0 : return 0
    elif n == 1 :return 1
    elif n == 2 :return 2
    while(n > 1):
        result += math.pow(2,(n-1))
        n -= math.pow(2,i)
    return int(result+i)

def numTrees(n):
    store = [1, 1]  # f(0),f(1)
    if n <= 1:
        return store[n]
    for m in range(2, n + 1):
        s = m - 1
        count = 0
        for i in range(m):
            count += store[i] * store[s - i]
        store.append(count)
    return store[n]


print(numTrees(4))