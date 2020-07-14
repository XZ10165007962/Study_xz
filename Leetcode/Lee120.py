
'''
给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。

相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。

'''

'''
例如，给定三角形：

[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）
'''

'''
将三角形的第一个数据作为最小路径的第一个数，每次去寻找与它相邻的数据中最小的数作为最小路径。
错误
[[-1],[2,3],[1,-1,-3]]这种情况下不会通过
'''
def minimumTotal1(triangle):
    nums = 0
    index = -1
    if not triangle:
        return 0
    for i in triangle:
        if index == -1:
            nums += i[0]
            index = 0
        else:
            if (nums+i[index]) < (nums + i[index+1]):
                nums += i[index]
            else:
                nums += i[index+1]
                index += 1
        print(nums)
        print('--------',index)
    return nums


'''
因为三角形最大的数组长度为n，因此使用长度为n的临时数组存储路径长度
临时数组的第一个元素为三角形每行第一个元素之和
每次首先更新dp[i]，因为只有一个元素能跟它进行相加
中间元素保证路径相加最小，从后向前更新
'''
def minimumTotal(triangle):
    n = len(triangle)
    dp = [0] * n
    dp[0] = triangle[0][0]
    for i in range(1,n):
        dp[i] = dp[i-1] + triangle[i][i]
        for j in range(i-1,0,-1):
            dp[j] = min(dp[j-1],dp[j]) + triangle[i][j]
        dp[0] += triangle[i][0]
    return min(dp)

