import numpy as np
data = [
    [1, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1]
]
m,n = np.array(data).shape
def getIslandNums(data):
    res = 0

    for i in range(m):
        for j in range(n):
            if (data[i][j] == 1):
                res += 1
                infect(data,i,j)
    return res

def infect(data,i,j):
    if(i < 0 or i >= m or j < 0 or j >= n or data[i][j] != 1):
        return
    print(i,j)
    data[i][j] = 2
    infect(data,i-1,j)
    infect(data,i+1,j)
    infect(data,i,j-1)
    infect(data,i,j+1)

print(getIslandNums(data))