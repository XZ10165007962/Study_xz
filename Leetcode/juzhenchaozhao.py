import numpy as np
data = [
    [1,2,3,4],
    [2,3,4,5],
    [3,4,5,6],
    [4,6,8,9]
]

def find_number(num,data):
    n,m = np.array(data).shape
    i = 0
    j = m -1
    while(i <= n -1 and j > -1):
        if (data[i][j] > num):
            j -= 1
        elif (data[i][j] < num):
            i += 1
        else:
            return True
    return False

if (find_number(9,data)):
    print('find')
else:
    print('not exist')