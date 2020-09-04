
data = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]
]

def printMatrixCircled(data):
    leftup = [0,0]
    rightdown = [len(data)-1,len(data)-1]
    while(leftup[0] < rightdown[0] and leftup[1] < rightdown[1]):
        i = leftup[0]
        j = leftup[1]
        while(j < rightdown[1]):
            print(data[i][j])
            j += 1
        while (i < rightdown[0]):
            print(data[i][j])
            i += 1
        while(j > leftup[1]):
            print(data[i][j])
            j -= 1
        while(i > leftup[0]):
            print(data[i][j])
            i -= 1
        leftup[0] += 1
        leftup[1] += 1
        rightdown[0] -= 1
        rightdown[1] -= 1

printMatrixCircled(data)