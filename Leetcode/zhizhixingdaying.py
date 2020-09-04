data = [
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18]
]


import numpy as np

def zigZagPrintMatrix(data):

    m,n = np.mat(data).shape
    if m :
        pass
    else:
        return
    leftdown = [0,0]
    rightup = [0,0]
    turnup = True
    while(leftdown[1] < n-1):
        printLine(leftdown,rightup,turnup,data)
        turnup = not turnup
        if (leftdown[0] < m -1):
            leftdown[0] += 1
        else:
            leftdown[1] += 1
        if (rightup[1] < n -1):
            rightup[1] += 1
        else:
            rightup[0] += 1
        print(leftdown)
        print(rightup)


def printLine(leftdown,rightup,turnup,data):
    if turnup:
        i = leftdown[0]
        j = leftdown[1]
        while(j <= rightup[1]):
            print(data[i][j])
            i -= 1
            j += 1
    else:
        i = rightup[0]
        j = rightup[1]
        while(i <= leftdown[0]):
            print(data[i][j])
            i += 1
            j -= 1

zigZagPrintMatrix(data)