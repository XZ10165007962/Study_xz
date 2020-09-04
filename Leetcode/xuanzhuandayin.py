
data = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]
]


def circledata(data):
    le = len(data)
    leftup = [0,0]
    rightdown = [le-1,le-1]
    while(leftup[0] < rightdown[0] and leftup[1] < rightdown[1]):
        p1=[leftup[0],leftup[1]]
        p2=[leftup[0],rightdown[1]]
        p3=[rightdown[0],rightdown[1]]
        p4 = [rightdown[0], leftup[1]]
        while(p1[1] < rightdown[1]):

            data[p1[0]][p1[1]] , data[p2[0]][p2[1]] = data[p2[0]][p2[1]],data[p1[0]][p1[1]]
            data[p1[0]][p1[1]] , data[p3[0]][p3[1]] = data[p3[0]][p3[1]], data[p1[0]][p1[1]]
            data[p1[0]][p1[1]] , data[p4[0]][p4[1]] = data[p4[0]][p4[1]], data[p1[0]][p1[1]]

            p1[1] += 1
            p2[0] += 1
            p3[1] -= 1
            p4[0] -= 1
        leftup[0] += 1
        leftup[1] += 1
        rightdown[0] -= 1
        rightdown[1] -= 1
    print(data)

circledata(data)