data = [5,2,6,7,1,8,9,0,3]

def insertionsort(data):
    if (not data or len(data) <= 1):
        print('没有要排序的数组')
    for i in range(len(data)):
        for j in range(i):
            if data[i] < data[j]:
                data[i],data[j] = data[j],data[i]

    return data

print(insertionsort(data))