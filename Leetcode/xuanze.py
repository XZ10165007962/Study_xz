data = [5,2,6,7,1,8,9,0,3]
data1 = []
def selectinsort(data):
    length = len(data)
    if (not data or length<=1):
        print('数组长度为空或者数组长度为1')
    for i in range(length):
        min = i
        for j in range(i+1,length):
            if data[j] < data[min]:
                min = j
        if (min != i):
            data[i],data[min]=data[min],data[i]
    return data
print(selectinsort(data))