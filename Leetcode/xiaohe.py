

data = [1,3,4,2,5]
def insertionsort(data):
    he = 0
    if (not data or len(data) <= 1):
        return 0
    else:
        for i in range(len(data)):
            flg = data[i]
            for j in range(i):
                if data[i] < data[j]:
                    data[i], data[j] = data[j], data[i]
            he += merge(flg , data)
    return he

def merge(flg,data):
    num = data.index(flg)
    he = 0
    for i in range(num-1,-1,-1):
        he += data[i]
    return he



print(insertionsort(data))