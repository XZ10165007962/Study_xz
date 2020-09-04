
data = [5,2,6,7,1,8,9,0,3]

def bubblesort(data):
    length = len(data)
    if (not data or length <= 1):
        return
    for i in range(length):
        for j in range(length-2,i-1,-1):
            if data[j] > data[j+1]:
                data[j],data[j+1] = data[j+1],data[j]
        print(data)

    return data

print(bubblesort(data))