
def partition(arr, low, high,privot):

    while(low < high):
        while(low < high and arr[high] >= privot):
            high -= 1
        arr[high],arr[low] = arr[low],arr[high]
        while(low < high and arr[low] <= privot):

            low += 1
        arr[high], arr[low] = arr[low], arr[high]

    return low


data1 = [4,5,9,6,3,2,1,41,58,96]
#partition(data,0,len(data)-1,8)
data = []
target = 8
data.append(target)
data.extend(data1)
partition(data,0,len(data)-1,target)
print(data)
