
data = [1,5,7,9,11,15,16,19,20,24,26,28,64,69,99]

def binarysearch(data,key):

    start = 0
    end = len(data)

    while(start <= end):
        mid = int((start + end) / 2)
        if (key < data[mid]):
            end = mid-1
        elif(key > data[mid]):
            start = mid+1
        else:
            print('查找到关键字%d，在数组的第%d个位置'%(key,mid))
            return mid
    print('没有查找到该数')
    return


binarysearch(data,0)