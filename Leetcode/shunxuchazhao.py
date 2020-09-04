
data = [7,8,4,2,1,5,9,6,3]

def sequential_search(data,key):
    n = len(data)

    for i in range(n):
        if data[i] == key:
            print('成功查找到 %d' % key)
            return key
    return


def sequential_search2(data,key):
    n = len(data)

    i = n-1

    while(data[i] != key):
        i -= 1
        if i == 0:
            if (data[i] == key):

                print('成功查找到 %d' % key)
                return
            else:

                print('没有查找到该数')
                return
    print('成功查找到 %d' % key)
    return

sequential_search2(data,1)