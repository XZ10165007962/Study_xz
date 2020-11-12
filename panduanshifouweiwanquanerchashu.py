class Node(object):
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

    def isW(self):
        flag = 0
        res = [self]
        while res:
            tmp = res.pop(0)
            if tmp.left:
                res.append(tmp.left)
            if tmp.right:
                res.append(tmp.right)
            if tmp.right and not tmp.left:
                return False
            if flag:
                if tmp.left or tmp.right:
                    return False
            if (tmp.left and not tmp.right) or (not tmp.left and not tmp.right):
                flag = 1
        return True
if __name__ == '__main__':
    '''tree = Node(1)
    tree.left = Node(2)
    tree.right = Node(3)
    tree.left.left = Node(4)
    tree.left.left.left = Node(8)
    tree.left.right = Node(5)
    tree.right.left = Node(6)
    tree.right.right = Node(7)
    print(tree.isW())'''

    import datetime
    import time
    '''curr_time = datetime.datetime.now()
    print('curr_time',curr_time)
    time_str = datetime.datetime.strftime(curr_time, "%Y%m%d%H%M%S")[:-2]
    print('time_str',time_str)
    print(type(time_str))
    timearray = datetime.datetime.strptime(str(time_str),"%Y%m%d%H%M%S")
    print('timearray',timearray)

    end_time = curr_time + datetime.timedelta(minutes=-1)
    print(end_time)
    print(end_time.timetuple())
    timestamp = int(time.mktime(end_time.timetuple()))
    print(timestamp)

    timearray = time.strptime(str(time_str), "%Y%m%d%H%M%S")
    timestamp = int(time.mktime(timearray))
    print(timestamp)
    time_local = time.localtime(timestamp)
    print(time_local)

    print(datetime.datetime.now())'''

    '''current_time = 20200706000000
    timearray = datetime.datetime.strptime(str(current_time), "%Y%m%d%H%M%S")
    print(timearray)
    current_time1 = current_time + 100
    print(current_time1)
    timearray = datetime.datetime.strptime(str(current_time1), "%Y%m%d%H%M%S")
    print(timearray)
    print('------')

    timearray = time.strptime(str(current_time), "%Y%m%d%H%M%S")
    timestamp = int(time.mktime(timearray))
    print(timestamp)
    timearray = time.strptime(str(current_time1), "%Y%m%d%H%M%S")
    timestamp = int(time.mktime(timearray))
    print(timestamp)'''

    '''timestamp = 1570774556514

    # 转换成localtime
    time_local = time.localtime(timestamp / 1000)
    # 转换成新的时间格式(精确到秒)
    dt = time.strftime("%Y%m%d%H%M%S", time_local)
    print(dt)  # 2019-10-11 14:15:56

    d = datetime.datetime.fromtimestamp(timestamp / 1000)
    # 精确到毫秒
    str1 = d.strftime("%Y%m%d%H%M%S")
    td_datetime = datetime.datetime.strptime(str1, '%Y%m%d%H%M%S')
    td_datetime = td_datetime + datetime.timedelta(days=2)
    print(td_datetime)
    print(str1)  # 2019-10-11 14:15:56.514000

    insert_time = datetime.datetime.fromtimestamp(timestamp / 1000)
    insert_time = insert_time.strftime("%Y%m%d%H%M%S")
    insert_time = datetime.datetime.strptime(insert_time, '%Y%m%d%H%M%S')
    insert_time = insert_time - datetime.timedelta(minutes=30)
    print(insert_time)

    print('---------')
    i = 1570774556514
    insert_time = datetime.datetime.fromtimestamp(i / 1000)
    insert_time = insert_time.strftime("%Y%m%d%H%M%S")
    insert_time = datetime.datetime.strptime(insert_time, '%Y%m%d%H%M%S')
    insert_time = insert_time + datetime.timedelta(minutes=2)
    print(insert_time)

    a = '10'
    if isinstance(a,int):
        print('True')
    else:
        print('False')
        print(a)
        print(type(int(a)))'''

    '''import pandas as pd
    data1 = [1,2,3]
    df1 = pd.DataFrame(data1,columns=['eg_data'])
    data2 = [3,4,5]
    df2 = pd.DataFrame(data2,columns=['hg_data'])
    data = pd.concat([df1,df2],axis=1)
    print(data)
    print(data.corr())'''

    '''current_time = 20200706000000
    timearray = datetime.datetime.strptime(str(current_time), "%Y%m%d%H%M%S")
    print(timearray.timestamp()*1000)

    tie = int(datetime.datetime.now().timestamp()) * 1000
    print( tie)'''

    import random
    import duojincheng

    for _ in range(10):
        a = random.random()
        print('a:',a)
        a = a*0.02+0.97
        print('new a:',a)
