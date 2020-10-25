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

    current_time = 20200706000000
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
    print(timestamp)

    '''import pandas as pd
    data1 = [1,2,3]
    df1 = pd.DataFrame(data1,columns=['eg_data'])
    data2 = [3,4,5]
    df2 = pd.DataFrame(data2,columns=['hg_data'])
    data = pd.concat([df1,df2],axis=1)
    print(data)
    print(data.corr())'''

