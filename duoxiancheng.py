import threading
import time,os

'''def run(n):
    print('task',n)
    time.sleep(1)
    print(n,'2s')
    time.sleep(1)
    print(n,'1s')
    time.sleep(1)
    print(n,'0s')
    time.sleep(1)

if __name__ == '__main__':
    t1 = threading.Thread(target=run,args=('t1',))
    t2 = threading.Thread(target=run, args=('t2',))
    t1.start()
    t1.join()
    t2.start()
    t2.join()'''


'''class MyThread(threading.Thread):
    def __init__(self,n):
        super(MyThread, self).__init__()
        self.n = n

    def run(self):
        print('task', self.n)
        time.sleep(1)
        print(self.n, '2s')
        time.sleep(1)
        print(self.n, '1s')
        time.sleep(1)
        print(self.n, '0s')
        time.sleep(1)

if __name__ == '__main__':
    t1 = MyThread('t1')
    t2 = MyThread('t2')
    t1.start()
    t1.join()
    t2.start()'''


'''def run(n):
    print('task', n)
    time.sleep(1)
    print(n, '2s')
    time.sleep(1)
    print(n, '1s')
    time.sleep(1)
    print(n, '0s')
    time.sleep(1)

if __name__ == '__main__':
    t = threading.Thread(target=run,args=('t1',))
    t.setDaemon(True)
    t.start()
    t.join()
    print('end')'''


'''g_num = 100

def work1():
    global g_num
    for i in range(3):
        g_num += 1
        print('in work1 g_num is : %d' % g_num)
        time.sleep(1)

def work2():
    global g_num
    print('in work2 g_num is : %d ' % g_num)

if __name__ == '__main__':
    t1 = threading.Thread(target=work1)

    t2 = threading.Thread(target=work2)
    t1.start()
    #time.sleep(1)
    t2.start()'''


'''def work(name):
    global n
    #lock.acquire()
    for i in range(10):
        n += 1
        time.sleep(0.5)
        print(name,n)

    #lock.release()


if __name__ == '__main__':
    lock = threading.Lock()
    n = 100
    l = []
    for i in range(10):
        p = threading.Thread(target=work,args=(i,))
        l.append(p)

    for p in l:
        p.start()
        p.join()'''


'''def Func(i):
    global fl_num
    lock.acquire()
    fl_num += 1
    time.sleep(1)
    print(fl_num)
    lock.release()

if __name__ == '__main__':
    fl_num = 0
    lock = threading.RLock()
    for i in range(10):
        t = threading.Thread(target=Func,args=(i,))
        t.start()'''


'''def run(n,semaphore):
    semaphore.acquire()  # 加锁
    time.sleep(1)
    print("run the thread:%s\n" % n)
    semaphore.release()

if __name__ == '__main__':
    num = 0
    semaphore = threading.BoundedSemaphore(5)
    for i in range(22):
        t = threading.Thread(target=run,args=('t-%s' %i,semaphore))
        t.start()
        print(threading.active_count())
    while threading.active_count() != 1:
        pass
    else:
        print('-----all threads done-----')'''

'''def loop():
    print('thread %s is running ...'%threading.current_thread().name)
    n = 0
    while n < 5:
        n = n+1
        print('thread %s >>> %s'%(threading.current_thread().name,n))
        time.sleep(1)
    print('thread %s ended'%threading.current_thread().name)

print('thread %s is running...'%threading.current_thread().name)
t = threading.Thread(target=loop,name='loopThread')
t.start()
t.join()
print('thread %s ended.' % threading.current_thread().name)'''

'''balance = 0
lock = threading.Lock()
def change_it(n):
    global balance
    balance = balance + n
    balance = balance - n

def run_thread(n):
    for i in range(2000000):
        try:
            lock.acquire()
            change_it(n)
        finally:
            lock.release()
t1 = threading.Thread(target=run_thread,args=(5,))
t2 = threading.Thread(target=run_thread,args=(8,))

t1.start()
t2.start()
t1.join()
t2.join()
print('main',balance)'''


import datetime

def get_timestamp(current_time):
    timearray=time.strptime(str(current_time),"%Y%m%d%H%M%S")
    print(timearray)
    print(time.mktime(timearray))
    print(time.gmtime(time.mktime(timearray)))
    timestamp=int(time.mktime(timearray))
    timestamp=timestamp*1000
    return timestamp

begin_time=20200706000000
print(get_timestamp(begin_time))