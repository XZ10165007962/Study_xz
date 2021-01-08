'''from threading import Thread,current_thread
import time

def run(n):
    print(current_thread().name,"run")
    time.sleep(n)
    print(current_thread().name,'sleep',n)

if __name__ == '__main__':
    t1 = Thread(target=run,args=(2,))
    t2 = Thread(target=run,args=(3,))
    t1.start()
    t2.start()'''

'''from threading import Thread,current_thread
import time

class MyThread(Thread):
    
    def __init__(self,n):
        super(MyThread, self).__init__()
        self.n = n

    def run(self):
        print(current_thread().name, "run")
        time.sleep(self.n)
        print(current_thread().name, 'sleep', self.n)

if __name__ == '__main__':
    t1 = MyThread(2)
    t2 = MyThread(3)
    print(t1.getName())
    print(t1.is_alive())
    t1.start()
    t2.start()'''

'''import threading
import time
def run(n):
    print(threading.current_thread().name,'run')
    time.sleep(n)
    print(threading.current_thread().name,'end')

if __name__ == '__main__':
    t1 = threading.Thread(target=run,args=(2,))
    t2 = threading.Thread(target=run,args=(3,))
    t2.setDaemon(True)#将t2设置为守护线程
    t1.start()
    t2.start()
    print('main end')#主代码块运行结束，等待非守护线程运行结束'''

'''import threading
import time
from threading import Lock

def work(count = 10):
    flag = False
    print('i am work for you')
    while True:
        try:
            lock.acquire()
            if len(cups) >= count:
                flag = True
            time.sleep(0.01)
            if not flag:
                cups.append(1)
        finally:
            lock.release()
        if flag:
            break
    print(len(cups))
if __name__ == '__main__':
    cups = []
    work_list = []
    lock = Lock()
    for i in range(10):
        t = threading.Thread(target=work,args=(1000,))
        work_list.append(t)
        t.start()

    print('main end')'''

'''from threading import Thread,current_thread,Lock,RLock
import time

def task1(lock1,lock2):
    if lock1.acquire():
        print('%s获取到lock1的锁'%current_thread().name)
        for i in range(5):
            print('%s ----------> %d'%(current_thread().name,i))
            time.sleep(0.01)
        if lock2.acquire():
            print('%s获取到lock1,lock2的锁'%current_thread().name)
            lock2.release()
        lock1.release()

def task2(lock1,lock2):
    if lock2.acquire():
        print('%s获取到lock2的锁' % current_thread().name)
        for i in range(5):
            print('%s ----------> %d' % (current_thread().name , i))
            time.sleep(0.01)
        if lock1.acquire():
            print('%s获取到lock1,lock2的锁' % current_thread().name)
            lock1.release()
        lock2.release()

if __name__ == '__main__':
    lock1 = lock2= RLock()
    t1 = Thread(target=task1,args=(lock1,lock2))
    t2 = Thread(target=task2,args=(lock1,lock2))

    t1.start()
    t2.start()'''

'''from queue import  Queue
from threading import Thread,current_thread
import time
import random

def producer(queue):
    print('{}开门了'.format(current_thread().name))
    foods = ['红烧狮子头', '香肠烤饭', '蒜蓉生蚝', '酸辣土豆丝', '肉饼']
    for i in range(20):
        food = random.choice(foods)
        print('{}正在加工中'.format(food))
        time.sleep(1)
        print('{}做完了'.format(food))
        queue.put(food)
    queue.put(None)

def consumer(queue):
    print('{}来吃饭了'.format(current_thread().name))
    while True:
        food = queue.get()
        if food:
            print('正在享用美食{}'.format(food))
            time.sleep(0.5)
        else:
            print('{}把饭店吃光了，走人。。。'.format(current_thread().name))
            break

if __name__ == '__main__':
    queue = Queue(8)
    t1 = Thread(target=producer,name = '肉饼',args=(queue,))
    t2 = Thread(target=consumer,name='坤坤',args=(queue,))

    t1.start()
    t2.start()'''


'''from threading import Thread,current_thread,Semaphore
import time
import random

def go_publice_wc():
    sem.acquire()
    print('{}正在上厕所'.format(current_thread().name))
    time.sleep(0.5)
    sem.release()

if __name__ == '__main__':
    sem = Semaphore(5)

    for i in range(20):
        t = Thread(target=go_publice_wc)
        t.start()'''

#利用event模拟红绿灯
'''from threading import Thread,Event
import time


def lighter():

    count = 0
    event.set() #初始值为绿灯
    while True:
        if 5 < count <= 10:
            event.clear() #红灯，清除标志位
            print("\33[41;1mred light is on...\033[0m")
        elif count > 10:
            event.set() #绿灯，设置标志位
            count = 0
        else:
            print("\33[42;1mgreen light is on...\033[0m")

        time.sleep(5)
        count += 1

def car(name):

    while True:
        if event.isSet():
            print("[%s] running..." % name)
            time.sleep(2)
        else:
            print("[%s] sees red light,waiting..." % name)
            event.wait()
            print("[%s] green light is on,start going..." % name)



if __name__ == '__main__':
    event = Event()

    light = Thread(target=lighter, )
    light.start()

    car = Thread(target=car, args=("MINI",))
    car.start()'''

'''import time
from threading import Thread, Condition,current_thread

def consumer():
    with condation:
        print('wait for product')
        condation.wait()
        print(current_thread().name,'get {}'.format(product))
        condation.notify()

def producer():
    product.append(10)
    with condation:
        condation.notify()
        print('notify')
        condation.wait()
    product.append(20)
    with condation:
        condation.notify()
        print('notify')

if __name__ == '__main__':
    condation = Condition()
    product = []

    c1 = Thread(target=consumer,name='c1')
    c2 = Thread(target=consumer,name='c2')
    p = Thread(target=producer,name='p')

    c1.start()
    c2.start()
    p.start()'''

'''import threading
from time import sleep

# 商品
product = 500
# 条件变量
con = threading.Condition(threading.Lock())


# 生产者类
# 继承Thread类
class Producer(threading.Thread):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def run(self):
        global product
        while True:
            # 如果获得了锁
            if con.acquire():
                # 处理产品大于等于500和小于500的情况
                if product > 100:
                    # 如果大于等于500，Producer不需要额外操作，于是挂起
                    con.wait()
                else:
                    product += 50
                    message = self.name + " produced 50 products."
                    print(message)
                    # 处理完成，发出通知告诉Consumer
                    con.notify()
                # 释放锁
                con.release()
                sleep(1)


# 消费者类
# 继承Thread类
class Consumer(threading.Thread):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def run(self):
        global product
        while True:
            # 如果获得了锁
            if con.acquire():
                # 处理product小于等于100和大于100的两种情况
                if product <= 100:
                    # 如果小于等于100，Consumer不需要额外操作，于是挂起
                    con.wait()
                else:
                    product -= 10
                    message = self.name + " consumed 10 products."
                    print(message)
                    # 处理完成，发出通知告诉Producer
                    con.notify()
                # 释放锁
                con.release()
                sleep(1)


def main():
    # 创建两个Producer
    for i in range(2):
        p = Producer('Producer-%d' % i)
        p.start()
    # 创建三个Consumer
    for i in range(3):
        c = Consumer('Consumer-%d' % i)
        c.start()


if __name__ == '__main__':
    main()'''

# 导入线程模块
'''import threading


def plyer_display():
    print('初始化通过完成，音视频同步完成，可以开始播放....')





def player_init(statu):
    while True:
        print(statu)
        try:
            # 设置超时时间，如果2秒内，没有达到障碍线程数量，
            # 会进入断开状态，引发BrokenBarrierError错误
            barrier.wait(2)
        except Exception as e:  # 断开状态，引发BrokenBarrierError错误
            # print("断开状态... ")
            continue
        else:
            print("xxxooyyyxxxooyyyxxxooyyy")
            break


if __name__ == '__main__':
    # 设置3个障碍对象
    barrier = threading.Barrier(3, action=plyer_display, timeout=None)
    statu_list = ["init ready", "video ready", "audio ready"]
    thread_list = list()
    for _ in range(3):
        for i in range(0, 3):
            t = threading.Thread(target=player_init, args=(statu_list[i],))
            t.start()

            thread_list.append(t)
            if i == 1:  # 重置状态
                print("不想看爱情片，我要看爱情动作片....")
                barrier.reset()

    for t in thread_list:
        t.join()'''


'''from concurrent.futures import ThreadPoolExecutor
import threading
import time

# 定义一个准备作为线程任务的函数
def action(max):
    my_sum = 0
    for i in range(max):
        print(threading.current_thread().name + '  ' + str(i))
        my_sum += i
    return my_sum
# 创建一个包含2条线程的线程池
pool = ThreadPoolExecutor(max_workers=2)
# 向线程池提交一个task, 50会作为action()函数的参数
future1 = pool.submit(action, 50)
# 向线程池再提交一个task, 100会作为action()函数的参数
future2 = pool.submit(action, 100)
# 判断future1代表的任务是否结束
print(future1.done())
time.sleep(3)
# 判断future2代表的任务是否结束
print(future2.done())
# 查看future1代表的任务返回的结果
print(future1.result())
# 查看future2代表的任务返回的结果
print(future2.result())
# 关闭线程池
pool.shutdown()'''

import threading


def _show_id_name():
    print(threading.current_thread().name)
    print(threading.current_thread().ident)
    print(threading.get_ident())


th = threading.Thread(target=_show_id_name, name='testing thread',
                      args=(), daemon=True)
th1 = threading.Thread(target=_show_id_name, name='testing1 thread',
                      args=(), daemon=True)

th.start()
th1.start()
th.join()
th1.join()