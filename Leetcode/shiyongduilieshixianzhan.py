class Queue(object):
    def __init__(self):
        self.items = []

    def size(self):
        return len(self.items)

    def inEmpty(self):
        return len(self.items) == 0

    def travel(self):
        for i in self.items:
            print(i, end=' ')
        print('')

    def push(self,item):
        self.items.append(item)

    def pop(self):
        if self.inEmpty():
            print('栈空')
            return None
        else:
            return self.items.pop(0)
class MyQueue(object):
    def __init__(self):
        self.queueData = Queue()
        self.queueHelp =  Queue()

    def push(self,item):
        self.queueData.push(item)

    def pop(self):
        if self.queueData.inEmpty() and self.queueHelp.inEmpty():
            print('2栈空')
            return None
        else:
            while self.queueData.size() > 1:
                self.queueHelp.push(self.queueData.pop())
            answer = self.queueData.pop()

            self.queueData,self.queueHelp = self.queueHelp,self.queueData

            return answer

    def printStr(self):
        self.queueData.travel()

if __name__ == '__main__':

    queue = MyQueue()
    queue.push(1)
    queue.push(2)
    queue.push(3)
    queue.printStr()

    print(queue.pop())

