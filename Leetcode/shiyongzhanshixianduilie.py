
class Stack(object):
    def __init__(self):
        self.items = []

    def size(self):
        return len(self.items)

    def inEmpty(self):
        return len(self.items) == 0

    def seek(self):
        return self.items[len(self.items) - 1]

    def pop(self):
        if self.inEmpty():
            print('栈空')
            return None
        return self.items.pop()

    def push(self,item):
        self.items.append(item)

    def travel(self):
        for i in self.items:
            print(i, end=' ')
        print('')

class MyStack(object):
    def __init__(self):
        self.getStack = Stack()
        self.putStack = Stack()

    def push(self,item):
        self.putStack.push(item)

    def pop(self):
        if self.putStack.inEmpty() and self.getStack.inEmpty():
            print('2栈空')
            return None
        else:
            if self.getStack.inEmpty():
                while not self.putStack.inEmpty():
                    self.getStack.push(self.putStack.pop())
            return self.getStack.pop()

    def printStr(self):
        return self.putStack.travel()

if __name__ == '__main__':
    queue = MyStack()
    queue.push(1)
    queue.push(2)
    queue.push(3)
    queue.printStr()

    print(queue.pop())

