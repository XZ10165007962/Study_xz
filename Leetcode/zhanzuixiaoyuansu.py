
class Stack(object):
    def __init__(self):
        self.items = []
    def empty(self):
        return len(self.items) == 0
    def size(self):
        return len(self.items)
    def seek(self):
        if not self.empty():
            return self.items[len(self.items) - 1]
        else:
            return None
    def pop(self):
        if not self.empty():
            return self.items.pop()
        else:
            print('栈空')
            return None
    def push(self,item):
        self.items.append(item)

class MyStack(object):
    def __init__(self):
        self.elemStack = Stack()
        self.minStack = Stack()
    def push(self,data):
        self.elemStack.push(data)
        if self.minStack.empty():
            self.minStack.push(data)
        else:
            if data<self.minStack.seek():
                self.minStack.push(data)

    def pop(self):
        if self.elemStack.empty():
            print('栈空')
            return None
        else:
            topData = self.elemStack.seek()
            self.elemStack.pop()
            if topData == self.minStack.seek():
                self.minStack.pop()
        return topData
    def mins(self):
        if self.minStack.empty():
            return 2 ** 32
        else:
            return self.minStack.seek()


if __name__=="__main__":
    stack = MyStack()
    stack.push(7)
    print("栈中最小值为：" + str(stack.mins()))
    stack.push(5)
    print("栈中最小值为：" + str(stack.mins()))
    stack.push(6)
    print("栈中最小值为：" + str(stack.mins()))
    stack.push(2)
    print("栈中最小值为：" + str(stack.mins()))
