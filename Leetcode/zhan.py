
class ArrayQueue(object):
    def __init__(self,initsize):
        if initsize < 0:
            raise Exception('wrong')
        self.arr = [0 for i in range(initsize)]
        self.size = 0
        self.start = 0
        self.end = 0

    def push(self,item):
        if self.size == len(self.arr):
            raise Exception('queue is full')
        self.arr[self.end] = item
        self.size += 1
        self.end = 0 if self.end == len(self.arr) -1 else self.end + 1
        return self.arr

    def pop(self):
        if self.size == 0:
            raise Exception('queue is empty')
        first = self.start
        self.start = 0 if self.start == len(self.arr) - 1 else self.start+1
        return self.arr[first]

    def peek(self):
        if self.size == 0:
            return None
        return self.arr[self.start]

if __name__ == '__main__':
    a = ArrayQueue(3)
    print(a.arr)
    a.push(1)
    a.push(2)
    b = a.push(3)
    print(b)
    c = a.pop()
    print(c)
    e = a.peek()
    print(e)