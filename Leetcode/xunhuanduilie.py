class Loopqueue(object):
    def __init__(self,length):
        self.head = 0
        self.tail = 0
        self.maxSize = length
        self.cnt = 0
        self.__list = [0] * length

    def isEmpty(self):
        return self.cnt == 0

    def isFull(self):
        return self.cnt == self.maxSize

    def push(self,data):
        if self.isFull():
            return False

        if self.isEmpty():
            self.__list[0] = data
            self.head = 0
            self.tail = 0
            self.cnt += 1
            return True
        self.tail = (self.tail+1)%self.maxSize
        self.cnt += 1
        self.__list[self.tail] = data
        return True

    def pop(self):
        if self.isEmpty():
            return False
        data = self.__list[self.head]
        self.head = (self.head+1)%self.maxSize
        self.cnt -= 1
        return data

    def clear(self):
        self.head = 0
        self.tail = 0
        self.cnt = 0
        return  True

    def __len__(self):
        return self.cnt

    def __str__(self):
        s = ''
        for i in range(self.cnt):
            index = (i+self.head)%self.maxSize
            s += str(self.__list[index]) + ' '
        return s

if __name__ == '__main__':
    a = Loopqueue(3)
    for i in range(3):
        a.push(i)

    print(a.__str__())
