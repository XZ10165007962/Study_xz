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
#if __name__ == '__main__':
    '''tree = Node(1)
    tree.left = Node(2)
    tree.right = Node(3)
    tree.left.left = Node(4)
    tree.left.left.left = Node(8)
    tree.left.right = Node(5)
    tree.right.left = Node(6)
    tree.right.right = Node(7)
    print(tree.isW())'''


