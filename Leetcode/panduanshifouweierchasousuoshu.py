class Node(object):
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

    def isBst(self):
        stack = []
        pos = self
        predata = float('-inf')
        while pos is not None or len(stack) > 0:
            if pos is not None:
                stack.append(pos)
                pos = pos.left
            else:
                pos = stack.pop()
                if pos.data < predata:
                    return False
                else:
                    predata = pos.data
                pos = pos.right
        return True

if __name__ == '__main__':
    tree = Node(4)
    tree.left = Node(2)
    tree.right = Node(6)
    tree.left.left = Node(1)
    tree.left.right = Node(3)
    tree.right.left = Node(5)
    tree.right.right = Node(7)
    print(tree.isBst())

