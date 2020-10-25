

class Node(object):
    def __init__(self,data=None,left=None,right=None):
        self.data = data
        self.left = left
        self.right = right

    #先序遍历1 2 4 5 3 6 7
    def preOrderTravese(self):
        stack = [self]
        while len(stack) > 0:
            print(self.data,end=' ')
            if self.right is not None:
                stack.append(self.right)
            if self.left is not None:
                stack.append(self.left)
            self = stack.pop()

    #中序遍历4 2 5 1 6 3 7
    def inOrderTraverse(self):
        stack = []
        pos = self
        while pos is not None or len(stack) > 0:
            if pos is not None:
                stack.append(pos)
                pos = pos.left
            else:
                pos = stack.pop()
                print(pos.data,end=' ')
                pos = pos.right

    #后序遍历7 3 6 1 5 2 4
    def postOrderTraverse(self):
        stack = [self]
        stack2 = []
        while len(stack) > 0:
            node = stack.pop()
            stack2.append(node)
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)

        while len(stack2) > 0:
            print(stack2.pop().data,end=' ')


if __name__ == '__main__':
    tree = Node(1)
    tree.left = Node(2)
    tree.right = Node(3)
    tree.left.left = Node(4)
    tree.left.right = Node(5)
    tree.left.right.right = Node(8)
    tree.right.left = Node(6)
    tree.right.right = Node(7)
    tree.preOrderTravese()
    print()
    tree.inOrderTraverse()
    print()
    tree.postOrderTraverse()