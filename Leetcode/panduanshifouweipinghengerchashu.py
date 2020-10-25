

class Node(object):
    def __init__(self, data=None):
        self.data = data
        self.left = None
        self.right = None

class ReturnData(object):
    def __init__(self, isB, h):
        self.isBalanced = isB
        self.height = h

def isBalancedBinaryTree(head):
    if not head:
        return ReturnData(True, 0)
    leftData = isBalancedBinaryTree(head.left)
    # leftData的最终返回值类型是上一句的终止条件，因为会遍历到最后
    if not leftData.isBalanced:
        return ReturnData(False, 0)
    rightData = isBalancedBinaryTree(head.right)
    if not rightData.isBalanced:
        return ReturnData(False, 0)
    if abs(leftData.height - rightData.height) > 1:
        return ReturnData(False, 0)
    return ReturnData(True, max(leftData.height, rightData.height) + 1)

def isB(head):
    return isBalancedBinaryTree(head).isBalanced

if __name__ == '__main__':

    head = Node(1)
    head.left = Node(2)
    head.right = Node(3)
    head.left.right = Node(4)
    head.right.right = Node(5)
    a = isB(head)
    print(a)


# 平衡二叉树：树中任何一个节点它左子树与右子树的高度差不超过1
# 利用递归手机信息
# 1 列出可能性 2 整理出返回类型
# 3 整个递归按照同样的结构，得到子树的信息 整合子树
# 的信息、加工出我的信息，往上返回、要求结构一致
'''class Node(object):
    def __init__(self, val=None):
        self.val = val
        self.left = None
        self.right = None

# 返回类型
class ReturnData(object):
    def __init__(self, isB, h):
        self.isB = isB
        self.h = h

def process(head):
    if not head:
        return ReturnData(True, 0)
    leftData = process(head.left)
    # leftData的最终返回值类型是上一句的终止条件，因为会遍历到最后
    if not leftData.isB:
        return ReturnData(False, 0)
    rightData = process(head.right)
    if not rightData.isB:
        return ReturnData(False, 0)
    if abs(leftData.h - rightData.h) > 1:
        return ReturnData(False, 0)
    return ReturnData(True, max(leftData.h, rightData.h) + 1)

def isB(head):
    return process(head).isB

# 两个返回语句中的类型规定了递归过后的返回值类型

if __name__ == '__main__':
    head = Node(1)
    head.left = Node(2)
    head.right = Node(3)
    head.left.right = Node(4)
    head.right.right = Node(5)
    a = isB(head)
    print(a)'''
