import numpy as np
import pandas as pd

class Node(object):
    def __init__(self,data=None,left=None,right=None):
        self.data = data
        self.left = left
        self.right = right

    def preOrderRecursive(self):
        if self.data != None:
            print(self.data , end=' ')
        if self.left != None:
            self.left.preOrderRecursive()
        if self.right != None:
            self.right.preOrderRecursive()

    def medOrderRecursive(self):
        if self.left != None:
            self.left.medOrderRecursive()
        if self.data != None:
            print(self.data,end=' ')
        if self.right != None:
            self.right.medOrderRecursive()

    def postOrderRecursive(self):
        if self.left != None:
            self.left.postOrderRecursive()
        if self.right != None:
            self.right.postOrderRecursive()
        if self.data != None:
            print(self.data,end=' ')


    #二叉树的高度
    def height(self):
        if self.data == None:
            return 0
        elif self.left == None and self.right == None:
            return 1
        elif self.left == None and self.right is not None:
            return 1 + self.right.height()
        elif self.left is not None and self.right == None:
            return 1 + self.left.height()
        else:
            return 1 + max(self.left.height() , self.right.height())

    # 层序遍历
    def levelorder(self):

        # 返回某个节点的左孩子
        def LChild_Of_Node(node):
            return node.left if node.left is not None else None
        # 返回某个节点的右孩子
        def RChild_Of_Node(node):
            return node.right if node.right is not None else None

        # 层序遍历列表
        level_order = []
        # 是否添加根节点中的数据
        if self.data is not None:
            level_order.append([self])

        # 二叉树的高度
        height = self.height()
        if height >= 1:
            # 对第二层及其以后的层数进行操作, 在level_order中添加节点而不是数据
            for _ in range(2, height + 1):
                level = []  # 该层的节点
                for node in level_order[-1]:
                    # 如果左孩子非空，则添加左孩子
                    if LChild_Of_Node(node):
                        level.append(LChild_Of_Node(node))
                    # 如果右孩子非空，则添加右孩子
                    if RChild_Of_Node(node):
                        level.append(RChild_Of_Node(node))
                # 如果该层非空，则添加该层
                if level:
                    level_order.append(level)
            # 取出每层中的数据
            for i in range(0, height):  # 层数
                for index in range(len(level_order[i])):
                    level_order[i][index] = level_order[i][index].data

        return level_order

if __name__ == '__main__':
    tree = Node(1)
    tree.left = Node(2)
    tree.right = Node(3)
    tree.left.left = Node(4)
    tree.left.right = Node(5)
    tree.right.left = Node(6)
    tree.right.right = Node(7)
    tree.preOrderRecursive()
    print()
    tree.medOrderRecursive()
    print()
    tree.postOrderRecursive()
    print()
    print('树的高度为',tree.height())
    print('层次遍历',tree.levelorder())



