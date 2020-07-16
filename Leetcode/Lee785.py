'''
给定一个无向图graph，当这个图为二分图时返回true。

如果我们能将一个图的节点集合分割成两个独立的子集A和B，并使图中的每一条边的两个节点一个来自A集合，一个来自B集合，我们就将这个图称为二分图。

graph将会以邻接表方式给出，graph[i]表示图中与节点i相连的所有节点。每个节点都是一个在0到graph.length-1之间的整数。这图中没有自环和平行边： graph[i] 中不存在i，并且graph[i]中没有重复的值。


'''
import numpy as np
def isBipartite(graph):
    nums = 0
    num1 = []
    num2 = []
    graph.reverse()
    while(graph):
        i = graph.pop()
        '''print('i',i)
        print('nums',nums)
        print('num1',num1)
        print('num2', num2)
        if nums not in num1 and nums not in num2:
            num1.append(nums)
        elif nums in num1 and nums not in num2:
            print('-----')
            pass
        elif nums not in num1 and nums  in num2:
            pass
        else:
            print('======')
            return False
        for j in i:
            print('j',j)
            if j not in num2 and j not in num1:
                num2.append(j)
            elif j in num2 and j not in num1:
                print('++++')
                pass
            elif j not in num2 and j in num1:
                pass
            else:
                print('***')
                return False'''
        if nums == 0:
            num1.append(nums)
            num2.extend(i)
            continue
        if nums not in num1 and nums not in num2:
            num1.append(nums)
        #i.remove(nums-1)
        for j in i:
            if j in num1:
                return False
            else:
                num2.append(j)
        nums += 1
    return True


def isBipartite1( graph):
    store = {}
    stack = []
    will = {i for i in range(len(graph))}
    while will:
        elem = will.pop()
        stack.append(elem)
        print(elem)
        if elem not in store: store[elem] = True
        while stack:
            idx = stack.pop()
            node = graph[idx]
            if idx in will: will.remove(idx)
            flag = store[idx]
            next_flag = not flag
            if not node: continue
            for n in node:
                if n not in store:
                    store[n] = next_flag
                    stack.append(n)
                else:
                    if store[n] != next_flag: return False
    return True

print(isBipartite1([[1,3], [0,2], [1,3], [0,2]]))



