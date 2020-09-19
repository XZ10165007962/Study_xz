

class Node(object):
    def __init__(self,elem,next_ = None):
        self.elem = elem
        self.next = next_

def isPalindrome(pHead):
    if pHead == None or pHead.next == None:
        print('不是回文结构')
        return

    pslow = pHead
    pfast = pHead
    stack = [pslow.elem]
    while True:
        if not pfast.next:
            mid = pslow
            break
        elif pfast and not pfast.next.next:
            mid = pslow.next
            break
        pslow = pslow.next
        pfast = pfast.next.next
        stack.append(pslow.elem)

    print(stack)
    while stack and mid:
        tmp = stack.pop()
        print(mid.elem, tmp)
        if mid.elem != tmp:
            print('不是回文结构')
            return
        mid = mid.next
    print('是回文结构')
    return


node1 = Node(1)
node2 = Node(2)
node3 = Node(3)
node4 = Node(3)
node5 = Node(2)
node6 = Node(1)
node1.next = node2
node2.next = node3
node3.next = node4
node4.next = node5
node5.next = node6

isPalindrome(node1)

