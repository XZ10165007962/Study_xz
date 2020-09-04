
class Node(object):
    def __init__(self,elem,next_=None):
        self.elem = elem
        self.next = next_

def reverseList(head):
    if head == Node or head.next == Node:
        return head
    pre = None
    next = None
    while(head != None):
        next = head.next
        head.next = pre
        pre = head
        head = next
    return pre

if __name__ == '__main__':
    l1 = Node(3)
    l1.next = Node(2)
    l1.next.next = Node(1)
    l1.next.next.next = Node(9)
    print(l1.elem, l1.next.elem, l1.next.next.elem, l1.next.next.next.elem)
    l = reverseList(l1)
    print(l.elem, l.next.elem, l.next.next.elem, l.next.next.next.elem)