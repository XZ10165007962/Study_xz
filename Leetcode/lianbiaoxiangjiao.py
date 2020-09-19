

class Node(object):
    def __init__(self,elem,next_ = None):
        self.elem = elem
        self.next = next_

def getLoopNode(head):
    if head == None or head.next == None:
        print('不存在相交节点')
        return None

    fast = head
    slow = head

    while(fast.elem != slow.elem):
        slow = slow.next
        if (fast.next == None or fast.next.next == None):
            print('不存在相交节点')
            return None
        else:
            fast = fast.next.next

    fast = head
    while (fast != slow):
        slow = slow.next
        fast = fast.next

    return slow

def getFirstIntersectNode(head1,head2):
    if (head1 == None or head2 == None):
        return None

    loop1 = getLoopNode(head1)
    loop2 = getLoopNode(head2)

    if (loop1 == None and loop2 == None):
        return noLoop(head1,head2)
    if (loop1 != None and loop2 != None):
        return bothLoop(head1,head2,loop1,loop2)

    return None

def bothLoop(head1,head2,loop1,loop2):
    cur1 = head1
    cur2 = head2
    if (loop1 == loop2):
        n = 0
        while(cur1.next != loop1):
            n += 1
            cur1 = cur1.next
        while(cur2.next != loop1):
            n -= 1
            cur2 = cur2.next

        if n > 0:
            cur1 = head1
            cur2 = head2
        else:
            cur1 = head2
            cur2 = head1
        n = abs(n)
        while(n != 0):
            cur1 = cur1.next
            n -= 1
        while(cur1 != cur2):
            cur1 = cur1.next
            cur2 = cur2.next

        return cur1
    cur1 = loop1.next
    while(cur1 != loop1):
        if cur1 == loop2:
            return loop1

        cur1 = cur1.next

    return None


def noLoop(head1,head2):

    cur1 = head1
    cur2 = head2
    n = 0
    while(cur1.next != None):
        n += 1
        cur1 = cur1.next
    while(cur2.next != None):
        n -= 1
        cur2 = cur2.next

    if (cur1 != cur2):
        return None

    if n > 0:
        cur1 = head1
        cur2 = head2
    else:
        cur1 = head2
        cur2 = head1
    n = abs(n)
    while (n != 0):
        cur1 = cur1.next
        n -= 1
    while (cur1 != cur2):
        cur1 = cur1.next
        cur2 = cur2.next

    return cur1

def printList(head):
    for i in range(51):
        print(head.elem , end='')
        head = head.next
    print()
    print('======')


if __name__ == '__main__':

    #1->2->[3]->4->5->6->7->[3]...
    head1 = Node(1)
    head1.next = Node(2)
    head1.next.next = Node(3)
    head1.next.next.next = Node(4)
    head1.next.next.next.next = Node(5)
    head1.next.next.next.next.next = Node(6)
    head1.next.next.next.next.next.next = Node(7)
    head1.next.next.next.next.next.next.next = head1.next.next

    #9->8->[6]->7->3->4->5->[6]...
    head2 = Node(9)
    head2.next = Node(8)
    head2.next.next = head1.next.next.next.next.next
    head2.next.next.next = head1.next.next.next.next.next.next
    head2.next.next.next.next = head1.next.next
    head2.next.next.next.next.next = head1.next.next.next
    head2.next.next.next.next.next.next = head1.next.next.next.next
    head2.next.next.next.next.next.next.next = head1.next.next.next.next.next

    printList(head1)
    printList(head2)
    print(getFirstIntersectNode(head1, head2).data)
    print("==================")

    #1->[2]->3->4->5->6->7->8->4...
    head3 = Node(1)
    head3.next = Node(2)
    head3.next.next = Node(3)
    head3.next.next.next = Node(4)
    head3.next.next.next.next = Node(5)
    head3.next.next.next.next.next = Node(6)
    head3.next.next.next.next.next.next = Node(7)
    head3.next.next.next.next.next.next.next = Node(8)
    head3.next.next.next.next.next.next.next.next = head1.next.next.next

    #9->0->[2]->3->4->5->6->7->8->4...
    head4 = Node(9)
    head4.next = Node(0)
    head4.next.next = head3.next
    head4.next.next.next = head3.next.next
    head4.next.next.next.next = head3.next.next.next
    head4.next.next.next.next.next = head3.next.next.next.next
    head4.next.next.next.next.next.next = head3.next.next.next.next.next
    head4.next.next.next.next.next.next.next = head3.next.next.next.next.next.next
    head4.next.next.next.next.next.next.next.next = head3.next.next.next.next.next.next.next
    head4.next.next.next.next.next.next.next.next.next = head3.next.next.next

    printList(head3)
    printList(head4)
    print(getFirstIntersectNode(head3, head4).data)
    print("==================")

    #1->[2]->3->4->5
    head5 = Node(1)
    head5.next = Node(2)
    head5.next.next = Node(3)
    head5.next.next.next = Node(4)
    head5.next.next.next.next = Node(5)
    # 6->[2]->3->4->5
    head6 = Node(6)
    head6.next = head5.next
    head6.next.next = head5.next.next
    head6.next.next.next = head5.next.next.next
    head6.next.next.next.next = head5.next.next.next.next

    print(getFirstIntersectNode(head5, head6).data)