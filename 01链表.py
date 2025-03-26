# time: 2025/1/14 10:06
# author: YanJP

class ListNode():
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 206. 反转链表
def reverseList( head):
    pre = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = pre
        pre = cur
        cur = nxt
    return pre

# 92. 反转链表 II
def reverseBetween(head, left, right):
    dummy = ListNode(next=head)
    p0 = dummy
    for _ in range(left - 1):
        p0 = p0.next
    pre = None
    cur = p0.next
    for _ in range(right - left + 1):
        nxt = cur.next
        cur.next = pre
        pre = cur
        cur = nxt
    p0.next.next = cur  #
    p0.next = pre
    return dummy.next

# 141. 环形链表
def hasCycle(head: ListNode) -> bool:
    fast=head
    slow=head
    while fast and fast.next:
        fast=fast.next.next
        slow=slow.next
        if fast==slow:
            return True
    return False

# 142. 环形链表 II
# 分析：快慢指针，快指针每次走两步，慢指针每次走一步，如果有环，快慢指针会在环内相遇，否则会到达链表尾部，返回None
# 环链表前部长度设为a，环起点到相遇点长度设为b，环剩下的长度为c，
# 快指针走的长度为a+b+k(b+c)，环长的整数倍（注意，在环内走的时候，计算相对速度，即慢指针在相遇点不动，快指针每次走一步）
# 慢指针走的长度为a+b，
# 因为快指针走的长度是慢指针的两倍，所以2a+2b = a+b+k(b+c)，==> a-c=(k-1)(b+c)
# 快指针和慢指针相遇后，慢指针再走c步，头结点走c步，然后：头结点走完c步之后，离环入口的距离是环长的整数倍，则两者一起走，一定会在入口相遇

# 还有一个点需要注意，慢指针进入环后，此时离相遇时，其移动的距离一定不会超过环的长度，（原因通过相对速度分析得出）
def detectCycle(head: ListNode) -> ListNode:
    fast=head
    slow=head
    while fast and fast.next:
        fast=fast.next.next
        slow=slow.next
        if fast is slow:
            while head is not slow:
                head=head.next
                slow=slow.next
            return slow
    return None


# 82. 删除排序链表中的重复元素 II
def deleteDuplicates(head):
    dummy = ListNode(next=head)
    cur = dummy
    while cur.next and cur.next.next:
        val = cur.next.val
        if val == cur.next.next.val:
            while cur.next and cur.next.val == val:
                cur.next = cur.next.next
        else:
            cur = cur.next
    return dummy.next  # 注意注意！！！

# 19. 删除链表的倒数第 N 个结点
def removeNthFromEnd(head, n: int):
    dummy=ListNode(next=head)
    right=dummy
    for _ in range(n):
        right=right.next
    left=dummy
    while right.next:
        left=left.next
        right=right.next
    left.next=left.next.next
    return dummy.next

# 24. 两两交换链表中的节点
def swapPairs( head ):
    node0=dummy=ListNode(next=head)
    node1=head
    while node1 and node1.next:
        node2=node1.next
        node3=node2.next

        node0.next=node2
        node2.next=node1
        node1.next=node3

        node0=node1
        node1=node3
    return dummy.next

# LGH 美团一面手撕
def reverseKGroup(head: ListNode, k: int):
    # 创建一个哑节点(dummy node)，它的next指向head
    dummy = ListNode(0)
    dummy.next = head
    # prev指针用于帮助反转子链表
    prev = dummy
    while True:
        # 检查是否还有至少k个节点剩余，如果不是，则直接返回结果
        check = prev
        for i in range(k):
            check = check.next
            if not check:
                return dummy.next
        # 反转k个节点
        start = prev.next  # 反转部分的开始节点
        then = start.next  # start之后的节点
        for i in range(k - 1):
            start.next = then.next
            then.next = prev.next
            prev.next = then
            then = start.next
        # 更新prev指针为start，即本轮反转部分的新尾部
        prev = start
    # return dummy.next