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
# 给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。
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
    p0.next.next = cur  # 注意，p0的next指向的是反转后链表的最后一个节点，此时这个节点需要指向right后面的节点，即为cur
    p0.next = pre # 最后，再将前半段与翻转后的头结点，即pre 连接起来
    return dummy.next
# 25. K 个一组翻转链表  每 k 个节点一组进行翻转，请你返回修改后的链表。
# LGH 美团一面手撕
def reverseKGroup(head: ListNode, k: int) -> ListNode:
    n=0
    cur=head
    while cur:
        n+=1
        cur=cur.next
    dummy=ListNode(next=head)
    p0=dummy
    pre=None
    cur=p0.next
    while n>=k:
        n-=k
        for _ in range(k):
            nxt=cur.next
            cur.next=pre
            pre=cur
            cur=nxt
        nxt=p0.next
        p0.next.next=cur
        p0.next=pre
        p0=nxt  # 相当于就是p0需要指向下次需要反正的K个节点的 前一个节点
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
# --------------删除链表----------------------------
# 237. 删除链表中的节点 不能访问头结点
def deleteNode(node):
    node.val=node.next.val
    node.next=node.next.next

# 19. 删除链表的倒数第 N 个结点
# 双向指针做法
def removeNthFromEnd(head, n: int):
    dummy=ListNode(next=head) # 一般来说，如果需要删除头结点，就需要设置dummy node
    right=dummy
    for _ in range(n):
        right=right.next
    left=dummy
    while right.next:  # 一直要循环到right指向最后一个节点，所以这里要写right.next
        left=left.next
        right=right.next
    left.next=left.next.next
    return dummy.next

# 83. 删除排序链表中的重复元素
# 给定一个已排序的链表的头 head ， 删除所有重复的元素，使每个元素只出现一次 。返回 已排序的链表
def deleteDuplicates( head):
    if head is None:
        return head
    cur=head
    while cur.next:
        if cur.val==cur.next.val:
            cur.next=cur.next.next
        else:
            cur=cur.next
    return head
# 82. 删除排序链表中的重复元素 II
# 给定一个已排序的链表的头 head ， 删除原始链表中所有重复数字的节点，只留下不同的数字 。返回 已排序的链表 。
def deleteDuplicatesII(head):
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


# 21. 合并两个有序链表
def mergeTwoLists(list1 , list2 ) :
    cur=dummy=ListNode()
    while list1 and list2:
        if list1.val<list2.val:
            cur.next=list1
            list1=list1.next
        else:
            cur.next=list2
            list2=list2.next
        cur=cur.next
    cur.next=list1 if list1 else list2
    return dummy.next

# 876. 链表的中间结点
# 找出并返回链表的中间结点。如果有两个中间结点，则返回第二个中间结点。
def middleNode(head):
    fast=head
    slow=head
    while fast and fast.next:
        fast=fast.next.next
        slow=slow.next
    return slow