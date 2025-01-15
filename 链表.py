# time: 2025/1/14 10:06
# author: YanJP

class ListNode():
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

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

def removeNthFromEnd(self, head: [ListNode], n: int):
    dummy=ListNode(next=head)
    right=dummy
    for _ in range(n):
        right=right.next
    left=dummy
    while right.next:
        left=left.next
        right=right.next
    left.next=left.next.next
    return dummy

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