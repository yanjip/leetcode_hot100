# time: 2025/2/24 11:07
# author: YanJP
from collections import defaultdict,Counter
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 560. 和为 K 的子数组（前缀和、哈希表）
def subarraySum( nums: list[int], k: int) -> int:
    # ans=0
    # for i in range(len(nums)):
    #     s = 0
    #     for j in range(i,len(nums)):
    #         s+=nums[j]
    #         if s==k:
    #             ans+=1
    # return ans

    #  如     1 1 0 1 1，k=2
    # 前缀和:0 1 2 2 3 4, 第二个2减0=2,得到一个子数组; 4减第一个2=2，得到一个子数组; 4-第二个2=2，得到一个子数组;...
    s=[0] * (len(nums)+1)
    for i, x in enumerate(nums):
        s[i + 1] = s[i] + x # 构建前缀和数组
    ans = 0
    cnt = defaultdict(int)  # 在一些数中找一个数，使用哈希表
    for sj in s:
        ans += cnt[sj - k] # 找前缀和为sj-k的个数
        cnt[sj] += 1
    return ans

# nums=list(map(int, input().strip().split()))
# k=int(input())
# print(subarraySum(nums, k))

# 56. 合并区间
def merge(intervals) :
    intervals.sort(key=lambda p:p[0])
    ans=[]
    for p in intervals:
        if ans and p[0]<=ans[-1][1]:
            ans[-1][1]=max(ans[-1][1], p[1])
        else:
            ans.append(p)
    return ans

# 双指针 数组
def rotate(nums:list[int], k: int) -> None:
    def reverse(i,j):
        while i<j:
            nums[i], nums[j]=nums[j], nums[i]
            i+=1
            j-=1
    n=len(nums)
    k %= n
    reverse(0,n-1)
    reverse(0,k-1)
    reverse(k,n-1)

# 240. 搜索二维矩阵 II
def searchMatrix(matrix: list[list[int]], target: int) -> bool:
    m, n =len(matrix), len(matrix[0])
    x, y =0, n-1
    while x<=m and y>=0:
        if matrix[x][y]==target:
            return True
        elif matrix[x][y]>target:
            y-=1
        else:
            x+=1
    return False

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
    cur.next=list1 or list2
    return dummy.next

# 443. 压缩字符串
def compress(chars) -> int:
    slow = 0
    fast = 0
    writer = 0
    while fast < len(chars):
        while fast < len(chars) and chars[fast] == chars[slow]:  # fast不断前进，直到到达不重复的位置
            fast += 1
        distance = fast - slow  # 此时二者的差就是重复区间长度
        chars[writer] = chars[slow]  # 覆写一下字母，比如a, b
        writer += 1  # 写完后前进一步来准备写下个内容
        if distance > 1:
            dist_str = str(distance)
            # writer
            for i in range(len(dist_str)):
                chars[writer] = dist_str[i]  # 开始写长度
                writer += 1

        slow = fast  # 慢指针初始化为下一个char序列的起点，以准备计算新长度distance

    chars = chars[:writer]  # 截断结果
    return writer

def compress2(chars) -> int:
    n = len(chars)
    i, j = 0, 0
    while i < n:
        start = i
        while i < n and chars[i] == chars[start]:
            i += 1
        chars[j] = chars[start]  # 覆写字母
        j += 1  # 写完后前进一步来准备写下个内容
        if i - start > 1:
            lst = str(i - start)   # 超过10后，就会出现两个字符
            for c in lst:
                chars[j] = c
                j += 1
    return j
# 200. 岛屿数量
def numIslands(grid: list[list[str]]) -> int:
    m,n=len(grid), len(grid[0])
    def dfs(i,j):
        if i<0 or i>=m or j<0 or j>=n or grid[i][j]!='1':
            return
        grid[i][j]='0'
        dfs(i+1,j)
        dfs(i-1,j)
        dfs(i,j+1)
        dfs(i,j-1)
    ans=1
    for i,x in enumerate(grid):
        for j, y in enumerate(x):
            if y=='1':
                dfs(i,j)
                ans+=1
    return ans