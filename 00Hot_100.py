# time: 2025/2/24 11:07
# author: YanJP
from collections import defaultdict,Counter
from functools import cache

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 560. 和为 K 的子数组（前缀和、哈希表）  注：子数组和子串是连续的，子序列不连续
# 请你统计并返回 该数组中和为 k 的子数组的个数 。
def subarraySum( nums: list[int], k: int) -> int:
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

# 189. 轮转数组 给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
# 输入: nums = [1,2,3,4,5,6,7], k = 3
# 输出: [5,6,7,1,2,3,4]
# 解释:# 向右轮转 1 步: [7,1,2,3,4,5,6]
# 双指针 数组
def rotate(nums:list[int], k: int) -> None:
    def reverse(i,j):
        while i<j:
            nums[i], nums[j]=nums[j], nums[i]
            i+=1
            j-=1
    n=len(nums)
    k %= n
    reverse(0,n-1) # 变成 [7,6,5, 4,3,2,1]
    reverse(0,k-1) # 变成 [5,6,7, 4,3,2,1]
    reverse(k,n-1) # 变成 [5,6,7, 1,2,3,4]

# 74. 搜索二维矩阵
# 每行中的整数从左到右按非严格递增顺序排列。
# 每行的第一个整数大于前一行的最后一个整数。
# 给你一个整数 target ，如果 target 在矩阵中，返回 true ；否则，返回 false 。
def searchMatrix(matrix, target: int) -> bool:
    m,n=len(matrix), len(matrix[0])
    i,j=0,n-1
    while i< m and j >=0:
        if matrix[i][j]==target:
            return True
        if matrix[i][j]<target:
            i+=1
        else:
            j-=1
    return False

# 240. 搜索二维矩阵 II
# 搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
# 每行的元素从左到右升序排列。
# 每列的元素从上到下升序排列。
def searchMatrixII(matrix: list[list[int]], target: int) -> bool:
    m, n =len(matrix), len(matrix[0])
    x, y =0, n-1
    while x<m and y>=0:
        if matrix[x][y]==target:
            return True
        if matrix[x][y]>target:
            y-=1
        else:
            x+=1
    return False

# 21. 合并两个有序链表  (手撕遇到过）
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
# 输入：chars = ["a","a","b","b","c","c","c"]
# 输出：返回 6 ，输入数组的前 6 个字符应该是：["a","2","b","2","c","3"]
# chars = ["a"] 输出：返回 1 而不是"a""1"
def compress(chars) -> int:
    n = len(chars)
    i, j = 0, 0
    while i < n:
        start = i
        while i < n and chars[i] == chars[start]:
            i += 1
        chars[j] = chars[start]  # 覆写字母
        j += 1  # 写完后前进一步来准备写下个内容
        if i - start > 1:
            lst = str(i - start)  # 超过10后，就会出现两个字符
            for c in lst:
                chars[j] = c
                j += 1
    return j
# print(compress(["a","b","b","c","c","c"]))

# 200. 岛屿数量
# 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
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

# 118. 杨辉三角
# 给定一个非负整数 numRows，生成「杨辉三角」的前 numRows 行
def generate( numRows: int) :
    ans=[[0]*i for i in range(1,numRows+1)]
    for i in range(numRows):
        ans[i][0]=ans[i][-1]=1
    for i in range(2,numRows):
        for j in range(1,i):
            ans[i][j]=ans[i-1][j-1]+ans[i-1][j]
    print(ans)
    return ans
# generate(5)

# 139. 单词拆分
# 输入: s = "applepenapple", wordDict = ["apple", "pen"]
# 输出: true
# 解释: 返回 true 因为 "applepenapple" 可以由 "apple" "pen" "apple" 拼接成。 注意，你可以重复使用字典中的单词。
# 本题状态个数等于 O(n)，单个状态的计算时间为 O(L^2)（L=wordDict中最长的字符串长度，注意判断子串是否在哈希集合中需要 O(L) 的时间），
# 所以记忆化搜索的时间复杂度为 O(nL^2)
def wordBreak(s: str, wordDict) -> bool:
    max_L=max(map(len, wordDict))
    wordDict=set(wordDict)
    # dfs(i)，表示能否把前缀 s[:i]（表示 s[0] 到 s[i−1] 这段子串）划分成若干段，使得每段都在 wordDict 中。
    @cache
    def dfs(i):
        if i==0:
            return True
        for j in range(i-1, max(i-max_L-1, -1),-1): # j的最小值就是取到0，范围是倒序的
            if s[j:i] in wordDict and dfs(j):
                return True
        return False
    return dfs(len(s))
# print(wordBreak(s = "applepenapple", wordDict = ["apple", "pen"]))
def wordBreak_dp(s: str, wordDict) -> bool:
    max_L = max(map(len, wordDict))
    wordDict = set(wordDict)
    n = len(s)
    dfs = [True] + [False] * n
    for i in range(1, n + 1): # 注意这里是n+1，因为s[n]取不到最后一个字符
        for j in range(i - 1, max(i - max_L - 1, -1), -1):
            if s[j:i] in wordDict and dfs[j]:
                dfs[i] = True
                break # 这里必须break
    return dfs[n]

# 79. 单词搜索
# 总结：自己写忽略了三个问题：一是什么时候返回True没写明白；二是递归入口是由mn种情况；三是剪枝的判断board[i][j]!=word[k]时就可直接返回False了
# 时间复杂度:O(mn3^k)，其中 m和n分别为grid 的行数和列数，k是word 的长度。
# 除了递归入口，其余递归至多有3个分支(因为至少有一个方向是之前走过的)，所以每次递归(回溯)的时间复杂度为O(3^k)，
# 一共回溯O(mn)次，所以时间复杂度为O(mn3^k)。
def exist( board, word: str) -> bool:
    lens=len(word)
    m,n=len(board), len(board[0])
    def dfs(i,j,k):
        if i<0 or i>=m or j<0 or j>=n or board[i][j]!=word[k]:
            return False
        if  k==lens-1:
            return True
        board[i][j]=1
        ans=dfs(i+1,j,k+1) or dfs(i-1,j,k+1) or dfs(i,j+1,k+1) or  dfs(i,j-1,k+1)
        board[i][j]=word[k]
        return ans
    ans=False
    for i in range(m):
        for j in range(n):
            ans=ans or dfs(i,j,0)
    return ans

def exist2(board, word: str) -> bool:
    m, n = len(board), len(board[0])
    def dfs(i: int, j: int, k: int) -> bool:
        if board[i][j] != word[k]:  # 匹配失败
            return False
        if k == len(word) - 1:  # 匹配成功！
            return True
        board[i][j] = ''  # 标记访问过
        for x, y in (i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j):  # 相邻格子
            if 0 <= x < m and 0 <= y < n and dfs(x, y, k + 1):
                return True  # 搜到了！
        board[i][j] = word[k]  # 恢复现场
        return False  # 没搜到
    return any(dfs(i, j, 0) for i in range(m) for j in range(n))

# 152. 乘积最大子数组
def maxProduct( nums: list[int]) -> int:
    n = len(nums)
    @cache
    def dfs(i: int) -> tuple:
        """ 返回以 nums[i] 结尾的 (最大乘积, 最小乘积) """
        if i == 0:
            return nums[0], nums[0]
        prev_max, prev_min = dfs(i - 1)
        # 由于 nums[i] 可能是负数，需要考虑翻转
        cur_max = max(nums[i], nums[i] * prev_max, nums[i] * prev_min)
        cur_min = min(nums[i], nums[i] * prev_max, nums[i] * prev_min)
        return cur_max, cur_min
    return max(dfs(i)[0] for i in range(n))

def maxProduct2(nums: list[int]) -> int:
    n = len(nums)
    dfs = [[0] * 2 for _ in range(n)]
    dfs[0][0] = dfs[0][1] = nums[0]
    for i in range(n - 1):
        dfs[i + 1][0] = max(nums[i + 1], nums[i + 1] * dfs[i][0], nums[i + 1] * dfs[i][1])
        dfs[i + 1][1] = min(nums[i + 1], nums[i + 1] * dfs[i][0], nums[i + 1] * dfs[i][1])
    return max(dfs[i][0] for i in range(n))

    # dfs = [[0] * 2 for _ in range(n)]
    # dfs[0][0] = dfs[0][1] = nums[0]
    # for i in range(1, n):                                 区别在于这里的取值和下面的nums[i]还是nums[i+1]
    #     dfs[i][0] = max(nums[i], nums[i] * dfs[i - 1][0], nums[i] * dfs[i - 1][1])
    #     dfs[i][1] = min(nums[i], nums[i] * dfs[i - 1][0], nums[i] * dfs[i - 1][1])
    # return max(dfs[i][0] for i in range(n))             也是对的

# 53. 最大子数组和 (腾讯面试题）
def maxSubArray( nums: list[int]) -> int:
    ans = -float("inf")
    min_pre_sum = pre_sum = 0
    for x in nums:
        pre_sum += x
        ans = max(ans, pre_sum - min_pre_sum)
        min_pre_sum = min(pre_sum, min_pre_sum)
    return ans

from functools import cache
def maxSubArray_dfs(nums: list[int]) -> int:
    n = len(nums)
    @cache
    def dfs(i):  # dfs定义为以 nums[i] 结尾的连续子数组的最大和
        if i == 0: return nums[i] # 因为子数组至少包含一个数
        return max(dfs(i - 1) + nums[i], nums[i])  # 和下面等价，+nums[i]可以卸载max函数后面
    return max(dfs(i) for i in range(n))
def maxSubArray_dp( nums: list[int]) -> int:
    f = [0] * len(nums)  # f[i] 表示以 nums[i] 结尾的连续子数组的最大和
    f[0] = nums[0]
    for i in range(1, len(nums)):
        f[i] = max(f[i - 1], 0) + nums[i]
    return max(f)

# 62. 不同路径
def uniquePaths(m: int, n: int) -> int:
    # @cache  # 缓存装饰器，避免重复计算 dfs 的结果（一行代码实现记忆化）
    # def dfs(i: int, j: int) -> int:
    #     if i < 0 or j < 0:
    #         return 0
    #     if i == 0 and j == 0:
    #         return 1
    #     return dfs(i - 1, j) + dfs(i, j - 1)
    #
    # return dfs(m - 1, n - 1)

    f = [[0] * (n + 1) for _ in range(m + 1)]
    f[1][1] = 1 # 也可以写成 f[0][1] = 1，这样就不需要下面判断i==0 and j==0
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                continue
            f[i + 1][j + 1] = f[i][j + 1] + f[i + 1][j]  # 由于+1，所以f[1][1]其实等价于上面的dfs(0,0)
    return f[m][n]

# 22. 括号生成  时间复杂度O(n*C(2n,n))
def generateParenthesis(n: int):
    m=2*n
    ans=[]
    path=['']*m
    def dfs(i,cont_left):
        if i==m:
            ans.append(''.join(path))
            return
        if cont_left<n:
            path[i]='('
            dfs(i+1,cont_left+1)
        if i-cont_left<cont_left: #右括号的个数小于左括号的个数
            path[i]=')'
            dfs(i+1,cont_left)
    dfs(0,0)
    return ans
def generateParenthesis2(n: int):
    m=2*n
    ans=[]
    path=[]
    def dfs(i,cont_left):
        if i==m:
            ans.append(''.join(path))
            return
        if cont_left<n:
            path.append('(')
            dfs(i+1,cont_left+1)
            path.pop()  #注意，这里也要pop恢复现场
        if i-cont_left<cont_left: #右括号的个数小于左括号的个数
            path.append=(')')
            dfs(i+1,cont_left)
            path.pop()
    dfs(0,0)
    return ans
# n=int(input().strip())
# print(generateParenthesis(n))


# 169. 多数元素
# 给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。给定的数组总是存在多数元素。
def majorityElement( nums:list[int]) -> int:
    # 投票法
    votes=0
    ans=0
    for x in nums:
        if votes==0: ans=x
        if x==ans:
            votes+=1
        else:
            votes-=1
    return ans

# 75. 颜色分类
# 给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地 对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
# 我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
def sortColors( nums: list[int]) -> None:
    # 维护两个指针，分别指向0和2
    # p0,i,p2=0,0,len(nums)-1
    # while i<=p2:
    #     if nums[i]==0:
    #         nums[i],nums[p0]=nums[p0],nums[i]
    #         p0+=1
    #         i+=1
    #     elif nums[i]==2:
    #         nums[i],nums[p2]=nums[p2],nums[i]
    #         p2-=1
    #     else:
    #         i+=1
    p0 = p1 = 0
    for i, x in enumerate(nums):
        nums[i] = 2
        if x <= 1:
            nums[p1] = 1
            p1 += 1
        if x == 0:
            nums[p0] = 0
            p0 += 1

# 31. 下一个排列 （hot100）
# 输入：nums = [1,7,3,5,4,2,1]
# 输出：[1,7,4, 1,2,3,5]
class Solution:
    def nextPermutation(self, nums:list[int]) -> None:
        n = len(nums)
        # 第一步：从右向左找到第一个小于右侧相邻数字的数 nums[i] (即i指向3,且i后面的元素一定是单调递减的： [1,7,|3|,5,4,2,1]）
        i = n - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1

        # 如果找到了，进入第二步；否则跳过第二步，反转整个数组
        if i >= 0:
            # 第二步：从右向左找到 nums[i] 右边最小的大于 nums[i] 的数 nums[j]
            j = n - 1
            while nums[j] <= nums[i]:
                j -= 1
            # 交换 nums[i] 和 nums[j]
            nums[i], nums[j] = nums[j], nums[i]     # 找到满足nums[j] > nums[i]的数，即4，交换后变成[1,7,|4|,5,|3|,2,1]

        # 第三步：反转 nums[i+1:]（如果上面跳过第二步，此时 i = -1）  翻转数组，变成[1,7,4, 1,2,3,5]
        # nums[i+1:] = nums[i+1:][::-1] 这样写也可以，但空间复杂度不是 O(1) 的
        left, right = i + 1, n - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
# 347. 前 K 个高频元素
# 给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。
# 输入: nums = [1,1,1,2,2,3], k = 2
# 输出: [1,2]
def topKFrequent( nums: list[int], k: int) -> list[int]:
    cnt=Counter(nums) # 第一步：统计每个元素的出现次数
    # top_k=cnt.most_common(k)
    # # 提取元素部分，忽略频率
    # result = [item[0] for item in top_k]

    max_cnt=max(cnt.values())
    # 第二步：把出现次数相同的元素，放到同一个桶中
    buckets=[[] for _ in range(max_cnt+1)]
    for x, c in cnt.items():
        buckets[c].append(x) # 关键是这里，例如出现5次的数字有[3,4]
    # 第三步：倒序遍历 buckets，把出现次数前 k 大的元素加入答案
    ans=[]
    for bucket in reversed(buckets):
        ans+=bucket
        if len(ans)==k:
            return ans

# 56. 合并区间
# 输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
# 输出：[[1,6],[8,10],[15,18]]
# 解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
def merge(intervals) :
    intervals.sort(key=lambda p:p[0])
    ans=[]
    for p in intervals:
        if ans and p[0]<=ans[-1][1]:
            ans[-1][1]=max(ans[-1][1], p[1])
        else:
            ans.append(p)
    return ans

# 字节二面手撕
# 外层循环的 M 次与内层循环的最坏 O(L) 次相乘。时间复杂度为 O (M×L)。
def solve():
    # 读取输入
    L, M = map(int, input().split())
    trees = [1] * (L + 1)   # 下标 0~L，每个位置初始都有树
    for _ in range(M):
        a, b = map(int, input().split())
        # 区间可能 a > b，取 min/max
        start, end = min(a, b), max(a, b)
        for i in range(start, end + 1):
            trees[i] = 0  # 移除树
    print(sum(trees))

# 先排序 再区间合并
def solve2():
    L, M = map(int, input().split())
    intervals = []
    for _ in range(M):
        a, b = map(int, input().split())
        # 标准化区间（确保start <= end）
        start, end = min(a, b), max(a, b)
        # 区间不能超出[0, L]范围（如果输入可能越界）
        start = max(0, start)
        end = min(L, end)
        intervals.append((start, end))

    # 若没有区间，直接返回所有树的数量
    if not intervals:
        print(L + 1)
        return

    # 1. 按区间起点排序
    intervals.sort()

    # 2. 合并重叠/相邻区间
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        # 若当前区间与上一个区间重叠或相邻（current.start <= last.end）
        if current[0] <= last[1]:
            # 合并为更大的区间（取最小start和最大end）
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    # 3. 计算被移除的树的总数
    removed = 0
    for s, e in merged:
        removed += e - s + 1  # 每个区间的树的数量

    # 剩余树的数量 = 总树数 - 被移除的数量
    print((L + 1) - removed)

#763. 划分字母区间
# 输入：s = "ababcbacadefegdehijhklij"
# 输出：[9,7,8]
# 解释：划分结果为 "ababcbaca"、"defegde"、"hijhklij" 。
# 例如字母 d 的区间为 [9,14]，片段要包含 d，必须包含区间 [9,14]，但区间 [9,14] 中还有其它字母 e,f,g，所以该片段也
# 必须包含这些字母对应的区间 e[10,15],f[11,11],g[13,13]，合并后得到区间 [9,15]。
def partitionLabels( s: str):
    last={c:i for i, c in enumerate(s)} # 每个字母最后出现的下标
    ans=[]
    start=end=0
    for i, c in enumerate(s):
        end=max(end, last[c]) # 更新当前区间右端点的最大值
        if end==i: # 当前区间合并完毕
            ans.append(end-start+1)
            start=i+1
    return ans # 下一个区间的左端点

# 55. 跳跃游戏
# 给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。
# 判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false 。
# 输入：nums = [3,2,1,0,4]
# 输出：false
def canJump( nums: list[int]) -> bool:
    mx=0
    for i, jump in enumerate(nums):
        if i > mx: # 无法到达 i
            return False
        mx = max(mx, i+jump)
    return True

# 45. 跳跃游戏 II
# 返回到达 nums[n - 1] 的最小跳跃次数。测试用例保证可以到达 nums[n - 1]。每个元素 nums[i] 表示从索引 i 向后跳转的最大长度。
# 输入: nums = [2,3,1,1,4]
# 输出: 2 步到达
# 问：为什么代码只需要遍历到 n−2？
# 当 i=n−2 时，如果 i<curRight，说明可以到达 n−1；
#            如果 i=curRight，我们会造桥（因为一定有一个桥可以到达 n−1，说明nums[n-2]不可能为0，肯定≥1，ans+=1就得到答案），这样也可以到达 n−1。
# 所以无论是何种情况，都只需要遍历到 n−2。或者说，n−1 已经是终点了，你总不能在终点还打算造桥吧？
def jump(nums: list[int]) -> int:
    ans = 0
    cur_right = 0  # 已建造的桥的右端点
    next_right = 0  # 下一座桥的右端点的最大值
    for i in range(len(nums) - 1):
        # 遍历的过程中，记录下一座桥的最远点
        next_right = max(next_right, i + nums[i])
        if i == cur_right:  # 无路可走，必须建桥
            cur_right = next_right
            ans += 1
    return ans

# 146. LRU 缓存
# 请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
# OrderedDict = dict + 双向链表
from collections import OrderedDict
class LRUCache(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key, last = False)
        return self.cache[key]

    def put(self, key, value):
        self.cache[key]=value # 添加 key value 或者更新 value
        # last=False 表示移到链表头
        self.cache.move_to_end(key, last=False)
        if len(self.cache) > self.capacity:
            self.cache.popitem() # 去掉最后一本书