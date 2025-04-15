# time: 2025/2/24 11:07
# author: YanJP
from collections import defaultdict,Counter
from functools import cache

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 560. 和为 K 的子数组（前缀和、哈希表）  注：子数组和子串是连续的，子序列不连续
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

# 32. 最长有效括号  栈+贪心
def longestValidParentheses(s: str) -> int:
    stack=[]
    maxL=0
    n=len(s)
    tmp=[0]*n
    for i in range(n):
        if s[i]=='(':
            stack.append(i)
        else:
            if stack:
                j=stack.pop()
                tmp[i], tmp[j]=1, 1
    curL=0
    for num in tmp:
        if num:
            curL+=1
            maxL=max(maxL, curL)
        else: curL=0
    return maxL
def longestValidParentheses2(s: str) -> int:
    ans = 0
    stack = []
    for i, c in enumerate(s):
        if stack and s[stack[-1]] == '(' and c == ')': # 这样写要简洁一点
            stack.pop()
            if not stack:
                ans = max(ans, i + 1)
            else:
                ans = max(ans, i - stack[-1])
        else:
            stack.append(i)
    return ans

# 118. 杨辉三角
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

# 5. 最长回文子串
# 输入：s = "babad"
# 输出："bab"
# 解释："aba" 同样是符合题意的答案。
def longestPalindrome( s: str) -> str:
    n = len(s)
    length = 1 # 最长回文子串的长度
    start = 0  # 最长回文子串的起始位置
    dp = [[False] * n for _ in range(n)]    # dp[j][i]表示子串s[j:i]是否为回文串
    for i in range(n):
        # 以i为终点，往回枚举起点j
        for j in range(i, -1, -1):
            if i == j:
                dp[j][i] = True    # 一个字符，一定为回文串
            elif i == j + 1:
                dp[j][i] = (s[i] == s[j])  # 两个字符，取决于二者是否相等
            else:
                dp[j][i] = (s[i] == s[j]) and dp[j+1][i-1]  # 两个字符以上，首先端点两个字符要相等，其次[j+1, i-1]也要为回文串
            # [j,i]为回文串且长度更大，更新
            if dp[j][i] and (i - j + 1) > length:
                length = i - j + 1
                start = j
        return s[start: start + length] # 截取最长回文子串

# 169. 多数元素
# 给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。
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
    p0,i,p2=0,0,len(nums)-1
    while i<=p2:
        if nums[i]==0:
            nums[i],nums[p0]=nums[p0],nums[i]
            p0+=1
            i+=1
        elif nums[i]==2:
            nums[i],nums[p2]=nums[p2],nums[i]
            p2-=1
        else:
            i+=1

# 31. 下一个排列
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
