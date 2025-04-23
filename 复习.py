# time: 2025/4/21 23:09
# author: YanJP
from collections import deque
from collections import defaultdict,Counter
from functools import cache


def canFinish(numCourses: int, prerequisites) -> bool:
    inder=[0]*numCourses
    adja=[[] for _ in range(numCourses)]
    queue=deque()
    for cur,pre in prerequisites:
        inder[cur]+=1
        adja[pre].append(cur)
    for i in range(numCourses):
        if inder[i]==0:
            queue.append(i)
    while queue:
        pre=queue.popleft()
        inder[pre]-=1
        numCourses-=1
        for cur in adja[pre]:
            inder[cur]-=1
            if inder[cur]==0:
                queue.append(cur)
    return not numCourses
# 416. 分割等和子集
# 输入：nums = [1,5,11,5]
# 输出：true
# 解释：数组可以分割成 [1, 5, 5] 和 [11] 。
# 时间复杂度：O(ns)，其中 n 是 nums 的长度，s 是 nums 的元素和（的一半）。由于每个状态只会计算一次，
# 动态规划的时间复杂度 = 状态个数 × 单个状态的计算时间。本题状态个数等于 O(ns)，单个状态的计算时间为 O(1)
def canPartition(nums: list[int]) -> bool:
    s = sum(nums)
    if s % 2 == 1: return False
    n = len(nums)
    @cache
    def dfs(i, c):
        if i < 0:
            # return True if c==0 else False
            return c == 0
        if nums[i] > c:
            return dfs(i - 1, c)
        return dfs(i - 1, c) or dfs(i - 1, c - nums[i])
    return dfs(n - 1, s//2)
# 560. 和为 K 的子数组（前缀和、哈希表）  注：子数组和子串是连续的，子序列不连续
def subSum(nums: list[int], k: int):
    s=[0]*(len(nums)+1)
    for i in range(len(nums)):
        s[i+1]=s[i]+nums[i]
    cnt=defaultdict(int)
    ans=0
    for sj in s:
        ans+=cnt[sj-k]
        cnt[sj]+=1 # 前缀和为sj 的个数
    return  ans

# 1143. 最长公共子序列
# 计算复杂度：O(mn) 状态个数 (mn)* 每个状态的计算时间(1)
def longestCommonSubsequence_huisu(text1: str, text2: str) -> int:
    m=len(text1)
    n=len(text2)
    # dfs(i, j) 的定义：
    # 计算 text1 前 i+1 个字符和 text2 前 j+1 个字符的最长公共子序列长度。
    @cache
    def dfs(i,j):
        if i<0 or j<0:
            return 0
        if text1[i]==text2[j]:
            return dfs(i-1,j-1)+1
        return max(dfs(i-1,j),dfs(i,j-1))
    return dfs(m-1,n-1)
def longestCommonSubsequence(text1: str, text2: str) -> int:
    m,n=len(text1),len(text2)
    f=[[0]*(n+1) for _ in range(m+1)]
    for i,x in enumerate(text1):
        for j,y in enumerate(text2):
            if x==y:
                f[i+1][j+1]=f[i][j]+1
            else:
                f[i+1][j+1]=max(f[i][j+1],f[i+1][j])
    return f[m][n]
# 72. 编辑距离
def minDistance(word1: str, word2: str) -> int:
    m = len(word1)
    n = len(word2)
    @cache
    def dfs(i,j):
        if i<0 :
            return j+1  # i<0等价于word1为空，所以对应的操作次数是将添加word2的所有字符，从0-j，所以为j+1
        if j<0:
            return i+1
        if word1[i]==word2[j]:
            return dfs(i-1,j-1)
        # 分别对应插入、删除、替换、 （从后往前分析）
        # i不变，j-1：可以理解为word1的第i个字符后边插入word2的第j个字符，因此当前匹配了，j-1
        # i-1， j不变：可以理解为删除word1的第i个字符，因此当前结果等于word1的i-1处的结果，j不变
        # i-1，j-1：可以理解为替换word1的第i个字符为word2的第j个字符，当前字符匹配，i，j分别往前移动-1
        return min(dfs(i,j-1), dfs(i-1,j), dfs(i-1,j-1))+1
    return dfs(m-1,n-1) # dfs(i,j)都是子问题

# 300. 最长递增子序列 LIS
# 思路1: 选或不选 为了比大小，需要知道上一个选的数字(最长公共子序列那个题适合用这种方法)
# 思路2: 枚举选哪个 比较当前选的数字和下一个要选的数字(当前题采用的方法，可以少维护一个变量）
def lengthOfLIS(nums: list[int]) -> int:  # 时间复杂度：O(n^2)
    n=len(nums)
    ans=0
    @cache
    def dfs(i):  #dfs(i)表示：以nums[i] 结尾的LIS长度
        res=0
        for j in range(i):
            if nums[j]<nums[i]:
                res=max(res, dfs(j))
        return res+1  # 注意，+1是写在这里，表示算上当前i这个元素
    for i in range(n):
        ans=max(ans,dfs(i))
    return ans
def lengthOfLIS_ditui(nums: list[int]) -> int:
    n=len(nums)
    dfs=[0]*(n)
    for i in range(n):
        for j in range(i):
            if nums[j]<nums[i]:
                dfs[i]=max(dfs[i], dfs[j])
        dfs[i]+=1
    return max(dfs)
# 674. 最长连续递增序列
# 最长递增子数组（Longest Continuous Increasing Subsequence, LCIS）
def find_length_of_lcis_dfs(nums):
    n = len(nums)
    if n == 0:
        return 0
    @cache
    def dfs(i):  # dfs(i) 表示以 nums[i] 结尾的最长连续递增子数组的长度
        if i == 0:
            return 1  # 初始条件：单个元素长度为1
        if nums[i] > nums[i - 1]:
            return dfs(i - 1) + 1  # 连续递增，长度+1
        else:
            return 1  # 不递增，重置长度为1
    return max(dfs(i) for i in range(len(nums))) # 遍历所有位置，取最大值

# 53. 最大子数组和 (腾讯面试题）三种解法
def maxSubArray( nums: list[int]) -> int:
    min_pre=0
    ans=-float('inf')
    pre=0
    for x in nums:
        pre+=x
        ans=max(ans, pre-min_pre)
        min_pre=min(min_pre, pre)
    return ans
def maxSubArray_dfs(nums: list[int]) -> int:
    n=len(nums)
    @cache
    def dfs(i):
        if i<0:
            return 0
        return max(dfs(i-1)+nums[i], nums[i])
    return max(dfs(i) for i in range(n))
# 152. 乘积最大子数组 （对比上面一个题）
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
# 买卖股票的最佳时机 只能买卖一次股票
def maxProfit_once( prices: list[int]) -> int:
    if not prices: return 0
    maxPro=0
    minPro=float("inf")
    for p in prices:
        if p < minPro:
            minPro=p
        maxPro=max(p-minPro,maxPro)
    return maxPro

# 122. 买卖股票的最佳时机 II  可以多次买卖股票
def maxProfit( prices: list[int]) -> int:
    res = 0
    for i in range(1, len(prices)):
        res += max(0, prices[i] - prices[i - 1])
    return res
# ------------------------区间动态规划-----------------------------------------
# 516. 最长回文子序列
# 输入：s = "cbbd"
# 输出：2
# 解释：一个可能的最长回文子序列为 "bb" 。
# 状态个数O(n^2) 总时间复杂度：O(n^2)
def longestPalindromeSubseq(s: str) -> int:
    n=len(s)
    @cache
    def dfs(i,j):  # dfs(i,j)表示字符串 s[i..j] 的最长回文子序列的长度。
        if i>j:
            return 0
        if i==j:
            return 1
        if s[i]==s[j]:
            return dfs(i+1,j-1)+2
        return max(dfs(i+1,j),dfs(i,j-1))
    return dfs(0,n-1)

# 5. 最长回文子串
# 输入：s = "babad"
# 输出："bab"
# 解释："aba" 同样是符合题意的答案。
def longestPalindrome_dfs( s: str) -> str:
    lenth=1
    start=0
    @cache
    def dfs(j,i):  # dp[j][i]表示子串s[j:i]是否为回文串
        # if  j>i:
        #     return False  # 多余的判断 删去就行
        if i == j:
            return True
        elif i - 1 == j:
            return s[i] == s[j]
        else:
            return dfs(j+1, i-1)  and s[i] == s[j]
    for i in range(len(s)):
        for j in range(i): #逆序写也正确：for j in range(i, -1, -1)
            if dfs(j, i) and i-j+1>lenth:
                start=j
                lenth=i-j+1
    return s[start:start+lenth]
