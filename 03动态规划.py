# time: 2025/2/18 9:35
# author: YanJP
import bisect
from collections import Counter, defaultdict
from functools import cache
import time
from math import isqrt

# -------------------------03动态规划.py------------------------------------

# 198. 打家劫舍
def rob(nums: list[int]):  # 超时，时间复杂度是指数级别
    def dfs(i):
        if i<0:  # 不能写等于0
            return 0
        return max(dfs(i-1),dfs(i-2)+nums[i])
    return dfs(len(nums)-1)
def rob2(nums: list[int]):
    n=len(nums)
    # 手动实现cache数组
    cache=[-1]*n
    def dfs(i):
        if i<0:  # 不能写等于0
            return 0
        if cache[i]!=-1:
            return cache[i]
        ans=max(dfs(i-1),dfs(i-2)+nums[i])
        cache[i]=ans
        return ans
    return dfs(n-1)
# 递推写法
def rob3(nums: list[int]) -> int:
    n=len(nums)
    f=[0]*(n+2)
    for i, x in enumerate(nums):
        f[i+2]=max(f[i+1],f[i]+x)
    return f[n+1]
    # f0,f1=0,0
    # for i, x in enumerate(nums):
    #     newf=max(f1,f0+x)
    #     f0=f1
    #     f1=newf
    # return newf
# nums=list(map(int,input().strip().split()))
# print(rob(nums))

#  70. 爬楼梯
# 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
# 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
def climbStairs( n: int) -> int:
    # 对于不加缓存的dfs写法:每次调用会分裂成两个子问题，递归树是一棵二叉树。时间复杂度 = O(2^n)。
    #     记忆化搜索dfs: 时间复杂度 = O(n)。
    # @cache
    # def dfs(i):
    #     if i==0 or i==1:  # 不知道dfs(0)应该是多少时，可以倒推1=dfs(1)   2=dfs(2)=dfs(2-1)+dfs(2-2)
    #         return 1
    #     return dfs(i-1)+dfs(i-2)
    # return dfs(n)

    # dfs[i] 表示到达第 i 阶的方法数。
    dfs = [0] * (n + 2)
    dfs[1], dfs[0] = 1, 1
    for i in range(n):
        dfs[i + 2] = dfs[i + 1] + dfs[i]
    return dfs[n] # 爬 n 阶楼梯的解应存储在 dfs[n]，而不是返回dfs[-1]。

    # 以下写法也是正确的，因为其实最终只需要返回 dfs[n].
    # 到底初始化dfs大小多少,就是看返回的值到底是 dfs[n] 还是 dfs[n+1]。
    # dfs = [0] * (n + 1)
    # dfs[1], dfs[0] = 1, 1
    # for i in range(n - 1):
    #     dfs[i + 2] = dfs[i + 1] + dfs[i]
    # return dfs[-1] #or dfs[n]

    # dfs=[0]*(n+1)
    # dfs[0]=dfs[1]=1
    # for i in range(2,n+1):
    #     dfs[i]=dfs[i-1]+dfs[i-2]
    # return dfs[n]


# 377. 组合总和 Ⅳ
# 给你一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。请你从 nums 中找出并返回总和为 target 的元素组合的个数。
# 输入：nums = [1,2,3], target = 4
# 输出：7
# (1, 1, 1, 1)
# (1, 1, 2)
# (1, 2, 1)
# (1, 3)
# (2, 1, 1)
# (2, 2)
# (3, 1)
# 请注意，顺序不同的序列被视作不同的组合。
def combinationSum4(nums: list[int], target: int) -> int:
    @cache
    def dfs(i):
        if i==0: return 1
        total=0
        for x in nums:
            if x<=i:
                total+=dfs(i-x)
        return total
    return dfs(target )
def combinationSum4_2(nums: list[int], target: int) -> int:
    f = [1] + [0] * target
    for i in range(1, target + 1):
        total = 0
        for x in nums:
            if x <= i:
                total += f[i - x]
        f[i] = total
    return f[target]

# 39. 组合总和 hot100 （回溯）
# 给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，
# 找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合(可以 无限制重复被选取 ) ，并以列表形式返回。你可以按 任意顺序 返回这些组合。
# [2,2,3]和[2,3,2]属于一种组合  (和上题不一样）
# 输入：candidates = [2,3,6,7], target = 7
# 输出：[[2,2,3],[7]]
def combinationSum( candidates, target: int) :
    ans = []
    path = []
    n = len(candidates)
    def dfs(start, s):
        if s==target:
            ans.append(path[:])
            return
        if s>target: return
        for j in range(start, n):
            path.append(candidates[j])
            dfs(j,s+candidates[j])
            path.pop()
    dfs(0, 0)
    return ans

#  01 背包问题 (选或不选） 返回能够选择的最大价值
def zero_one_Bag(capacity, weights: list[int], values: list[int]):
    n=len(weights)
    @cache
    def dfs(i,c):
        if i<0: return 0
        if weights[i]>c:  # 只能不选 选不了
            return dfs(i-1,c)
        return max(dfs(i-1,c),values[i]+dfs(i-1,c-weights[i]))
    return dfs(n-1,capacity)

# 494. 目标和
# 分析：假设合法的方案中，所有的正数之和为 p, 那么所有的负数之和为-(sum(nums)-p). 若要满足目标和，则有
# p - (sum(nums)-p) = target   ==>      p= (sum(nums)+target)/2   也就是说sum(nums)+target必须为偶数，否则没有合法的方案
# 此时问题可以看做一个背包问题，背包容量为p，背包中放nums中的元素，问有多少种方案使得背包中元素和为p
def findTargetSumWays(nums: list[int], target: int):
    target+=sum(nums)
    n=len(nums)
    if target%2 or target<0:
        return 0
    target//=2
    def zero_one_bag(i,target):
        if i<0:  # 必须要遍历完所有元素
            return 1 if target==0 else 0
        ans=zero_one_bag(i-1,target)+zero_one_bag(i-1,target-nums[i])
        return ans
    return zero_one_bag(n-1,target)
def findTargetSumWays2(nums: list[int], target: int):
    target+=sum(nums)
    n=len(nums)
    if target%2 or target<0:
        return 0
    target//=2
    dfs=[[0]*(target+1) for _ in range(n+1)]  # 使用 n+1 可以确保在遍历所有元素时不会出现索引越界的问题. 从0-target一共target+1个数
    dfs[0][0]=1  # 很重要，对应上述dfs做法，其中i<0 and target==0 的情况,答案为1。但由于上面一行索引加了1，因此对应的是i==0 的情况
    for i, x in enumerate(nums):
        for  j in range(target+1):
            if j<nums[i]:
                dfs[i+1][j]=dfs[i][j]
            else:
                dfs[i+1][j]=dfs[i][j]+dfs[i][j-nums[i]]
    print(dfs)
    return dfs[-1][-1] # 等价与dfs[-1][-1]

# nums=[1,1,1,1,1]
# target=int(input().strip())
# print(findTargetSumWays2(nums,target))

# 322. 零钱兑换  完全背包问题 返回能凑成amount的最小的硬币数
def coinChange(coins: list[int], amount: int) -> int:
    n=len(coins)
    @cache
    def dfs(i,target):
        if i<0:
            return 0 if target==0 else float("inf")
        if target<coins[i]:
            return dfs(i-1,target)
        ans=min(dfs(i-1,target),dfs(i,target-coins[i])+1)
        return ans
    # ans=dfs(n-1,amount)
    # return ans if ans<float("inf") else -1

    dfs=[[float("inf")]*(amount+1) for _ in range(n+1)]
    dfs[0][0]=0
    for i in range(n):
        for j in range(amount+1):
            if j<coins[i]: # 一定要注意这个条件的使用，表示当前容量不能使用当前硬币
                dfs[i+1][j]=dfs[i][j]
            else:
                dfs[i+1][j]=min(dfs[i][j],dfs[i+1][j-coins[i]]+1)
    return dfs[-1][-1] if dfs[-1][-1]<float("inf") else -1
def coinChange2(coins: list[int], amount: int) -> int:
    dfs=[float("inf")]*(amount+1)
    dfs[0] = 0
    for c in coins:
        for j in range(c,amount+1):
            dfs[j]=min(dfs[j],dfs[j-c]+1)
    return dfs[-1] if dfs[-1]<float("inf") else -1
coins=[3,7,405,436]
# amount=int(input().strip())
# print(coinChange2(coins,amount))


# 边界条件的处理：
    # 最大价值和：返回0（无物品可选）。--状态转移使用max
    # 最小价值和：如果c == 0，返回0；否则返回float("inf")（无解）。--状态转移使用min
    # 方案数：如果c == 0，返回1；否则返回0（无方案）。--状态转移使用 加法

# 279. 完全平方数 （完全背包问题）
# 给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。
# 输入：n = 13
# 输出：2
# 解释：13 = 4 + 9
@cache  # 写在外面，多个测试数据之间可以共享，减少计算量
def dfs(i, j):  # 定义 dfs(i,j) 表示从前 i 个完全平方数中选一些数（可以重复选），满足元素和恰好等于 j，最少要选的数字个数。
    if i == 0:
        return float('inf') if j else 0
    if j < i * i:
        return dfs(i - 1, j)  # 只能不选
    return min(dfs(i - 1, j), dfs(i, j - i * i) + 1)


class Solution:
    def numSquares(self, n: int) -> int:
        return dfs(isqrt(n), n)  # 递归入口为 dfs(⌊n⌋,n)


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
def canPartition2(nums: list[int]) -> bool:
    s = sum(nums)
    if s % 2 == 1: return False
    n = len(nums)
    s //= 2
    dfs = [[False] * (s + 1) for _ in range(n + 1)]
    dfs[0][0] = True
    for i, x in enumerate(nums):
        for j in range(s + 1):
            if x > j:
                dfs[i + 1][j] = dfs[i][j]
            else:
                dfs[i + 1][j] = dfs[i][j] or dfs[i][j - x]
    return dfs[-1][-1]

# https://www.bilibili.com/video/BV1TM4y1o7ug
# 1143. 最长公共子序列 LCS  线性DP问题
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
def longestCommonSubsequence2(text1: str, text2: str) -> int:  # 注意和上面解法的区别
    len1=len(text1)
    len2=len(text2)
    dfs=[[0]*(len2+1) for _ in range(len1+1)]
    for i,x in enumerate(text1):
        for j, y in enumerate(text2):
            if x==y:
                dfs[i+1][j+1]=dfs[i][j]+1
            else:
                dfs[i+1][j+1]=max(dfs[i][j+1],dfs[i+1][j])
    return dfs[-1][-1]
# text1  = input().strip()
# text2  = input().strip()
# print(longestCommonSubsequence(text1,text2))

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
    return dfs(m-1,n-1)

def minDistance_ditui(word1: str, word2: str) -> int:
    m = len(word1)
    n = len(word2)
    # dfs[i][j] 表示将 word1 的前 i 个字符转换为 word2 的前 j 个字符所需的最小操作数。
    dfs=[[0]*(n+1) for _ in range(m+1)]
    # 初始化n+1是因为需要取0-n的n+1个数！
    dfs[0]=[i for i in range(n+1)] # dfs[0]=list(range(n+1))   比如word1是空串，word2是"abc", 则dfs[0][3]=3
    for i, x in enumerate(word1):
        dfs[i+1][0]=i+1  # 和上面同理，比如word1是"abc", word2是空串，则dfs[3][0]=3
        for j, y in enumerate(word2):
            if x==y:
                dfs[i+1][j+1]=dfs[i][j]
            else:
                dfs[i+1][j+1]=min(dfs[i+1][j],dfs[i][j+1],dfs[i][j])+1
    return dfs[-1][-1]
# word1=input().strip()
# word2=input().strip()
# print(minDistance_ditui(word1,word2))

# 583. 两个字符串的删除操作 非hot100
# 给定两个单词 word1 和 word2 ，返回使得 word1 和  word2 相同所需的最小步数。
# 每步 可以删除任意一个字符串中的一个字符。
def minDistance_583( word1: str, word2: str) -> int:
    m=len(word1)
    n=len(word2)
    @cache
    def dfs(i,j):
        if i < 0: return j + 1  # 删除 word2 的剩余 j+1 个字符
        if j < 0: return i + 1  # 删除 word1 的剩余 i+1 个字符
        if word1[i]==word2[j]:
            return dfs(i-1,j-1)
        return min(dfs(i-1,j)+1, dfs(i,j-1)+1, dfs(i-1,j-1)+2)
    return dfs(m-1,n-1)
# word1 = "sea"
# word2 = "eat"
# print(minDistance_583(word1,word2)) #2  第一步将 "sea" 变为 "ea" ，第二步将 "eat "变为 "ea"

def minDistance_ditui2( word1: str, word2: str) -> int:
    m=len(word1)
    n=len(word2)
    dp=[[0]*(n+1) for _ in range(m+1)]
    for j, y in enumerate(word2):
        dp[0][j+1]=j+1
    for i,x in enumerate(word1):
        dp[i+1][0]=i+1
        for j, y in enumerate(word2):
            if x==y:
                dp[i+1][j+1]=dp[i][j]
            else:
                dp[i+1][j+1]=min(dp[i+1][j]+1, dp[i][j+1]+1, dp[i][j]+2)
    return dp[-1][-1]

# 300. 最长递增子序列 LIS ——也是线性DP问题
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
    # return max(dfs(i) for i in range(n))
def lengthOfLIS_ditui(nums: list[int]) -> int:
    n=len(nums)
    dfs=[0]*(n)
    for i in range(n):
        for j in range(i):
            if nums[j]<nums[i]:
                dfs[i]=max(dfs[i], dfs[j])
        dfs[i]+=1
    return max(dfs)

# 贪心+二分：
def lengthOfLIS_greedy(nums: list[int]) -> int:
    g=[]
    # g维护的就是最长子序列，需要随时更新最小值，从而扩展最长子序列
    for x in nums:
        j=bisect.bisect_left(g, x) # bisect_left返回小于等于x的最大的索引
        if j==len(g):
            g.append(x)
        else:
            g[j]=x
    return len(g)


# nums = [10,9,2,5,3,7,101,18]
# start_time = time.perf_counter()
# print(lengthOfLIS(nums))
# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(f"程序运行时间: {elapsed_time:.6f} 秒")

# 674. 最长连续递增序列
# 最长递增子数组（Longest Continuous Increasing Subsequence, LCIS）
def find_length_of_lcis_dfs(nums):
    n = len(nums)
    if n == 0:
        return 0
    ans = 0
    @cache
    def dfs(i):  # dfs(i) 表示以 nums[i] 结尾的最长连续递增子数组的长度
        if i == 0:
            return 1  # 初始条件：单个元素长度为1
        if nums[i] > nums[i - 1]:
            return dfs(i - 1) + 1  # 连续递增，长度+1
        else:
            return 1  # 不递增，重置长度为1
    for i in range(n):
        ans = max(ans, dfs(i))  # 遍历所有位置，取最大值
    return ans
def find_length_of_lcis_dfs2(nums):
    if not nums:
        return 0
    max_length = 1  # 至少长度为1
    def dfs(start, current_length):
        nonlocal max_length
        if start == len(nums) - 1:
            return
        if nums[start + 1] > nums[start]:
            current_length += 1
            max_length = max(max_length, current_length)
            dfs(start + 1, current_length)
        else:
            dfs(start + 1, 1)  # 重置current_length

    dfs(0, 1)  # 从第0个元素开始，初始current_length=1
    return max_length

# 560. 和为 K 的子数组（前缀和、哈希表）  注：子数组和子串是连续的，子序列不连续
# 数组可能是负数
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

# 53. 最大子数组和 (腾讯面试题）三种解法
# 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
# 输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
# 输出：6
# 解释：连续子数组 [4,-1,2,1] 的和最大，为 6
def maxSubArray( nums: list[int]) -> int:
    ans = -float("inf")
    min_pre_sum = pre_sum = 0
    for x in nums:
        pre_sum += x
        ans = max(ans, pre_sum - min_pre_sum) # 一定要先计算ans，再更新min_pre
        min_pre_sum = min(pre_sum, min_pre_sum)
    return ans

def maxSubArray_dfs(nums: list[int]) -> int:
    n = len(nums)
    @cache       # 可以对比一下前面的最长连续子序列
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

# 变形题：求这个最大连续子数组的长度
def maxSubArray_length(nums: list[int]) -> int:
    n = len(nums)
    # 定义 dfs(i) 返回一个元组 (current_max_sum, current_len)：
    # current_max_sum：以 nums[i] 结尾的连续子数组的最大和（保持原逻辑）。
    # current_len：该最大和子数组的长度。
    @cache
    def dfs(i):  # 返回 (max_sum, length)
        if i == 0:
            return (nums[i], 1)
        prev_sum, prev_len = dfs(i - 1)
        # 比较两种情况：加入前一个子数组 vs 以 nums[i] 作为新起点
        if prev_sum + nums[i] > nums[i]:
            return (prev_sum + nums[i], prev_len + 1)
        else:
            return (nums[i], 1)

    max_sum = float('-inf')
    max_len = 0
    for i in range(n):
        current_sum, current_len = dfs(i)
        if current_sum > max_sum:
            max_sum = current_sum
            max_len = current_len
    return max_len

# 152. 乘积最大子数组 （对比上面一个题）
# 输入: nums = [2,3,-2,4]
# 输出: 6
# 解释: 子数组 [2,3] 有最大乘积 6。
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

def maxProduct3(nums: list[int]) -> int:
    fmax=fmin=1
    ans=-float('inf')
    for x in nums:
        tmp_max=fmax
        fmax=max(x, x*fmax, x* fmin)
        fmin=min(x, x*tmp_max, x* fmin)
        ans=max(ans,fmax)
    return ans

# ---------------------状态机DP：买卖股票系列---------------------------------------------
# 121. 买卖股票的最佳时机 只能买卖一次股票
# 输入：[7,1,5,3,6,4]
# 输出：5 （6-1）
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
def maxProfit2( prices: list[int]) -> int:
    n=len(prices)
    @cache
    def dfs(i,hold):  # dfs(i,hold)表示第i天持有股票或者不持有股票，hold为True表示持有股票，为False表示不持有股票的最大利润
        # 定义dfs(i,0)表示到第i天*结束*时，未持有股票的最大利润定义; dfs(i,1)表示到第i天结束时，持有股票的最大利润
        # 由于第i-1天的结束就是第i天的开始, dfs(i-1,·)也表示到第i天开始时的最大利润
        if i<0:
            return -float("inf") if hold else 0
        # 如果第i天持有股票，对应的选择一是：在这一天不操作，因此返回的dfs(i,True)对应的是dfs(i-1,True)
        #     对应的选择二是：在这一天选择买入股票，因此返回的dfs(i,True)对应是dfs(i-1,False) - prices[i]
        if hold:
            return max(dfs(i-1,hold),dfs(i-1,False)-prices[i])  # 如果题目改成含冷冻期(卖出后冷冻一天,即买入股票前一天不能卖出)，则直接改成dfs(i-2,False)-prices[i]

        # 如果第i天不持有股票，对应的选择一是：在这一天没有操作，因此对应的是dfs(i-1,False)
        #   对应的选择二是：在这一天卖出了股票，因此对应是dfs(i-1,True)+prices[i]
        return max(dfs(i-1,False),dfs(i-1,True)+prices[i])

    return dfs(n-1,False) # 最后一天肯定是不持有股票利润最大
def maxProfit2_ditui( prices: list[int]) -> int:
    n=len(prices)
    dfs=[[0]*2 for _ in range(n+1)]
    dfs[0][1]=-float("inf")
    for i,x in enumerate(prices):
        dfs[i+1][0]=max(dfs[i][0],dfs[i][1]+x)
        dfs[i+1][1]=max(dfs[i][1],dfs[i][0]-x)
    return dfs[-1][0]
# prices = [7,1,5,3,6,4]
# print(maxProfit2_ditui(prices))

# 188. 买卖股票的最佳时机 IV —— 最多可以完成 k 笔交易
def maxProfitIV(k: int, prices: list[int]) -> int:
    n=len(prices)
    @cache
    def dfs(i,j,hold):
        if j<0:  # 这个判断语句一定要写在最前面，不然会报错
            return -float("inf")
        if i<0:
            return -float("inf") if hold else 0
        if hold:
            return max(dfs(i-1,j,hold),dfs(i-1,j-1,False)-prices[i])
        return max(dfs(i-1,j,hold),dfs(i-1,j,True)+prices[i])
    return dfs(n-1,k,False)

def maxProfitIV_2(k: int, prices: list[int]) -> int:
    n=len(prices)
    dfs=[[[-float("inf")]*2 for _ in range(k+2)] for _ in range(n+1)]
    for j in range(1,k+2): # 为什么是k+2？因为对应回溯中j的取值是-1到k，一共k+2中情况
        dfs[0][j][0]=0  # 初始化刚开始时不持有股票的情况，所以为0
    for i, p in enumerate(prices):
        for j in range(1,k+2):
            dfs[i+1][j][0]=max(dfs[i][j][0],dfs[i][j-1][1]+p)
            dfs[i+1][j][1]=max(dfs[i][j][1],dfs[i][j][0]-p)
    return dfs[-1][-1][0]
# k=int(input())
# prices=list(map(int,input().split()))
# print(maxProfitIV_2(k,prices))

# ------------------------区间动态规划-----------------------------------------
# 516. 最长回文子序列
# 输入：s = "cbbd"
# 输出：2
# 解释：一个可能的最长回文子序列为 "bb" 。
# 状态个数O(n^2) 总时间复杂度：O(n^2)
def longestPalindromeSubseq(s: str) -> int:
    n=len(s)
    @cache
    def dfs(i,j):  # dfs(i,j)表示字符串 s[i..j] 的最长回文子序列的长度。(包括了j）
        if i>j:
            return 0
        if i==j:
            return 1
        if s[i]==s[j]:
            return dfs(i+1,j-1)+2
        return max(dfs(i+1,j),dfs(i,j-1))
    return dfs(0,n-1)
def longestPalindromeSubseq_ditui(s: str) -> int:
    n=len(s)
    dfs=[[0]*n for _ in range(n)]
    for i in range(n-1,-1,-1):  # i要逆序
        dfs[i][i]=1
        for j in range(i+1,n): # 注意，这里即使i的最大值为n-1，对应的j=n，所以不会执行这个循环，因此不存在数组越界问题；同理，i的最小值为0，对应的j=1,因此不存在数组越界问题
            if s[i]==s[j]:
                dfs[i][j]=dfs[i+1][j-1]+2
            else:
                dfs[i][j]=max(dfs[i+1][j],dfs[i][j-1])
    return dfs[0][-1]
# s=input().strip()
# print(longestPalindromeSubseq(s))

# 还可以用回溯法求解 （假设要列举所有回文子序列，再找最长） 但确实会超时
def allPalindromeSubseq(s: str):
    n = len(s)
    result = []
    def backtrack(i, path):
        # 记录当前路径是否为回文子序列
        if path and path == path[::-1]:
            result.append(path)
        # 递归终止
        if i == n:
            return
        # 选择当前字符
        backtrack(i + 1, path + s[i])
        # 不选择当前字符
        backtrack(i + 1, path)
    backtrack(0, "")
    # 从所有回文子序列中找最长
    # return max(result, key=len) if result else "" 返回最长的串
    return max(len(i) for i in result) if result else ""

# 5. 最长回文子串
# 输入：s = "babad"
# 输出："bab"
# 解释："aba" 同样是符合题意的答案。
# 时间复杂度为 O(n²)（n 为输入字符串的长度），空间复杂度为 O(1)
def longestPalindrome( s: str) -> str:
    n = len(s)
    ans = ''
    def expand(l, r):
        # 以[l，r]为中心向两侧扩展
        while l >= 0 and r < n and s[l] == s[r]:
            nonlocal ans
            if len(s[l:r + 1]) > len(ans):
                ans = s[l:r + 1]
            l -= 1
            r += 1
    for i in range(n):
        expand(i, i)  # 奇数长度
        expand(i, i + 1)  # 偶数长度
    return ans

# 5. 最长回文子串 （参考最长回文子序列的写法，但复杂度过高）
def longestPalindromeSubstring(s: str) -> str:
    n = len(s)
    @cache
    def dfs(i, j):
        # 返回 s[i..j] 的最长回文子串，如果不存在返回 ""
        if i > j:
            return ""
        if i == j:
            return s[i]
        if s[i] == s[j]:
            inner = dfs(i + 1, j - 1)
            # 如果内部是完整的回文，拼接首尾
            if len(inner) == j - i - 1:
                return s[i] + inner + s[j]
        # 如果 s[i] != s[j]，这个区间不能作为回文子串
        # 但我们仍然要尝试在两个子区间中寻找较长的回文子串。
        left = dfs(i, j - 1)
        right = dfs(i + 1, j)
        return left if len(left) >= len(right) else right
    return dfs(0, n - 1)
# print(longestPalindromeSubstring("abbbca"))

# 下面返回的是回文子串的长度（上面返回的是回文子串）
def longestPalindromeSubstr(s: str) -> int:
    n = len(s)
    @cache
    def dfs(i, j):
        if i > j:
            return 0
        if i == j:
            return 1
        if s[i] == s[j]:
            inner_len = dfs(i + 1, j - 1)
            inner_full_len = j - i - 1
            if inner_len == inner_full_len:
                return inner_len + 2
        # 如果s[i] != s[j]或者内部子串不是完全回文，则取左右子串的最大值
        return max(dfs(i + 1, j), dfs(i, j - 1))
    return dfs(0, n - 1)
# print(longestPalindromeSubstr("abbbca"))

# 百度手撕 输出s的所有连续回文子串
def continuous_huiwen(s):
    n = len(s)
    out = []
    def expand(l, r):
    # 以[L，r]为中心向两侧扩展，找到所有回文并按发现顺序加入
        while l >= 0 and r < n and s[l] == s[r]:
            out.append(s[l:r + 1])
            l -= 1
            r += 1
    for i in range(n):
        expand(i, i)    # 奇数长度
        expand(i, i + 1)  # 偶数长度
    return out


# 1039. 多边形三角剖分的最低得分
def minScoreTriangulation(values: list[int]) -> int:
    # 时间复杂度：状态个数O(n^2)*每个状态计算的复杂度O(n)=O(n^3)
    n=len(values)
    @cache
    def dfs(i,j):
        if i+1==j:
            return 0
        res = float("inf")
        for k in range(i+1,j):
            res=min(res, dfs(i,k)+dfs(k,j)+values[i]*values[j]*values[k])
        return res
    return dfs(0,n-1)
def minScoreTriangulation_ditui(values: list[int]) -> int:
    n=len(values)
    dfs=[[0]*n for _ in range(n)]
    for i in range(n-3,-1,-1):
        for j in range(i+2,n):   # 为什么是i+2到n？for k in range(i+1,j)根据上面这个代码，j只能i+2开始
            res=float("inf")
            for k in range(i+1,j):
                res=min(res,dfs[i][k]+dfs[k][j]+values[i]*values[j]*values[k])
            dfs[i][j]=res
    return dfs[0][-1]


# ----------------------------树形动态规划------------------------------
# 树dp通常不存在重叠子问题从而无需考虑重复计算问题，通常不需要显式地存储cache中间计算结果，子问题的解都通过递归隐式地返回给了子问题
# 树dp和普通dp的计算的过程都是自顶向下的，即从根节点/最终目标出发，不断分解成更小的子问题直到边界，然后有些二叉树题目比如说这里的二叉树直径问题，
# 我们在递归的递的过程中自顶向下计算子树最大链长，在归的过程中自底向上计算子树直径，一直到达根节点，最终答案就被计算出来了。
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 543. 二叉树的直径
# 在递归的递的过程中*自顶向下*计算子树最大链长，在归的过程中*自底向上*计算子树直径，一直到达根节点，最终答案就被计算出来了。
def diameterOfBinaryTree(root: TreeNode) -> int:
    ans=0
    def dfs(node):  # 注意 这里没有记忆化，是因为没有*重叠*子问题，只要遍历完所有节点，ans就是最长路径(树dp 通常每个状态/节点只会访问一次，所以不需要记忆化。)
        if node is None:
            return -1
        left_len=dfs(node.left)+1
        right_len=dfs(node.right)+1
        nonlocal ans
        ans=max(ans,left_len+right_len) # 以当前节点拐弯的最长路径，注意，不是数点，是数边，所以并没有重复计算
        return max(left_len,right_len)  # 要思考返回的到底是什么？dfs(node)的目的是获取以node为根节点的最大深度(不拐弯)，所以返回的是left_len和right_len中的较大值
    dfs(root)
    return ans

# 124. 二叉树中的最大路径和
# 同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。
# 属于DP解法。如果是回溯法，会尝试枚举所有可能的路径（如从每个节点出发，探索所有向下的路径组合），并记录其中的最大值。
# 但这样会产生大量重复计算（同一子树被多次遍历），且逻辑上是 “穷举所有路径” 而非 “用子问题解推导”。
def maxPathSum(root: [TreeNode]):
    ans=-float("inf")
    def dfs(node):
        if node is None:
            return 0
        left_len=dfs(node.left)
        right_len=dfs(node.right)
        nonlocal ans
        ans=max(ans,left_len+right_len+node.val) # 以当前节点 **拐弯** 的最长路径
        # 要思考返回的到底是什么？dfs(node)的目的是获取以node为子节点的最大路径和(不拐弯)
        return max(max(left_len,right_len,0)+node.val,0)   # 如果以当前节点值作为子树的路径和为负数，则告诉父节点不选该子树
    dfs(root) # 枚举所有节点作为 拐弯的节点
    return ans

# 437. 路径总和 III （hot100）
# 给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。
# 路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
def pathSum(root, targetSum: int) -> int:
    def dfs(node, cur_sum):
        if node is None:
            return 0
        cnt=0
        if cur_sum+node.val==targetSum:
            cnt+=1
            # cur_sum=0
        cnt+=dfs(node.left, cur_sum+node.val)
        cnt+=dfs(node.right, cur_sum+node.val)
        return cnt
    def double(node): # 每个node为起点
        if node is None:
            return 0
        total= dfs(node,0)
        total+=double(node.left)
        total+=double(node.right)
        return total
    return double(root)

# 前缀和解法
def pathSum2(root, targetSum: int) -> int:
    ans = 0
    cnt = defaultdict(int)
    cnt[0] = 1  # 类似560的解法
    def dfs(node, s):
        if node is None:
            return
        nonlocal ans
        s += node.val
        ans += cnt[s - targetSum]
        cnt[s] += 1
        dfs(node.left, s)
        dfs(node.right, s)
        cnt[s] -= 1
    dfs(root, 0)
    return ans
# 687. 最长同值路径
# 返回 最长的路径的长度 ，这个路径中的 每个节点具有相同值 。(可以拐弯）
def longestUnivaluePath(root: [TreeNode]) -> int:
    ans=0
    def dfs(node):
        if node is None:
            return -1  # # 下面 +1 后，对于叶子节点就刚好是 0
        left_len=dfs(node.left)+1
        right_len=dfs(node.right)+1
        if node.left and node.left.val!=node.val: left_len=0
        if node.right and node.right.val!=node.val: right_len=0
        nonlocal ans
        ans=max(ans, left_len+right_len)
        return max(left_len, right_len)
    dfs(root)
    return ans

# 337. 打家劫舍 III
# 每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。
# 如果 两个直接相连的房子在同一天晚上被打劫 ，房屋将自动报警。 给定二叉树的 root 。返回 在不触动警报的情况下 ，小偷能够盗取的最高金额 。
# 选=左不选＋右不选＋当前节点值
# 不选=max（左选,左不选)+max(右选,右不选)
def rob4(root: [TreeNode]) -> int:
    # def dfs(node, choose):  超时
    #     if node is None:
    #         return 0
    #     if choose:
    #         ans=dfs(node.left, False)+dfs(node,False)+node.val
    #     else:
    #         ans=max(dfs(node.left,True),dfs(node.left,False))+ max(dfs(node.right,True),dfs(node.right,False))
    #     return ans
    # return max(dfs(root,True),dfs(root,False))
    def dfs(node):
        if node is None:
            return 0,0
        left_choose,left_not_choose=dfs(node.left)
        right_choose,right_not_choose=dfs(node.right)
        choose=node.val+left_not_choose+right_not_choose
        not_choose=max(left_choose,left_not_choose)+max(right_choose,right_not_choose)
        return choose,not_choose
    return max(dfs(root))

# 3186. 施咒的最大总伤害 (非hot100）
# 已知魔法师使用伤害值为 power[i] 的咒语时，他们就 不能 使用伤害为
# power[i] - 2 ，power[i] - 1 ，power[i] + 1 或者 power[i] + 2 的咒语。
# 输入：power = [1,1,3,4]
# 输出：6
# 解释：可以使用咒语 0，1，3，伤害值分别为 1，1，4，总伤害值为 6 。
def maximumTotalDamage( power: list[int]) -> int:
    # n=len(power)
    # ans=0
    # def dfs(j,dic):
    #     if j>=n:
    #         res=sum(path)
    #         nonlocal ans
    #         ans=max(ans,res)
    #     dfs(j+1,dic) # 不选
    #     if power[j] not in dic:
    #         path.append(power[j])
    #         tmp=[power[j]+1,power[j]-1,power[j]-2,power[j]+2]
    #         dic+=tmp
    #         dfs(j,dic)
    #         dic-=tmp
    # for i in range(n):
    #     path=[]
    #     dfs(i,[])
    # return ans
    cnt=Counter(power)
    a=sorted(cnt.keys()) # 已经按升序排序了，所以下面只需要满足 a[j-1]<x-2就可继续往前递归（即a[j-1]不能比x-2还要大
    print(a) # [1, 2, 3, 4]
    @cache
    def dfs(i): # 定义 dfs(i) 表示从 a[0] 到 a[i] 中选择，可以得到的伤害值之和的最大值。
        if i<0:
            return 0
        x=a[i]
        j=i
        while j > 0 and a[j-1]>=x-2:  # 不能写成a[j]>x-2  # 为什么是j-1，因为后面j-1要进入递归，所以需要判断
            j-=1
        return max(dfs(i-1),dfs(j-1)+x*cnt[x])
    return dfs(len(a)-1)
# power = [1,1,3,2,4]
# print(maximumTotalDamage(power))

# 62. 不同路径 hot
# 个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
# 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
# 问总共有多少条不同的路径？
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

# 64. 最小路径和
# 给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
def minPathSum( grid: list[list[int]]):
    m,n=len(grid),len(grid[0])
    @cache
    def dfs(i: int, j: int):
        if i < 0 or j < 0:
            return float('inf')
        if i == 0 and j == 0: # 需不需要这个判断条件可以带入i=0,j=0去看，如果没有的话，下面的val直接返回无穷大了
            return grid[i][j]
        vals=min(dfs(i - 1, j),dfs(i, j - 1))+ grid[i][j]
        return  vals
    return dfs(m - 1, n - 1)
def minPathSum2(grid: list[list[int]]):
    m, n = len(grid), len(grid[0])
    dfs = [[float('inf')] * (n + 1) for _ in range(m + 1)]
    for i, row in enumerate(grid):
        for j, x in enumerate(row):
            if i == j == 0:
                dfs[1][1] = x
            else:
                dfs[i + 1][j + 1] = min(dfs[i][j + 1], dfs[i + 1][j]) + x
    return dfs[-1][-1]
# LCP 34. 二叉树染色
# def maxValue(root: TreeNode, k: int) -> int:
#     def dfs(node,k):
#         if node is None:
#             return 0

# 判断是否是2的n次方  （5_17 高通手撕）
def isPowerOfTwo(n: int) -> bool:
    # 位运算法  例如：8 & 7 = 1000 & 0111 = 0000。
    # return n > 0 and (n & (n - 1)) == 0

    # 递推求解
    # if n <= 0:
    #     return False
    # while n % 2 == 0:
    #     n //= 2
    # return n == 1

    # dfs求解
    if n <= 0:
        return False
    if n == 1:
        return True
    if n % 2 != 0:
        return False
    return isPowerOfTwo(n // 2)
# 为什么打家劫舍需要单独定义一个dfs函数，而上面这个题不需要？
# 1. 因为 rob 本身的输入是整个数组，而递归的状态是 “考虑前 i 个房子”。 rob 的函数签名固定：只收一个数组。
# 2. 这道题递归的状态就是 n 本身，不需要额外的下标或数组。 每次递归直接传入新的 n // 2，输入输出和原函数的签名完全一致。

# 139. 单词拆分
# 输入: s = "applepenapple", wordDict = ["apple", "pen"]
# 输出: true
# 解释: 返回 true 因为 "applepenapple" 可以由 "apple" "pen" "apple" 拼接成。 注意，你可以重复使用字典中的单词。
# 本题状态个数等于 O(n)，单个状态的计算时间为 O(L^2)（L=wordDict中最长的字符串长度，注意判断子串是否在哈希集合中需要 O(L) 的时间），
# 所以记忆化搜索的时间复杂度为 O(nL^2)
def wordBreak2(s: str, wordDict) -> bool:
    n = len(s)
    wordDict = set(wordDict)  # 转换为集合提高查询效率
    @cache
    def dfs(i): # dp[i] 表示 s[0:i] 是否能被拆分。
        if i == 0:
            return True  # 空字符串可以被拆分（base case）
        # for j in range(i - 1, -1, -1):  # 从 i-1 倒序遍历到 0
        # 关于下面为什么要有for循环，可以联想到状态转移图，dfs(i)可以由多个不同的dfs(j)转移过来，所以用for循环依次遍历j
        for j in range(i):
            if s[j:i] in wordDict and dfs(j): # 也就是说字符串被分成了两部分，前部分为dfs(j)的判断（不包括j），后部分为s[j:n]判断是否在wordDict中
                return True
        return False
    return dfs(n)  # 判断整个字符串 s[0:n] 是否能被拆分
# print(wordBreak(s = "applepenapple", wordDict = ["apple", "pen"]))

def wordBreak_ditui(s: str, wordDict: list[str]) -> bool:
    n = len(s)
    wordDict = set(wordDict)
    m = len(wordDict)
    dp = [False] * (n+1)
    dp[0] = True
    for i in range(1, n+1):
        for j in range(i):
            if dp[j] and s[j:i] in wordDict:
                dp[i] = True
                break
    return dp[n]
