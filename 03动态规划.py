# time: 2025/2/18 9:35
# author: YanJP
# 198. 打家劫舍
from collections import Counter
from functools import cache
import time

def rob(nums: list[int]):  # 超时，时间复杂度是指数级别
    def dfs(i):
        if i<0:  # 不能写等于0
            return 0
        return max(dfs(i-1),dfs(i-2)+nums[i])
    return dfs(len(nums)-1)
def rob2(nums: list[int]):
    n=len(nums)
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
def rob3(self, nums: list[int]) -> int:
    n=len(nums)
    # f=[0]*(n+2)
    # for i, x in enumerate(nums):
    #     f[i+2]=max(f[i+1],f[i]+x)
    # return f[n+1]
    f0,f1=0,0
    for i, x in enumerate(nums):
        newf=max(f1,f0+x)
        f0=f1
        f1=newf
    return newf
# nums=list(map(int,input().strip().split()))
# print(rob(nums))

#  70. 爬楼梯
def climbStairs( n: int) -> int:
    # @cache
    # def dfs(i):
    #     if i==0 or i==1:
    #         return 1
    #     return dfs(i-1)+dfs(i-2)
    # return dfs(n)

    # dfs=[0]*(n+1)
    # dfs[1],dfs[0]=1,1
    # for i in range(0,n-1):
    #     dfs[i+2]=dfs[i+1]+dfs[i]
    # return dfs[n]

    dfs=[0]*(n+1)
    dfs[0]=dfs[1]=1
    for i in range(2,n+1):
        dfs[i]=dfs[i-1]+dfs[i-2]
    return dfs[n]

# 377. 组合总和 Ⅳ
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
        if i<0:
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
    dfs=[[0]*(target+1) for _ in range(n+1)]  # 使用 n+1 可以确保在遍历所有元素时不会出现索引越界的问题
    dfs[0][0]=1  # 很重要，对应上述dfs做法，其中i<0 and target==0 的情况,答案为1。但由于上面一行索引加了1，因此对应的是i==0 的情况
    for i in range(n):
        for  j in range(target+1):
            if j<nums[i]:
                dfs[i+1][j]=dfs[i][j]
            else:
                dfs[i+1][j]=dfs[i][j]+dfs[i][j-nums[i]]
    print(dfs)
    return dfs[-1][-1]

# nums=[1,1,1,1,1]
# target=int(input().strip())
# print(findTargetSumWays2(nums,target))

# 322. 零钱兑换  完全背包问题 返回能凑成amount的最小的硬币数
def coinChange(coins: list[int], amount: int) -> int:
    n=len(coins)
    # def dfs(i,target):  # 回溯写法--超时
    #     if i<0:
    #         return 0 if target==0 else float("inf")
    #     if target<coins[i]:
    #         return dfs(i-1,target)
    #     ans=min(dfs(i-1,target),dfs(i,target-coins[i])+1)
    #     return ans
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

# 416. 分割等和子集
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
    dfs = [[False] * (s + 1) for _ in range(n + 1)]
    dfs[0][0] = True
    for i, x in enumerate(nums):
        for j in range(s + 1):
            if x > j:
                dfs[i + 1][j] = dfs[i][j]
            else:
                dfs[i + 1][j] = dfs[i][j] or dfs[i][j - x]
    return dfs[-1][-1]
# 1143. 最长公共子序列
# 计算复杂度：O(mn) 状态个数 (mn)* 每个状态的计算时间(1)
def longestCommonSubsequence_huisu(text1: str, text2: str) -> int:
    m=len(text1)
    n=len(text2)
    @cache
    def dfs(i,j):
        if i<0 or j<0:
            return 0
        if text1[i]==text2[j]:
            return dfs(i-1,j-1)+1
        return max(dfs(i-1,j),dfs(i,j-1))
    return dfs(m-1,n-1)
def longestCommonSubsequence(text1: str, text2: str) -> int:
    len1=len(text1)
    len2=len(text2)
    dfs=[[0]*(len2+1) for _ in range(len1+1)]
    for i in range(1,len1+1):
        for j in range(1,len2+1):
            if text1[i-1]==text2[j-1]:  # 这里要写减一，不然会越界！！！
                dfs[i][j]=dfs[i-1][j-1]+1
            else:
                dfs[i][j]=max(dfs[i-1][j],dfs[i][j-1])
    return dfs[-1][-1]
def longestCommonSubsequence2(text1: str, text2: str) -> int:  # 注意和上面解法的区别
    len1=len(text1)
    len2=len(text2)
    dfs=[[0]*(len2+1) for _ in range(len1+1)]
    for i,x in enumerate(text1):
        for j, y in enumerate(text1):
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
    # dfs=[[0]*(n+1) for _ in range(m+1)]
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
    dfs[0]=[i for i in range(n+1)]
    for i, x in enumerate(word1):
        dfs[i+1][0]=i+1
        for j, y in enumerate(word2):
            if x==y:
                dfs[i+1][j+1]=dfs[i][j]
            else:
                dfs[i+1][j+1]=min(dfs[i+1][j],dfs[i][j+1],dfs[i][j])+1
    return dfs[-1][-1]
# word1=input().strip()
# word2=input().strip()
# print(minDistance_ditui(word1,word2))

# 583. 两个字符串的删除操作
def minDistance_583( word1: str, word2: str) -> int:
    m=len(word1)
    n=len(word2)
    @cache
    def dfs(i,j):
        if i<0 or j<0:
            return 0
        if word1[i]==word2[j]:
            return dfs(i-1,j-1)
        return min(dfs(i-1,j)+1, dfs(i,j-1)+1, dfs(i-1,j-1)+2)
    return dfs(m-1,n-1)
word1 = "sea"
word2 = "eat"
print(minDistance_583(word1,word2))

# 300. 最长递增子序列 LIS
# 思路1: 选或不选 为了比大小，需要知道上一个选的数字
# 思路2: 枚举选哪个 比较当前选的数字和下一个要选的数字
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
    return dfs[-1]
# nums = [10,9,2,5,3,7,101,18]
# start_time = time.perf_counter()
# print(lengthOfLIS(nums))
# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(f"程序运行时间: {elapsed_time:.6f} 秒")

# 最长递增子数组（Longest Continuous Increasing Subsequence, LCIS）
def find_length_of_lcis_dfs(nums):
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

# 53. 最大子数组和 (腾讯面试题）三种解法
def maxSubArray( nums: list[int]) -> int:
    ans = -float("inf")
    min_pre_sum = pre_sum = 0
    for x in nums:
        pre_sum += x
        ans = max(ans, pre_sum - min_pre_sum)
        min_pre_sum = min(pre_sum, min_pre_sum)
    return ans
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
def maxProfit2( prices: list[int]) -> int:
    n=len(prices)
    @cache
    def dfs(i,hold):  # dfs(i,hold)表示第i天持有股票或者不持有股票，hold为True表示持有股票，为False表示不持有股票的最大利润
        # 定义dfs(i,0)表示到第i天*结束*时，未持有股票的最大利润定义; dfs(i,1)表示到第i天结束时，持有股票的最大利润
        # 由于第i-1天的结束就是第i天的开始, dfs(i-1,·)也表示到第i天开始时的最大利润
        if i<0:
            return -float("inf") if hold else 0
        if hold:
            return max(dfs(i-1,hold),dfs(i-1,False)-prices[i])  # 如果题目改成含冷冻期(卖出后冷冻一天,即买入股票前一天不能卖出)，则直接改成dfs(i-2,False)-prices[i]
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
        ans=max(ans,left_len+right_len) # 以当前节点拐弯的最长路径
        return max(left_len,right_len)  # 要思考返回的到底是什么？dfs(node)的目的是获取以node为子节点的最大深度(不拐弯)，所以返回的是left_len和right_len中的较大值
    dfs(root)
    return ans

# 124. 二叉树中的最大路径和
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
        return max(max(left_len,right_len)+node.val,0)   # 如果以当前节点值作为子树的路径和为负数，则告诉父节点不选该子树
    dfs(root) # 枚举拐弯的节点
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

# 3186. 施咒的最大总伤害
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
    a=sorted(cnt.keys())
    @cache
    def dfs(i):
        if i<0:
            return 0
        x=a[i]
        j=i
        while j > 0 and a[j-1]>=x-2:  # 不能写成a[j]>x-2  # 为什么是j-1，因为后面j-1要进入递归，所以需要判断
            j-=1
        return max(dfs(i-1),dfs(j-1)+x*cnt[x])
    return dfs(len(a)-1)
# power = [1,1,3,4]
# print(maximumTotalDamage(power))
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

# 64. 最小路径和
# 给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
def minPathSum( grid: list[list[int]]) -> int:
    m,n=len(grid),len(grid[0])
    @cache
    def dfs(i: int, j: int) -> int:
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
#         i