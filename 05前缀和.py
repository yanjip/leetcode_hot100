# time: 2025/2/28 20:36
# author: YanJP
from collections import defaultdict
from typing import List


# 53. 最大子数组和 (腾讯面试题）
def maxSubArray( nums: List[int]) -> int:
    ans = -float("inf")
    min_pre_sum = pre_sum = 0
    for x in nums:
        pre_sum += x
        ans = max(ans, pre_sum - min_pre_sum)
        min_pre_sum = min(pre_sum, min_pre_sum)
    return ans

from functools import cache
def maxSubArray_dfs(nums: List[int]) -> int:
    n = len(nums)
    @cache
    def dfs(i):  # dfs定义为以 nums[i] 结尾的连续子数组的最大和
        if i == 0: return nums[i] # 因为子数组至少包含一个数
        return max(dfs(i - 1) + nums[i], nums[i])  # 和下面等价，+nums[i]可以卸载max函数后面
    return max(dfs(i) for i in range(n))
def maxSubArray_dp( nums: List[int]) -> int:
    f = [0] * len(nums)  # f[i] 表示以 nums[i] 结尾的连续子数组的最大和
    f[0] = nums[0]
    for i in range(1, len(nums)):
        f[i] = max(f[i - 1], 0) + nums[i]
    return max(f)

# 560. 和为 K 的子数组 ——返回 该数组中和为 k 的子数组的个数
def subarraySum( nums: List[int], K: int) -> int:
    s=[0]*(len(nums)+1) # 这里必须写成+1，因为第一次遍历时会执行cnt[sj]+=1，得到cnt[0]=1
    for i,x in enumerate(nums):
        s[i+1] = s[i]+x
    ans=0
    cnt=defaultdict(int)
    for sj in s:
        ans += cnt[sj-K]
        cnt[sj]+=1
    return ans
# print(subarraySum([1,2,3],2))

# 一次遍历解法：一边计算前缀和，一边遍历前缀和。
def subarraySum_once(nums: List[int], K: int) -> int:
    ans = s = 0
    cnt = defaultdict(int)
    cnt[0] = 1  # 对应上面sj=1时，cnt[sj]+=1
    for x in nums:
        s += x             #1. 计算前缀和
        ans += cnt[s - K]  #2. 更新答案
        cnt[s] += 1        #3. 更新哈希表中对应前缀和的个数
    return ans


# 437. 路径总和 III
# 给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。
# 路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
# 解法一：递归解法
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
    def double(node):
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