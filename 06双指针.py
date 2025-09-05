# time: 2025/3/2 12:29
# author: YanJP
from collections import Counter


# 11. 盛最多水的容器
def maxArea(height: list[int]) -> int:
    i, j, res = 0, len(height) - 1, 0
    while i < j:
        res = max(min(height[i], height[j]) * (j - i), res)
        # 核心思想：在固定两端后，每轮向内移动短板，所有消去的状态都不会导致面积最大值丢失
        if height[i] < height[j]:
            i += 1
        else:
            j -= 1
    return res

#42. 接雨水
# 前缀和分解
def trap(height: list[int]) -> int:
    n = len(height)
    pre = [0] * n
    suf = [0] * n
    pre[0] = height[0]
    suf[-1] = height[-1]
    for i in range(1, n):
        pre[i] = max(pre[i - 1], height[i])
    for i in range(n - 2, -1, -1):
        suf[i] = max(suf[i + 1], height[i])
    ans = 0
    for h, p, s in zip(height, pre, suf):
        ans += min(s, p) - h
    return ans

# 双向双指针
def trap2(height: list[int]) -> int:
    # 优化空间复杂度O(1)
    n=len(height)
    left=0
    right=n-1
    ans=0
    pre_max=0
    suf_max=0
    while left< right:
        pre_max=max(pre_max,height[left])
        suf_max=max(suf_max,height[right])
        if pre_max<suf_max:
            ans+=pre_max-height[left]
            left+=1
        else:
            ans+=suf_max-height[right]
            right-=1
    return ans


# 2824. 统计和小于目标的下标对数目
def countPairs(nums: list[int], target: int) -> int:
    nums.sort()
    ans = 0
    left = 0
    right = len(nums) - 1
    while left < right:
        if nums[left] + nums[right] < target:
            ans += (right - left)
            left+=1
        elif nums[left] + nums[right] >= target:
            right -= 1
    return ans
nums=[-6,2,5,-2,-7,-1,3]
print(countPairs(nums,-2))

# 3. 无重复字符的最长子串 (腾讯手撕题）
def lengthOfLongestSubstring( s):
    left=0
    ans=0
    # 记录当前窗口中每个字符的出现次数。
    count=Counter() # 也可以写成 defaultdict(int)
    for right,x in enumerate(s):
        count[x]+=1
        while count[x]>1: # 出现重复了，left对应的cnt可以-1，然后移动left
            count[s[left]]-=1 # 有可能s[left]等于x，此时count[x]==2，所以这行不能写成=0
            left+=1
        ans=max(ans,right-left+1)
    return ans

# print(length_of_longest_substring("abcabcbb"))  # 输出: 3

