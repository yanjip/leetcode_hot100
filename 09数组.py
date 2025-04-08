# time: 2025/4/4 8:56
# author: YanJP
from collections import defaultdict,Counter

# 167. 两数之和 II - 输入有序数组 （也可认为是相向双指针）
def twoSum(numbers, target: int) :
    left = 0
    right = len(numbers) - 1
    while True:
        s = numbers[left] + numbers[right]
        if s == target:
            break
        elif s > target:
            right -= 1
        else:
            left += 1
    return [left + 1, right + 1]

# 15. 三数之和
def threeSum(nums):
    nums.sort()
    ans=[]
    for i in range(len(nums)-2):
        x=nums[i]
        if i>0 and x==nums[i-1]:
            continue
        if x+nums[i+1]+nums[i+2]>0:
            break
        if x+nums[-2]+nums[-1]<0:
            continue
        j=i+1
        k=len(nums)-1
        while j<k:
            s=x+nums[j]+nums[k]
            if s>0:
                k-=1
            elif s<0:
                j+=1
            else:
                ans.append([x,nums[j],nums[k]])
                j+=1 # 不能有重复的三元组，所以当前的j不能用了
                while j<k and nums[j]==nums[j-1]:
                    j+=1
                k-=1
                while j<k and nums[k]==nums[k+1]:
                    k-=1
    return ans




# ------------------------------滑动窗口类型题------------------------------------------------------
# 209. 长度最小的子数组
# 给定一个含有 n 个正整数的数组和一个正整数 target。找出该数组中满足其总和大于等于 target 的长度最小的 子数组，并返回其长度
# 时间复杂度：O(n)，因为left和right都是至多+=1执行n次
def minSubArrayLen(target: int, nums: list[int]) -> int:
    ans=len(nums)+1
    left=0
    s=0
    for right, x in enumerate(nums):
        s+=x
        while s>=target: # 滑动窗口的数值是满足条件的；  left往后一位或者多位都可能满足条件，所以更新答案在while内
            ans=min(ans, right-left+1)
            s-=nums[left]
            left+=1 # 移动left前要保证总和大于等于 target
    return ans if ans<=len(nums) else 0

def minSubArrayLen2(target: int, nums: list[int]) -> int:
    ans=len(nums)+1
    left=0
    s=0
    for right, x in enumerate(nums):
        s+=x
        # 注意，这里不需要判断left<=right, 因为如果left<=right, 那么s-nums[left]一定是≤0的，因此不会进入while循环
        while s-nums[left]>=target: # 因为left越往后，越容易满足题目条件
            s-=nums[left]
            left+=1 # 移动left前要保证总和大于等于 target
        if s>=target:
            ans=min(ans, right-left+1)
    return ans if ans<=len(nums) else 0

# 713. 乘积小于 K 的子数组
# 给你一个整数数组 nums 和一个整数 k ，请你返回子数组内所有元素的乘积严格小于 k 的连续子数组的数目。
def numSubarrayProductLessThanK( nums: list[int], k: int) -> int:
    if k<=1: return 0
    prob=1
    ans=0
    left=0
    for right, x in enumerate(nums):
        prob*=x
        while prob>=k: # 要保证滑动窗口内的乘积小于k，不满足就移动left； left往后一位不一定能满足条件，所以需要while循环
            prob/=nums[left]
            left+=1
        ans+=(right-left+1) # 这里注意，[l,r] [l+1,r] ...[r,r]都是满足条件的子数组
    return ans

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

# 一次遍历解法：一边计算前缀和，一边遍历前缀和。
def subarraySum_once(nums: list[int], K: int) -> int:
    ans = s = 0
    cnt = defaultdict(int)
    cnt[0] = 1  # 对应上面sj=1时，cnt[sj]+=1
    for x in nums:
        s += x             #1. 计算前缀和
        ans += cnt[s - K]  #2. 更新答案
        cnt[s] += 1        #3. 更新哈希表中对应前缀和的个数
    return ans

# 3. 无重复字符的最长子串
# 给定一个字符串 s ，请你找出其中不含有重复字符的 最长 子串 的长度。
def lengthOfLongestSubstring( s):
    left=0
    ans=0
    count=Counter()
    for right,x in enumerate(s):
        count[x]+=1
        while count[x]>1:
            count[s[left]]-=1
            left+=1
        ans=max(ans,right-left+1)
    return ans