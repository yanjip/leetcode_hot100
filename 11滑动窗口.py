# time: 2025/9/18 10:25
# author: YanJP
from collections import defaultdict, Counter, deque


# ------------------------------11滑动窗口.py------------------------------------------------------
# 239. 滑动窗口最大值 （单调队列）
# 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。
# 你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
# 返回 滑动窗口中的最大值 。
def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
    ans=[]
    q=deque() # 存在的是下标，从左往右递减的队列
    for i, x in enumerate(nums):
        while q and nums[q[-1]]<=x: # x太大了 就存x pop队列的数
            q.pop()
        q.append(i)
        if i-q[0]+1>k: # pop出（左边）的元素
            q.popleft()
        if i>=k-1: # 从满足窗口大小后就开始存答案
            ans.append(nums[q[0]])
    return ans

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
# 和560不同的是，本题由于算乘积，且数组均大于等于1，因此，前缀乘积是单调增的，只能逐步往后扩展找到满足的子数组
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
# 数组可能是负数
def subarraySum( nums: list[int], k: int) -> int:
    #  如     1 1 0 1 1，k=2
    # 前缀和:0 1 2 2 3 4, 第二个2减0=2,得到一个子数组; 4减第一个2=2，得到一个子数组; 4-第二个2=2，得到一个子数组;...
    s=[0] * (len(nums)+1) # 这里必须写成+1，因为第一次遍历时会执行cnt[sj]+=1，得到cnt[0]=1
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
    # 记录当前窗口中每个字符的出现次数。
    count=Counter() # 也可以写成 defaultdict(int)
    for right,x in enumerate(s):
        count[x]+=1
        while count[x]>1: # 出现重复了，left对应的cnt可以-1，然后移动left
            count[s[left]]-=1 # 有可能s[left]等于x，此时count[x]==2，所以这行不能写成=0
            left+=1
        ans=max(ans,right-left+1)
    return ans

# 438. 找到字符串中所有字母异位词
# 给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。
# 输入: s = "cbaebabacd", p = "abc"
# 输出: [0,6]
# 起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
# 起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。
def findAnagrams(s: str, p: str) -> list[int]:
    ans = []
    cnt = Counter(p)  # 统计 p 的每种字母的出现次数
    left = 0
    for right, c in enumerate(s):
        cnt[c] -= 1  # 右端点字母进入窗口
        while cnt[c] < 0:  # 字母 c 太多了
            cnt[s[left]] += 1  # 左端点字母离开窗口
            left += 1
        if right - left + 1 == len(p):  # s' 和 p 的每种字母的出现次数都相同
            ans.append(left)  # s' 左端点下标加入答案
    return ans

# 76. 最小覆盖子串
# 给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。
# 如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
# 输入：s = "ADOBECODEBANC", t = "ABC"
# 输出："BANC"
# 解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。
def minWindow(s: str, t: str) -> str:
    ans_left, ans_right = -1, len(s)
    cnt_s = Counter()
    cnt_t = Counter(t)
    left = 0
    for right, c in enumerate(s):
        cnt_s[c] += 1
        # cnt_s >= cnt_t 表示对于 cnt_t 中的每一个键（字符），cnt_s 中对应键的计数都大于或等于 cnt_t 中的计数。
        while cnt_s >= cnt_t:
            if right - left < ans_right - ans_left:
                ans_left, ans_right = left, right
            cnt_s[s[left]] -= 1  # 如果子串涵盖 t，就不断移动左端点 left 直到不涵盖为止。
            left += 1
    return "" if ans_left < 0 else s[ans_left:ans_right + 1]