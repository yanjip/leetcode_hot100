# time: 2025/2/24 11:07
# author: YanJP
from collections import defaultdict,Counter


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
