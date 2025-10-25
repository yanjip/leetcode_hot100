# time: 2025/4/4 8:56
# author: YanJP
from collections import defaultdict,Counter
from typing import List
# -----------------------09数组.py--------------------------

# 1. 两数之和
# 输入：nums = [3,2,4], target = 6
# 输出：[1,2]
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        idx = {}
        for j, x in enumerate(nums):
            if target - x in idx:
                return [idx[target - x], j]
            idx[x] = j

# 49. 字母异位词分组
# 输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
# 输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
def groupAnagrams(strs):
    table = {}
    for s in strs:
        s_ = "".join(sorted(s))
        if s_ not in table:
            table[s_] = [s]
        else:
            table[s_].append(s)
    return list(table.values())

# 167. 两数之和 II - 输入有序数组 （也可认为是相向双指针） （非hot100
# 输入：numbers = [2,7,11,15], target = 9
# 输出：[1,2]
# 解释：2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。返回 [1, 2] 。
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
# 给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，
# 同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组。
# 输入：nums = [-1,0,1,2,-1,-4]
# 输出：[[-1,-1,2],[-1,0,1]]
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

# 240. 搜索二维矩阵 II 和上题解法一模一样
# 搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
# 每行的元素从左到右升序排列。
# 每列的元素从上到下升序排列。
# 时间复杂度：O(m+n)
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


# 238. 除自身以外数组的乘积
# 给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。
# 请 不要使用除法，且在 O(n) 时间复杂度内完成此题。
# 输入: nums = [1,2,3,4]
# 输出: [24,12,8,6]
# 前后缀分解 如果知道了 i 左边所有数的乘积，以及 i 右边所有数的乘积，就可以算出 answer[i]。
# 定义 pre[i] 表示从 nums[0] 到 nums[i−1] 的乘积。
# 定义 suf[i] 表示从 nums[i+1] 到 nums[n−1] 的乘积。
def productExceptSelf( nums: List[int]) -> List[int]:
    n = len(nums)
    pre = [1] * n
    suf = [1] * n

    for i in range(1, n):
        pre[i] = pre[i - 1] * nums[i - 1] # [1, 1, 2, 6]

    for i in range(n - 2, -1, -1):
        suf[i] = suf[i + 1] * nums[i + 1] # [24, 12, 4, 1]

    return [p * s for p, s in zip(pre, suf)]
# productExceptSelf([2,3,4])

# 矩阵置零
# 给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
def setZeroes(matrix: List[List[int]]) -> None:
    m=len(matrix)
    n=len(matrix[0])
    def design(x,y):
        for i in range(n):
            matrix[x][i]=0
        for i in range(m):
            matrix[i][y]=0
    mark=[]
    for i in range(m):
        for j in range(n):
            if matrix[i][j]==0:
                mark.append([i,j])
    for x in mark:
        design(x[0],x[1])

# 54. 螺旋矩阵
# 给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
def spiralOrder(matrix: List[List[int]]) -> List[int]:
    DIRS = (0, 1), (1, 0), (0, -1), (-1, 0)  # 右下左上
    m,n=len(matrix),len(matrix[0])
    ans=[]
    i=j=di=0
    for _ in range(m*n):
        ans.append(matrix[i][j])
        matrix[i][j]=None
        x = i + DIRS[di][0]
        y = j + DIRS[di][1]  # 下一步的位置
        if x<0 or x>=m or y<0 or y>=n or matrix[x][y] is None:
            di=(di+1)%4
        i+=DIRS[di][0]
        j+=DIRS[di][1]
    return ans
print(spiralOrder(matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]))