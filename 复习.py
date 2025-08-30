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
# 总结
# 和为 K 的子数组：前缀和求解

# 最长公共子序列LCS：dfs(m-1,n-1) 选或不选
# dp[i][j] = dp[i-1][j-1]+1 if s1[i] == s2[j] else max(dp[i-1][j], dp[i][j-1])


# 最长递增子序列LCS：max(dfs(i)) 枚举选哪个的思路 (以i结尾
# for j in range(i): if nums[j] < nums[i]: res=max(res, dfs(j)) return res+1

# 最长连续递增子序列LICS：max(dfs(i)) (以i结尾
# if i==0 return 1 if nums[i]>nums[i-1] return dfs(i-1)+1 else return 1

# 最长子数组和 前缀和（维护一个最小的前缀和）+dfs max(dfs(i))
# dfs[i]=max(dp[i-1]+nums[i], nums[i]) if i<0 return 0

# 最长回文子序列：dfs(0,n-1) 表示s[i..j]的最长回文子序列
# if i>j: return 0 if i==j: return 1 if s[i]==s[j]: return dfs(i+1,j-1)+2 else: return max(dfs(i+1,j), dfs(i,j-1))

# 最长回文子串：dfs(j,i) 表示子串s[j:i]是否为回文串 j \in [0,i] 且for循环遍历
# 入参：if dfs(j,i) and i-j+1>max_len: start=j max_len=i-j+1 return s[start:start+max_len]
# dfs：if i == j: return True elif i - 1 == j: return s[i] == s[j] else: return dfs(j+1, i-1) and s[i] == s[j]

# 01背包：dfs(n-1, cap) 选或不选
# if weights[i]>c: return dfs(i-1, cap) else: return max(dfs(i-1, cap), dfs(i-1, cap-weights[i])+values[i])

# 目标和：返回的是方案数 target+=sum(nums) target//=2
# 入参：dfs(n-1,target)
# if i<0: return 1 if target==0 else 0 return dfs(i-1,target)+dfs(i-1,target-nums[i])

# 零钱兑换 返回凑成amount的最小硬币数量 完全背包
# 答案：ans=dfs(n-1, amount) return ans if ans<=inf else -1
# if i<0: return 0 if target==0 else inf. if target<coins[i]: return dfs(i-1, target). return min(dfs(i-1, target), dfs(i, target-coins[i])+1)

# 分割等和子集 返回是否可以分割 选或不选的思路
# 答案： dfs(n-1, s//2)
# if i<0: return s==0. if nums[i]>s return dfs(i-1,s). return dfs(i-1, s) or dfs(i-1, s-nums[i])

# 子集回溯：枚举选哪个 dfs(0)
# 每次都要执行append：ans.append(path[:]). for j in range(i,n): path.append(nums[j]) dfs(j+1) pop

# 分割回文串 枚举选哪个 dfs(0)
# if i==n: ans.append(path[:]) for j in range(i,n): if s[i:j+1]==s[i:j+1][::-1]: path.append(s[i:j+1]) dfs(j+1) path.pop()
# 总结
# 和为 K 的子数组：前缀和求解

# 最长公共子序列LCS：dfs(m-1,n-1) 选或不选
# dp[i][j] = dp[i-1][j-1]+1 if s1[i] == s2[j] else max(dp[i-1][j], dp[i][j-1])


# 最长递增子序列LCS：max(dfs(i)) 枚举选哪个的思路 (以i结尾
# for j in range(i): if nums[j] < nums[i]: res=max(res, dfs(j)) return res+1

# 最长连续递增子序列LICS：max(dfs(i)) (以i结尾
# if i==0 return 1 if nums[i]>nums[i-1] return dfs(i-1)+1 else return 1

# 最长子数组和 前缀和（维护一个最小的前缀和）+dfs max(dfs(i))
# dfs[i]=max(dp[i-1]+nums[i], nums[i]) if i<0 return 0

# 最长回文子序列：dfs(0,n-1) 表示s[i..j]的最长回文子序列
# if i>j: return 0 if i==j: return 1 if s[i]==s[j]: return dfs(i+1,j-1)+2 else: return max(dfs(i+1,j), dfs(i,j-1))

# 最长回文子串：dfs(j,i) 表示子串s[j:i]是否为回文串 j \in [0,i] 且for循环遍历
# 入参：if dfs(j,i) and i-j+1>max_len: start=j max_len=i-j+1 return s[start:start+max_len]
# dfs：if i == j: return True elif i - 1 == j: return s[i] == s[j] else: return dfs(j+1, i-1) and s[i] == s[j]

# 01背包：dfs(n-1, cap) 选或不选
# if weights[i]>c: return dfs(i-1, cap) else: return max(dfs(i-1, cap), dfs(i-1, cap-weights[i])+values[i])

# 目标和：返回的是方案数 target+=sum(nums) target//=2
# 入参：dfs(n-1,target)
# if i<0: return 1 if target==0 else 0 return dfs(i-1,target)+dfs(i-1,target-nums[i])

# 零钱兑换 返回凑成amount的最小硬币数量 完全背包
# 答案：ans=dfs(n-1, amount) return ans if ans<=inf else -1
# if i<0: return 0 if target==0 else inf. if target<coins[i]: return dfs(i-1, target). return min(dfs(i-1, target), dfs(i, target-coins[i])+1)

# 分割等和子集 返回是否可以分割 选或不选的思路
# 答案： dfs(n-1, s//2)
# if i<0: return s==0. if nums[i]>s return dfs(i-1,s). return dfs(i-1, s) or dfs(i-1, s-nums[i])
# ——————————————————————————回溯问题————————————————————————————————
# 子集回溯：枚举选哪个 dfs(0)
# 每次都要执行append：ans.append(path[:]). for j in range(i,n): path.append(nums[j]) dfs(j+1) pop

# 分割回文串 枚举选哪个 dfs(0)
# if i==n: ans.append(path[:]) for j in range(i,n): if s[i:j+1]==s[i:j+1][::-1]: path.append(s[i:j+1]) dfs(j+1) path.pop()

# 单词搜索 两层for循环入参 ans=ans or dfs(i,j,k=0) k==len(word)-1时说明匹配 i in range(m) j in range(n)
# 超出边界或不匹配： return False. if k==lens-1: return True. board[i][j]=1(标记访问过） ans=dfs(i+1,j,k+1) or dfs(i-1,j,k+1) or dfs(i,j+1,k+1) or dfs(i,j-1,k+1) board[i][j]=word[k]
# borad[i][j] = word[k] returnans

# 组合：枚举选哪个 后序选择比较方便 dfs(n)
# d=k-len(path) if i<d: return if len(path==k): ans.append(path[:]) return. for j in range(i,0,-1): path.append(j) dfs(j-1) path.pop()

# 组合总和 给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，
# 找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合。可重复选
# 入参：dfs(start=0,s=0)
# if s==target: ans.append(path[:]) return. if s>target: return. for j in range(start,n): path.append(candidates[j]) dfs(j,s+candidates[j]) path.pop()

# 括号生成. 数字 n 代表生成括号的对数 入参：dfs(0,cnt_left).
# if i==2*n: ans.append(path[:]) return. if cnt_left<n: path[i]=( dfs(i+1,cnt_left+1). if i-cnt_left<cnt_left: path[i]=) dfs(i+1,cnt_left)

# 全排列 dfs(0, set(nums))
# if i==n:ans.append(path[:]) return. for j in sets: path[i]=j dfs(i+1, sets-{j})

# N皇后 记录col=【0】*n 数组，第i个元素的值 == 第i行的皇后所在的列
# 入参：dfs(row=0, set(range(n))
# if row==n: 添加答案 return. for c in s: if valid(row,c)： col[row]=c dfs(row+1, s-{c})
# 其中valid(row,c): for R in range(row): C=col[R] if row+c==R+C or row-c==R-C: return False. 最后return True


# ——————————————————————————二叉树———————————————————————————————————
# 二叉树的最大深度
# left=maxDepth(root.left) right=maxDepth(root.right) return max(left,right)+1

# 二叉树的右视图   入参：dfs(root, depth=0) ans=[]
# if len(ans)==depth: ans.append(node.val). dfs(node.right, depth+1) dfs(node.left, depth+1)

# 验证二叉搜索树  前序遍历做法（先访问根节点值)
# if root is None: x=root.val. return left<x<right and isValid(root.left, left, x) and isValid(root.right, x, right)

# 二叉树的最近公共祖先 调用自身
# if root is None or root==p or root==q: return root
# left=lowestCommonAncestor(root.left, p, q) right=lowestCommonAncestor(root.right, p, q)
# if left and right: return root
# return left if left else right 只有一侧非空，则返回非空的那侧

# 二叉树的中序遍历 入参：dfs(root) ans=[]
# dfs(node.left) ans.append(node.val) dfs(node.right)

# 二叉树的层序遍历  BFS 使用队列求解
# ans=[] q=deque([root])
# while q: vals=[] for _ in range(len(q)): node=q.popleft() vals.append(node.val)
# if node.left: q.append(node.left) if node.right: q.append(node.right). for循环后：ans.append(vals)

# 二叉搜索树中第 K 小的元素 入参：dfs(root)
# if node is None: return -1. left=dfs(node.left).
# if left!=k-1: return left. k-=1 if k==0: return node.val. return dfs(node.right)
# 解法2：直接层序遍历，然后取第k个元素

# 二叉树展开为链表 头插法 调用自身
# def flatten(root: Optional[TreeNode]) -> None:
# if root is None: return. self.flatten(root.right). self.flatten(root.left) root.left=None root.right=self.head. self.head=root

# 二叉树的直径 入口：dfs(root)
# if node is None: return -1. left=dfs(node.left)+1 right=dfs(node.right)+1 ans=max(ans, left+right) return max(left,right)

# ————————————————————————————链表——————————————————————————————————
# 反转链表：pre=None cur=head
# while cur: nxt=cur.next cur.next=pre pre=cur cur=nxt return pre

# 删除链表的倒数第 N 个结点 同向双指针
# dummgy=ListNode(next=head) right=dummy for _ in range(n): right=right.next
# left=dummy while right.next: left=left.next right=right.next.
# left.next=left.next.next return dummy.next

# 两两交换链表中的节点
# 维护node1-node3四个节点， node0看成pre，node1看成cur 然后node0.next=node2 node2

# 合并两个有序链表
# cur=dummy=ListNode()
# while list1 and list2: if list1.val<list2.val: cur.next=list1 list1=list1.next
# else: cur.next=list2 list2=list2.next
# cur=cur.next 退出while循环：cur.next=list1 if list1 else list2 return dummy.next

# 560. 和为 K 的子数组（前缀和、哈希表）
# def subarraySum_once(nums: list[int], K: int) -> int:
#     ans = s = 0
#     cnt = defaultdict(int)
#     cnt[0] = 1  # 对应上面sj=1时，cnt[sj]+=1
#     for x in nums:
#         s += x             #1. 计算前缀和
#         ans += cnt[s - K]  #2. 更新答案
#         cnt[s] += 1        #3. 更新哈希表中对应前缀和的个数
#     return ans

# 无重复字符的最长子串  滑动窗口
# def lengthOfLongestSubstring( s):
#     left=0
#     ans=0
#     # 记录当前窗口中每个字符的出现次数。
#     count=Counter() # 也可以写成 defaultdict(int)
#     for right,x in enumerate(s):
#         count[x]+=1
#         while count[x]>1: # 出现重复了，left对应的cnt可以-1，然后移动left
#             count[s[left]]-=1 # 有可能s[left]等于x，此时count[x]==2，所以这行不能写成=0
#             left+=1
#         ans=max(ans,right-left+1)
#     return ans

# 滑动窗口最大值 （单调队列）
# def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
#     ans=[]
#     q=deque() # 存在的是下标
#     for i, x in enumerate(nums):
#         while q and nums[q[-1]]<=x: # x太大了 就存x pop队列的数
#             q.pop()
#         q.append(i)
#         if i-q[0]+1>k:
#             q.popleft()
#         if i>=k-1: # 从满足窗口大小后就开始存答案
#             ans.append(nums[q[0]])
#     return ans

# —————————————————————————————二分查找——————————————————————————
# def low_bound(nums, target): # 找到第一个大于等于≥target的索引
#     left,right=0,len(nums)-1
#     while left<=right:
#         mid=(left+right)//2
#         if nums[mid]<target: left=mid+1
#         else: right=mid-1
#     return left
# 变形：如果求找到第一个>target的索引，则可看成求第一个≥(target+1)的索引
