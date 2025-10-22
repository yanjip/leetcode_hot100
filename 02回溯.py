# time: 2025/2/8 10:54
# author: YanJP
from collections import Counter

# ------------02回溯.py------------
# ------------回溯可分为子集型回溯、组合型回溯、排列型回溯-------------------------

# 17. 电话号码的字母组合
# 计算复杂度分析：对于每个字母组合，需要执行一次递归调用。字母组合的数量最大为4^n
# 每次递归中，主要操作是更新 path[i] 和将 path 转换为字符串。这些操作的时间复杂度是 O(n)。
# 因此，时间复杂度是 O(n * 4^n)。
def letterCombinations(digits: str) :
    ans=[]
    n=len(digits)
    if len(digits) == 0:
        return []
    Phone = ['', '', 'abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
    path=['']*n
    def dfs(i):
        if i==n:
            ans.append(''.join(path))
            return
        for c in Phone[int(digits[i])]:
            path[i]=c
            # 因为每次递归到 i，一定会修改 path【i】，那么递归到终点时，每个 path【i】 都被覆盖成要枚举的字母，所以不需要恢复现场。
            dfs(i+1)
    dfs(0)
    return ans

# 78. 子集
# 时间复杂度取决于生成的所有可能的子集的数量和每次递归的时间复杂度。时间复杂度O(n*2^n) （选或者不选的角度分析 所以是2^n）
def subsets(nums):  # 从结果的角度进行回溯 for循环执行
    ans = []
    path = [] # 全局变量，所有后面要用copy ()
    n = len(nums)
    def dfs(i):
        ans.append(path.copy())  #  每执行一次dfs，都会得到一个结果 注意注意，这里没有return
        for j in range(i, n): # 这里可以避免子集重复
            path.append(nums[j])
            dfs(j + 1)
            path.pop()
    dfs(0)
    return ans

def subsets2(nums):  # 不同的写法 从输入的角度（选还是不选）进行回溯 时间复杂度O(n*2^n)
    ans = []
    path = []
    n = len(nums)
    def dfs(i):
        if i == n:
            ans.append(path.copy())
            return
        dfs(i + 1)  # 两种情况（选择当前元素进行回溯，或不选择）

        path.append(nums[i])
        dfs(i + 1)
        path.pop()
    dfs(0)
    return ans
# print(subsets2([1,2,3]))

# 90. 子集 II (含有重复元素)
# 输入：nums = [1,2,2]
# 输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]
#如果直接套用子集代码，结果是： [[],[1],[1,2],[1,2,2],[1,2],[2],[2,2],[2]]
def subsetsWithDup(nums):
    nums.sort()
    ans = []
    path = []
    n = len(nums)
    def dfs(i):
        ans.append(path.copy())   # 这里没有 return
        for j in range(i, n):
            if j>i and nums[j]==nums[j-1]: # 上一次如果递归了2，这次的值还是2的话就跳过
                continue
            path.append(nums[j])
            dfs(j + 1)
            path.pop()
    dfs(0)
    return ans

# 131. 分割回文串  （更适合采用枚举选哪个的方法求解，而括号生成那题更适合选或不选的方法）
# 输入：s = "aab"
# 输出：[["a","a","b"],["aa","b"]]
# 思路是枚举逗号的位置，等价于枚举回文子串最后一个字符的位置，即s[i...j]
def partition( s: str):
    # 如果第一次选择的是"a",第二次是"ab"，发现不是回文串，所以ab就不会继续向下递归，到不了i==n的判断，因此不用担心ans添加了错误结果
    ans=[]
    path=[]
    n=len(s)
    def dfs(i): # 每一次dfs就是选择一个子串
        if i==n:
            ans.append(path.copy())
            return
        for j in range(i,n):
            # 注意，不能写成for j in range(i+1,n) t=s[i:j]，因为最大j=n-1时，是s[i:j]=s[i:n-1],娶不到最后一个值
            t=s[i:j+1] # 为什么是j+1？如果写成了j，j==i时，t为空；j==n-1时，也取不到最后一个值。也就是说必须是j+1
            if t==t[::-1]:
                path.append(t)
                dfs(j+1)
                path.pop()
    dfs(0)
    return ans
def partition2( s: str):
    ans=[]
    path=[]
    n=len(s)
    def dfs(i,start):
        if i==n:
            ans.append(path.copy())
            return
        # 不选 i 和 i+1 之间的逗号（i=n-1 时一定要选）
        # 确保在处理到字符串末尾时强制分割，防止未处理的子串导致错误结果。若去掉该条件，递归会在末尾生成空路径，导致答案错误。
        if i<n-1:
            dfs(i+1,start) # 跳过 i
        t = s[start:i + 1]
        if t == t[::-1]:
            path.append(t)
            dfs(i + 1, i + 1)
            path.pop()
    dfs(0,0)
    return ans
# inp_=input().strip()
# print(partition2(inp_))

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


# 79. 单词搜索
# 给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。同一个单元格内的字母不允许被重复使用。
# 总结：自己写忽略了三个问题：一是什么时候返回True没写明白；二是递归入口是由mn种情况；三是剪枝的判断board[i][j]!=word[k]时就可直接返回False了
# 时间复杂度:O(mn3^k)，其中 m和n分别为grid 的行数和列数，k是word 的长度。
# 除了递归入口，其余递归至多有3个分支(因为至少有一个方向是之前走过的)，所以每次递归(回溯)的时间复杂度为O(3^k)，
# 一共回溯O(mn)次，所以时间复杂度为O(mn3^k)。
def exist( board, word: str) -> bool:
    lens=len(word)
    m,n=len(board), len(board[0])
    def dfs(i,j,k):
        if i<0 or i>=m or j<0 or j>=n or board[i][j]!=word[k]:
            return False
        if  k==lens-1:
            return True
        board[i][j]=1 # 能到这一行，说明board[i][j]==word[k] 标记访问过。不能删掉，否则可能会走回头路
        ans=dfs(i+1,j,k+1) or dfs(i-1,j,k+1) or dfs(i,j+1,k+1) or  dfs(i,j-1,k+1)
        board[i][j]=word[k] # 恢复现场
        return ans
    ans=False
    for i in range(m):
        for j in range(n):
            ans=ans or dfs(i,j,0)
    return ans

def exist2(board, word: str) -> bool:
    m, n = len(board), len(board[0])
    def dfs(i: int, j: int, k: int) -> bool:
        if board[i][j] != word[k]:  # 匹配失败
            return False
        if k == len(word) - 1:  # 匹配成功！
            return True
        board[i][j] = ''  # 标记访问过
        for x, y in (i, j - 1), (i, j + 1), (i - 1, j), (i + 1, j):  # 相邻格子
            if 0 <= x < m and 0 <= y < n and dfs(x, y, k + 1):
                return True  # 搜到了！
        board[i][j] = word[k]  # 恢复现场，其他情况可能再次搜到
        return False  # 没搜到
    return any(dfs(i, j, 0) for i in range(m) for j in range(n))
# --------------------------(组合回溯问题）--------------------------------------------
# 77. 组合
# 给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
# 输入：n = 4, k = 2
# 输出：[ [2,4],
#   [3,4],
#   [2,3],
#   [1,2],
#   [1,3],
#   [1,4],]
# 时间复杂度：O(Cnk * k )
# 和组合回溯的区别：{1,2}和{2,1}是同一种组合，但是排列型回溯问题中，{1,2}和{2,1}是不同的排列。
def combine2(n: int, k: int): # 从大到小选  时间复杂度：叶子节点有Cnk，路径长度为k，所以时间复杂度是O(k*Cnk)
    ans = []
    path=[]
    def dfs(i):
        d=k-len(path)  # 还需要选择的个数
        # 下面是剪枝，剪掉不用的分支，减少递归次数，提高效率
        if i<d:
            return
        if len(path)==k:
            ans.append(path.copy())
            return
        for j in range(i,0,-1):  # 从大到小选，避免重复，选的是1-n的数，不是下标
            path.append(j)
            dfs(j-1)
            path.pop()
    dfs(n)
    return ans
def combine(n: int, k: int):
    ans = []
    path=[]
    def dfs(i):
        d=k-len(path)  # 还需要选择的个数
        # 下面是剪枝，剪掉不用的分支，减少递归次数，提高效率
        if n-i+1<d: # n-i+1表示当前还可选的个数，比如[1,2,3| 4,5], i指向4，当前只能选择4,5 也就是5-4+1=2，若果2<d，剪掉
            return
        if len(path)==k:
            ans.append(path.copy())
            return
        for j in range(i,n+1):  # 从小到大选，避免重复
            path.append(j)
            dfs(j+1)
            path.pop()
    dfs(1)
    return ans
def combine3(n: int, k: int):  # 使用选和不选的思路解决
    ans = []
    path = []
    def dfs(i):
        d = k - len(path)
        if d == 0:
            ans.append(path[:])
            return
        if i > d:  # 很关键啊
            dfs(i - 1)
        path.append(i)
        dfs(i - 1)
        path.pop()
    dfs(n)
    return ans
# n,k=map(int,input().strip().split())
# print(combine2(n,k))

# 39. 组合总和 只有这个是hot100的
# 给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，
# 找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合(可以 无限制重复被选取 ) ，并以列表形式返回。你可以按 任意顺序 返回这些组合。
# [2,2,3]和[2,3,2]属于一种组合
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
# print(combinationSum([1,2,3], 4))
# 按照下一题的写法求解（效率还没上一题优化好）
def combinationSum3( candidates, target: int):
    candidates.sort(reverse=True)
    ans = []
    path = []
    n = len(candidates)
    def dfs(start):
        s = sum(path)
        if s == target:
            ans.append(path[:])
            return
        if s > target: return
        if target - s < candidates[start]: return
        for j in range(start, -1, -1):
            path.append(candidates[j])
            dfs(j)
            path.pop()
    dfs(n - 1)
    return ans
def combinationSum2( candidates, target: int):
    ans = []
    path = []
    n = len(candidates)
    def dfs(i, s):
        if s == target:
            ans.append(path[:])
            return
        if s > target or i == n: return
        dfs(i + 1, s)
        # 选择当前的数
        path.append(candidates[i])
        dfs(i, s + candidates[i])
        path.pop()
    dfs(0, 0)
    return ans
# ---------------直接采用DFS（递归）的解法-------------
# 递归特点：
# 每次调用 dfs()，都在探索一种可能的路径。
# 每一层函数调用就是在尝试“添加一个数”，再交给下一层决定。
# 有明确的终止条件（if total == target）。
def combinationSum(candidates, target):
    def dfs(start, path, total):
        if total == target:
            res.append(path)
            return
        if total > target:
            return
        for i in range(start, len(candidates)):
            dfs(i, path + [candidates[i]], total + candidates[i])

    res = []
    dfs(0, [], 0)
    return res
# ---------------采用回溯的解法-------------
# 回溯的关键点：
# 使用了 path.pop() 把当前的选择撤回，这是“回溯”的核心操作。
# 如果你不 pop，你在多个递归之间就会带着错误的状态。
# 回溯 = DFS + 状态回退。
def combinationSum(candidates, target):
    def backtrack(start, path, total):
        if total == target:
            res.append(path[:])  # 不能写 res.append(path)
            return
        if total > target:
            return
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            backtrack(i, path, total + candidates[i])  # 注意是 i，不是 i+1
            path.pop()  # 回溯

    res = []
    backtrack(0, [], 0)
    return res
# ---------------采用动态规划的解法-------------
# DP思路：
# 把所有和为 i 的组合都保存到 dp[i]。
# dp[t] += dp[t-c] + [c] 表示可以从前面的状态转移过来。
# 典型的完全背包问题，是一种“记忆型”的方法。
def combinationSum(candidates, target):
    dp = [[] for _ in range(target + 1)]
    dp[0] = [[]]

    for c in candidates:
        for t in range(c, target + 1):
            for comb in dp[t - c]:
                dp[t].append(comb + [c])

    return dp[target]


# 40. 组合总和 II （含重复元素）
# candidates 中的每个数字在每个组合中只能使用 一次 。
# 输入: candidates = [10,1,2,7,6,1,5], target = 8,
# 输出: [[1,1,6],[1,2,5],[1,7],[2,6]]
# 直接套用的答案为：[[5,1,2],[5,2,1],[1,6,1],[1,7],[6,2],[7,1]]，产生重复
def combinationSum2_1(candidates, target: int) :
    candidates.sort(reverse=True)
    ans = []
    path=[]
    n=len(candidates)
    def dfs(i):
        ss = sum(path)
        if ss == target:
            ans.append(path.copy())
            return
        elif ss > target:
            return
        # elif target - ss < candidates[i]:
        elif ss + candidates[i] > target:
            return
        for j in range(i,-1,-1):
            # 如果不加这个判断，DFS 在同一层循环里会把前一个 1 和后一个 1 都作为起点再递归一次，最后得到完全一样的组合，导致结果里有重复解。
            # 这句判断的含义是：
            # 在同一层循环中，如果某个数和它右边的数相等（candidates[j] == candidates[j+1]），并且 j < i 说明它不是本层第一个被选择的数，就跳过它。
            # 这样保证了相同数值在同一层递归里只会被用一次。
            # candidates = [2,1,1], target=3
            # 如果不加去重条件，可能会得到两个 [1,2]。
            # 加了之后，就只会保留一个 [1,2]。
            if j<i and candidates[j]==candidates[j+1]:
                continue
            path.append(candidates[j])
            dfs(j-1)
            path.pop()
    dfs(n-1)
    return ans
# print(combinationSum2_1(candidates = [10,1,2,7,6,1,5], target = 8))
# 另一种写法
def combinationSum2_2(candidates, target: int):
    candidates.sort()
    ans = []
    path = []
    n = len(candidates)
    def dfs(i, ss):
        if ss == target:
            ans.append(path.copy())
            return
        if ss > target or i >= n:
            return
        for j in range(i, n):  # 修正1：循环到n，而不是写成n-1
            if j > i and candidates[j] == candidates[j - 1]: # 修正2：比较j和j-1，而不是写成j和j+1
                continue
            path.append(candidates[j])
            dfs(j + 1, ss + candidates[j])
            path.pop()
    dfs(0, 0)
    return ans
# print(combinationSum2_1([2,5,2,1,2],5))

# 216. 组合总和 III
# 找出所有相加之和为 n 的 k 个数的组合，且满足下列条件：
# 只使用数字1到9；每个数字 最多使用一次
# 输入: k = 3, n = 7
# 输出: [[1,2,4]]
def combinationSum3_1(k,n):
    ans=[]
    path=[]
    def dfs(i,t):
        if t<0:
            return
        if t==0 and len(path)==k:
            ans.append(path.copy())
            return
        for j in range(i,0,-1):
            path.append(j)
            dfs(j-1,t-j)
            path.pop()
    dfs(9,n)
    return ans
def combinationSum3_2(k,n):  # 选或者不选的思路
    ans = []
    path = []
    def dfs(i, t):
        d = k - len(path)
        if t < 0 or t > (i * 2 - d + 1) * d // 2: return  # 必须加上后面这个剪枝条件，不然会maxrecursion error
        if t == 0 and d == 0:
            ans.append(path[:])
            return
        if i > d: dfs(i - 1, t)
        path.append(i)
        dfs(i - 1, t - i)
        path.pop()

    dfs(9, n)
    return ans
# k,n=map(int,input().strip().split())
# print(combinationSum3(k,n))

# 22. 括号生成  时间复杂度O(n*C(2n,n))
def generateParenthesis(n: int):
    m=2*n
    ans=[]
    path=['']*m
    def dfs(i,cont_left):
        if i==m:
            ans.append(''.join(path))
            return
        if cont_left<n:
            path[i]='('
            dfs(i+1,cont_left+1)
        if i-cont_left<cont_left: #右括号的个数小于左括号的个数
            path[i]=')'
            dfs(i+1,cont_left)
    dfs(0,0)
    return ans
def generateParenthesis2(n: int):
    m=2*n
    ans=[]
    path=[]
    def dfs(i,cont_left):
        if i==m:
            ans.append(''.join(path))
            return
        if cont_left<n:
            path.append('(')
            dfs(i+1,cont_left+1)
            path.pop()  #注意，这里也要pop恢复现场
        if i-cont_left<cont_left: #右括号的个数小于左括号的个数
            path.append=(')')
            dfs(i+1,cont_left)
            path.pop()
    dfs(0,0)
    return ans
# n=int(input().strip())
# print(generateParenthesis(n))

# --------------------------------------排列型回溯问题 （典型：N皇后）------------------------------------------
# 和组合回溯的区别：{1,2}和{2,1}是同一种组合，但是排列型回溯问题中，{1,2}和{2,1}是不同的排列。
# 46. 全排列
def permute(nums):
    # 时间复杂度O(n*n!)
    ans=[]
    n=len(nums)
    path=[0]*n
    def dfs(i,s):
        if i==n:
            ans.append(path.copy())
        for j in s:
            path[i]=j
            dfs(i+1,s-{j}) # s-{j}表示删除当前元素，避免重复 s.copy().remove(j)
    dfs(0,set(nums))
    return ans
# print(permute([1,2,3]))

# 改成长度为k的全排列
def permute_k(nums, k):
    ans = []
    path = []
    def dfs(s):
        if len(path) == k:  # 达到长度 k
            ans.append(path.copy())
            return
        for j in s:
            path.append(j)
            dfs(s - {j})  # 去掉已用元素
            path.pop()
    dfs(set(nums))
    return ans
# print(permute_k([1,2,3],2))

def permute2(nums):
    ans=[]
    n=len(nums)
    path=[0]*n
    on_path=[False]*n
    def dfs(i):
        if i==n:
            ans.append(path[:])
            return
        for j in range(n):
            if not on_path[j]:
                path[i] = nums[j]
                on_path[j] = True
                dfs(i + 1)
                on_path[j] = False
    dfs(0)
    return ans
# nums=list(map(int,input().strip().split()))
# print(permute(nums))

# 47. 全排列 II
# 输入：nums = [1,1,2]
# 输出： [[1,1,2],[1,2,1],[2,1,1]]
def permuteUnique(nums):
    cnt=Counter(nums)
    ans=[]
    n=len(nums)
    s=set(nums)
    path=[]
    def dfs(i):
        if i==n:
            ans.append(path.copy())
            return
        for x in s:
            if cnt[x]<=0:
                continue
            cnt[x]-=1
            path.append(x)
            dfs(i+1)
            path.pop()
            cnt[x]+=1
    dfs(0)
    return ans

# 51. N 皇后
def solveNQueens( n: int):
    ans = []
    col = [0] * n  # 每一个元素记录皇后在第i行，第col[i]列    # 第i个元素的值 == 第i行的皇后所在的列
    def valid(row,c):
        for former_r in range(row):
            former_col=col[former_r]
            if row+c==former_col+former_r or row-c==former_r-former_col:
                return False
        return True
    def dfs(row,s):
        if row==n:
            tmp=[]
            for c in col:
                tmp.append('.'*c+'Q'+'.'*(n-c-1))
            ans.append(tmp)
            return
        for c in s:
            if valid(row,c):
                col[row]=c
                dfs(row+1,s-{c})
    dfs(0,set(range(n)))
    return ans
def solveNQueens2(n: int):
    ans = []
    col = [0] * n  # 每一个元素记录皇后在第i行，第col[i]列
    m=2*n-1  # 从0-(2*n-2) 共2n-1个数。因为row+c最大值就是2n-2 最小值是0
    dig1=[False]*m
    dig2=[False]*m
    on_Path=[False]*n
    def dfs(row):
        if row==n:
            tmp=[]
            for c in col:
                tmp.append('.'*c+'Q'+'.'*(n-c-1))
            ans.append(tmp)
            return
        for c in range(n):
            if not on_Path[c] and not dig1[row+c] and not dig2[row-c]:
                col[row]=c
                on_Path[c]=dig1[row+c]=dig2[row-c]=True
                dfs(row+1)
                on_Path[c]=dig1[row+c]=dig2[row-c]=False
    dfs(0)
    return ans
# n=int(input().strip())
# print(solveNQueens2(n))

