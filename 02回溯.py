# time: 2025/2/8 10:54
# author: YanJP
from collections import Counter


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
# 时间复杂度取决于生成的所有可能的子集的数量和每次递归的时间复杂度。时间复杂度O(n*2^n)
def subsets(nums):  # 从结果的角度进行回溯 for循环执行
    ans = []
    path = [] # 全局变量，所有后面要用copy ()
    n = len(nums)
    def dfs(i):
        ans.append(path.copy())  #  每执行一次dfs，都会得到一个结果
        for j in range(i, n): # 这里可以避免子集重复
            path.append(nums[j])
            dfs(j + 1)
            path.pop()
    dfs(0)
    return ans

def subsets2(self, nums):  # 不同的写法 从输入的角度（选还是不选）进行回溯 时间复杂度O(n*2^n)
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

# 131. 分割回文串  （更适合采用枚举选哪个的方法求解，而括号生成那题更适合选或不选的方法）
# 输入：s = "aab"
# 输出：[["a","a","b"],["aa","b"]]
def partition( s: str):
    ans=[]
    path=[]
    n=len(s)
    def dfs(i):
        if i==n:
            ans.append(path.copy())
            return
        for j in range(i,n):
            # 注意，不能写成for j in range(i+1,n) t=s[i:j]，因为最大j=n-1时，是s[i:j]=s[i:n-1],娶不到最后一个值
            t=s[i:j+1]
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
            dfs(i+1,start)
        t = s[start:i + 1]
        if t == t[::-1]:
            path.append(t)
            dfs(i + 1, i + 1)
            path.pop()
    dfs(0,0)
    return ans
# inp_=input().strip()
# print(partition2(inp_))

# 90. 子集 II
# 输入：nums = [1,2,2]
# 输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]
#如果直接套用子集代码，结果是： [[],[1],[1,2],[1,2,2],[1,2],[2],[2,2],[2]]
def subsetsWithDup(nums: list[int]):
    nums.sort()
    ans = []
    path = []
    n = len(nums)
    def dfs(i):
        ans.append(path.copy())
        for j in range(i, n):
            if j>i and nums[j]==nums[j-1]:
                continue
            path.append(nums[j])
            dfs(j + 1)
            path.pop()
    dfs(0)
    return ans

# 79. 单词搜索
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
        board[i][j]=1
        ans=dfs(i+1,j,k+1) or dfs(i-1,j,k+1) or dfs(i,j+1,k+1) or  dfs(i,j-1,k+1)
        board[i][j]=word[k]
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
        board[i][j] = word[k]  # 恢复现场
        return False  # 没搜到
    return any(dfs(i, j, 0) for i in range(m) for j in range(n))
# --------------------------(组合回溯问题）--------------------------------------------
# 77. 组合
# 时间复杂度：O(Cnk * k )
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
        for j in range(i,0,-1):  # 从小到大选，避免重复
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


# 39. 组合总和
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

# 40. 组合总和 II （含重复元素）
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
        elif target - ss < candidates[i]:
            return
        for j in range(i,-1,-1):
            if j<i and candidates[j]==candidates[j+1]:
                continue
            path.append(candidates[j])
            dfs(j-1)
            path.pop()
    dfs(n-1)
    return ans
# print(combinationSum2_1([2,5,2,1,2],5))

# 216. 组合总和 III
# 找出所有相加之和为 n 的 k 个数的组合，且满足下列条件：
# 只使用数字1到9；每个数字 最多使用一次
# 输入: k = 3, n = 7
# 输出: [[1,2,4]]
def combinationSum3(k,n):
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
def permute(nums: list[int]):
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
def permute2(nums: list[int]):
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

