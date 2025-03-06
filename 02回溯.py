# time: 2025/2/8 10:54
# author: YanJP

# 78. 子集
def subsets(self, nums):
    ans = []
    path = [] # 全局变量，所有后面要用copy ()
    n = len(nums)
    def dfs(i):
        ans.append(path.copy())  #  每执行一次dfs，都会得到一个结果
        if i == n:
            return
        for j in range(i, n): # 这里可以避免子集重复
            path.append(nums[j])
            dfs(j + 1)
            path.pop()
    dfs(0)
    return ans

def subsets2(self, nums):  # 不同的写法
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

# 131. 分割回文串
def partition( s: str):
    ans=[]
    path=[]
    n=len(s)
    def dfs(i):
        if i==n:
            ans.append(path.copy())
            return
        for j in range(i,n):
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

# 77. 组合
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
def combine2(n: int, k: int): # 从大到小选
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
# n,k=map(int,input().strip().split())
# print(combine2(n,k))


# 216. 组合总和 III
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
# k,n=map(int,input().strip().split())
# print(combinationSum3(k,n))

# 22. 括号生成
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

# 46. 全排列
def permute(nums: list[int]):
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
# nums=list(map(int,input().strip().split()))
# print(permute(nums))

# 51. N 皇后
def solveNQueens( n: int):
    ans = []
    col = [0] * n  # 每一个元素记录皇后在第i行，第col[i]列
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

