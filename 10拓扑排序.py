# time: 2025/4/21 11:05
# author: YanJP
from collections import deque


# 拓扑排序是对有向无环图（DAG）中所有顶点进行线性排序，使得对于图中任意一条有向边 (u, v)，顶点 u 在序列中总是位于顶点 v 之前。

# 207. 课程表
# 给你一个有向图，判断图中是否有环。
# 输入：numCourses = 2, prerequisites = [[1,0]]
# 输出：true
# 解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0 。这是可能的。
# https://www.bilibili.com/video/BV1hZc5eWEsV/ 视频讲的非常好
def canFinish(numCourses: int, prerequisites) -> bool:
    # 广度优先遍历
    indegr = [0] * numCourses # 入度表 第i个位置的入度为indegr[i]，等于0表示当前i课程没有依赖的课程
    adjacency = [[] for _ in range(numCourses)]  # 其实就是建立出度表，修完pre课程就有资格修adjacency[pre]存的课程
    queue = deque()
    for cur, pre in prerequisites:
        indegr[cur] += 1
        adjacency[pre].append(cur)
    for i in range(numCourses):
        if indegr[i] == 0: queue.append(i)

    while queue:
        pre = queue.popleft()
        numCourses -= 1
        for cur in adjacency[pre]:
            indegr[cur] -= 1
            if indegr[cur] == 0:
                queue.append(cur) # 入度为0的节点入队
    # 若课程安排图中存在环，一定有节点的入度始终不为 0，也就是说indegr始终有两个元素不为0。这时候queue就append不了所有节点，所以numCourses就一直不会减为0，所以最后会返回False
    return not numCourses

# print(canFinish(6, [[3, 0], [3, 1], [4, 1], [4, 2], [5, 3], [5, 4]]))
# print(canFinish(4, [[3, 1], [1, 3]]))

# 994. 腐烂的橘子
# 在给定的 m x n 网格 grid 中，每个单元格可以有以下三个值之一：
# 值 0 代表空单元格；
# 值 1 代表新鲜橘子；
# 值 2 代表腐烂的橘子。
# 每分钟，腐烂的橘子 周围 4 个方向上相邻 的新鲜橘子都会腐烂。
# 返回 直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1 。
def orangesRotting(grid) -> int:
    q = []
    ans = 0
    m, n = len(grid), len(grid[0])
    fresh = 0  # 新鲜橘子的个数
    for i, row in enumerate(grid):
        for j, x in enumerate(row):
            if x == 1:
                fresh += 1
            elif x == 2:
                q.append((i, j))
    if fresh == 0: return 0
    while q and fresh:
        ans += 1
        q1 = q
        q = []
        for x, y in q1:
            for i, j in (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1):
                if 0 <= i < m and 0 <= j < n and grid[i][j] == 1:
                    fresh -= 1
                    grid[i][j] = 2
                    q.append((i, j))
    return -1 if fresh else ans

# 200. 岛屿数量
# 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
def numIslands(grid: list[list[str]]) -> int:
    m,n=len(grid), len(grid[0])
    def dfs(i,j):
        if i<0 or i>=m or j<0 or j>=n or grid[i][j]!='1':
            return
        grid[i][j]='0'
        dfs(i+1,j)
        dfs(i-1,j)
        dfs(i,j+1)
        dfs(i,j-1)
    ans=0
    for i,x in enumerate(grid):
        for j, y in enumerate(x):
            if y=='1':
                dfs(i,j)
                ans+=1
    return ans