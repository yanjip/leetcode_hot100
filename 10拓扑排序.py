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
    indegr = [0] * numCourses # 入度表
    adjacency = [[] for _ in range(numCourses)]  # 其实就是建立出度表
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
                queue.append(cur)
    # 若课程安排图中存在环，一定有节点的入度始终不为 0。这时候queue就append不了所有节点，所以numCourses就一直不会减为0，所以最后会返回False
    return not numCourses

