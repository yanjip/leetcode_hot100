# time: 2025/2/27 9:59
# author: YanJP
from collections import deque
# 一、栈（Stack）的典型应用
# 1. 递归实现：每次递归调用压栈，返回时弹栈（如阶乘、斐波那契数列）。
# 2. 括号匹配：用栈检查括号是否成对（如 {[()]}）。
# 3. 处理运算符优先级（如 3 + 4 * 2 → 3 4 2 * +）。
# 4. 前进/后退功能：用双栈实现（一个栈保存后退页面，另一个保存前进页面）。
# 5. 文本编辑器撤销（Undo）：栈存储操作历史。

# 二、队列（Queue）的典型应用
# 1. 任务调度与消息传递：CPU 任务队列：操作系统按FIFO调度进程（如先到先服务算法）。
# 2. 广度优先搜索（BFS）
# 3. 缓冲区管理：打印队列，多个打印任务按提交顺序执行。

# 三、堆的典型应用
# 优先队列：快速获取或删除最高/低优先级元素（如任务调度）。（解决许多与优先级相关的问题）
# 堆排序：时间复杂度 O(nlogn)，原地排序但不稳定。
# Top K 问题：用最小堆维护当前最大的 K 个元素（或最大堆维护最小的 K 个）。



# 单调栈：要计算的内容涉及到上一个或者下一个更大或者更小的元素

# 739. 每日温度
# 每个元素入栈至多1次，出栈也是至多一次，因此时间复杂度O(n)
# 返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。
# 输入: temperatures = [30,40,50,60]
# 输出: [1,1,1,0]
def dailyTemperatures(temperatures: list[int]) -> list[int]:
    st=[] # 单调栈  存放下标 从下往上为递减的栈
    ans=[0]*len(temperatures)
    for i in range(len(temperatures)-1,-1,-1):
        t=temperatures[i]
        while st and t>=temperatures[st[-1]]:  # 先pop更新单调栈，再push当前值，即下面的append
            st.pop()
        if st:
            ans[i]=st[-1]-i
        st.append(i)
    return ans
def dailyTemperatures_forword(temperatures: list[int]) -> list[int]:
    st=[] # 单调栈  存放下标 从下往上为递减的栈
    ans=[0]*len(temperatures)
    for i, t in enumerate(temperatures):
        while st and t>temperatures[st[-1]]:
            j=st.pop()
            ans[j]=i-j  # 这里是ans[j]不要写错了
        st.append(i)
    return ans
# temperatures=list(map(int,input().strip().split(',')))
# print(dailyTemperatures_forword(temperatures))

# 20. 有效的括号
# 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
def isValid( s: str) -> bool:
    if len(s)%2: return False
    st=[]
    mp={')':'(', '}':'{', ']':'['}
    for ss in s:
        if ss not in mp:
            st.append(ss)  # 此时存的是左括号({[
        elif not st or st.pop() != mp[ss]: # 如果ss是右括号会执行，以防的是括号不匹配的情况(]
            return False
    return not st # 以防的是全是左括号的情况({

# 入栈对应的右括号，且不使用map的写法
def isValid2( s: str) -> bool:
    if len(s) % 2:  # s 长度必须是偶数
        return False
    st = []
    for c in s:
        if c == '(':
            st.append(')')  # 入栈对应的右括号
        elif c == '[':
            st.append(']')
        elif c == '{':
            st.append('}')
        elif not st or st.pop() != c:  # c 是右括号
            return False  # 没有左括号，或者左括号类型不对
    return not st  # 所有左括号必须匹配完毕


# 42. 接雨水
def trap(height: list[int]) -> int:
    ans=0
    st=[]
    for i, h in enumerate(height):
        while st and h>=height[st[-1]]:
            botton_height=height[st.pop()]
            if len(st)==0:
                break
            left_h=height[st[-1]]  # 刚才栈顶的下一个元素，作为左边高度
            ans+= (min(left_h,h)-botton_height)*(i-st[-1]-1)
        st.append(i)
    return ans
# height = [0,1,0,2,1,0,1,3,2,1,2,1]
# print(trap(height))

# 239. 滑动窗口最大值 （单调队列）
def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
    ans=[]
    q=deque() # 存在的是下标，从左往右递增的队列
    for i, x in enumerate(nums):
        while q and nums[q[-1]]<=x: # x太大了 就存x pop队列的数
            q.pop()
        q.append(i)
        if i-q[0]+1>k: # pop出最新（左边）的元素
            q.popleft()
        if i>=k-1: # 从满足窗口大小后就开始存答案
            ans.append(nums[q[0]])
    return ans

# 394. 字符串解码
# 输入：s = "3[a]2[bc]"
# 输出："aaabcbc"
def decodeString( s):
    stack=[]
    res=""
    num=0
    for c in s:
        if c.isdigit():
            num=num*10+int(c)
        elif c=='[':
            stack.append((res, num))
            res,num="", 0
        elif c==']':
            pre_res, pre_num=stack.pop()
            res=pre_res+res*pre_num
        else:
            res+=c
    return res

# 找到某个特定下标 i 的左边 小于 heights[i] 的元素的 下标
def find_left_smaller_index(heights, i):
    stack = []  # 单调递增栈，保存的是下标, 例如对应的值为[2,4,5,7]

    for idx in range(i):
        while stack and heights[stack[-1]] >= heights[i]:
            stack.pop()
    if stack:
        return stack[-1]  # 返回左边第一个比 heights[i] 小的下标
    else:
        return -1  # 没有符合条件的元素

# 给定一个数组heights，找到小于height[i]的左边最近比它小的高度，右边比它小的最近高度的 下标。
def find_left_small(heights: list[int]):
    n = len(heights)
    left = [-1] * n  # 维护的是一个从下往上递增的子序列，即找到比h[i]小的下标
    st = []
    for i, h in enumerate(heights):
        while st and heights[st[-1]] >= h:
            st.pop()
        if st:
            left[i] = st[-1]
        st.append(i)  # 存下标
    right = [n] * n
    st.clear()
    for i in range(n - 1, -1, -1):
        h = heights[i]
        while st and h <= heights[st[-1]]:
            st.pop()
        if st:
            right[i] = st[-1]
        st.append(i)
    print(left, right)
    return left, right
# find_left_small([2,1,5,6,2,3])

# 84. 柱状图中最大的矩形
# 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
# 求在该柱状图中，能够勾勒出来的矩形的最大面积。
# 解题思路就是找到矩形的左右下标，然后计算面积求最大
def largestRectangleArea(heights) -> int:
    left, right=find_left_small(heights)
    ans=0
    for h, l, r in zip(heights, left, right):
        ans = max(ans, h * (r - l - 1))
    return ans

# -----------------------堆----------------------------------------------------------
# 1. 是一个完全二叉树：除最后一层外，其他层节点必须填满，最后一层节点从左到右排列
# 2. 性质
# 父节点：parent(i) = (i - 1) // 2
# 左孩子：left_child(i) = 2i + 1
# 右孩子：right_child(i) = 2i + 2
# 最小堆（Min-Heap）：每个节点的值 ≤ 其子节点的值（根节点是最小值）。
# 最大堆（Max-Heap）：每个节点的值 ≥ 其子节点的值（根节点是最大值）。

