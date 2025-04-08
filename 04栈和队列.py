# time: 2025/2/27 9:59
# author: YanJP
from collections import deque
# 单调栈：要计算的内容涉及到上一个或者下一个更大或者更小的元素

# 739. 每日温度
# 每个元素入栈至多1次，出栈也是至多一次，因此时间复杂度O(n)
# 返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。
# 输入: temperatures = [30,40,50,60]
# 输出: [1,1,1,0]
def dailyTemperatures(temperatures: list[int]) -> list[int]:
    st=[] # 单调栈  存放下标
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
    st=[] # 单调栈  存放下标
    ans=[0]*len(temperatures)
    for i, t in enumerate(temperatures):
        while st and t>temperatures[st[-1]]:
            j=st.pop()
            ans[j]=i-j  # 这里是ans[j]不要写错了
        st.append(i)
    return ans
# temperatures=list(map(int,input().strip().split(',')))
# print(dailyTemperatures_forword(temperatures))



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
height = [0,1,0,2,1,0,1,3,2,1,2,1]
print(trap(height))

# 239. 滑动窗口最大值 （单调队列）
def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
    ans=[]
    q=deque() # 存在的是下标
    for i, x in enumerate(nums):
        while q and nums[q[-1]]<=x:
            q.pop()
        q.append(i)
        if i-q[0]+1>k:
            q.popleft()
        if i>=k-1:
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