# time: 2025/2/27 9:59
# author: YanJP
from collections import deque


# 739. 每日温度
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
# height = [0,1,0,2,1,0,1,3,2,1,2,1]
# print(trap(height))

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