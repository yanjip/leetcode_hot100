# time: 2025/3/5 9:58
# author: YanJP

# 左右均为闭区间写法
def low_bound(nums, target): # 找到第一个大于等于target的索引
    left=0
    right=len(nums)-1
    while left<=right:
        mid=(left+right)//2
        if nums[left]<target:
            left=mid+1
        else:
            right=mid-1
    return left  # 第一个大于等于target的索引 （循环不变量，left-1始终是小于target的，right+1始终是大于等于target的）