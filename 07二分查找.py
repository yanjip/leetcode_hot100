# time: 2025/3/5 9:58
# author: YanJP

# 左右均为闭区间写法
def low_bound(nums, target): # 找到第一个大于等于target的索引
    left=0
    right=len(nums)-1
    while left<=right:
        mid=(left+right)//2
        if nums[mid]<target:
            left=mid+1
        else:
            right=mid-1
    return left  # 第一个大于等于target的索引 （循环不变量，循环到最后，left-1始终是小于target的，right+1始终是大于等于target的）

# 上述为≥tgt，其他形式可以转换成该形式： 例子[5, 7, 7, 8, 8, 10], tg=8 （递增序列）
#   求>tg 的第一个数:   ≥(tg+1)  等价于先找到第一个≥(8+1)，即index=5的10，即是所求的数。
#   求<tg的最后一个数:  (≥tg)-1  等价于先找到index=3(从0开始算的)的tg 8，然后再取它右边的数。即找到index=2的数7，即是所求的数。
#   求≤tg的最后一个数:  (>tg)-1  等价于先找到第一个>tg，即index=5的10，再取它左边的数。即找到index=4的数8，即是所求的数。

# 34. 在排序数组中查找元素的第一个和最后一个位置
# 输入：nums = [5,7,7,8,8,10], target = 8
# 输出：[3,4]
def searchRange(nums, target):
    start=low_bound(nums,target)
    if start==len(nums) or nums[start]!=target:
        return [-1,-1]
    end=low_bound(nums,target+1)-1
    return [start,end]

# 给定一个非升序的有序数组和一个target，返回数组中等于target的个数。
# 腾讯二面
def count_target(nums, target):
    if not nums:  # 空数组处理
        return 0
    def find_left(nums, target):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] > target:
                left = mid + 1
            elif nums[mid] < target:
                right = mid - 1
            else:  # nums[mid] == target
                right = mid - 1  # 继续向左找
        return left if nums[left] == target else -1
    def find_right(nums, target):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] > target:
                left = mid + 1
            elif nums[mid] < target:
                right = mid - 1
            else:  # nums[mid] == target
                left = mid + 1  # 继续向右找
        return right if nums[right] == target else -1

    left_idx = find_left(nums, target)
    if left_idx == -1:  # 没找到target
        return 0
    right_idx = find_right(nums, target)
    return right_idx - left_idx + 1

# 测试
# nums = [5, 5, 5, 4, 3, 2, 2, 2, 1]
# target = 1
# result = count_target(nums, target)
# print(f"数组 {nums} 中等于 {target} 的个数是: {result}")  # 输出: 3

# 35. 搜索插入位置
def searchInsert( nums: list[int], target: int) -> int:
    def low_bound(nums, target): # 找到第一个大于等于target的索引
        left=0
        right=len(nums)-1
        while left<=right:
            mid=(left+right)//2
            if nums[left]<target:
                left=mid+1
            else:
                right=mid-1
        return left

    return low_bound(nums,target-1)+1
print(searchInsert([1,3,5,6],2))

# 153. 寻找旋转排序数组中的最小值（设计一个时间复杂度为 O(log n) 的算法
# 输入：nums = [3,4,5,1,2]
# 输出：1
# 解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。
def findMin(nums: list[int]) -> int:
    # 只需要比较 x 和 nums[n−1] 的大小关系，就间接地知道了 x 和数组最小值的位置关系
    left = 0
    n = len(nums)
    right = n - 2
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] < nums[-1]:
            right = mid - 1
        else:
            left = mid + 1
    return nums[left]

# 百度二面题 求float x的立方根
def main(x):
    # 处理0的立方根
    if x == 0:
        return 0.0
    # 确定符号并取绝对值
    sign = 1 if x > 0 else -1
    x_abs = abs(x)
    # 统一设置初始区间：确保low < high
    # 当 |x| ≥ 1 时：立方根范围在 [0, |x|] 之间
    # 当 0 < |x| < 1 时：立方根范围在 [0,1] 之间
    low = 0.0
    high = max(1.0, x_abs)  # 处理0 < |x| < 1的情况
    # 二分查找
    while high - low > 1e-8:
        mid = (low + high) / 2
        mid_cubed = mid ** 3
        if mid_cubed < x_abs:
            low = mid
        else:
            high = mid
    # 最终结果在low和high之间
    result = (low + high) / 2 * sign
    return round(result, 3)
# ans = main(64.1)
# print(ans)