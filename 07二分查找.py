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

# 33. 搜索旋转排序数组