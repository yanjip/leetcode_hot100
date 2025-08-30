def count(nums, target):
    if not nums:
        return 0
        
    # Find first occurrence
    left, right = 0, len(nums)-1
    first = -1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            if mid == 0 or nums[mid-1] != target:
                first = mid
                break
            else:
                right = mid - 1
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    # Find last occurrence
    left, right = 0, len(nums)-1
    last = -1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            if mid == len(nums)-1 or nums[mid+1] != target:
                last = mid
                break
            else:
                left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    
    return last - first + 1 if first != -1 and last != -1 else 0
nums=[1, 3,5]
print(count(nums,2))
