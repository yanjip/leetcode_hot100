# time: 2025/3/2 12:29
# author: YanJP

# 2824. 统计和小于目标的下标对数目
def countPairs(nums: list[int], target: int) -> int:
    nums.sort()
    ans = 0
    left = 0
    right = len(nums) - 1
    while left < right:
        if nums[left] + nums[right] < target:
            ans += (right - left)
            left+=1
        elif nums[left] + nums[right] >= target:
            right -= 1
    return ans
nums=[-6,2,5,-2,-7,-1,3]
print(countPairs(nums,-2))

# 3. 无重复字符的最长子串 (腾讯手撕题）
def length_of_longest_substring(s: str) -> int:
    # 使用集合来存储当前窗口中的字符
    char_set = set()
    left = 0  # 左指针
    max_length = 0  # 记录最长子串的长度

    for right in range(len(s)):
        # 如果当前字符已经在集合中，移动左指针直到移除重复字符
        while s[right] in char_set:  # 这里一定是while ，而不是if。比如set为ab时，下一个 s[right]=b，那么while循环会一直执行，直到把前面的set清空，添加当前的b
            char_set.remove(s[left])
            left += 1

        # 将当前字符加入集合
        char_set.add(s[right])

        # 更新最大长度
        max_length = max(max_length, right - left + 1)

    return max_length
# 测试示例
print(length_of_longest_substring("abcabcbb"))  # 输出: 3