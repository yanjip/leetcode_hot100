# time: 2025/3/2 12:29
# author: YanJP

# 11. 盛最多水的容器
def maxArea(height: list[int]) -> int:
    i, j, res = 0, len(height) - 1, 0
    while i < j:
        res = max(min(height[i], height[j]) * (j - i), res)
        # 核心思想：在固定两端后，每轮向内移动短板，所有消去的状态都不会导致面积最大值丢失
        if height[i] < height[j]:
            i += 1
        else:
            j -= 1
    return res

#42. 接雨水
def trap(height: list[int]) -> int:
    n = len(height)
    pre = [0] * n
    suf = [0] * n
    pre[0] = height[0]
    suf[-1] = height[-1]
    for i in range(1, n):
        pre[i] = max(pre[i - 1], height[i])
    for i in range(n - 2, -1, -1):
        suf[i] = max(suf[i + 1], height[i])
    ans = 0
    for h, p, s in zip(height, pre, suf):
        ans += min(s, p) - h
    return ans

def trap2(height: list[int]) -> int:
    # 优化空间复杂度O(1)
    n=len(height)
    left=0
    right=n-1
    ans=0
    pre_max=0
    suf_max=0
    while left< right:
        pre_max=max(pre_max,height[left])
        suf_max=max(suf_max,height[right])
        if pre_max<suf_max:
            ans+=pre_max-height[left]
            left+=1
        else:
            ans+=suf_max-height[right]
            right-=1
    return ans


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