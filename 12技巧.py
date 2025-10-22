# time: 2025/9/18 16:03
# author: YanJP
# 169. 多数元素
# 给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。给定的数组总是存在多数元素。
from collections import Counter

# -----------------------012技巧.py--------------------------
# 169. 多数元素
# 给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。给定的数组总是存在多数元素。
def majorityElement( nums:list[int]) -> int:
    # 投票法
    votes=0
    ans=0
    for x in nums:
        if votes==0: ans=x
        if x==ans:
            votes+=1
        else:
            votes-=1
    return ans

# 75. 颜色分类
# 给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地 对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
# 我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
def sortColors( nums: list[int]) -> None:
    # 维护两个指针，分别指向0和2
    # p0,i,p2=0,0,len(nums)-1
    # while i<=p2:
    #     if nums[i]==0:
    #         nums[i],nums[p0]=nums[p0],nums[i]
    #         p0+=1
    #         i+=1
    #     elif nums[i]==2:
    #         nums[i],nums[p2]=nums[p2],nums[i]
    #         p2-=1
    #     else:
    #         i+=1
    p0 = p1 = 0
    for i, x in enumerate(nums):
        nums[i] = 2
        if x <= 1:
            nums[p1] = 1
            p1 += 1
        if x == 0:
            nums[p0] = 0
            p0 += 1

# 31. 下一个排列 （hot100）
# 输入：nums = [1,7,3,5,4,2,1]
# 输出：[1,7,4, 1,2,3,5]
class Solution:
    def nextPermutation(self, nums:list[int]) -> None:
        n = len(nums)
        # 第一步：从右向左找到第一个小于右侧相邻数字的数 nums[i] (即i指向3,且i后面的元素一定是单调递减的： [1,7,|3|,5,4,2,1]）
        i = n - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1

        # 如果找到了，进入第二步；否则跳过第二步，反转整个数组
        if i >= 0:
            # 第二步：从右向左找到 nums[i] 右边最小的大于 nums[i] 的数 nums[j]
            j = n - 1
            while nums[j] <= nums[i]:
                j -= 1
            # 交换 nums[i] 和 nums[j]
            nums[i], nums[j] = nums[j], nums[i]     # 找到满足nums[j] > nums[i]的数，即4，交换后变成[1,7,|4|,5,|3|,2,1]

        # 第三步：反转 nums[i+1:]（如果上面跳过第二步，此时 i = -1）  翻转数组，变成[1,7,4, 1,2,3,5]
        # nums[i+1:] = nums[i+1:][::-1] 这样写也可以，但空间复杂度不是 O(1) 的
        left, right = i + 1, n - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

# -----------------------贪心--------------------------------
#763. 划分字母区间
# 把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。答案返回每段的长度
# 输入：s = "ababcbacadefegdehijhklij"
# 输出：[9,7,8]
# 解释：划分结果为 "ababcbaca"、"defegde"、"hijhklij" 。
# 例如字母 d 的区间为 [9,14]，片段要包含 d，必须包含区间 [9,14]，但区间 [9,14] 中还有其它字母 e,f,g，所以该片段也
# 必须包含这些字母对应的区间 e[10,15],f[11,11],g[13,13]，合并后得到区间 [9,15]。
def partitionLabels( s: str):
    last={c:i for i, c in enumerate(s)} # 每个字母最后出现的下标
    ans=[]
    start=end=0
    for i, c in enumerate(s):
        end=max(end, last[c]) # 更新当前区间右端点的最大值
        if end==i: # 当前区间合并完毕
            ans.append(end-start+1)
            start=i+1
    return ans # 下一个区间的左端点

# 55. 跳跃游戏
# 给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。
# 判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false 。
# 输入：nums = [3,2,1,0,4]
# 输出：false
def canJump( nums: list[int]) -> bool:
    mx=0
    for i, jump in enumerate(nums):
        if i > mx: # 无法到达 i
            return False
        mx = max(mx, i+jump)
    return True

# 45. 跳跃游戏 II
# 返回到达 nums[n - 1] 的最小跳跃次数。测试用例保证可以到达 nums[n - 1]。每个元素 nums[i] 表示从索引 i 向后跳转的最大长度。
# 输入: nums = [2,3,1,1,4]
# 输出: 2 步到达
# 问：为什么代码只需要遍历到 n−2？
# 当 i=n−2 时，如果 i<curRight，说明可以到达 n−1；
#            如果 i=curRight，我们会造桥（因为一定有一个桥可以到达 n−1，说明nums[n-2]不可能为0，肯定≥1，ans+=1就得到答案），这样也可以到达 n−1。
# 所以无论是何种情况，都只需要遍历到 n−2。或者说，n−1 已经是终点了，你总不能在终点还打算造桥吧？
def jump(nums: list[int]) -> int:
    ans = 0
    cur_right = 0  # 已建造的桥的右端点
    next_right = 0  # 下一座桥的右端点的最大值
    for i in range(len(nums) - 1):
        # 遍历的过程中，记录下一座桥的最远点
        next_right = max(next_right, i + nums[i])
        if i == cur_right:  # 无路可走，必须建桥
            cur_right = next_right
            ans += 1
    return ans

# 146. LRU 缓存
# 请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
# OrderedDict = dict + 双向链表
from collections import OrderedDict
class LRUCache(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key, last = False)
        return self.cache[key]

    def put(self, key, value):
        self.cache[key]=value # 添加 key value 或者更新 value
        # last=False 表示移到链表头
        self.cache.move_to_end(key, last=False)
        if len(self.cache) > self.capacity:
            self.cache.popitem() # 去掉最后一本书
