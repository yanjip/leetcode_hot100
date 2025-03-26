# time: 2025/3/14 11:10
# author: YanJP

class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 199. 二叉树的右视图
def rightSideView(root):
    ans=[]
    def f(node,depth):
        if node is None:
            return
        if len(ans)==depth:
            ans.append(node.val)
        f(node.right,depth+1)
        f(node.left, depth+1)
    f(root,0)
    return ans

# 110. 平衡二叉树
def isBalanced(root):
    def get_h(node):
        if node is None:
            return 0
        left=get_h(node.left)
        if left==-1: return -1
        right=get_h(node.right)
        if right==-1 or abs(left-right)>1:
            return -1
        return max(left, right)+1
    return get_h(root)!=-1

# 98. 验证二叉搜索树
# 前序遍历做法（先访问根节点值，再访问左子树，再访问右子树）
def isValidBST(root,left=float('-inf'),right=float('inf')):
    if root is None:
        return True
    x=root.val
    return left<x<right and isValidBST(root.left, left,x) and isValidBST(root.right,x,right)
# 中序遍历
def isValidBST2(root):
    pre=float('-inf')
    def dfs(node):
        if node is None:
            return True
        if not dfs(node.left):
            return False
        global pre
        if node.val<=pre:
            return False
        pre=node.val
        return dfs(node.right)
    return dfs(root)

# 236. 二叉树的最近公共祖先
def lowestCommonAncestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    if root is None:
        return None
    if root==p or root==q:
        return root
    # 2. 在左右子树中递归查找
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    # 3. 根据左右子树返回结果进行判断
    if left and right:
        # p 和 q 分别在左右子树，则当前 root 即为最近公共祖先
        return root
    else:
        # 只有一侧非空，则返回非空的那侧
        return left if left else right

# 235. 二叉搜索树的最近公共祖先
def lowestCommonAncestor_search_Tree(root, p, q):
    """
    :type root: TreeNode
    :type p: TreeNode
    :type q: TreeNode
    :rtype: TreeNode
    """
    x = root.val
    if x<p.val and x<q.val:
        return lowestCommonAncestor(root.right,p,q)
    if x>p.val and x>q.val:
        return lowestCommonAncestor(root.left,p,q)
    return root

# 94. 二叉树的中序遍历
def inorderTraversal(root) :
    ans=[]
    def dfs(node):
        if node is None:
            return
        dfs(node.left)
        ans.append(node.val)
        dfs(node.right)
    dfs(root)
    return ans

# 104. 二叉树的最大深度
def maxDepth(root) -> int:
    if root is None:
        return 0
    left=maxDepth(root.left)
    right=maxDepth(root.right)
    return max(left,right)+1

# 226. 翻转二叉树
def invertTree(root) :
    if root is None:
        return None
    left = invertTree(root.left)  # 翻转左子树
    right = invertTree(root.right)  # 翻转右子树
    root.left = right  # 交换左右儿子
    root.right = left
    return root


# 102. 二叉树的层序遍历
def levelOrder(root):
    if root is None:
        return []
    ans=[]
    cur=[root]
    while cur:
        vals=[]
        nxt=[]
        for n in cur:
            if n.left: nxt.append(n.left)
            if n.right: nxt.append(n.right)
            vals.append(n.val)
        ans.append(vals)
        cur=nxt
    return ans
from collections import deque
def levelOrder2(root):
    if root is None:
        return []
    ans=[]
    q=deque([root])
    while q:
        vals=[]
        for _ in range(len(q)):
            n=q.popleft()
            if n.left: q.append(n.left)
            if n.right: q.append(n.right)
            vals.append(n.val)
        ans.append(vals)
    return ans

# 513. 找树左下角的值
def findBottomLeftValue(root):
    q=deque([root])
    while q:
        node=q.popleft()
        if node.right:
            q.append(node.right)
        if node.left:
            q.append(node.left)
    return node.val

# 230. 二叉搜索树中第 K 小的元素
def kthSmallest(root, k: int) -> int:
    def dfs(node):
        if node is None:
            return -1
        left = dfs(node.left)  # 直接返回答案的写法
        if left != -1:
            return left
        nonlocal k
        k -= 1             #!!!!!!!!!!!!!!!!!dfs就是从上往下递归，从下往上返回答案
        if k == 0:
            return node.val
        return dfs(node.right)  # 右子树会返回答案或者 -1
    return dfs(root)

def kthSmallest2(root, k):
    ans = []
    global ans
    def dfs(node):
        if node is None:
            return
        dfs(node.left)
        ans.append(node.val)
        dfs(node.right)
        if len(ans) >= k:
            return
    dfs(root)
    return ans[k - 1]

# 114. 二叉树展开为链表
# 方法一：分治求解
#     1
#    / \
#   2   5
#  / \   \
# 3   4   6
    # 对于节点 3，它没有左子树和右子树，所以返回 3。
    # 对于节点 4，它没有左子树和右子树，所以返回 4。
    # 对于节点 2，它的左子树是 3，右子树是 4。展开后，链表为 2 -> 3 -> 4，返回 4。
def flatten(root) -> None:
    """
    Do not return anything, modify root in-place instead.
    """
    if root is None:
        return None
    left_tail=flatten(root.left)
    right_tail=flatten(root.right)
    if left_tail:
        left_tail.right=root.right
        root.right=root.left
        root.left=None
    return right_tail or left_tail or root  # 要注意顺序

# # 方法二：头插法
class Solution:
    head=None
    def flatten(self, root) -> None:
        if root is None:
            return
        self.flatten(root.right)
        self.flatten(root.left)
        root.left=None
        root.right=self.head
        self.head=root


#105. 从前序与中序遍历序列构造二叉树
def buildTree(preorder: list[int], inorder: list[int]):
    if preorder==[]:
        return None
    head=TreeNode(val=preorder[0])
    index=inorder.index(preorder[0])
    head.left=buildTree(preorder[1:index+1], inorder[:index])
    head.right=buildTree(preorder[index+1:], inorder[index+1:])  # 不管是前序还是中序，index+1之后都是右子树
    return head

# 437. 路径总和 III
def pathSum(root, targetSum: int) -> int:
    def dfs(node, cur_sum):
        if node is None:
            return 0
        cnt=0
        if cur_sum+node.val==targetSum:
            cnt+=1
            # cur_sum=0
        cnt+=dfs(node.left, cur_sum+node.val)
        cnt+=dfs(node.right, cur_sum+node.val)
        return cnt
    def double(node):
        if node is None:
            return 0
        total= dfs(node,0)
        total+=double(node.left)
        total+=double(node.right)
        return total
    return double(root)