from typing import List, Optional
from collections import defaultdict


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        m, n = len(matrix), len(matrix[0])
        row_tag = any(matrix[0][j] == 0 for j in range(n))
        column_tag = any(matrix[i][0] == 0 for i in range(m))
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
                
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0

        if row_tag:
            for i in range(n):
                matrix[0][i] = 0
        if column_tag:
            for i in range(m):
                matrix[i][0] = 0


    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m , n = len(matrix), len(matrix[0])
        l, r, t, d = 0, n-1, 0, m-1
        all_num = m * n
        res = [] 
        while all_num > 0:
            for i in range(l, r+1):
                if all_num == 0:
                    break
                res.append(matrix[t][i])
                all_num -= 1

            t += 1
            for i in range(t, d+1):
                res.append(matrix[i][r])
                all_num -= 1
            r -=1
            for i in range(r, l-1, -1):
                if all_num == 0:
                    break
                res.append(matrix[d][i])
                all_num -= 1
            d -=1
            for i in range(d, t-1, -1):
                if all_num == 0:
                    break
                res.append(matrix[i][l])
                all_num -= 1
            l += 1
        return res
    
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # 水平翻转
        n = len(matrix)
        for i in range(n // 2):
            for j in range(n):
                matrix[i][j], matrix[n-i-1][j] =   matrix[n-i-1][j], matrix[i][j]
        
        for i in range(n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        '''
        右上角为root节点
        '''
        m, n = len(matrix), len(matrix[0])
        i, j = n-1, 0 
        while i >= 0 and j < m:
            if matrix[j][i] > target:
                i -= 1
            elif matrix[j][i] < target:
                j += 1
            else:
                return True
        return False

    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        cur = dummy
        while list1 and list2:
            if list1.val < list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2
                list2 = list2.next
            cur = cur.next
        if list1:
            cur.next = list1
        if list2:
            cur.next = list2
        return dummy.next



    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        pivot = 0
        dummy = ListNode()
        cur = dummy
        while l1 or l2 or pivot:
            tmp = ListNode()
            val = 0
            if l1:
                val += l1.val
                l1 = l1.next
            if l2:
                val += l2.val
                l2 = l2.next
            val += pivot
            pivot = val // 10
            tmp.val = val % 10
            cur.next = tmp
            cur = cur.next
        return dummy.next
            

    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode()
        dummy.next = head
        fast, slow  = head, dummy
        while n:
            fast = fast.next
            n -= 1
        while fast:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next

    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        dummy.next = head
        cur = dummy 
        while cur.next and cur.next.next: 
            tmp1 = cur.next.next.next
            tmp2 = cur.next 
            cur.next = tmp2.next
            cur.next.next = tmp2
            tmp2.next = tmp1
            cur = tmp2
        return dummy.next

    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        def reverse(head, tail) ->Optional[ListNode]:
            pre = None
            cur = head 
            while cur != tail:
                tmp = cur.next
                cur.next = pre
                pre = cur
                cur = tmp
            return pre
        a = b = head 
        count = k
        while count and b:
            b = b.next
            count -= 1
        if count > 0:
            return head
        new_head = reverse(a, b)
        a.next = self.reverseKGroup(b, k)
        return new_head
            
        dummy.next = head
        pre = dummy

    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if head is None:
            return None
        cur = head
        dic = {}
        while cur:
            dic[cur] = Node(cur.val)
            cur = cur.next
        cur = head 
        while cur:
            dic[cur].next = dic.get(cur.next)
            dic[cur].random = dic.get(cur.random)
            cur = cur.next
        return dic.get(head)
    # def sortList(self, head: ListNode) -> ListNode:
    #     if not head or not head.next: return head # termination.
    #     # cut the LinkedList at the mid index.
    #     slow, fast = head, head.next
    #     while fast and fast.next:
    #         fast, slow = fast.next.next, slow.next
    #     mid, slow.next = slow.next, None # save and cut.
    #     # recursive for cutting.
    #     left, right = self.sortList(head), self.sortList(mid)
    #     # merge `left` and `right` linked list and return it.
    #     h = res = ListNode(0)
    #     while left and right:
    #         if left.val < right.val: h.next, left = left, left.next
    #         else: h.next, right = right, right.next
    #         h = h.next
    #     h.next = left if left else right
    #     return res.next
    def sortList(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head
        fast, slow = head.next, head
        while fast and fast.next:
            fast, slow = fast.next.next, slow.next
        mid = slow.next 
        slow.next = None 
        left, right = self.sortList(head), self.sortList(mid)
        dummy = ListNode(0)
        cur = dummy
        while left and right:
            if left.val < right.val:
                cur.next = left
                left = left.next
            else:
                cur.next = right
                right = right.next
            cur = cur.next
        cur.next = left if left else right
        return dummy.next

    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        import heapq
        queue = []
        dummy = ListNode()
        cur = dummy
        for i in range(len(lists)):
            if lists[i]:
                heapq.heappush(queue, (lists[i].val, i))
                lists[i] = lists[i].next
        
        while queue:
            val, idx = heapq.heappop(queue)
            cur.next = ListNode(val)
            cur = cur.next
            if lists[idx]:
                heapq.heappush(queue, (lists[idx].val, idx))
                lists[idx] = lists[idx].next
        return dummy.next
    


    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        def dfs(node):
            if not node:
                return 
            if node.left:
                dfs(node.left)
            res.append(node.val)
            if node.right:
                dfs(node.right)
        dfs(root)
        return res
        

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        depth = 0
        if not root:
            return 0
        def dfs(node, depth):
            if not node:
                return depth
            depth += 1
            return max(dfs(node.left, depth), dfs(node.right, depth))
        return dfs(root, depth)

            

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None 
        def dfs(node):
            if not node:
                return 
            node.left, node.right = node.right, node.left
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        return root
    

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        if not root.left and not root.right:
            return True
        def dfs(node1, node2) -> bool:
            if not node1 and not node2:
                return True
            if not node1 or not node2:
                return False 
            if node1.val != node2.val:
                return False
            return dfs(node1.left, node2.right) and dfs(node1.right, node2.left)
        return dfs(root.left, root.right)


    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        queue = [root]
        res = []
        while queue:
            level_size = len(queue)
            tmp = []
            for i in range(level_size):
                node = queue.pop(0)
                tmp.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(tmp)
        return res
    
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        a, b = headA, headB
        while a != b:
            a = a.next if a else headB
            b = b.next if b else headA
        return a

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = None
        while head:
            tmp = head.next
            head.next = dummy
            dummy = head
            head = tmp
        return dummy

    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        def find_middle(head):
            slow, fast = head, head
            while fast and fast.next:
                fast = fast.next.next
            slow = slow.next
            return slow
        def reverse(head):
            pre = None
            cur = head
            while cur:
                tmp = cur.next
                cur.next = pre
                pre = cur
                cur = tmp
            return pre
        middle = find_middle(head)
        reverse_middle = reverse(middle)
        while reverse_middle and head:
            if head.val != reverse_middle.val:
                return False
            head = head.next
            reverse_middle = reverse_middle.next
        return True
        
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
        
    
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow, fast = head, head
        while 1:
            if not fast or not fast.next:
                return None
            fast, slow = fast.next.next, slow.next
            if slow == fast:
                break

        fast = head 
        while fast != slow:
            fast = fast.next
            slow = slow.next
        return slow

    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def dfs(left, right):
            if left > right:
                return None 
            mid = (left + right + 1) // 2
            root = TreeNode(nums[mid])
            root.left = dfs(left, mid - 1)
            root.right = dfs(mid + 1, right)
            return root
        return dfs(0, len(nums) - 1)

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        import math
        pre = -math.inf
        def dfs(node):
            if not node:
                return True
            if not dfs(node.left) or  node.val <= pre:
                return False
            self.pre = node.val
            return dfs(node.right)
        return dfs(root)


    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.k = k
        self.res = None
        def dfs(node):
            if node is None:
                return 
            dfs(node.left)
            if self.k == 0: return
            self.k -= 1
            if self.k == 0: 
                self.res = node.val
            dfs(node.right)
        dfs(root)
        return self.res

                
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        map = {}
        queue = [(root, 0)]
        max_depth = -1
        while queue:
            node, depth = queue.pop(0)
            if node:
                max_depth = max(max_depth, depth)
                map[max_depth] = node.val
                queue.append((node.left, depth + 1))
                queue.append((node.right, depth + 1))
        return [map[i] for i in range(max_depth + 1)]
    
    
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        cur = root
        while cur:
            if cur.left:
                pre = nxt = cur.left
                while pre.right:
                    pre = pre.right
                pre.right = cur.right
                cur.left = None 
                cur.right = nxt 
            cur = cur.right
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        dic = {}
        for i in range(len(inorder)):
            dic[inorder[i]] = i
        def dfs(root, left, right):
            if left > right:
                return 
            root_idx = dic[preorder[root]]
            node = TreeNode(preorder[root])
            node.left = dfs(root + 1, left, root_idx - 1)
            node.right = dfs(root + (root_idx - left + 1), root_idx + 1, right)
            return node
        return dfs(0, 0, len(inorder) - 1)

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        ans = 0 
        cnt = defaultdict(int)
        cnt[0] = 1
        def dfs(node, cur):
            if node is None:
                return 
            nonlocal ans
            cur += node.val
            ans += cnt[cur - targetSum]
            cnt[cur] += 1
            dfs(node.left, cur)
            dfs(node.right, cur)
            cnt[cur] -= 1
        dfs(root, 0)
        return ans

    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if not root or p == root or q == root:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left if left else right
            
    def permute(self, nums: List[int]) -> List[List[int]]:
        self.res = []
        used = defaultdict(bool)
        self.path = []
        def dfs(num:int):
            if num == len(nums):
                self.res.append(self.path)
                return 
            for i in range(len(nums)):
                if used[nums[i]]:
                    continue
                used[nums[i]] = True
                self.path.append(nums[i])
                dfs(num + 1)
                self.path.pop()
                used[nums[i]] = False
        dfs(0)
        return self.res

    def subsets(self, nums: List[int]) -> List[List[int]]:
        self.res = []
        self.path = []
        def dfs(num:int):
            tmp = copy.deepcopy(self.path)
            self.res.append(tmp)
            if num >= len(nums):
                return 
            for i in range(num, len(nums)):
                self.path.append(nums[i])
                dfs(i + 1)
                self.path.pop()
        dfs(0)
        return self.res

    def letterCombinations(self, digits: str) -> List[str]:
        maps = {
		'2': "abc",
		'3': "def",
		'4': "ghi",
		'5': "jkl",
		'6': "mno",
		'7': "pqrs",
		'8': "tuv",
		'9': "wxyz",
        }
        res = []
        path = []
        def dfs(num:int):
            if num == len(digits):
                res.append("".join(path))
                return
            for i in range(num, len(digits)):
                digit = digits[i]
                chars = maps[digit]
                for char in chars:
                    path.append(char)
                    dfs(num + 1)
                    path.pop()
        dfs(0)
        return res

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:

        self.res = [] 
        self.path = []
        candidates.sort()
        start = 0
        def dfs(start, target, path):
            if target == 0:
                self.res.append(copy.deepcopy(self.path))
                return 
                if candidates[i] > target:
            for i in range(start, len(candidates)):
                    break
                path.append(candidates[i])
                dfs(i, target - candidates[i], path)
                path.pop()
        dfs(start, target, self.path)
        return self.res

    def generateParenthesis(self, n: int) -> List[str]:
        self.res = []
        path = []
        def dfs(open:int, close:int):
            if open == n and close == n:
                self.res.append("".join(path))
                return 
            if open < n:
                path.append("(")
                dfs(open + 1, close)
                path.pop()
            if close < open:
                path.append(")")
                dfs(open, close + 1)
                path.pop()
        dfs(0, 0)
        return self.res
    
    def exist(self, board: List[List[str]], word: str) -> bool:
        n, m = len(board), len(board[0])
        used = [[False] * m for _ in range(n)]
        x, y = [], []
        for i in range(n):
            for j in range(m):
                if board[i][j] == word[0]:
                    x.append(i)
                    y.append(j)
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        self.flag = False
        def dfs(start, i, j):
            if start == len(word):
                self.flag = True
                return True
            for idx in range(4):
                tmpx = i + directions[idx][0]
                tmpy = j + directions[idx][1]
                if tmpx >= 0 and tmpx < n and tmpy >= 0 and tmpy < m and not used[tmpx][tmpy] and board[tmpx][tmpy] == word[start]:
                    used[tmpx][tmpy] = True
                    dfs(start + 1, tmpx, tmpy)
                    used[tmpx][tmpy] = False

        for i in range(len(x)):
            used[x[i]][y[i]] = True
            dfs(1, x[i], y[i])
            if self.flag:
                return True
            used[x[i]][y[i]] = False
        return False

    def partition(self, s: str) -> List[List[str]]:
        self.res = []
        self.path = []
        def huiwen(s):
            right = len(s) - 1
            left = 0
            while left < right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            return True
        def dfs(start):
            if start == len(s):
                self.res.append(copy.deepcopy(self.path))
                return 
            for i in range(start, len(s)):
                if huiwen(s[start:i+1]):
                    self.path.append(s[start:i+1])
                    dfs(i+1)
                    self.path.pop()
        dfs(0)
        return self.res

    
    
class LRUNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.pre = None
        self.next = None
    
class LRUCache:

    def __init__(self, capacity: int):
        self.cap = capacity
        self.dummy = LRUNode(0, 0)
        self.dummy.pre = self.dummy
        self.dummy.next = self.dummy
        self.hashmap = {}
    
    def remove(self, node):
        node.pre.next = node.next
        node.next.pre = node.pre
    
    def put_node(self, node):
        node.next = self.dummy.next
        node.pre = self.dummy
        node.pre.next = node 
        node.next.pre = node


    def get_node(self, key):
        if key not in self.hashmap:
            return None
        node = self.hashmap[key]
        self.remove(node)
        self.put_node(node)
        return node

    def get(self, key: int) -> int:
        node = self.get_node(key)
        return node.value if node else -1
        
    def put(self, key: int, value: int) -> None:
        node = self.get_node(key)
        if node:
            node.value = value
            return 
        node = LRUNode(key, value)
        self.hashmap[key] = node
        self.put_node(node)
        if len(self.hashmap) > self.cap:
            last_node = self.dummy.pre
            self.remove(last_node)
            del self.hashmap[last_node.key]
    

        

if __name__ == "__main__":
    matrix = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]]
    print(Solution().searchMatrix(matrix, 19))

