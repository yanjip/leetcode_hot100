# time: 2025/1/10 13:40
# author: YanJP
import sys

# 1. 单个整数或浮点数
# n = int(input().strip())
# print(n)

# 2. 一组以空格分隔的整数或浮点数
# a = [int(i) for i in input().strip().split()]
# print(a)
numbers = list(map(int, input().strip().split()))
print(numbers)

# 3. 多组输入直到EOF（文件结束）
# for line in sys.stdin:
#     a, b = map(int, line.strip().split())
#     print(a + b)

# 4. 矩阵或二维数组输入
# rows, cols = map(int, input().strip().split())
# matrix = []
# for _ in range(rows):
#     row = list(map(int, input().strip().split()))
#     matrix.append(row)
# # 打印矩阵
# for row in matrix:
#     print(' '.join(str(x) for x in row))

# # 5. 字符串输入
s = input().strip()
print(s)


# 6. 读取一行
lst = list(map(int, sys.stdin.readline().strip().split()))
data = sys.stdin.read().split()

#7. 读取所有输入行
input = sys.stdin.read().split()
T = int(input[0]) #笫一行是测试组数
for t in range(1, T + 1):
    n = int(input[t])

