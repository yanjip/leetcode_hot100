# time: 2025/1/10 13:40
# author: YanJP

# 1. 单个整数或浮点数
# n = int(input().strip())
# print(n)

# 2. 一组以空格分隔的整数或浮点数
# a = [int(i) for i in input().strip().split()]
# print(a)
# numbers = list(map(int, input().strip().split()))
# print(numbers)

# 3. 多组输入直到EOF（文件结束）
# import sys
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