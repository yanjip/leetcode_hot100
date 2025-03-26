# time: 2025/3/3 19:13
# author: YanJP
# import numpy as np
#
# # 创建一个二维数组和一个一维数组
# a = np.array([[1, 2, 3], [4, 5, 6]])
# b = np.array([[1, 0, 1]])
#
# # 使用广播机制进行加法运算
# result = b- a  # 数组维度[2,3] - or + [1,3] = [2,3]
#
# print("Array A:\n", a)
# print("Array B:\n", b)
# print("Result of A + B:\n", result)


# def decrypt_string(s):
#     t = []  # 解密后的字符串
#     p = 0   # 记录位移的整数
#
#     for char in s:
#         if char.isdigit():
#             x = int(char)
#             if p == 0:
#                 p = x
#             else:
#                 p = 10 * p + x
#         else:
#             # 先将字符串左移 p 位
#             if p > 0:
#                 t = t[p:] + t[:p]
#                 p = 0  # 重置 p
#             # 对 t 进行修改
#             if char == 'R':
#                 t = t[::-1]  # 反转字符串
#             else:
#                 t.append(char)  # 添加到字符串结尾
#
#     return ''.join(t)
#
# # 处理多组测试数据
# T = int(input())
# for _ in range(T):
#     s = input().strip()
#     result = decrypt_string(s)
#     print(result)
import sys

def decrypt_string(s):
    t = []  # 解密后的字符串
    p = 0   # 记录位移的整数

    for char in s:
        if char.isdigit():
            x = int(char)
            if p == 0:
                p = x
            else:
                p = 10 * p + x
        else:
            # 先将字符串左移 p 位
            if p > 0:
                t = t[p:] + t[:p]
                p = 0  # 重置 p
            # 对 t 进行修改
            if char == 'R':
                t = t[::-1]  # 反转字符串
            else:
                t.append(char)  # 添加到字符串结尾

    return ''.join(t)


def main():
    # 读取所有输入行
    # lines = sys.stdin.read().splitlines()
    #
    # # 第一行为测试数据组数 T
    # T = int(lines[0])
    # print(T)
    # # 处理每组测试数据
# # 处理多组测试数据
    T = int(input())
    for _ in range(T):
        s = input().strip()
        result = decrypt_string(s)
        print(result)



if __name__ == "__main__":
    main()


