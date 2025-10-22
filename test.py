import heapq

heap = []
heapq.heappush(heap, 3)  # 堆变为 [3]
heapq.heappush(heap, 1)  # 堆变为 [1, 3]（自动调整，1 上浮到堆顶）
heapq.heappush(heap, 2)  # 堆变为 [1, 3, 2]（2 插入后调整）
heapq.heappush(heap, 2)  # 堆变为 [1, 3, 2]（2 插入后调整）

print(heap)