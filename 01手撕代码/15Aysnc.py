# time: 2025/5/28 10:30
# author: YanJP
import asyncio
# --------------------------- await 调用其他异步函数
async def fetch_data():
    await asyncio.sleep(1)
    return "data"

async def main():
    result = await fetch_data()
    print(result)

# asyncio.run(main())

# ------------------------并发执行多个任务：asyncio.gather
async def task(name, sec):
    await asyncio.sleep(sec)
    return f"{name} done"

async def main1():
    # task("A", 2) 和 task("B", 1) 会一起调度。
    # 第 1 秒后 B 完成，第 2 秒后 A 完成。
    # 所以只等最大那个协程时间（不是总和），提高了效率。
    results = await asyncio.gather(
        task("A", 2),
        task("B", 1),
    )
    print(results)

# asyncio.run(main1())
# ---------------------------------- 创建任务：asyncio.create_task

async def task2(name, sec):
    await asyncio.sleep(sec)
    print(f"{name} end")
    # return f"{name} done"

async def main2():
    # asyncio.create_task(coroutine) 会将协程包装成一个 Task 并立即调度。
    t1 = asyncio.create_task(task2("A", 3))
    t2 = asyncio.create_task(task2("B", 1))
    # await t1  # 只 await t1，事件循环在**空闲**时仍然会继续执行其他“就绪”的任务（包括 t2）。
    # await t2

    await asyncio.gather(t1, t2)

if __name__ == '__main__':
    # asyncio.run(main())
    # asyncio.run(main1())
    asyncio.run(main2())
