# time: 2025/4/19 11:01
# author: YanJP
class Animal():
    num=0
    def __getattr__(self):
        return 1
    def __init__(self):
        pass
a=Animal()
print(a.num)