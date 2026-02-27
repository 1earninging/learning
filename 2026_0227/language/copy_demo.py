#!/usr/bin/env python3
# Day1 交付：演示浅拷贝/深拷贝差异 + 复现典型 bug

import copy


def demo_aliasing():
    print("== demo_aliasing ==")
    a = [1, 2]
    b = a
    a.append(3)  # 原地修改
    print("a:", a)
    print("b:", b, "(b 也变了，因为 b 和 a 指向同一个对象)")
    print()


def demo_rebind():
    print("== demo_rebind ==")
    a = [1, 2]
    b = a
    a = [3, 4]  # 重新绑定
    print("a:", a)
    print("b:", b, "(b 不变，因为只是 a 换了标签)")
    print()


def demo_shallow_vs_deepcopy():
    print("== demo_shallow_vs_deepcopy ==")
    a = [[1], [2]]
    shallow = copy.copy(a)  # 或 a[:]，只复制外层 list
    deep = copy.deepcopy(a)

    a[0].append(99)  # 修改内层 list（可变对象）
    print("a     :", a)
    print("shallow:", shallow, "(浅拷贝共享内层对象，所以也看到 99)")
    print("deep  :", deep, "(深拷贝不共享内层对象，所以不受影响)")
    print()


def demo_default_arg_bug():
    print("== demo_default_arg_bug ==")

    def f(x=[]):
        x.append(1)
        return x

    print("f()  ->", f())
    print("f()  ->", f(), "(第二次调用继续污染同一个默认列表)")
    print("f([])->", f([]), "(显式传新列表则不会共享)")
    print()


if __name__ == "__main__":
    demo_aliasing()
    demo_rebind()
    demo_shallow_vs_deepcopy()
    demo_default_arg_bug()

