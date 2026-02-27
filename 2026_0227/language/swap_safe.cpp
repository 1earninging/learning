#include <iostream>
#include <string>
#include <utility>
#include <vector>

// Day1 交付：安全 swap（含自交换）+ 使用 move 避免深拷贝
template <class T>
void my_swap(T& a, T& b) {
    if (&a == &b) return; // 自交换：直接返回
    T tmp = std::move(a);
    a = std::move(b);
    b = std::move(tmp);
}

int main() {
    // 1) 基本类型
    int x = 1, y = 2;
    my_swap(x, y);
    std::cout << "x=" << x << " y=" << y << "\n";

    // 2) 容器（move 的收益更明显：转移内部指针/容量，而不是逐元素拷贝）
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = {9, 8};
    my_swap(a, b);
    std::cout << "a.size=" << a.size() << " b.size=" << b.size() << "\n";

    // 3) 自交换：应当安全无副作用
    std::string s = "hello";
    my_swap(s, s);
    std::cout << "s=" << s << "\n";
    return 0;
}

