#include <iostream>
#include <vector>

class A {
    public:
        int a_;
        A() : a_(0) {}
        A(int a) : a_(a) {
            std::cout << "构造函数"  << std::endl;
        }
        A(const A &other) {
            a_ = other.a_;
            std::cout << "拷贝构造函数"  << std::endl;
        }

        A(const A && other) {
            a_ = other.a_;
            std::cout << "移动构造函数"  << std::endl;
        }
        ~A() {
            // std::cout << "析构函数"  << std::endl;
        }
};

int main() {
    std::vector<int> test;
    for (int i = 0; i < 100; i++) {
        test.push_back(i);
        std::cout << test.size() << ", capicity is " << test.capacity() << std::endl;
    }
    std::cout << "======================" << std::endl;

    std::vector<A> temp;
    for (int i = 0; i < 10; i++) {
        temp.push_back(i);
        std::cout << temp.size() << ", capicity is " << temp.capacity() << ", vecotr ptr is : " << &temp << std::endl;
    }

    std::cout << "======================" << std::endl;
    // push_back vs emplace_back
    std::vector<A> test2;
    test2.reserve(10);
    test2.push_back(1);
    test2.emplace_back(2);

    std::cout << "======================" << std::endl;
    std::vector<int> test3(1000);
    std::cout<< "test3 size : " << test3.size() << ", capicity is " << test3.capacity() << std::endl;

    test3.clear();
    std::cout<< "After clear, test3 size : " << test3.size() << ", capicity is " << test3.capacity() << std::endl;

    std::vector<int> ().swap(test3);
    std::cout<< "After swap, test3 size : " << test3.size() << ", capicity is " << test3.capacity() << std::endl;

}
