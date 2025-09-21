#include <iostream>
#include <memory>
#include <vector>
using namespace std;

// ================= 1. 没有虚析构函数的问题类 =================
class BadBase {
public:
    BadBase(const string& name) : name_(name) {
        cout << "BadBase constructor: " << name_ << endl;
    }
    
    // 注意：这里没有 virtual！！！
    ~BadBase() {
        cout << "BadBase destructor: " << name_ << endl;
    }
    
    virtual void speak() const {
        cout << name_ << " speaks from BadBase" << endl;
    }

protected:
    string name_;
};

class BadDerived : public BadBase {
private:
    int* data_;  // 动态分配的内存
    
public:
    BadDerived(const string& name, int value) : BadBase(name) {
        data_ = new int(value);  // 分配内存
        cout << "BadDerived constructor: " << name_ << ", allocated memory for value " << *data_ << endl;
    }
    
    ~BadDerived() {
        cout << "BadDerived destructor: " << name_ << ", freeing memory for value " << *data_ << endl;
        delete data_;  // 释放内存
    }
    
    void speak() const override {
        cout << name_ << " speaks from BadDerived with value " << *data_ << endl;
    }
};

// ================= 2. 正确的虚析构函数类 =================
class GoodBase {
public:
    GoodBase(const string& name) : name_(name) {
        cout << "GoodBase constructor: " << name_ << endl;
    }
    
    // 关键：virtual 析构函数！
    virtual ~GoodBase() {
        cout << "GoodBase destructor: " << name_ << endl;
    }
    
    virtual void speak() const {
        cout << name_ << " speaks from GoodBase" << endl;
    }

protected:
    string name_;
};

class GoodDerived : public GoodBase {
private:
    int* data_;  // 动态分配的内存
    
public:
    GoodDerived(const string& name, int value) : GoodBase(name) {
        data_ = new int(value);  // 分配内存
        cout << "GoodDerived constructor: " << name_ << ", allocated memory for value " << *data_ << endl;
    }
    
    ~GoodDerived() override {
        cout << "GoodDerived destructor: " << name_ << ", freeing memory for value " << *data_ << endl;
        delete data_;  // 释放内存
    }
    
    void speak() const override {
        cout << name_ << " speaks from GoodDerived with value " << *data_ << endl;
    }
};

// ================= 3. 演示非虚析构函数的问题 =================
void demonstrateBadDestruction() {
    cout << "\n=== ❌ 没有虚析构函数的问题 ===" << endl;
    cout << "通过基类指针删除派生类对象..." << endl;
    
    {
        BadBase* ptr = new BadDerived("BadObject", 42);
        ptr->speak();
        
        cout << "\n删除对象..." << endl;
        delete ptr;  // ⚠️ 危险！只调用基类析构函数
    }
    
    cout << "❌ 注意：BadDerived的析构函数没有被调用！" << endl;
    cout << "❌ 这会导致内存泄漏！data_指向的内存没有被释放！" << endl;
}

void demonstrateGoodDestruction() {
    cout << "\n=== ✅ 虚析构函数的正确行为 ===" << endl;
    cout << "通过基类指针删除派生类对象..." << endl;
    
    {
        GoodBase* ptr = new GoodDerived("GoodObject", 42);
        ptr->speak();
        
        cout << "\n删除对象..." << endl;
        delete ptr;  // ✅ 正确！先调用派生类析构，再调用基类析构
    }
    
    cout << "✅ 析构顺序正确：先GoodDerived，后GoodBase" << endl;
    cout << "✅ 内存正确释放，没有泄漏！" << endl;
}

// ================= 4. 智能指针的情况 =================
void demonstrateSmartPointers() {
    cout << "\n=== 智能指针的情况 ===" << endl;
    
    cout << "\n--- 没有虚析构函数的情况 ---" << endl;
    {
        unique_ptr<BadBase> bad_ptr = make_unique<BadDerived>("BadSmart", 99);
        bad_ptr->speak();
        // unique_ptr析构时也会有同样问题！
    }
    
    cout << "\n--- 有虚析构函数的情况 ---" << endl;
    {
        unique_ptr<GoodBase> good_ptr = make_unique<GoodDerived>("GoodSmart", 99);
        good_ptr->speak();
        // 正确的析构顺序
    }
}

// ================= 5. 不需要虚析构函数的情况 =================
class NoPolymorphismBase {
public:
    NoPolymorphismBase() { cout << "NoPolymorphismBase constructor" << endl; }
    ~NoPolymorphismBase() { cout << "NoPolymorphismBase destructor" << endl; }  // 不需要virtual
    
    // 没有虚函数，不用于多态
    void doSomething() { cout << "Base doing something" << endl; }
};

class NoPolymorphismDerived : public NoPolymorphismBase {
public:
    NoPolymorphismDerived() { cout << "NoPolymorphismDerived constructor" << endl; }
    ~NoPolymorphismDerived() { cout << "NoPolymorphismDerived destructor" << endl; }
    
    void doSomething() { cout << "Derived doing something" << endl; }  // 隐藏，不是重写
};

void demonstrateNoPolymorphismCase() {
    cout << "\n=== 不需要虚析构函数的情况 ===" << endl;
    cout << "当不使用多态时，不需要虚析构函数：" << endl;
    
    {
        NoPolymorphismDerived obj;  // 直接使用派生类对象
        obj.doSomething();
        // 析构时会正确调用派生类然后基类的析构函数
    }
}

// ================= 主函数 =================
int main() {
    try {
        cout << "=== 虚析构函数重要性演示 ===" << endl;
        
        demonstrateBadDestruction();
        demonstrateGoodDestruction();
        demonstrateSmartPointers();
        demonstrateNoPolymorphismCase();
        
        cout << "\n=== 总结 ===" << endl;
        cout << "1. 当基类用于多态时，必须有虚析构函数" << endl;
        cout << "2. 虚析构函数确保正确的析构顺序：派生类 → 基类" << endl;
        cout << "3. 没有虚析构函数会导致内存泄漏和未定义行为" << endl;
        cout << "4. 如果类不用于多态，可以不需要虚析构函数" << endl;
        cout << "5. 现代C++推荐使用智能指针，但仍需要虚析构函数" << endl;
        
    } catch (const exception& e) {
        cout << "错误: " << e.what() << endl;
    }
    
    return 0;
}
