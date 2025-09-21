#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

// ================= 1. 虚函数和多态示例 =================
class Animal {
public:
    Animal(const string& name) : name_(name) {
        cout << "Animal constructor: " << name_ << endl;
    }
    
    virtual ~Animal() {
        cout << "Animal destructor: " << name_ << endl;
    }
    
    // 虚函数 - 支持多态
    virtual void makeSound() const {
        cout << name_ << " makes some sound" << endl;
    }
    
    // 纯虚函数 - 使Animal成为抽象类
    virtual void move() const = 0;
    
    const string& getName() const { return name_; }

private:
    string name_;
};

class Dog : public Animal {
public:
    Dog(const string& name) : Animal(name) {
        cout << "Dog constructor: " << getName() << endl;
    }
    
    ~Dog() override {
        cout << "Dog destructor: " << getName() << endl;
    }
    
    // 重写虚函数
    void makeSound() const override {
        cout << getName() << " barks: Woof!" << endl;
    }
    
    void move() const override {
        cout << getName() << " runs on four legs" << endl;
    }
};

class Cat : public Animal {
public:
    Cat(const string& name) : Animal(name) {
        cout << "Cat constructor: " << getName() << endl;
    }
    
    ~Cat() override {
        cout << "Cat destructor: " << getName() << endl;
    }
    
    void makeSound() const override {
        cout << getName() << " meows: Meow!" << endl;
    }
    
    void move() const override {
        cout << getName() << " walks silently" << endl;
    }
};

// ================= 2. RAII和资源管理示例 =================
class Resource {
public:
    Resource(const string& name) : name_(name) {
        cout << "Resource acquired: " << name_ << endl;
    }
    
    ~Resource() {
        cout << "Resource released: " << name_ << endl;
    }
    
    void use() const {
        cout << "Using resource: " << name_ << endl;
    }

private:
    string name_;
};

// ================= 3. 智能指针示例 =================
void demonstrateSmartPointers() {
    cout << "\n=== 智能指针示例 ===" << endl;
    
    // unique_ptr - 独占所有权
    {
        cout << "\n--- unique_ptr 示例 ---" << endl;
        auto dog1 = make_unique<Dog>("Buddy");
        dog1->makeSound();
        
        // 移动语义 - 所有权转移
        auto dog2 = move(dog1);
        if (!dog1) {
            cout << "dog1 现在为空" << endl;
        }
        dog2->move();
    } // dog2在这里自动释放
    
    // shared_ptr - 共享所有权
    {
        cout << "\n--- shared_ptr 示例 ---" << endl;
        shared_ptr<Cat> cat1 = make_shared<Cat>("Whiskers");
        cout << "cat1 引用计数: " << cat1.use_count() << endl;
        
        {
            shared_ptr<Cat> cat2 = cat1; // 共享所有权
            cout << "cat1 引用计数: " << cat1.use_count() << endl;
            cat2->makeSound();
        } // cat2离开作用域
        
        cout << "cat1 引用计数: " << cat1.use_count() << endl;
    } // cat1离开作用域，对象被释放
    
    // weak_ptr - 弱引用，解决循环引用
    {
        cout << "\n--- weak_ptr 示例 ---" << endl;
        shared_ptr<Dog> dog = make_shared<Dog>("Max");
        weak_ptr<Dog> weakDog = dog;
        
        cout << "shared_ptr 引用计数: " << dog.use_count() << endl;
        cout << "weak_ptr 是否过期: " << weakDog.expired() << endl;
        
        if (auto sharedDog = weakDog.lock()) {
            sharedDog->makeSound();
        }
    }
}

// ================= 4. RAII示例 =================
void demonstrateRAII() {
    cout << "\n=== RAII示例 ===" << endl;
    
    {
        cout << "\n--- 自动资源管理 ---" << endl;
        auto resource = make_unique<Resource>("Database Connection");
        resource->use();
        
        // 异常安全 - 即使抛出异常，资源也会被正确释放
        try {
            // 模拟可能抛异常的操作
            resource->use();
        } catch (...) {
            cout << "捕获异常，但资源仍会被正确释放" << endl;
        }
    } // resource在这里自动释放，无需手动管理
}

// ================= 5. 现代C++特性示例 =================
void demonstrateModernCpp() {
    cout << "\n=== 现代C++特性示例 ===" << endl;
    
    // Lambda表达式
    auto animals = vector<shared_ptr<Animal>>{
        make_shared<Dog>("Rex"),
        make_shared<Cat>("Fluffy"),
        make_shared<Dog>("Spot")
    };
    
    cout << "\n--- Lambda表达式和算法 ---" << endl;
    // 使用lambda表达式
    for_each(animals.begin(), animals.end(), [](const shared_ptr<Animal>& animal) {
        animal->makeSound();
        animal->move();
    });
    
    // auto关键字
    cout << "\n--- auto关键字 ---" << endl;
    auto message = "Hello, Modern C++!";
    auto number = 42;
    auto lambda = [](int x) { return x * 2; };
    
    cout << "message: " << message << endl;
    cout << "number: " << number << endl;
    cout << "lambda(5): " << lambda(5) << endl;
    
    // 范围for循环
    cout << "\n--- 范围for循环 ---" << endl;
    for (const auto& animal : animals) {
        cout << "动物名字: " << animal->getName() << endl;
    }
}

// ================= 主函数 =================
int main() {
    cout << "=== C++ 面试重点概念演示 ===" << endl;
    
    try {
        demonstrateSmartPointers();
        demonstrateRAII();
        demonstrateModernCpp();
        
        cout << "\n=== 程序正常结束 ===" << endl;
        
    } catch (const exception& e) {
        cout << "捕获异常: " << e.what() << endl;
    }
    
    return 0;
}
