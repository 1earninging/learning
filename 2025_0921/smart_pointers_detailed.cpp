#include <iostream>
#include <memory>
#include <vector>
#include <string>
using namespace std;

// ================= 智能指针内部实现机制详解 =================

// 测试类，用于观察构造和析构
class Resource {
public:
    Resource(const string& name) : name_(name) {
        cout << "🔨 Resource [" << name_ << "] 构造" << endl;
    }
    
    ~Resource() {
        cout << "💥 Resource [" << name_ << "] 析构" << endl;
    }
    
    void use() const {
        cout << "⚙️  使用资源: " << name_ << endl;
    }
    
    const string& getName() const { return name_; }

private:
    string name_;
};

// ================= 1. unique_ptr 内部实现机制 =================

template<typename T>
class MyUniquePtr {
private:
    T* ptr_;  // 存储原始指针
    
public:
    // 构造函数
    explicit MyUniquePtr(T* p = nullptr) : ptr_(p) {
        cout << "🔗 MyUniquePtr 构造，管理对象: " << ptr_ << endl;
    }
    
    // 析构函数 - RAII的核心
    ~MyUniquePtr() {
        if (ptr_) {
            cout << "🗑️  MyUniquePtr 析构，删除对象: " << ptr_ << endl;
            delete ptr_;
        }
    }
    
    // 禁止拷贝构造和拷贝赋值 - 确保独占所有权
    MyUniquePtr(const MyUniquePtr&) = delete;
    MyUniquePtr& operator=(const MyUniquePtr&) = delete;
    
    // 移动构造函数 - 转移所有权
    MyUniquePtr(MyUniquePtr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;  // 清空原对象
        cout << "📦 MyUniquePtr 移动构造，转移所有权" << endl;
    }
    
    // 移动赋值运算符
    MyUniquePtr& operator=(MyUniquePtr&& other) noexcept {
        if (this != &other) {
            delete ptr_;  // 删除当前管理的对象
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
            cout << "📦 MyUniquePtr 移动赋值，转移所有权" << endl;
        }
        return *this;
    }
    
    // 解引用运算符
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    
    // 获取原始指针
    T* get() const { return ptr_; }
    
    // 释放所有权
    T* release() {
        T* temp = ptr_;
        ptr_ = nullptr;
        return temp;
    }
    
    // 重置指针
    void reset(T* p = nullptr) {
        delete ptr_;
        ptr_ = p;
    }
    
    // 布尔转换
    explicit operator bool() const { return ptr_ != nullptr; }
};

void demonstrateUniquePtrInternals() {
    cout << "\n=== 🔐 unique_ptr 内部机制详解 ===" << endl;
    
    cout << "\n📋 unique_ptr 内部结构：" << endl;
    cout << "┌─────────────────────────┐" << endl;
    cout << "│      unique_ptr         │" << endl;
    cout << "├─────────────────────────┤" << endl;
    cout << "│ T* ptr_  (8字节)        │ ← 存储原始指针" << endl;
    cout << "└─────────────────────────┘" << endl;
    cout << "总大小: 8字节 (与原始指针相同!)" << endl;
    
    {
        cout << "\n🔨 创建 unique_ptr：" << endl;
        MyUniquePtr<Resource> ptr1(new Resource("Unique1"));
        
        cout << "\n📦 移动语义测试：" << endl;
        MyUniquePtr<Resource> ptr2 = move(ptr1);  // 移动构造
        
        if (!ptr1) {
            cout << "✅ ptr1 已转移所有权，现在为空" << endl;
        }
        
        if (ptr2) {
            cout << "✅ ptr2 获得所有权" << endl;
            ptr2->use();
        }
    }  // ptr2 析构时自动删除Resource
    
    cout << "\n💡 unique_ptr 特点：" << endl;
    cout << "• 零开销抽象：大小等于原始指针" << endl;
    cout << "• 独占所有权：不能拷贝，只能移动" << endl;
    cout << "• 自动管理：析构时自动删除对象" << endl;
    cout << "• 异常安全：任何情况下都能正确释放资源" << endl;
}

// ================= 2. shared_ptr 内部实现机制 =================

template<typename T>
struct ControlBlock {
    size_t ref_count;      // 强引用计数
    size_t weak_count;     // 弱引用计数
    T* ptr;                // 管理的对象指针
    
    ControlBlock(T* p) : ref_count(1), weak_count(0), ptr(p) {
        cout << "📊 ControlBlock 创建，ref_count=1, weak_count=0" << endl;
    }
    
    ~ControlBlock() {
        cout << "🗑️  ControlBlock 析构" << endl;
    }
    
    void addRef() {
        ++ref_count;
        cout << "📈 引用计数增加: " << ref_count << endl;
    }
    
    bool release() {
        --ref_count;
        cout << "📉 引用计数减少: " << ref_count << endl;
        
        if (ref_count == 0) {
            cout << "💥 强引用归零，删除管理的对象" << endl;
            delete ptr;
            ptr = nullptr;
            return weak_count == 0;  // 如果弱引用也为0，返回true表示可以删除控制块
        }
        return false;
    }
    
    void addWeakRef() {
        ++weak_count;
        cout << "📈 弱引用计数增加: " << weak_count << endl;
    }
    
    bool releaseWeak() {
        --weak_count;
        cout << "📉 弱引用计数减少: " << weak_count << endl;
        return ref_count == 0 && weak_count == 0;
    }
};

template<typename T>
class MySharedPtr {
private:
    T* ptr_;                      // 指向管理的对象
    ControlBlock<T>* control_;    // 指向控制块
    
public:
    explicit MySharedPtr(T* p = nullptr) {
        if (p) {
            ptr_ = p;
            control_ = new ControlBlock<T>(p);
        } else {
            ptr_ = nullptr;
            control_ = nullptr;
        }
        cout << "🔗 MySharedPtr 构造" << endl;
    }
    
    // 拷贝构造函数 - 增加引用计数
    MySharedPtr(const MySharedPtr& other) : ptr_(other.ptr_), control_(other.control_) {
        if (control_) {
            control_->addRef();
        }
        cout << "📋 MySharedPtr 拷贝构造" << endl;
    }
    
    // 拷贝赋值运算符
    MySharedPtr& operator=(const MySharedPtr& other) {
        if (this != &other) {
            // 释放当前资源
            if (control_ && control_->release()) {
                delete control_;
            }
            
            // 共享新资源
            ptr_ = other.ptr_;
            control_ = other.control_;
            if (control_) {
                control_->addRef();
            }
        }
        cout << "📋 MySharedPtr 拷贝赋值" << endl;
        return *this;
    }
    
    ~MySharedPtr() {
        if (control_) {
            if (control_->release()) {
                delete control_;
            }
        }
        cout << "🗑️  MySharedPtr 析构" << endl;
    }
    
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    T* get() const { return ptr_; }
    
    size_t use_count() const {
        return control_ ? control_->ref_count : 0;
    }
    
    explicit operator bool() const { return ptr_ != nullptr; }
};

void demonstrateSharedPtrInternals() {
    cout << "\n=== 🤝 shared_ptr 内部机制详解 ===" << endl;
    
    cout << "\n📋 shared_ptr 内部结构：" << endl;
    cout << "┌─────────────────────────────────────┐" << endl;
    cout << "│            shared_ptr               │" << endl;
    cout << "├─────────────────────────────────────┤" << endl;
    cout << "│ T* ptr_       (8字节)               │ ← 指向管理的对象" << endl;
    cout << "│ ControlBlock* control_ (8字节)      │ ← 指向控制块" << endl;
    cout << "└─────────────────────────────────────┘" << endl;
    cout << "总大小: 16字节 (是原始指针的2倍)" << endl;
    
    cout << "\n📊 ControlBlock 结构：" << endl;
    cout << "┌─────────────────────────────────────┐" << endl;
    cout << "│          ControlBlock               │" << endl;
    cout << "├─────────────────────────────────────┤" << endl;
    cout << "│ size_t ref_count  (强引用计数)      │" << endl;
    cout << "│ size_t weak_count (弱引用计数)      │" << endl;
    cout << "│ T* ptr           (管理的对象指针)   │" << endl;
    cout << "│ ...其他数据(删除器、分配器等)        │" << endl;
    cout << "└─────────────────────────────────────┘" << endl;
    
    {
        cout << "\n🔨 创建第一个 shared_ptr：" << endl;
        MySharedPtr<Resource> ptr1(new Resource("Shared1"));
        cout << "引用计数: " << ptr1.use_count() << endl;
        
        {
            cout << "\n📋 拷贝构造第二个 shared_ptr：" << endl;
            MySharedPtr<Resource> ptr2 = ptr1;  // 拷贝构造
            cout << "ptr1 引用计数: " << ptr1.use_count() << endl;
            cout << "ptr2 引用计数: " << ptr2.use_count() << endl;
            
            {
                cout << "\n📋 赋值创建第三个 shared_ptr：" << endl;
                MySharedPtr<Resource> ptr3(nullptr);
                ptr3 = ptr2;  // 拷贝赋值
                cout << "三个 shared_ptr 都指向同一对象，引用计数: " << ptr3.use_count() << endl;
            }  // ptr3 析构
            cout << "ptr3 析构后，引用计数: " << ptr1.use_count() << endl;
        }  // ptr2 析构
        cout << "ptr2 析构后，引用计数: " << ptr1.use_count() << endl;
    }  // ptr1 析构，引用计数归零，删除对象
    
    cout << "\n💡 shared_ptr 特点：" << endl;
    cout << "• 共享所有权：多个指针可以指向同一对象" << endl;
    cout << "• 引用计数：自动跟踪有多少指针指向对象" << endl;
    cout << "• 线程安全：引用计数操作是原子的" << endl;
    cout << "• 内存开销：比原始指针大2倍" << endl;
}

// ================= 3. weak_ptr 和循环引用问题 =================

class Parent;
class Child;

class Parent {
public:
    string name;
    shared_ptr<Child> child;
    
    Parent(const string& n) : name(n) {
        cout << "👨 Parent [" << name << "] 构造" << endl;
    }
    
    ~Parent() {
        cout << "💥 Parent [" << name << "] 析构" << endl;
    }
};

class Child {
public:
    string name;
    weak_ptr<Parent> parent;  // 使用 weak_ptr 打破循环引用!
    // shared_ptr<Parent> parent;  // 如果用这个会造成循环引用
    
    Child(const string& n) : name(n) {
        cout << "👶 Child [" << name << "] 构造" << endl;
    }
    
    ~Child() {
        cout << "💥 Child [" << name << "] 析构" << endl;
    }
    
    void visitParent() {
        if (auto p = parent.lock()) {  // 尝试获取强引用
            cout << "👶 " << name << " 访问父亲: " << p->name << endl;
        } else {
            cout << "😢 " << name << " 的父亲已经不存在了" << endl;
        }
    }
};

void demonstrateWeakPtrInternals() {
    cout << "\n=== 🔗 weak_ptr 内部机制详解 ===" << endl;
    
    cout << "\n📋 weak_ptr 内部结构：" << endl;
    cout << "┌─────────────────────────────────────┐" << endl;
    cout << "│            weak_ptr                 │" << endl;
    cout << "├─────────────────────────────────────┤" << endl;
    cout << "│ T* ptr_       (8字节)               │ ← 指向管理的对象" << endl;
    cout << "│ ControlBlock* control_ (8字节)      │ ← 指向同一个控制块" << endl;
    cout << "└─────────────────────────────────────┘" << endl;
    cout << "总大小: 16字节 (与shared_ptr相同)" << endl;
    
    cout << "\n🔄 循环引用问题演示：" << endl;
    
    {
        cout << "\n👨‍👩‍👧 创建父子关系：" << endl;
        auto parent = make_shared<Parent>("爸爸");
        auto child = make_shared<Child>("小明");
        
        // 建立双向关系
        parent->child = child;          // Parent -> Child (强引用)
        child->parent = parent;         // Child -> Parent (弱引用!!)
        
        cout << "\nparent 引用计数: " << parent.use_count() << endl;
        cout << "child 引用计数: " << child.use_count() << endl;
        
        cout << "\n👶 子对象访问父对象：" << endl;
        child->visitParent();
        
        cout << "\n🔍 weak_ptr 不影响引用计数：" << endl;
        cout << "parent 引用计数仍然是: " << parent.use_count() << endl;
        
    }  // parent 和 child 都离开作用域
    
    cout << "\n✅ 成功避免了循环引用，对象正确析构！" << endl;
    
    cout << "\n💡 weak_ptr 特点：" << endl;
    cout << "• 不拥有对象：不影响引用计数" << endl;
    cout << "• 安全访问：可以检查对象是否还存在" << endl;
    cout << "• 打破循环：解决shared_ptr的循环引用问题" << endl;
    cout << "• 大小相同：与shared_ptr相同的内存布局" << endl;
}

// ================= 4. 循环引用问题对比 =================

class BadParent;  // 演示错误的循环引用

class BadChild {
public:
    string name;
    shared_ptr<BadParent> parent;  // 错误：使用shared_ptr造成循环引用
    
    BadChild(const string& n) : name(n) {
        cout << "👶 BadChild [" << name << "] 构造" << endl;
    }
    
    ~BadChild() {
        cout << "💥 BadChild [" << name << "] 析构" << endl;
    }
};

class BadParent {
public:
    string name;
    shared_ptr<BadChild> child;
    
    BadParent(const string& n) : name(n) {
        cout << "👨 BadParent [" << name << "] 构造" << endl;
    }
    
    ~BadParent() {
        cout << "💥 BadParent [" << name << "] 析构" << endl;
    }
};

void demonstrateCircularReference() {
    cout << "\n=== 🔄 循环引用问题对比 ===" << endl;
    
    cout << "\n❌ 错误示例：shared_ptr 循环引用" << endl;
    {
        auto badParent = make_shared<BadParent>("坏爸爸");
        auto badChild = make_shared<BadChild>("坏小孩");
        
        badParent->child = badChild;     // Parent -> Child (强引用)
        badChild->parent = badParent;    // Child -> Parent (强引用) ← 问题所在！
        
        cout << "badParent 引用计数: " << badParent.use_count() << endl;  // 2
        cout << "badChild 引用计数: " << badChild.use_count() << endl;    // 2
        
        cout << "\n离开作用域..." << endl;
    }  // badParent 和 badChild 离开作用域，但引用计数不会归零！
    
    cout << "😱 注意：没有看到析构消息！对象泄漏了！" << endl;
    cout << "\n🔍 循环引用原理：" << endl;
    cout << "badParent.use_count() = 2 (badParent变量 + badChild->parent)" << endl;
    cout << "badChild.use_count() = 2  (badChild变量 + badParent->child)" << endl;
    cout << "当变量离开作用域时，引用计数只减少到1，永远不会归零！" << endl;
}

// ================= 5. 性能和使用场景对比 =================

void demonstratePerformanceComparison() {
    cout << "\n=== ⚡ 智能指针性能对比 ===" << endl;
    
    cout << "\n📊 内存开销对比：" << endl;
    cout << "┌─────────────────┬────────────┬─────────────────┐" << endl;
    cout << "│   指针类型      │   大小     │   额外开销      │" << endl;
    cout << "├─────────────────┼────────────┼─────────────────┤" << endl;
    cout << "│ 原始指针 T*     │   8字节    │      无         │" << endl;
    cout << "│ unique_ptr<T>   │   8字节    │      无         │" << endl;
    cout << "│ shared_ptr<T>   │  16字节    │ ControlBlock    │" << endl;
    cout << "│ weak_ptr<T>     │  16字节    │ 共享ControlBlock │" << endl;
    cout << "└─────────────────┴────────────┴─────────────────┘" << endl;
    
    cout << "\n⚡ 运行时开销：" << endl;
    cout << "• unique_ptr: 零开销 (编译时优化为原始指针)" << endl;
    cout << "• shared_ptr: 原子操作开销 (引用计数增减)" << endl;
    cout << "• weak_ptr: lock()操作开销 (需要检查有效性)" << endl;
    
    cout << "\n🎯 使用场景总结：" << endl;
    cout << "\n🔐 unique_ptr 使用场景：" << endl;
    cout << "• 独占资源所有权" << endl;
    cout << "• 函数返回堆对象" << endl;
    cout << "• PIMPL惯用法" << endl;
    cout << "• 容器中存储多态对象" << endl;
    cout << "• 零开销要求的场景" << endl;
    
    cout << "\n🤝 shared_ptr 使用场景：" << endl;
    cout << "• 多个对象需要共享同一资源" << endl;
    cout << "• 对象生命周期复杂，难以确定所有者" << endl;
    cout << "• 需要在多个线程间共享对象" << endl;
    cout << "• 观察者模式的实现" << endl;
    cout << "• 缓存系统" << endl;
    
    cout << "\n🔗 weak_ptr 使用场景：" << endl;
    cout << "• 打破shared_ptr的循环引用" << endl;
    cout << "• 观察者模式中的观察者" << endl;
    cout << "• 缓存中的临时引用" << endl;
    cout << "• 父子关系中的反向引用" << endl;
    cout << "• 需要检查对象是否还存在" << endl;
}

// ================= 主函数 =================

int main() {
    cout << "=== 🧠 C++ 智能指针内部机制深度解析 ===" << endl;
    
    try {
        demonstrateUniquePtrInternals();
        demonstrateSharedPtrInternals();
        demonstrateWeakPtrInternals();
        demonstrateCircularReference();
        demonstratePerformanceComparison();
        
        cout << "\n=== 🎓 核心要点总结 ===" << endl;
        cout << "1. unique_ptr: 独占 + 零开销 + 移动语义" << endl;
        cout << "2. shared_ptr: 共享 + 引用计数 + 线程安全" << endl;
        cout << "3. weak_ptr: 观察 + 打破循环 + 安全访问" << endl;
        cout << "4. 选择原则: 能用unique就用unique，需要共享才用shared" << endl;
        cout << "5. 循环引用: 用weak_ptr在适当位置打断循环" << endl;
        
    } catch (const exception& e) {
        cout << "错误: " << e.what() << endl;
    }
    
    return 0;
}
