#include <iostream>
#include <memory>
#include <typeinfo>
using namespace std;

// ================= 虚函数表(vtable)详细机制解析 =================

class Base {
public:
    Base(int value) : base_data_(value) {
        cout << "Base constructor, base_data_ = " << base_data_ << endl;
    }
    
    virtual ~Base() {
        cout << "Base destructor" << endl;
    }
    
    // 虚函数1
    virtual void virtualFunc1() const {
        cout << "Base::virtualFunc1() - base_data_ = " << base_data_ << endl;
    }
    
    // 虚函数2  
    virtual void virtualFunc2() const {
        cout << "Base::virtualFunc2() - base_data_ = " << base_data_ << endl;
    }
    
    // 纯虚函数
    virtual void pureVirtual() const = 0;
    
    // 非虚函数
    void nonVirtualFunc() const {
        cout << "Base::nonVirtualFunc() - base_data_ = " << base_data_ << endl;
    }

private:
    int base_data_;
};

class Derived1 : public Base {
public:
    Derived1(int base_val, int derived_val) : Base(base_val), derived_data_(derived_val) {
        cout << "Derived1 constructor, derived_data_ = " << derived_data_ << endl;
    }
    
    ~Derived1() override {
        cout << "Derived1 destructor" << endl;
    }
    
    // 重写虚函数1
    void virtualFunc1() const override {
        cout << "Derived1::virtualFunc1() - derived_data_ = " << derived_data_ << endl;
    }
    
    // 不重写virtualFunc2，使用基类版本
    
    // 实现纯虚函数
    void pureVirtual() const override {
        cout << "Derived1::pureVirtual() - derived_data_ = " << derived_data_ << endl;
    }
    
    // 派生类特有函数
    void derived1OnlyFunc() const {
        cout << "Derived1::derived1OnlyFunc() - derived_data_ = " << derived_data_ << endl;
    }

private:
    int derived_data_;
};

class Derived2 : public Base {
public:
    Derived2(int base_val, string str) : Base(base_val), derived_str_(str) {
        cout << "Derived2 constructor, derived_str_ = " << derived_str_ << endl;
    }
    
    ~Derived2() override {
        cout << "Derived2 destructor" << endl;
    }
    
    // 重写所有虚函数
    void virtualFunc1() const override {
        cout << "Derived2::virtualFunc1() - derived_str_ = " << derived_str_ << endl;
    }
    
    void virtualFunc2() const override {
        cout << "Derived2::virtualFunc2() - derived_str_ = " << derived_str_ << endl;
    }
    
    void pureVirtual() const override {
        cout << "Derived2::pureVirtual() - derived_str_ = " << derived_str_ << endl;
    }

private:
    string derived_str_;
};

// ================= 模拟vtable结构 =================

struct VTableEntry {
    const char* function_name;
    void* function_ptr;  // 简化表示，实际是函数指针
};

void simulateVTableStructure() {
    cout << "\n=== 📋 虚函数表结构模拟 ===" << endl;
    
    cout << "\n🎯 理论上的vtable结构：" << endl;
    cout << "┌─────────────────────────────────────┐" << endl;
    cout << "│              Base类                 │" << endl;  
    cout << "├─────────────────────────────────────┤" << endl;
    cout << "│ Base vtable:                        │" << endl;
    cout << "│ [0] -> Base::~Base()               │" << endl;
    cout << "│ [1] -> Base::virtualFunc1()        │" << endl;
    cout << "│ [2] -> Base::virtualFunc2()        │" << endl;
    cout << "│ [3] -> Base::pureVirtual()         │" << endl;
    cout << "└─────────────────────────────────────┘" << endl;
    
    cout << "\n┌─────────────────────────────────────┐" << endl;
    cout << "│             Derived1类               │" << endl;  
    cout << "├─────────────────────────────────────┤" << endl;
    cout << "│ Derived1 vtable:                    │" << endl;
    cout << "│ [0] -> Derived1::~Derived1()       │" << endl;
    cout << "│ [1] -> Derived1::virtualFunc1() ✏️  │" << endl;
    cout << "│ [2] -> Base::virtualFunc2()        │" << endl;
    cout << "│ [3] -> Derived1::pureVirtual() ✏️   │" << endl;
    cout << "└─────────────────────────────────────┘" << endl;
    
    cout << "\n┌─────────────────────────────────────┐" << endl;
    cout << "│             Derived2类               │" << endl;  
    cout << "├─────────────────────────────────────┤" << endl;
    cout << "│ Derived2 vtable:                    │" << endl;
    cout << "│ [0] -> Derived2::~Derived2()       │" << endl;
    cout << "│ [1] -> Derived2::virtualFunc1() ✏️  │" << endl;
    cout << "│ [2] -> Derived2::virtualFunc2() ✏️  │" << endl;
    cout << "│ [3] -> Derived2::pureVirtual() ✏️   │" << endl;
    cout << "└─────────────────────────────────────┘" << endl;
    
    cout << "\n✏️ 表示该项被派生类重写了" << endl;
}

// ================= 对象内存布局展示 =================

void simulateObjectLayout() {
    cout << "\n=== 🧠 对象内存布局模拟 ===" << endl;
    
    Derived1 obj1(100, 200);
    Derived2 obj2(300, "Hello");
    
    cout << "\n🎯 对象在内存中的布局：" << endl;
    
    cout << "\n📦 Derived1 对象 obj1:" << endl;
    cout << "┌─────────────────────────────┐ ← obj1的内存地址" << endl;
    cout << "│ vptr (虚函数表指针)          │ → 指向Derived1的vtable" << endl;
    cout << "├─────────────────────────────┤" << endl;  
    cout << "│ base_data_ = 100            │ ← Base类的成员" << endl;
    cout << "├─────────────────────────────┤" << endl;
    cout << "│ derived_data_ = 200         │ ← Derived1类的成员" << endl;
    cout << "└─────────────────────────────┘" << endl;
    
    cout << "\n📦 Derived2 对象 obj2:" << endl;
    cout << "┌─────────────────────────────┐ ← obj2的内存地址" << endl;
    cout << "│ vptr (虚函数表指针)          │ → 指向Derived2的vtable" << endl;
    cout << "├─────────────────────────────┤" << endl;  
    cout << "│ base_data_ = 300            │ ← Base类的成员" << endl;
    cout << "├─────────────────────────────┤" << endl;
    cout << "│ derived_str_ = \"Hello\"      │ ← Derived2类的成员" << endl;
    cout << "└─────────────────────────────┘" << endl;
    
    cout << "\n💡 关键点：" << endl;
    cout << "• 每个对象的第一个成员通常是vptr（虚函数指针）" << endl;
    cout << "• vptr指向该对象类型对应的vtable" << endl;
    cout << "• 不同类型的对象有不同的vptr值" << endl;
}

// ================= 详细的函数调用过程模拟 =================

void simulateVirtualCallProcess() {
    cout << "\n=== 🔍 虚函数调用过程详细模拟 ===" << endl;
    
    Derived1 obj1(10, 20);
    Derived2 obj2(30, "Test");
    
    cout << "\n准备调用过程..." << endl;
    Base* ptr1 = &obj1;  // 基类指针指向Derived1对象
    Base* ptr2 = &obj2;  // 基类指针指向Derived2对象
    
    cout << "\n🎯 调用 ptr1->virtualFunc1() 的详细过程：" << endl;
    cout << "┌─ Step 1: 编译器生成的伪代码 ─┐" << endl;
    cout << "│ // ptr1->virtualFunc1();    │" << endl;
    cout << "│ auto vptr = ptr1->vptr;     │" << endl;
    cout << "│ auto vtable = *vptr;        │" << endl;
    cout << "│ auto func = vtable[1];      │ ← virtualFunc1在索引1" << endl;
    cout << "│ func(ptr1);                 │ ← 调用实际函数" << endl;
    cout << "└─────────────────────────────┘" << endl;
    
    cout << "\n🔍 执行步骤分解：" << endl;
    cout << "1️⃣ ptr1 = " << ptr1 << " (指向Derived1对象)" << endl;
    cout << "2️⃣ 通过ptr1找到对象，读取vptr" << endl;
    cout << "3️⃣ vptr指向Derived1的vtable" << endl;
    cout << "4️⃣ 在vtable[1]位置找到Derived1::virtualFunc1" << endl;
    cout << "5️⃣ 调用Derived1::virtualFunc1(ptr1)" << endl;
    
    cout << "\n▶️ 实际执行结果：" << endl;
    ptr1->virtualFunc1();  // 实际调用
    
    cout << "\n🎯 调用 ptr2->virtualFunc1() 的详细过程：" << endl;
    cout << "📍 同样的代码，不同的结果：" << endl;
    cout << "1️⃣ ptr2 = " << ptr2 << " (指向Derived2对象)" << endl;
    cout << "2️⃣ 通过ptr2找到对象，读取vptr" << endl; 
    cout << "3️⃣ vptr指向Derived2的vtable" << endl;
    cout << "4️⃣ 在vtable[1]位置找到Derived2::virtualFunc1" << endl;
    cout << "5️⃣ 调用Derived2::virtualFunc1(ptr2)" << endl;
    
    cout << "\n▶️ 实际执行结果：" << endl;
    ptr2->virtualFunc1();  // 实际调用
}

// ================= 对比非虚函数调用 =================

void compareNonVirtualCall() {
    cout << "\n=== ⚡ 对比：非虚函数的调用过程 ===" << endl;
    
    Derived1 obj(40, 50);
    Base* ptr = &obj;
    
    cout << "\n🎯 调用 ptr->nonVirtualFunc() 的过程：" << endl;
    cout << "┌─ 编译时就确定的调用 ─┐" << endl;
    cout << "│ // 编译器直接生成：   │" << endl;
    cout << "│ Base::nonVirtualFunc(ptr); │" << endl;
    cout << "└─────────────────────┘" << endl;
    
    cout << "\n🔍 执行步骤：" << endl;
    cout << "1️⃣ 编译器看到ptr是Base*类型" << endl;
    cout << "2️⃣ nonVirtualFunc不是virtual" << endl;
    cout << "3️⃣ 直接调用Base::nonVirtualFunc" << endl;
    cout << "4️⃣ 不需要查vtable，效率更高" << endl;
    
    cout << "\n▶️ 实际执行结果：" << endl;
    ptr->nonVirtualFunc();  // 总是调用Base版本
    
    cout << "\n💡 关键区别：" << endl;
    cout << "• 虚函数：运行时通过vtable查找 → 动态绑定" << endl;
    cout << "• 非虚函数：编译时直接确定 → 静态绑定" << endl;
}

// ================= vtable的继承和覆盖机制 =================

void demonstrateVTableInheritance() {
    cout << "\n=== 🧬 vtable的继承和覆盖机制 ===" << endl;
    
    cout << "\n🎯 vtable构建过程：" << endl;
    
    cout << "\n1️⃣ Base类创建vtable：" << endl;
    cout << "   Base::vtable[0] = &Base::~Base" << endl;
    cout << "   Base::vtable[1] = &Base::virtualFunc1" << endl;
    cout << "   Base::vtable[2] = &Base::virtualFunc2" << endl;
    cout << "   Base::vtable[3] = nullptr (纯虚函数)" << endl;
    
    cout << "\n2️⃣ Derived1类继承并修改vtable：" << endl;
    cout << "   Derived1::vtable[0] = &Derived1::~Derived1  ✏️覆盖" << endl;
    cout << "   Derived1::vtable[1] = &Derived1::virtualFunc1  ✏️覆盖" << endl;
    cout << "   Derived1::vtable[2] = &Base::virtualFunc2  📋继承" << endl;
    cout << "   Derived1::vtable[3] = &Derived1::pureVirtual  ✏️实现" << endl;
    
    cout << "\n3️⃣ Derived2类完全重写vtable：" << endl;
    cout << "   Derived2::vtable[0] = &Derived2::~Derived2  ✏️覆盖" << endl;
    cout << "   Derived2::vtable[1] = &Derived2::virtualFunc1  ✏️覆盖" << endl;
    cout << "   Derived2::vtable[2] = &Derived2::virtualFunc2  ✏️覆盖" << endl;
    cout << "   Derived2::vtable[3] = &Derived2::pureVirtual  ✏️实现" << endl;
    
    cout << "\n🧪 验证继承行为：" << endl;
    Derived1 d1(1, 2);
    Base* ptr = &d1;
    
    cout << "\n调用 ptr->virtualFunc2() (Derived1没有重写)：" << endl;
    ptr->virtualFunc2();  // 调用Base::virtualFunc2
    
    cout << "\n💡 说明：Derived1的vtable[2]仍然指向Base::virtualFunc2" << endl;
}

// ================= 性能分析 =================

void demonstratePerformanceImpact() {
    cout << "\n=== ⚡ 性能影响分析 ===" << endl;
    
    cout << "\n🎯 调用开销对比：" << endl;
    cout << "┌─────────────────┬─────────────────┬─────────────────┐" << endl;
    cout << "│   调用类型      │     开销        │     原因        │" << endl;
    cout << "├─────────────────┼─────────────────┼─────────────────┤" << endl;
    cout << "│ 普通函数调用     │     最低        │ 直接跳转        │" << endl;
    cout << "│ 非虚成员函数     │     很低        │ 编译时绑定      │" << endl;
    cout << "│ 虚函数调用       │   轻微额外      │ vtable查找      │" << endl;
    cout << "└─────────────────┴─────────────────┴─────────────────┘" << endl;
    
    cout << "\n🔍 虚函数额外开销包括：" << endl;
    cout << "• 一次内存读取（读取vptr）" << endl;
    cout << "• 一次vtable索引计算" << endl;
    cout << "• 一次间接函数调用" << endl;
    cout << "• 每个对象额外8字节存储vptr（64位系统）" << endl;
    
    cout << "\n💡 实际影响：" << endl;
    cout << "• 现代CPU预测分支很准确，性能影响很小" << endl;
    cout << "• 多态带来的设计灵活性远超性能损失" << endl;
    cout << "• 只有在极其频繁的调用中才需要考虑" << endl;
}

// ================= 主函数 =================

int main() {
    try {
        cout << "=== 🔬 虚函数表(vtable)详细机制解析 ===" << endl;
        
        simulateVTableStructure();
        simulateObjectLayout(); 
        simulateVirtualCallProcess();
        compareNonVirtualCall();
        demonstrateVTableInheritance();
        demonstratePerformanceImpact();
        
        cout << "\n=== 🎓 深度总结 ===" << endl;
        cout << "1. vtable是每个类的静态数据，存储虚函数地址" << endl;
        cout << "2. vptr是每个对象的成员，指向其类型的vtable" << endl;
        cout << "3. 虚函数调用 = 对象→vptr→vtable→函数地址→调用" << endl;
        cout << "4. 继承时vtable被复制，重写的函数地址会被替换" << endl;
        cout << "5. 这是动态绑定和多态的核心实现机制" << endl;
        cout << "6. 轻微性能开销换来巨大的设计灵活性" << endl;
        
    } catch (const exception& e) {
        cout << "错误: " << e.what() << endl;
    }
    
    return 0;
}
