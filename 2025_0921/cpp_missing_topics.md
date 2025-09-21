# 🔍 C++ 面试知识点补充清单

## 📋 当前学习计划已覆盖的知识点
✅ **面向对象编程** (继承、多态、虚函数、抽象类)  
✅ **智能指针与内存管理** (unique_ptr, shared_ptr, weak_ptr, RAII)  
✅ **STL容器与算法** (vector, map, 迭代器, 算法库)  
✅ **多线程编程** (thread, mutex, condition_variable)  
✅ **设计模式** (单例、工厂、观察者)  
✅ **关键字理解** (virtual, explicit, const基础)

---

## ⚠️ 重要遗漏的高频面试知识点

### 🔥 **优先级1 - 必须掌握**

#### 1. **模板编程 (Template Programming)** ⭐⭐⭐⭐⭐
**为什么重要**: C++最核心特性，面试必考，STL基础

**关键考点**:
- 函数模板vs类模板
- 模板参数推导 (Template Argument Deduction)
- 模板特化 (Template Specialization) 
- 可变参数模板 (Variadic Templates)
- SFINAE (Substitution Failure Is Not An Error)
- 模板元编程基础

**面试常见问题**:
```cpp
// 面试题：实现一个泛型的min函数
template<typename T>
T min(T a, T b) { return a < b ? a : b; }

// 进阶：如何处理不同类型的参数？
template<typename T, typename U>
auto min(T a, U b) -> decltype(a < b ? a : b);
```

#### 2. **引用 vs 指针深度对比** ⭐⭐⭐⭐⭐
**为什么重要**: 基础概念，但面试官爱深挖

**关键对比**:
| 特性 | 引用(Reference) | 指针(Pointer) |
|------|----------------|---------------|
| **初始化** | 必须初始化 | 可以不初始化 |
| **重新指向** | 不能 | 可以 |
| **空值** | 不能为空 | 可以为nullptr |
| **算术运算** | 不支持 | 支持 |
| **内存占用** | 不占用额外空间 | 占用8字节 |
| **多层间接** | 不支持 | 支持指针的指针 |

#### 3. **左值引用 vs 右值引用** ⭐⭐⭐⭐⭐
**为什么重要**: 现代C++核心，移动语义基础

**核心概念**:
```cpp
// 左值引用 - 绑定到左值
int a = 10;
int& lref = a;        // OK

// 右值引用 - 绑定到右值  
int&& rref = 20;      // OK
int&& rref2 = std::move(a); // 强制转换为右值

// 万能引用 - 模板中的神奇特性
template<typename T>
void func(T&& param);  // 既可以接受左值也可以接受右值
```

#### 4. **操作符重载 (Operator Overloading)** ⭐⭐⭐⭐
**为什么重要**: 面试常要求现场实现

**重点操作符**:
```cpp
class MyString {
public:
    // 赋值操作符
    MyString& operator=(const MyString& other);
    MyString& operator=(MyString&& other) noexcept;
    
    // 比较操作符
    bool operator==(const MyString& other) const;
    bool operator<(const MyString& other) const;
    
    // 下标操作符
    char& operator[](size_t index);
    const char& operator[](size_t index) const;
    
    // 输入输出操作符 (友元函数)
    friend std::ostream& operator<<(std::ostream& os, const MyString& str);
};
```

#### 5. **异常处理与异常安全** ⭐⭐⭐⭐
**为什么重要**: 企业级代码必备，RAII相关

**关键概念**:
```cpp
// 异常安全等级
// 1. 基本保证 - 不泄露资源
// 2. 强保证 - 操作要么成功要么回滚
// 3. 不抛出保证 - 绝不抛出异常

// RAII + 异常安全示例
class SafeResource {
    Resource* ptr;
public:
    SafeResource() : ptr(new Resource()) {}
    ~SafeResource() { delete ptr; }  // 即使异常也能清理
    
    // 拷贝控制需要考虑异常安全
    SafeResource(const SafeResource& other) {
        ptr = new Resource(*other.ptr); // 可能抛出异常
    }
};
```

### 🔴 **优先级2 - 重要补充**

#### 6. **const correctness (const正确性)** ⭐⭐⭐⭐
```cpp
// const的各种用法
const int* p1;        // 指向常量的指针
int* const p2;        // 常量指针
const int* const p3;  // 指向常量的常量指针

// const成员函数
class MyClass {
    int value;
public:
    int getValue() const { return value; }      // const成员函数
    void setValue(int v) { value = v; }         // 非const成员函数
};

// mutable关键字
class Counter {
    mutable int count = 0;  // 即使在const函数中也可以修改
public:
    void increment() const { ++count; }  // const函数中修改mutable成员
};
```

#### 7. **类型转换详解** ⭐⭐⭐⭐
```cpp
// C++四种显式类型转换
Base* base = new Derived();

// 1. static_cast - 编译时检查的安全转换
Derived* d1 = static_cast<Derived*>(base);

// 2. dynamic_cast - 运行时检查的安全转换（需要虚函数）
Derived* d2 = dynamic_cast<Derived*>(base);  // 失败返回nullptr

// 3. const_cast - 移除const属性
const int* cp = &value;
int* p = const_cast<int*>(cp);

// 4. reinterpret_cast - 底层位模式转换（危险）
int* ip = reinterpret_cast<int*>(base);
```

#### 8. **完美转发 (Perfect Forwarding)** ⭐⭐⭐
```cpp
// 问题：如何在模板函数中保持参数的值类别？
template<typename T>
void wrapper(T&& param) {
    // 错误的转发方式
    func(param);  // param总是左值
    
    // 完美转发
    func(std::forward<T>(param));  // 保持原始值类别
}
```

#### 9. **lambda表达式深入** ⭐⭐⭐
```cpp
// 捕获方式详解
int x = 10, y = 20;

auto lambda1 = [=]() { return x + y; };      // 按值捕获
auto lambda2 = [&]() { return x + y; };      // 按引用捕获  
auto lambda3 = [x, &y]() { return x + y; };  // 混合捕获
auto lambda4 = [=, &y]() { return x + y; };  // 默认按值，y按引用

// C++14 初始化捕获
auto lambda5 = [z = x + y]() { return z; };

// mutable lambda
auto lambda6 = [x]() mutable { return ++x; };
```

### 🟡 **优先级3 - 了解即可**

#### 10. **编译和链接过程** ⭐⭐⭐
- **预处理** → **编译** → **汇编** → **链接**
- 头文件包含机制
- 静态链接 vs 动态链接
- 符号解析过程

#### 11. **内存对齐和padding** ⭐⭐⭐
```cpp
struct BadStruct {
    char c;     // 1字节
    int i;      // 4字节，但需要4字节对齐，所以前面有3字节padding
    char c2;    // 1字节  
    double d;   // 8字节，需要8字节对齐，前面有7字节padding
}; // 总大小：24字节，而不是14字节

struct GoodStruct {
    double d;   // 8字节
    int i;      // 4字节
    char c;     // 1字节
    char c2;    // 1字节
}; // 总大小：16字节
```

#### 12. **枚举类型进化** ⭐⭐⭐
```cpp
// C风格枚举的问题
enum Color { RED, GREEN, BLUE };
enum Size { SMALL, MEDIUM, LARGE };
int x = RED;  // 隐式转换为int，可能不是期望行为

// C++11 强类型枚举
enum class NewColor { RED, GREEN, BLUE };
enum class NewSize { SMALL, MEDIUM, LARGE };
// int y = NewColor::RED;  // 编译错误！不能隐式转换
NewColor c = NewColor::RED;  // 必须显式使用
```

#### 13. **union的现代用法** ⭐⭐
```cpp
// 传统union
union Data {
    int i;
    float f;
    char str[20];
};

// 现代用法：std::variant (C++17)
std::variant<int, float, std::string> data;
data = 42;
data = 3.14f;
data = "hello";
```

#### 14. **友元函数和友元类** ⭐⭐
```cpp
class MyClass {
private:
    int secret = 42;
    
    // 友元函数可以访问私有成员
    friend void globalFunc(const MyClass& obj);
    
    // 友元类可以访问所有成员
    friend class FriendClass;
};

void globalFunc(const MyClass& obj) {
    std::cout << obj.secret;  // 可以访问private成员
}
```

---

## 🔧 如何整合到学习计划中

### **方案1: 扩展为7天计划**
- **Day 6**: 模板编程 + 类型推导 + SFINAE
- **Day 7**: 异常处理 + 操作符重载 + 类型转换

### **方案2: 强化现有5天计划**
- **Day 1**: 增加引用vs指针、const correctness
- **Day 2**: 增加左值右值引用、完美转发
- **Day 3**: 增加模板编程基础
- **Day 4**: 增加异常处理与异常安全
- **Day 5**: 增加操作符重载、类型转换

### **方案3: 创建专项突破清单**
针对每个遗漏知识点，创建30分钟速成专题：
1. **模板编程30分钟速成**
2. **引用vs指针深度对比**  
3. **移动语义完全指南**
4. **异常安全最佳实践**
5. **操作符重载实战**

---

## 📊 知识点重要性评分

| 知识点 | 面试频率 | 难度 | 重要性 | 建议学习时长 |
|--------|----------|------|--------|--------------|
| 模板编程 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 必须 | 2-3小时 |
| 引用vs指针 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 必须 | 1小时 |
| 左值右值 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 重要 | 1.5小时 |
| 操作符重载 | ⭐⭐⭐⭐ | ⭐⭐⭐ | 重要 | 1小时 |
| 异常处理 | ⭐⭐⭐ | ⭐⭐⭐ | 重要 | 1小时 |
| const正确性 | ⭐⭐⭐⭐ | ⭐⭐ | 重要 | 30分钟 |
| 类型转换 | ⭐⭐⭐ | ⭐⭐ | 了解 | 30分钟 |

---

## 🎯 建议行动方案

**推荐方案**: **强化现有5天计划** + **周末专项突破**

1. **立即行动**: 在原5天计划每天增加30-45分钟专门学习遗漏知识点
2. **周末深化**: 用周末时间深入学习模板编程和移动语义
3. **面试前突击**: 针对具体公司要求，重点复习相关知识点

**具体时间分配建议**:
```
Day 1: +30分钟学习引用vs指针、const正确性
Day 2: +45分钟学习左值右值引用、移动语义  
Day 3: +45分钟学习模板编程基础
Day 4: +30分钟学习异常处理
Day 5: +30分钟学习操作符重载、类型转换

周末: 2-3小时深入模板编程和高级特性
```

这样既不会打乱原有计划，又能确保知识点的完整性！
