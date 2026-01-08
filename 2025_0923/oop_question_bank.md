# C++ 面向对象结构化题库（0923）

> 面试导向，覆盖：类与对象、封装、继承、多态、构造/析构/拷贝/移动、指针 vs 引用、const 正确性、重载/隐藏/重写、关键字。每题给出简要答案或要点，附若干代码型题目。

## 1. 类与对象（基础）
- 问：类与对象的区别？
  - 答：类是对象的抽象蓝图；对象是类的具体实例，拥有独立状态。
- 问：C++ 编译器默认生成哪些特殊成员函数？
  - 答：默认构造、析构、拷贝构造、拷贝赋值、移动构造、移动赋值（满足条件时）。
- 问：何时需要 `=delete` 或 `=default`？
  - 答：禁止不期望的拷贝/移动时用 `=delete`；保留默认语义且显式文档化时用 `=default`。
- 问：对象大小如何受虚函数影响？
  - 答：含虚函数的对象通常增加一指针大小（vptr），具体取决于实现和 ABI。
- 问：成员初始化顺序由谁决定？
  - 答：由成员声明顺序决定，与初始化列表书写顺序无关。
- 问：RAII 是什么？
  - 答：以对象生命周期管理资源的手法，构造获取、析构释放，确保异常安全。

## 2. 封装
- 问：`private`/`protected`/`public` 的访问边界？
  - 答：private：类内/友元；protected：类内/友元/派生类；public：所有可见处。
- 问：`protected` 的利弊？
  - 答：便于派生类复用，但扩大耦合面；谨慎暴露只给必要的受保护接口。
- 问：如何在保持封装下进行测试？
  - 答：友元、接口注入、PIMPL/桥接、构造注入依赖、对外暴露可观察状态。
- 问：何时使用 `friend`？
  - 答：运算符重载（对称访问）、测试夹具、紧密耦合但控制在局部范围。

## 3. 继承
- 问：`public`/`protected`/`private` 继承影响？
  - 答：影响基类成员在派生类中的可见性和对外暴露；public 保持外部 is-a 关系。
- 问：构造/析构顺序？
  - 答：构造：基类→成员→派生；析构：派生→成员→基类。虚基类最先构造、最后析构。
- 问：菱形继承与虚拟继承代价？
  - 答：通过虚拟继承共享一次基类实例，代价是对象布局和访问间接性更复杂。
- 问：覆盖/隐藏/重载区别？
  - 答：覆盖：同签名、虚函数、不同类层级；隐藏：同名遮蔽不同签名；重载：同作用域同名不同参数。
- 问：为何基类应有虚析构？
  - 答：通过基类指针删除派生对象时确保派生析构被调用，防资源泄漏。

## 4. 多态
- 问：动态绑定发生条件？
  - 答：通过基类（引用/指针）调用虚函数；非虚函数与对象表达式静态绑定。
- 问：vtable/vptr 的作用？
  - 答：vptr 指向类的虚函数表 vtable，实现运行时调度；影响对象大小与调用开销。
- 问：构造/析构期间调用虚函数会怎样？
  - 答：会静态绑定到当前构造/析构阶段的类版本，派生覆盖不会被调用。
- 问：`override`/`final` 的最佳实践？
  - 答：所有重写都加 `override`；必要时用 `final` 禁止进一步重写或继承。

## 5. 构造/析构/拷贝/赋值/移动
- 问：三/五/零法则？
  - 答：三：自定义析构则自定义拷贝构造与赋值；五：再加移动构造与移动赋值；零：仅用资源成员对象（如智能指针）而不自定义特殊成员。
- 问：`std::move` vs `std::forward`？
  - 答：move 无条件转换为右值；forward 保留值类别，仅用于转发引用。
- 问：RVO/NRVO 与移动的关系？
  - 答：发生（N）RVO 时可消除拷贝/移动；禁用优化或无法触发时才需要移动。
- 问：异常安全等级？
  - 答：basic：不变式保持；strong：要么成功要么无副作用；nothrow：不抛异常。

## 6. 指针 vs 引用
- 问：引用能否重新绑定？
  - 答：不能；引用一旦绑定不可变。指针可重指向。
- 问：何时用引用/指针/智能指针？
  - 答：引用用于非空、无所有权参数；裸指针用于观察；智能指针表达所有权（unique/shared）。
- 问：顶层/底层 const 区分？
  - 答：顶层修饰对象本身；底层修饰所指向对象或通过类型传播的部分。

## 7. const 正确性
- 问：`const` 成员函数语义？
  - 答：承诺不修改可见状态，`this` 为 `T const*`；允许改 `mutable`。
- 问：`const T*` vs `T* const`？
  - 答：前者指向常量；后者常量指针（指针值不能改）。

## 8. 重载/隐藏/重写
- 问：为何常用 `using Base::func;`？
  - 答：引入基类重载集合，避免派生同名函数隐藏基类其他重载。
- 问：重载决议优先级？
  - 答：精确匹配 > 提升转换 > 标准转换 > 用户自定义转换。

## 9. 关键字
- 问：`explicit` 的意义？
  - 答：禁止单实参构造的隐式转换，避免意外构造。
- 问：`static` 成员初始化与 ODR 注意点？
  - 答：类内仅声明，类外定义；内联变量或 `constexpr` 可在类内定义。
- 问：`volatile` 在 C++ 的真实用途？
  - 答：仅用于避免编译器优化掉对外设/内存映射 I/O 的访问，不提供原子性。

---

# 代码题（判断/改错/输出）

## 题1：隐藏 vs 覆盖
```cpp
struct Base {
    void f(int) { std::cout << "Base::f(int)\n"; }
    virtual void g() { std::cout << "Base::g()\n"; }
};
struct Der : Base {
    using Base::f;           // A
    void f(double) { std::cout << "Der::f(double)\n"; }
    void g() override { std::cout << "Der::g()\n"; }
};
int main() {
    Der d; Base& b = d;
    d.f(1);      // (1)
    d.f(1.0);    // (2)
    b.g();       // (3)
}
```
- 问：A 处 `using` 去掉会怎样？(1)(2)(3) 输出分别为何？
- 答：去掉 `using`，`Der::f(double)` 隐藏 `Base::f(int)`；(1) 进行双精度匹配到 `Der::f(double)`；(2) 同上；(3) 动态绑定到 `Der::g()`。

## 题2：构造/析构期虚函数
```cpp
struct B {
    B() { h(); }
    virtual ~B() { h(); }
    virtual void h() { std::cout << "B::h\n"; }
};
struct D : B {
    void h() override { std::cout << "D::h\n"; }
};
int main(){ D d; }
```
- 问：输出？为何？
- 答：两次都是 `B::h`。构造/析构期间虚调度绑定到当前阶段类的版本。

## 题3：三/五/零法则与移动
```cpp
struct Holder {
    std::unique_ptr<int> p;
    Holder() = default;
    Holder(const Holder&) = delete;               // (a)
    Holder& operator=(const Holder&) = delete;    // (b)
    Holder(Holder&&) noexcept = default;          // (c)
    Holder& operator=(Holder&&) noexcept = default; // (d)
};
```
- 问：为何 (a)(b) 要删？何时允许拷贝？
- 答：独占所有权不可拷贝；可自定义“深拷贝”语义时才允许。

## 题4：const 正确性与重载
```cpp
struct S {
    int x{0};
    int get() const { return x; }
    int& get() { return x; }
};
int main(){
    S s; const S cs{};
    s.get() = 42;        // (1)
    auto a = cs.get();   // (2)
}
```
- 问：(1)(2) 是否通过，推导类型是什么？
- 答：(1) 选用非常量重载，返回 `int&` 可赋值；(2) 选常量重载，返回 `int`。

## 题5：菱形继承与虚拟继承
```cpp
struct A { int a{1}; };
struct B : virtual A {};
struct C : virtual A {};
struct D : B, C {};
int main(){ D d; std::cout << d.a; }
```
- 问：是否有二义性？如何消除？
- 答：使用虚拟继承共享一个 `A` 子对象，访问 `d.a` 唯一，无二义。

---

建议使用：先刷“概念题”→做“代码题”→将答案口述一遍。每题 1–3 分钟。

---

## 进阶问答与原理解释（更深入）

- 问：vptr/vtable 的内存与调度原理？多继承下有什么变化？
  - 答：每个含虚函数的对象包含至少一个隐藏指针 vptr，指向所属最具体动态类型的 vtable。vtable 是按“虚函数声明顺序”存放的函数指针表，重写时沿槽位覆盖。构造/析构时对象的 vptr 会在各阶段被设置为当前阶段的类的 vtable，因此在构造/析构期调用虚函数会静态落到当前阶段类的版本。多继承下可能存在多个 vptr（每个有虚子对象一份），指针向上转型时编译器插入指针调整以定位正确的子对象 vptr。

- 问：对象切片（object slicing）是什么？
  - 答：以值传递或按值赋值基类对象时，派生类中超出的状态被截断，仅保留基类子对象部分。切片后多态性丢失。避免：以引用/指针（最好是智能指针）传递多态对象。

- 问：值类别（lvalue/xvalue/prvalue）与引用折叠规则？
  - 答：lvalue 表示具名可定位对象；xvalue 为“将亡值”（如 `std::move(x)`）；prvalue 为纯右值（字面量、返回未绑定对象的表达式）。折叠规则：`T& & -> T&`，`T& && -> T&`，`T&& & -> T&`，`T&& && -> T&&`。模板形参 `T&&` 若接收实参为 lvalue 则折叠为 `T&`（转发引用/通用引用），配合 `std::forward<T>` 保留值类别。

- 问：C++17 对 prvalue 的变化与 RVO/NRVO？
  - 答：C++17 使某些场景下的拷贝省略成为强制（如直接返回临时构造的对象），表达式结果不再需要临时物化，直接初始化目标。NRVO 仍是可选优化。若禁用拷贝省略（如编译器选项），则将观察到移动或拷贝构造发生。

- 问：虚函数 vs CRTP（静态多态）何时选用？
  - 答：运行时多态（需要跨动态类型边界、抽象接口、插件架构）选虚函数；编译期静态多态（性能敏感、内联、无需运行时分发）可用 CRTP/概念/策略模式。静态多态减少 vtable 开销但牺牲 ABI 稳定与二进制边界灵活性。

- 问：析构函数是否应抛异常？
  - 答：不建议。析构在栈展开异常传播中再次抛异常会调用 `std::terminate`。若必须报告失败：捕获并记录、设置状态、或提供显式 `close()` API 返回错误码。

- 问：PIMPL 的动机与代价？
  - 答：隐藏实现细节、降低编译耦合、稳定 ABI；代价是一次间接访问开销与堆分配，可配合小对象优化/自定义内存池缓解。

- 问：`final` 与 `override` 的组合与最佳实践？
  - 答：所有重写都写 `override` 让编译器校验签名；不希望继续重写时在函数或类上加 `final`，可作为二进制稳定承诺与意图表达。

---

# 更多代码题与详解（进阶）

## 题6：值类别与重载匹配
```cpp
struct X {};
void f(X&)      { std::cout << "f(lvalue)\n"; }
void f(const X&){ std::cout << "f(const lvalue)\n"; }
void f(X&&)     { std::cout << "f(rvalue)\n"; }
int main(){
    X x; const X cx{};
    f(x);              // (1)
    f(std::move(x));   // (2)
    f(cx);             // (3)
    f(std::move(cx));  // (4)
}
```
- 解析：
  - (1) 选择 `f(X&)`，非常量左值优先匹配非常量左值引用。
  - (2) 选择 `f(X&&)`，`std::move(x)` 产生 xvalue（右值引用可绑定）。
  - (3) 选择 `f(const X&)`，常量左值只能绑定到常量引用。
  - (4) 仍选择 `f(const X&)`，`std::move(cx)` 是 `const X` 的 xvalue，不能绑定到 `X&&`（非常量），因此退化到 `const&`。

## 题7：完美转发与引用折叠
```cpp
void use(int&)      { std::cout << "use(lvalue)\n"; }
void use(const int&){ std::cout << "use(const lvalue)\n"; }
void use(int&&)     { std::cout << "use(rvalue)\n"; }

template <typename T>
void wrapper(T&& t){
    use(std::forward<T>(t));
}

int main(){
    int x = 0;
    wrapper(x);       // (1)
    wrapper(42);      // (2)
    const int cx = 1;
    wrapper(cx);      // (3)
}
```
- 解析：
  - (1) `T` 推导为 `int&`，`T&&` 折叠成 `int&`，`forward<T>(t)` 保留左值，调 `use(int&)`。
  - (2) `T` 推导为 `int`，`T&&` 为 `int&&`，`forward` 保留右值，调 `use(int&&)`。
  - (3) `T` 推导为 `const int&`，折叠为 `const int&`，调用 `use(const int&)`。

## 题8：成员函数引用限定符
```cpp
struct S {
    std::string id;
    std::string info() &  { return "L:" + id; }
    std::string info() && { return "R:" + id; }
};
int main(){
    S s{"x"};
    std::cout << s.info() << "\n";           // (1)
    std::cout << std::move(s).info() << "\n"; // (2)
}
```
- 解析：
  - (1) `s` 是左值，调用 `info() &`，输出 `L:x`。
  - (2) `std::move(s)` 是 xvalue，调用 `info() &&`，输出 `R:x`。引用限定符可区分对象值类别以优化实现（如复用资源）。

## 题9：对象切片与多态丢失
```cpp
struct B { virtual void g(){ std::cout << "B::g\n"; } };
struct D : B { void g() override { std::cout << "D::g\n"; } };

void call_by_value(B b){ b.g(); }
void call_by_ref(B& b){ b.g(); }

int main(){
    D d;
    call_by_value(d); // (1)
    call_by_ref(d);   // (2)
}
```
- 解析：
  - (1) 发生切片，`B b` 只保留基类子对象，调用 `B::g`。
  - (2) 引用保持动态类型为 `D`，虚调用至 `D::g`。

## 题10：虚析构与资源释放
```cpp
struct B { ~B(){ std::cout << "~B\n"; } };   // 非虚析构
struct D : B { ~D(){ std::cout << "~D\n"; } };
int main(){
    B* p = new D{};
    delete p; // (1)
}
```
- 解析：
  - (1) 未定义行为风险：只调用 `~B`，`~D` 不被调用导致资源泄漏。应将基类析构函数设为 `virtual ~B(){}` 以确保通过基类指针删除派生对象时进行动态析构。

## 题11：RVO/NRVO 与可观察构造
```cpp
struct T {
    T(){ std::cout << "ctor\n"; }
    T(const T&){ std::cout << "copy\n"; }
    T(T&&) noexcept { std::cout << "move\n"; }
};
T make(){ return T{}; }       // 可能强制省略（C++17）
T make_nrvo(){ T t; return t; } // NRVO（可选优化）
int main(){
    T a = make();
    T b = make_nrvo();
}
```
- 解析：
  - 在 C++17 中，`a` 的构造常见仅打印 `ctor`（强制消除拷贝）；`b` 若触发 NRVO 也仅 `ctor`，否则可能 `ctor`+`move`。使用编译器选项（如 `-fno-elide-constructors`）可观察不同路径。

## 题12：多继承与指针调整
```cpp
struct A { virtual void fa(){} };
struct B { virtual void fb(){} };
struct C : A, B {};
int main(){
    C c; A* pa = &c; B* pb = &c; // (1)
}
```
- 解析：
  - (1) `&c` 转为 `A*`/`B*` 时编译器会插入不同的指针偏移以指向相应的基类子对象；`C` 对象可能包含多个 vptr。调用虚函数时将使用各自子对象的 vptr 进行分发。

