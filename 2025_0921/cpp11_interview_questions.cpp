#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <list>
#include <algorithm>

using namespace std;

// ================= 1. auto相关面试题 =================

void autoInterviewQuestions() {
    cout << "=== 🚗 auto相关面试题集合 ===" << endl;
    
    // 面试题1：auto推导规则陷阱
    cout << "\n📋 面试题1：下面代码的输出是什么？" << endl;
    cout << "---代码---" << endl;
    cout << "const int x = 10;" << endl;
    cout << "auto y = x;" << endl;
    cout << "y = 20;  // 这行代码合法吗？" << endl;
    cout << "---分析---" << endl;
    
    const int x = 10;
    auto y = x;        // y是int，不是const int！
    y = 20;           // 合法！因为auto忽略了顶层const
    cout << "✅ 合法！auto忽略顶层const，y是int类型" << endl;
    cout << "x = " << x << ", y = " << y << endl;
    
    // 面试题2：auto与引用的陷阱
    cout << "\n📋 面试题2：引用推导陷阱" << endl;
    cout << "---代码---" << endl;
    cout << "int a = 100;" << endl;
    cout << "int& ref = a;" << endl;
    cout << "auto b = ref;" << endl;
    cout << "b = 200;" << endl;
    cout << "cout << a;  // 输出什么？" << endl;
    cout << "---分析---" << endl;
    
    int a = 100;
    int& ref = a;
    auto b = ref;      // b是int，不是int&！
    b = 200;          // 只修改了b，不影响a
    cout << "✅ 输出100！auto忽略了引用，b是独立的拷贝" << endl;
    cout << "a = " << a << ", b = " << b << endl;
    
    // 面试题3：auto与数组退化
    cout << "\n📋 面试题3：数组类型推导" << endl;
    cout << "---代码---" << endl;
    cout << "int arr[5] = {1,2,3,4,5};" << endl;
    cout << "auto p = arr;" << endl;
    cout << "cout << sizeof(arr) << ' ' << sizeof(p);" << endl;
    cout << "---分析---" << endl;
    
    int arr[5] = {1,2,3,4,5};
    auto p = arr;      // p是int*，不是int[5]！
    cout << "✅ 数组退化为指针！" << endl;
    cout << "sizeof(arr) = " << sizeof(arr) << ", sizeof(p) = " << sizeof(p) << endl;
    
    // 面试题4：auto与初始化列表
    cout << "\n📋 面试题4：初始化列表推导" << endl;
    cout << "---代码---" << endl;
    cout << "auto list1 = {1, 2, 3};    // 推导为什么类型？" << endl;
    cout << "auto list2{1, 2, 3};       // 这个呢？" << endl;
    cout << "---分析---" << endl;
    
    auto list1 = {1, 2, 3};    // std::initializer_list<int>
    // auto list2{1, 2, 3};    // C++17前是initializer_list<int>，C++17后编译错误
    cout << "✅ list1是initializer_list<int>" << endl;
    cout << "✅ list2在不同C++标准下行为不同（陷阱！）" << endl;
    
    // 面试题5：auto在模板中的应用
    cout << "\n📋 面试题5：模板中的auto使用" << endl;
    cout << "---场景：写一个通用的容器大小检查函数---" << endl;
    
    auto checkSize = [](const auto& container) {
        cout << "容器大小: " << container.size() << endl;
        return container.size() > 0;
    };
    
    vector<int> vec = {1, 2, 3};
    string str = "hello";
    
    cout << "✅ 使用auto可以写出泛型lambda：" << endl;
    checkSize(vec);
    checkSize(str);
}

// ================= 2. decltype相关面试题 =================

void decltypeInterviewQuestions() {
    cout << "\n=== 🔍 decltype相关面试题集合 ===" << endl;
    
    // 面试题6：decltype与表达式类型
    cout << "\n📋 面试题6：表达式类型推导" << endl;
    cout << "---代码---" << endl;
    cout << "int x = 10;" << endl;
    cout << "decltype(x) a;      // 什么类型？" << endl;
    cout << "decltype((x)) b;    // 什么类型？" << endl;
    cout << "---分析---" << endl;
    
    int x = 10;
    decltype(x) a = 0;        // int
    decltype((x)) b = x;      // int& ！注意括号的影响
    
    a = 100;  // 不影响x
    b = 200;  // 影响x
    cout << "✅ decltype(x) = int, decltype((x)) = int&" << endl;
    cout << "✅ 括号会影响decltype的结果！" << endl;
    cout << "x = " << x << ", a = " << a << endl;
    
    // 面试题7：decltype与函数调用
    cout << "\n📋 面试题7：函数调用表达式" << endl;
    cout << "---代码---" << endl;
    cout << "int func() { return 42; }" << endl;
    cout << "decltype(func()) result = func();" << endl;
    cout << "---分析---" << endl;
    
    auto func = []() -> int { return 42; };
    decltype(func()) result = func();  // int
    cout << "✅ decltype(func())获取函数返回类型" << endl;
    cout << "result = " << result << endl;
    
    // 面试题8：decltype(auto)的使用
    cout << "\n📋 面试题8：decltype(auto)应用" << endl;
    cout << "---场景：完美转发返回类型---" << endl;
    
    auto getValue = [](bool flag) -> int& {
        static int value = 100;
        return value;
    };
    
    auto forwardCall1 = [&](bool flag) -> decltype(auto) {
        return getValue(flag);  // 完美转发返回类型
    };
    
    auto forwardCall2 = [&](bool flag) -> auto {
        return getValue(flag);  // 返回值类型，丢失引用
    };
    
    cout << "✅ decltype(auto)保持返回类型的完整性" << endl;
    int& ref1 = forwardCall1(true);   // OK，返回引用
    int val2 = forwardCall2(true);    // 返回值，不是引用
    
    ref1 = 999;  // 会修改原始value
    cout << "通过decltype(auto)修改后的值: " << getValue(true) << endl;
    
    // 面试题9：decltype与重载函数
    cout << "\n📋 面试题9：重载函数类型推导" << endl;
    cout << "---陷阱：decltype不能直接用于重载函数---" << endl;
    
    auto add1 = [](int a, int b) { return a + b; };
    auto add2 = [](double a, double b) { return a + b; };
    
    // decltype(add1)不能推导重载函数，但lambda可以
    cout << "✅ 需要通过函数调用或具体上下文来推导类型" << endl;
}

// ================= 3. 范围for循环相关面试题 =================

void rangeForInterviewQuestions() {
    cout << "\n=== 🔄 范围for循环相关面试题集合 ===" << endl;
    
    // 面试题10：迭代器失效陷阱
    cout << "\n📋 面试题10：迭代器失效陷阱（重要！）" << endl;
    cout << "---危险代码---" << endl;
    cout << "vector<int> vec = {1, 2, 3};" << endl;
    cout << "for (auto& element : vec) {" << endl;
    cout << "    vec.push_back(element * 2);  // 危险！" << endl;
    cout << "}" << endl;
    cout << "---分析---" << endl;
    
    cout << "❌ 这会导致未定义行为！" << endl;
    cout << "✅ 原因：范围for循环内修改容器大小会使迭代器失效" << endl;
    cout << "✅ 解决方案：使用传统for循环或者先收集要添加的元素" << endl;
    
    // 正确的做法
    vector<int> vec = {1, 2, 3};
    vector<int> toAdd;
    for (const auto& element : vec) {
        toAdd.push_back(element * 2);
    }
    vec.insert(vec.end(), toAdd.begin(), toAdd.end());
    
    cout << "正确修改后的容器: ";
    for (int val : vec) cout << val << " ";
    cout << endl;
    
    // 面试题11：性能陷阱
    cout << "\n📋 面试题11：性能陷阱对比" << endl;
    cout << "---代码对比---" << endl;
    
    vector<string> words = {"hello", "world", "C++", "programming", "language"};
    
    cout << "方法1（性能差）：" << endl;
    cout << "for (string word : words) { /* 每次拷贝string */ }" << endl;
    
    cout << "方法2（性能好）：" << endl;  
    cout << "for (const string& word : words) { /* 引用，无拷贝 */ }" << endl;
    
    cout << "方法3（自动推导）：" << endl;
    cout << "for (const auto& word : words) { /* 推荐写法 */ }" << endl;
    
    // 面试题12：临时对象的生命周期
    cout << "\n📋 面试题12：临时对象陷阱" << endl;
    cout << "---危险代码---" << endl;
    cout << "for (const auto& element : getVector()) {" << endl;
    cout << "    // getVector()返回临时对象" << endl;
    cout << "}" << endl;
    cout << "---分析---" << endl;
    
    auto getVector = []() {
        return vector<int>{1, 2, 3, 4, 5};
    };
    
    cout << "✅ C++11保证临时对象在范围for循环中的生命周期" << endl;
    cout << "✅ 但要注意返回引用的情况可能有陷阱" << endl;
    
    for (const auto& element : getVector()) {
        cout << element << " ";
    }
    cout << endl;
    
    // 面试题13：自定义类型的范围for支持
    cout << "\n📋 面试题13：如何让自定义类支持范围for？" << endl;
    
    class MyRange {
        vector<int> data;
    public:
        MyRange() : data{1, 2, 3, 4, 5} {}
        
        // 需要提供begin()和end()方法
        auto begin() { return data.begin(); }
        auto end() { return data.end(); }
        auto begin() const { return data.begin(); }
        auto end() const { return data.end(); }
    };
    
    MyRange range;
    cout << "自定义类型的范围for循环: ";
    for (auto value : range) {
        cout << value << " ";
    }
    cout << endl;
    
    cout << "✅ 需要实现begin()和end()方法" << endl;
}

// ================= 4. 综合性面试题 =================

void combinedInterviewQuestions() {
    cout << "\n=== 🎯 综合性面试题 ===" << endl;
    
    // 面试题14：三者结合的复杂场景
    cout << "\n📋 面试题14：写一个泛型函数，统计容器中满足条件的元素个数" << endl;
    cout << "要求：使用auto, decltype, 范围for循环" << endl;
    
    auto countIf = [](const auto& container, auto predicate) -> decltype(container.size()) {
        decltype(container.size()) count = 0;
        for (const auto& element : container) {
            if (predicate(element)) {
                ++count;
            }
        }
        return count;
    };
    
    vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto evenCount = countIf(numbers, [](int n) { return n % 2 == 0; });
    
    vector<string> words = {"hello", "world", "C++", "auto", "decltype"};
    auto longWordCount = countIf(words, [](const string& word) { return word.length() > 4; });
    
    cout << "✅ 偶数个数: " << evenCount << endl;
    cout << "✅ 长单词个数: " << longWordCount << endl;
    
    // 面试题15：类型推导的边界情况
    cout << "\n📋 面试题15：这些声明都合法吗？" << endl;
    
    cout << "auto x;                    // ❌ 编译错误：must be initialized" << endl;
    cout << "auto y = {1, 2};          // ✅ initializer_list<int>" << endl;  
    cout << "auto z = {1, 2.0};        // ❌ 编译错误：mixed types" << endl;
    
    // auto x;                    // 编译错误
    auto y = {1, 2};          // OK
    // auto z = {1, 2.0};        // 编译错误
    
    cout << "✅ auto必须初始化，初始化列表类型必须一致" << endl;
}

// ================= 5. 实际编程题 =================

void practicalCodingQuestions() {
    cout << "\n=== 💻 实际编程题 ===" << endl;
    
    cout << "\n📋 编程题1：实现一个通用的查找函数" << endl;
    cout << "要求：返回第一个满足条件的元素的迭代器" << endl;
    
    auto findIf = [](auto& container, auto predicate) -> decltype(container.begin()) {
        for (auto it = container.begin(); it != container.end(); ++it) {
            if (predicate(*it)) {
                return it;
            }
        }
        return container.end();
    };
    
    vector<int> nums = {1, 3, 5, 8, 9, 12};
    auto it = findIf(nums, [](int n) { return n % 2 == 0; });
    
    if (it != nums.end()) {
        cout << "✅ 找到第一个偶数: " << *it << endl;
    }
    
    cout << "\n📋 编程题2：实现一个类型安全的打印函数" << endl;
    cout << "要求：能打印任何支持范围for循环的容器" << endl;
    
    auto safePrint = [](const auto& container) {
        cout << "容器内容: [";
        bool first = true;
        for (const auto& element : container) {
            if (!first) cout << ", ";
            cout << element;
            first = false;
        }
        cout << "]" << endl;
    };
    
    vector<int> intVec = {1, 2, 3, 4};
    list<string> stringList = {"hello", "world", "C++"};
    
    safePrint(intVec);
    safePrint(stringList);
}

// ================= 6. 面试官最爱问的陷阱题 =================

void interviewerFavoriteTraps() {
    cout << "\n=== 🕳️  面试官最爱的陷阱题 ===" << endl;
    
    cout << "\n📋 陷阱题1：auto与万能引用" << endl;
    cout << "template<typename T>" << endl;
    cout << "void func(T&& param) {" << endl;
    cout << "    auto local = param;" << endl;
    cout << "}" << endl;
    cout << "✅ auto总是按值拷贝，即使param是引用" << endl;
    
    cout << "\n📋 陷阱题2：decltype与成员变量" << endl;
    cout << "struct S { int x; };" << endl;
    cout << "S obj;" << endl;
    cout << "decltype(S::x) a;        // int" << endl;
    cout << "decltype(obj.x) b;       // int" << endl;
    cout << "decltype((obj.x)) c;     // int& ！" << endl;
    cout << "✅ 成员访问表达式加括号会变成引用" << endl;
    
    cout << "\n📋 陷阱题3：范围for与const容器" << endl;
    cout << "const vector<int> vec = {1,2,3};" << endl;
    cout << "for (auto& x : vec) {    // 编译错误！" << endl;
    cout << "    x = 10;" << endl;
    cout << "}" << endl;
    cout << "✅ const容器的元素也是const，不能用非const引用" << endl;
    
    cout << "\n📋 陷阱题4：auto与数组参数" << endl;
    cout << "void func(int arr[10]) {" << endl;
    cout << "    auto x = arr;        // x是int*，不是int[10]" << endl;
    cout << "}" << endl;
    cout << "✅ 数组参数实际是指针，auto推导为指针类型" << endl;
}

int main() {
    cout << "=== 🎓 C++11三特性深度面试题集合 ===" << endl;
    
    autoInterviewQuestions();
    decltypeInterviewQuestions();
    rangeForInterviewQuestions();
    combinedInterviewQuestions();
    practicalCodingQuestions();
    interviewerFavoriteTraps();
    
    cout << "\n=== 📚 面试准备总结 ===" << endl;
    cout << "🎯 重点掌握：auto的推导规则和限制" << endl;
    cout << "🎯 重点掌握：decltype与auto的差异" << endl;
    cout << "🎯 重点掌握：范围for的性能考虑和陷阱" << endl;
    cout << "🎯 重点掌握：三者结合的实际应用场景" << endl;
    cout << "🎯 重点掌握：各种边界情况和陷阱题" << endl;
    
    return 0;
}



