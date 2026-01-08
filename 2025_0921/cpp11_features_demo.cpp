#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>

using namespace std;

// ================= 1. auto 关键字详解 =================

void demonstrateAuto() {
    cout << "\n=== 🚗 auto 关键字详解 ===" << endl;
    
    // 1.1 基本类型推导
    cout << "\n1️⃣ 基本类型推导：" << endl;
    auto a = 42;              // int
    auto b = 3.14;            // double  
    auto c = 3.14f;           // float
    auto d = 'A';             // char
    auto e = "hello";         // const char*
    auto f = string("world"); // string
    auto g = true;            // bool
    
    cout << "auto a = 42;           类型: " << typeid(a).name() << endl;
    cout << "auto b = 3.14;         类型: " << typeid(b).name() << endl;
    cout << "auto c = 3.14f;        类型: " << typeid(c).name() << endl;
    
    // 1.2 复杂类型推导
    cout << "\n2️⃣ 复杂类型推导：" << endl;
    vector<int> vec = {1, 2, 3, 4, 5};
    auto it = vec.begin();    // vector<int>::iterator
    
    map<string, int> myMap = {{"apple", 5}, {"banana", 3}};
    auto mapIt = myMap.begin(); // map<string, int>::iterator
    
    cout << "迭代器类型自动推导，无需写复杂的类型声明" << endl;
    
    // 1.3 函数返回类型推导
    cout << "\n3️⃣ 函数返回类型推导：" << endl;
    auto func = []() { return 42; };  // lambda表达式
    auto result = func();             // int
    
    cout << "Lambda 返回值: " << result << endl;
    
    // 1.4 智能指针
    cout << "\n4️⃣ 智能指针简化：" << endl;
    auto ptr = make_unique<string>("智能指针");  // unique_ptr<string>
    auto sharedPtr = make_shared<int>(100);      // shared_ptr<int>
    
    cout << "智能指针内容: " << *ptr << ", " << *sharedPtr << endl;
    
    // 1.5 auto的限制和注意事项
    cout << "\n5️⃣ auto的重要规则：" << endl;
    
    // auto会忽略顶层const
    const int x = 10;
    auto y = x;        // y是int，不是const int
    // y = 20;         // 这是合法的！
    
    // 如果想保持const，需要显式指定
    const auto z = x;  // z是const int
    
    cout << "auto会忽略顶层const，需要注意" << endl;
    
    // auto不能推导引用类型
    int original = 100;
    auto copy = original;      // copy是int，不是引用
    auto& ref = original;      // 需要显式指定&才是引用
    
    copy = 200;    // 不影响original
    ref = 300;     // 影响original
    cout << "original = " << original << ", copy = " << copy << endl;
}

// ================= 2. decltype 关键字详解 =================

void demonstrateDecltype() {
    cout << "\n=== 🔍 decltype 关键字详解 ===" << endl;
    
    // 2.1 基本用法 - 获取表达式的类型
    cout << "\n1️⃣ 基本类型获取：" << endl;
    int a = 42;
    double b = 3.14;
    
    decltype(a) x = 100;       // x的类型是int
    decltype(b) y = 2.718;     // y的类型是double
    decltype(a + b) z = a + b; // z的类型是double（int + double = double）
    
    cout << "decltype(a) x = 100;        x = " << x << endl;
    cout << "decltype(a + b) z = a + b;  z = " << z << endl;
    
    // 2.2 与auto的区别
    cout << "\n2️⃣ decltype vs auto 对比：" << endl;
    const int constValue = 50;
    
    auto autoVar = constValue;        // int（忽略const）
    decltype(constValue) decltypeVar = constValue; // const int（保持const）
    
    // autoVar = 60;           // 合法
    // decltypeVar = 60;       // 编译错误！const不能修改
    
    cout << "auto会忽略const，decltype会保持原始类型" << endl;
    
    // 2.3 引用类型推导
    cout << "\n3️⃣ 引用类型推导：" << endl;
    int original = 100;
    int& ref = original;
    
    auto autoFromRef = ref;        // int（忽略引用）
    decltype(ref) decltypeRef = original; // int&（保持引用）
    
    autoFromRef = 200;    // 不影响original
    decltypeRef = 300;    // 影响original
    cout << "original = " << original << endl;
    
    // 2.4 函数返回类型推导
    cout << "\n4️⃣ 函数返回类型推导：" << endl;
    auto func1 = [](int x) { return x * 2; };
    
    // 使用decltype获取函数返回类型
    decltype(func1(5)) result = func1(10);  // int
    cout << "函数返回值: " << result << endl;
}

// ================= 3. 范围for循环详解 =================

void demonstrateRangeFor() {
    cout << "\n=== 🔄 范围for循环详解 ===" << endl;
    
    // 3.1 基本语法
    cout << "\n1️⃣ 基本语法演示：" << endl;
    vector<int> numbers = {1, 2, 3, 4, 5};
    
    cout << "传统for循环: ";
    for (size_t i = 0; i < numbers.size(); ++i) {
        cout << numbers[i] << " ";
    }
    cout << endl;
    
    cout << "范围for循环: ";
    for (int num : numbers) {  // 按值拷贝
        cout << num << " ";
    }
    cout << endl;
    
    // 3.2 引用版本（避免拷贝）
    cout << "\n2️⃣ 引用版本（性能优化）：" << endl;
    vector<string> words = {"hello", "world", "C++", "programming"};
    
    cout << "按值拷贝（可能慢）: ";
    for (string word : words) {  // 每次都拷贝string对象
        cout << word << " ";
    }
    cout << endl;
    
    cout << "按引用（高效）: ";
    for (const string& word : words) {  // 不拷贝，只是引用
        cout << word << " ";
    }
    cout << endl;
    
    // 3.3 修改元素
    cout << "\n3️⃣ 修改容器元素：" << endl;
    vector<int> values = {1, 2, 3, 4, 5};
    
    cout << "修改前: ";
    for (int val : values) {
        cout << val << " ";
    }
    cout << endl;
    
    // 使用非const引用来修改元素
    for (int& val : values) {  // 注意：必须是非const引用
        val *= 2;
    }
    
    cout << "修改后: ";
    for (int val : values) {
        cout << val << " ";
    }
    cout << endl;
    
    // 3.4 map容器的范围for循环
    cout << "\n4️⃣ map容器的范围for循环：" << endl;
    map<string, int> scores = {
        {"Alice", 95},
        {"Bob", 87},
        {"Charlie", 92}
    };
    
    cout << "学生成绩单：" << endl;
    for (const auto& pair : scores) {  // auto自动推导为pair<const string, int>
        cout << pair.first << ": " << pair.second << "分" << endl;
    }
    
    // C++17 结构化绑定（更简洁）
    #if __cplusplus >= 201703L
    cout << "\nC++17 结构化绑定版本：" << endl;
    for (const auto& [name, score] : scores) {
        cout << name << ": " << score << "分" << endl;
    }
    #endif
    
    // 3.5 数组的范围for循环
    cout << "\n5️⃣ 数组的范围for循环：" << endl;
    int arr[] = {10, 20, 30, 40, 50};
    
    cout << "数组元素: ";
    for (int element : arr) {
        cout << element << " ";
    }
    cout << endl;
    
    // 3.6 自定义类型的范围for循环
    cout << "\n6️⃣ 初始化列表的范围for循环：" << endl;
    cout << "直接遍历初始化列表: ";
    for (int val : {100, 200, 300, 400}) {
        cout << val << " ";
    }
    cout << endl;
}

// ================= 4. 三者结合的实际应用 =================

void demonstrateCombinedUsage() {
    cout << "\n=== 🎯 三者结合的实际应用 ===" << endl;
    
    // 4.1 复杂容器的遍历
    cout << "\n1️⃣ 复杂容器遍历：" << endl;
    map<string, vector<int>> studentGrades = {
        {"Alice", {95, 87, 92}},
        {"Bob", {78, 82, 85}},
        {"Charlie", {88, 91, 94}}
    };
    
    cout << "学生所有成绩：" << endl;
    for (const auto& student : studentGrades) {  // auto + 范围for
        cout << student.first << "的成绩: ";
        for (auto grade : student.second) {      // auto + 范围for
            cout << grade << " ";
        }
        cout << endl;
    }
    
    // 4.2 算法库结合使用
    cout << "\n2️⃣ 与算法库结合：" << endl;
    vector<int> data = {5, 2, 8, 1, 9, 3};
    
    // 使用auto来存储lambda表达式
    auto isEven = [](int n) { return n % 2 == 0; };
    
    // 使用decltype获取迭代器类型
    auto evenIt = find_if(data.begin(), data.end(), isEven);
    
    if (evenIt != data.end()) {
        cout << "找到的第一个偶数: " << *evenIt << endl;
    }
    
    // 4.3 函数模板中的应用
    cout << "\n3️⃣ 泛型编程应用：" << endl;
    
    auto printContainer = [](const auto& container) {  // 泛型lambda
        cout << "容器内容: ";
        for (const auto& element : container) {  // 范围for + auto
            cout << element << " ";
        }
        cout << endl;
    };
    
    vector<int> intVec = {1, 2, 3};
    vector<string> stringVec = {"a", "b", "c"};
    
    printContainer(intVec);
    printContainer(stringVec);
}

// ================= 5. 最佳实践和注意事项 =================

void demonstrateBestPractices() {
    cout << "\n=== 💡 最佳实践和注意事项 ===" << endl;
    
    cout << "\n1️⃣ auto 最佳实践：" << endl;
    cout << "✅ 适合使用auto的场景：" << endl;
    cout << "   - 复杂的迭代器类型" << endl;
    cout << "   - 智能指针" << endl;
    cout << "   - Lambda表达式" << endl;
    cout << "   - 模板函数的返回类型" << endl;
    
    cout << "\n❌ 不建议使用auto的场景：" << endl;
    cout << "   - 简单的基本类型（可读性考虑）" << endl;
    cout << "   - 需要明确类型转换的场合" << endl;
    cout << "   - 接口函数的参数和返回值" << endl;
    
    cout << "\n2️⃣ 范围for循环最佳实践：" << endl;
    cout << "✅ 只读遍历：使用 const auto&" << endl;
    cout << "✅ 修改元素：使用 auto&" << endl;
    cout << "✅ 简单类型：可以使用auto（按值）" << endl;
    cout << "❌ 避免：在循环中修改容器结构" << endl;
    
    // 演示错误用法
    cout << "\n⚠️  常见错误示例：" << endl;
    vector<int> vec = {1, 2, 3, 4, 5};
    
    cout << "错误：在范围for中修改容器大小会导致未定义行为" << endl;
    // for (auto& element : vec) {
    //     vec.push_back(element * 2);  // 危险！可能导致崩溃
    // }
}

// ================= 主函数 =================

int main() {
    cout << "=== 🎉 C++11 现代特性详解 ===" << endl;
    
    try {
        demonstrateAuto();
        demonstrateDecltype();  
        demonstrateRangeFor();
        demonstrateCombinedUsage();
        demonstrateBestPractices();
        
        cout << "\n=== 📚 学习总结 ===" << endl;
        cout << "🚗 auto: 自动类型推导，简化代码" << endl;
        cout << "🔍 decltype: 获取表达式类型，保持原始类型特性" << endl;  
        cout << "🔄 范围for: 简化容器遍历，提高代码可读性" << endl;
        cout << "🎯 三者结合: 让C++代码更现代、更简洁、更安全" << endl;
        
    } catch (const exception& e) {
        cout << "错误: " << e.what() << endl;
    }
    
    return 0;
}



