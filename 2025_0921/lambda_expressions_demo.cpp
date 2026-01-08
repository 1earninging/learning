#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <numeric>
#include <memory>

using namespace std;

// ================= 1. Lambda表达式基础语法 =================

void lambdaBasicSyntax() {
    cout << "=== 🎯 Lambda表达式基础语法 ===" << endl;
    
    // 1.1 最简单的lambda
    cout << "\n1️⃣ 最简单的lambda：" << endl;
    auto simpleLambda = []() {
        cout << "Hello Lambda!" << endl;
    };
    simpleLambda();  // 调用
    
    // 1.2 带参数的lambda
    cout << "\n2️⃣ 带参数的lambda：" << endl;
    auto addLambda = [](int a, int b) {
        return a + b;
    };
    cout << "5 + 3 = " << addLambda(5, 3) << endl;
    
    // 1.3 显式指定返回类型
    cout << "\n3️⃣ 指定返回类型：" << endl;
    auto divLambda = [](double a, double b) -> double {
        if (b != 0) return a / b;
        return 0.0;
    };
    cout << "10.0 / 3.0 = " << divLambda(10.0, 3.0) << endl;
    
    // 1.4 lambda语法结构详解
    cout << "\n📋 Lambda语法结构：" << endl;
    cout << "[capture](parameters) -> return_type { body }" << endl;
    cout << "- [capture]: 捕获子句" << endl;
    cout << "- (parameters): 参数列表（可省略）" << endl;
    cout << "- -> return_type: 返回类型（可省略）" << endl;
    cout << "- { body }: 函数体" << endl;
}

// ================= 2. 捕获机制详解 =================

void lambdaCaptureDemo() {
    cout << "\n=== 📦 Lambda捕获机制详解 ===" << endl;
    
    int x = 10;
    int y = 20;
    string message = "Hello";
    
    // 2.1 按值捕获
    cout << "\n1️⃣ 按值捕获 [=]：" << endl;
    auto captureByValue = [=]() {
        cout << "捕获的值: x=" << x << ", y=" << y << ", message=" << message << endl;
        // x = 100;  // 编译错误！按值捕获的变量是const的
    };
    captureByValue();
    
    x = 999;  // 修改原始值
    cout << "修改x后再次调用：" << endl;
    captureByValue();  // 输出的x还是10，因为是按值捕获
    
    // 2.2 按引用捕获
    cout << "\n2️⃣ 按引用捕获 [&]：" << endl;
    auto captureByReference = [&]() {
        cout << "捕获的引用: x=" << x << ", y=" << y << endl;
        x += 1;  // 可以修改原始变量
    };
    captureByReference();
    cout << "lambda修改后x的值: " << x << endl;
    
    // 2.3 混合捕获
    cout << "\n3️⃣ 混合捕获：" << endl;
    int a = 100, b = 200;
    auto mixedCapture = [x, &y, a](int param) {  // x按值，y按引用，a按值
        cout << "x=" << x << ", y=" << y << ", a=" << a << ", param=" << param << endl;
        y += 10;  // 可以修改y（按引用捕获）
        // x = 50;  // 编译错误！x是按值捕获的
    };
    mixedCapture(500);
    cout << "lambda修改后y的值: " << y << endl;
    
    // 2.4 不捕获任何变量
    cout << "\n4️⃣ 空捕获 []：" << endl;
    auto noCapture = [](int value) {
        return value * value;
    };
    cout << "5的平方: " << noCapture(5) << endl;
    
    // 2.5 this指针捕获（在类中使用）
    cout << "\n5️⃣ this指针捕获（类中使用）：" << endl;
    cout << "在类的成员函数中可以使用[this]或[=]捕获this指针" << endl;
}

// ================= 3. mutable Lambda =================

void mutableLambdaDemo() {
    cout << "\n=== 🔄 Mutable Lambda详解 ===" << endl;
    
    int counter = 0;
    
    // 3.1 普通按值捕获lambda（不能修改捕获的变量）
    cout << "\n1️⃣ 普通按值捕获（不可修改）：" << endl;
    auto normalLambda = [counter]() {
        cout << "counter = " << counter << endl;
        // counter++;  // 编译错误！按值捕获的变量是const的
    };
    normalLambda();
    
    // 3.2 mutable lambda（可以修改按值捕获的变量）
    cout << "\n2️⃣ Mutable lambda（可修改捕获值）：" << endl;
    auto mutableLambda = [counter](int increment) mutable {
        counter += increment;
        cout << "lambda内部counter = " << counter << endl;
        return counter;
    };
    
    cout << "调用前原始counter: " << counter << endl;
    mutableLambda(5);
    mutableLambda(3);
    cout << "调用后原始counter: " << counter << endl;  // 还是0，因为是按值捕获
    
    // 3.3 mutable的实际应用场景
    cout << "\n3️⃣ Mutable的实际应用：" << endl;
    auto idGenerator = [id = 0]() mutable {  // C++14初始化捕获
        return ++id;
    };
    
    cout << "生成ID: " << idGenerator() << ", " << idGenerator() << ", " << idGenerator() << endl;
}

// ================= 4. Lambda与STL算法结合 =================

void lambdaWithSTL() {
    cout << "\n=== 🧮 Lambda与STL算法结合使用 ===" << endl;
    
    vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    cout << "原始数据: ";
    for (int n : numbers) cout << n << " ";
    cout << endl;
    
    // 4.1 find_if - 查找第一个偶数
    cout << "\n1️⃣ find_if查找第一个偶数：" << endl;
    auto it = find_if(numbers.begin(), numbers.end(), [](int n) {
        return n % 2 == 0;
    });
    if (it != numbers.end()) {
        cout << "找到第一个偶数: " << *it << endl;
    }
    
    // 4.2 count_if - 统计偶数个数
    cout << "\n2️⃣ count_if统计偶数个数：" << endl;
    auto evenCount = count_if(numbers.begin(), numbers.end(), [](int n) {
        return n % 2 == 0;
    });
    cout << "偶数个数: " << evenCount << endl;
    
    // 4.3 transform - 数据转换
    cout << "\n3️⃣ transform数据转换：" << endl;
    vector<int> squares;
    transform(numbers.begin(), numbers.end(), back_inserter(squares), [](int n) {
        return n * n;
    });
    cout << "平方数: ";
    for (int n : squares) cout << n << " ";
    cout << endl;
    
    // 4.4 for_each - 遍历操作
    cout << "\n4️⃣ for_each遍历操作：" << endl;
    cout << "所有奇数: ";
    for_each(numbers.begin(), numbers.end(), [](int n) {
        if (n % 2 == 1) {
            cout << n << " ";
        }
    });
    cout << endl;
    
    // 4.5 sort - 自定义排序
    cout << "\n5️⃣ sort自定义排序：" << endl;
    vector<string> words = {"apple", "banana", "cherry", "date", "elderberry"};
    cout << "按长度排序前: ";
    for (const auto& word : words) cout << word << " ";
    cout << endl;
    
    sort(words.begin(), words.end(), [](const string& a, const string& b) {
        return a.length() < b.length();  // 按长度排序
    });
    
    cout << "按长度排序后: ";
    for (const auto& word : words) cout << word << " ";
    cout << endl;
    
    // 4.6 accumulate - 累积操作
    cout << "\n6️⃣ accumulate累积操作：" << endl;
    auto sum = accumulate(numbers.begin(), numbers.end(), 0, [](int acc, int n) {
        return acc + n * n;  // 计算平方和
    });
    cout << "平方和: " << sum << endl;
}

// ================= 5. Lambda vs 函数对象对比 =================

// 传统函数对象
class MultiplyFunctor {
private:
    int factor;
public:
    MultiplyFunctor(int f) : factor(f) {}
    int operator()(int x) const {
        return x * factor;
    }
};

void lambdaVsFunctionObject() {
    cout << "\n=== ⚔️ Lambda vs 函数对象对比 ===" << endl;
    
    int factor = 3;
    vector<int> data = {1, 2, 3, 4, 5};
    
    // 方法1：传统函数对象
    cout << "\n1️⃣ 传统函数对象：" << endl;
    vector<int> result1;
    transform(data.begin(), data.end(), back_inserter(result1), MultiplyFunctor(factor));
    cout << "函数对象结果: ";
    for (int n : result1) cout << n << " ";
    cout << endl;
    
    // 方法2：Lambda表达式
    cout << "\n2️⃣ Lambda表达式：" << endl;
    vector<int> result2;
    transform(data.begin(), data.end(), back_inserter(result2), [factor](int x) {
        return x * factor;
    });
    cout << "Lambda结果: ";
    for (int n : result2) cout << n << " ";
    cout << endl;
    
    // 对比分析
    cout << "\n📊 对比分析：" << endl;
    cout << "函数对象优势：" << endl;
    cout << "- 可以有复杂的状态管理" << endl;
    cout << "- 可以重用" << endl;
    cout << "- 类型明确" << endl;
    
    cout << "\nLambda优势：" << endl;
    cout << "- 代码简洁，就地定义" << endl;
    cout << "- 可以捕获局部变量" << endl;
    cout << "- 编译器优化更好" << endl;
    cout << "- 现代C++风格" << endl;
}

// ================= 6. 高级Lambda特性 =================

void advancedLambdaFeatures() {
    cout << "\n=== 🚀 高级Lambda特性 ===" << endl;
    
    // 6.1 泛型Lambda（C++14）
    cout << "\n1️⃣ 泛型Lambda（C++14）：" << endl;
    auto genericLambda = [](auto a, auto b) {
        return a + b;
    };
    cout << "整数相加: " << genericLambda(5, 3) << endl;
    cout << "浮点相加: " << genericLambda(2.5, 1.7) << endl;
    cout << "字符串相加: " << genericLambda(string("Hello"), string(" World")) << endl;
    
    // 6.2 初始化捕获（C++14）
    cout << "\n2️⃣ 初始化捕获（C++14）：" << endl;
    auto initCapture = [counter = 0](int increment) mutable {
        counter += increment;
        return counter;
    };
    cout << "计数器: " << initCapture(1) << ", " << initCapture(2) << ", " << initCapture(3) << endl;
    
    // 6.3 Lambda递归
    cout << "\n3️⃣ Lambda递归：" << endl;
    auto factorial = [](int n) {
        function<int(int)> fac = [&fac](int x) -> int {
            return x <= 1 ? 1 : x * fac(x - 1);
        };
        return fac(n);
    };
    cout << "5的阶乘: " << factorial(5) << endl;
    
    // 6.4 Lambda作为函数参数
    cout << "\n4️⃣ Lambda作为函数参数：" << endl;
    auto processVector = [](const vector<int>& vec, function<int(int)> processor) {
        vector<int> result;
        for (int value : vec) {
            result.push_back(processor(value));
        }
        return result;
    };
    
    vector<int> numbers = {1, 2, 3, 4, 5};
    auto doubled = processVector(numbers, [](int x) { return x * 2; });
    auto squared = processVector(numbers, [](int x) { return x * x; });
    
    cout << "原数组: ";
    for (int n : numbers) cout << n << " ";
    cout << "\n翻倍: ";
    for (int n : doubled) cout << n << " ";
    cout << "\n平方: ";
    for (int n : squared) cout << n << " ";
    cout << endl;
}

// ================= 7. 实际应用场景 =================

void practicalLambdaApplications() {
    cout << "\n=== 💼 Lambda实际应用场景 ===" << endl;
    
    // 7.1 事件处理回调
    cout << "\n1️⃣ 事件处理回调模拟：" << endl;
    auto eventHandler = [](const string& eventType, const string& data) {
        cout << "处理事件: " << eventType << ", 数据: " << data << endl;
    };
    
    eventHandler("click", "button1");
    eventHandler("keypress", "Enter");
    
    // 7.2 数据过滤和处理管道
    cout << "\n2️⃣ 数据处理管道：" << endl;
    vector<int> data = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10};
    
    // 过滤正数
    vector<int> positives;
    copy_if(data.begin(), data.end(), back_inserter(positives), [](int n) {
        return n > 0;
    });
    
    // 转换为平方
    vector<int> squares;
    transform(positives.begin(), positives.end(), back_inserter(squares), [](int n) {
        return n * n;
    });
    
    // 求和
    int sum = accumulate(squares.begin(), squares.end(), 0);
    
    cout << "原数据: ";
    for (int n : data) cout << n << " ";
    cout << "\n正数: ";
    for (int n : positives) cout << n << " ";
    cout << "\n平方: ";
    for (int n : squares) cout << n << " ";
    cout << "\n总和: " << sum << endl;
    
    // 7.3 自定义比较器
    cout << "\n3️⃣ 自定义比较器：" << endl;
    // 使用function<bool(const string&, const string&)>作为比较器类型
    map<string, int, function<bool(const string&, const string&)>> wordMap(
        [](const string& a, const string& b) {
            return a.length() < b.length();  // 按长度排序
        }
    );
    
    wordMap["hello"] = 5;
    wordMap["world"] = 5;
    wordMap["C++"] = 3;
    wordMap["programming"] = 11;
    
    cout << "按单词长度排序的map:" << endl;
    for (const auto& pair : wordMap) {
        cout << pair.first << ": " << pair.second << endl;
    }
    
    // 7.4 延迟计算
    cout << "\n4️⃣ 延迟计算：" << endl;
    auto lazyCalculation = [](bool shouldCalculate) {
        return [shouldCalculate](int x, int y) {
            if (shouldCalculate) {
                cout << "执行复杂计算..." << endl;
                return x * x + y * y;
            } else {
                cout << "跳过计算" << endl;
                return 0;
            }
        };
    };
    
    auto calc = lazyCalculation(true);
    cout << "结果: " << calc(3, 4) << endl;
}

// ================= 8. Lambda常见陷阱和注意事项 =================

void lambdaTipsAndTraps() {
    cout << "\n=== ⚠️ Lambda常见陷阱和注意事项 ===" << endl;
    
    // 8.1 悬空引用陷阱
    cout << "\n1️⃣ 悬空引用陷阱：" << endl;
    cout << "⚠️  危险：返回捕获局部变量引用的lambda" << endl;
    cout << "auto createDanglingLambda() {" << endl;
    cout << "    int local = 42;" << endl;
    cout << "    return [&local]() { return local; }; // 危险！" << endl;
    cout << "}" << endl;
    cout << "✅ 解决：按值捕获或确保变量生命周期" << endl;
    
    // 8.2 按值捕获的性能考虑
    cout << "\n2️⃣ 按值捕获的性能考虑：" << endl;
    cout << "对于大对象，按值捕获会导致拷贝开销" << endl;
    
    string largeString(1000, 'A');  // 大字符串
    
    // 不好的做法
    auto badLambda = [largeString](int x) {  // 拷贝大对象
        return largeString.length() + x;
    };
    
    // 好的做法
    auto goodLambda = [&largeString](int x) {  // 引用，无拷贝
        return largeString.length() + x;
    };
    
    cout << "✅ 对大对象使用引用捕获更高效" << endl;
    
    // 8.3 mutable的误解
    cout << "\n3️⃣ Mutable的常见误解：" << endl;
    int value = 10;
    auto mutableDemo = [value](int x) mutable {
        value += x;  // 只修改lambda内部的副本
        return value;
    };
    
    cout << "调用前value: " << value << endl;
    cout << "lambda返回: " << mutableDemo(5) << endl;
    cout << "调用后value: " << value << endl;  // 原值未改变
    cout << "⚠️  mutable只影响lambda内部的副本，不影响原变量" << endl;
    
    // 8.4 lambda与auto的配合
    cout << "\n4️⃣ Lambda与auto的最佳实践：" << endl;
    cout << "✅ 推荐：auto lambda = []() { /* ... */ };" << endl;
    cout << "✅ 推荐：function<int(int)> lambda = [](int x) { return x * 2; };" << endl;
    cout << "⚠️  注意：每个lambda都有唯一的类型，即使代码相同" << endl;
}

// ================= 主函数 =================

int main() {
    cout << "=== 🎭 C++11 Lambda表达式全面解析 ===" << endl;
    
    try {
        lambdaBasicSyntax();
        lambdaCaptureDemo();
        mutableLambdaDemo();
        lambdaWithSTL();
        lambdaVsFunctionObject();
        advancedLambdaFeatures();
        practicalLambdaApplications();
        lambdaTipsAndTraps();
        
        cout << "\n=== 📚 Lambda表达式学习总结 ===" << endl;
        cout << "🎯 基础语法: [capture](params) -> return_type { body }" << endl;
        cout << "🎯 捕获机制: 按值[=]、按引用[&]、混合捕获" << endl;
        cout << "🎯 STL结合: 完美配合算法库，简化代码" << endl;
        cout << "🎯 实际应用: 回调、事件处理、数据流水线" << endl;
        cout << "🎯 注意事项: 生命周期、性能、类型推导" << endl;
        cout << "🎯 现代特性: 泛型lambda、初始化捕获" << endl;
        
        cout << "\n✨ Lambda让C++代码更简洁、更现代、更强大！" << endl;
        
    } catch (const exception& e) {
        cout << "错误: " << e.what() << endl;
    }
    
    return 0;
}
