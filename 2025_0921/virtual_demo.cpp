#include <iostream>
#include <vector>
#include <memory>
using namespace std;

// ================= 演示虚函数和纯虚函数的区别 =================

// 1. 包含纯虚函数的抽象基类
class Shape {
public:
    Shape(const string& name) : name_(name) {}
    
    // 虚函数 - 提供默认实现，子类可以选择重写或使用默认实现
    virtual void display() const {
        cout << "这是一个 " << name_ << " 形状" << endl;
    }
    
    // 纯虚函数 - 强制子类必须实现
    virtual double getArea() const = 0;
    virtual double getPerimeter() const = 0;
    
    // 虚析构函数 - 确保正确的多态析构
    virtual ~Shape() = default;
    
protected:
    string name_;
};

// 2. 具体实现类 - Rectangle
class Rectangle : public Shape {
private:
    double width_, height_;
    
public:
    Rectangle(double w, double h) : Shape("矩形"), width_(w), height_(h) {}
    
    // 重写虚函数 (可选)
    void display() const override {
        cout << "这是一个宽=" << width_ << ", 高=" << height_ << "的矩形" << endl;
    }
    
    // 必须实现纯虚函数
    double getArea() const override {
        return width_ * height_;
    }
    
    double getPerimeter() const override {
        return 2 * (width_ + height_);
    }
};

// 3. 具体实现类 - Circle
class Circle : public Shape {
private:
    double radius_;
    static constexpr double PI = 3.14159;
    
public:
    Circle(double r) : Shape("圆形"), radius_(r) {}
    
    // 选择不重写display()，使用基类的默认实现
    
    // 必须实现纯虚函数
    double getArea() const override {
        return PI * radius_ * radius_;
    }
    
    double getPerimeter() const override {
        return 2 * PI * radius_;
    }
};

// 4. 演示不完整的实现类（故意注释掉一个纯虚函数的实现）
/*
class IncompleteShape : public Shape {
public:
    IncompleteShape() : Shape("不完整形状") {}
    
    // 只实现了一个纯虚函数，缺少另一个
    double getArea() const override {
        return 0.0;
    }
    
    // 注释掉这个实现，编译器会报错
    // double getPerimeter() const override { return 0.0; }
};
*/

// ================= 演示函数 =================
void demonstrateVirtualFunctions() {
    cout << "=== 虚函数 vs 纯虚函数演示 ===" << endl;
    
    // 1. 不能实例化抽象类
    cout << "\n1. 抽象类不能被实例化：" << endl;
    // Shape shape("test");  // 编译错误！抽象类不能实例化
    cout << "Shape是抽象类，包含纯虚函数，无法直接创建对象" << endl;
    
    // 2. 创建具体实现类的对象
    cout << "\n2. 创建具体实现类对象：" << endl;
    auto rect = make_unique<Rectangle>(5.0, 3.0);
    auto circle = make_unique<Circle>(2.0);
    
    // 3. 虚函数的多态行为
    cout << "\n3. 虚函数的多态行为：" << endl;
    vector<unique_ptr<Shape>> shapes;
    shapes.push_back(make_unique<Rectangle>(4.0, 6.0));
    shapes.push_back(make_unique<Circle>(3.0));
    
    for (const auto& shape : shapes) {
        // 调用虚函数 - 可能使用基类实现或派生类重写
        shape->display();
        
        // 调用纯虚函数 - 必须是派生类的实现
        cout << "面积: " << shape->getArea() << endl;
        cout << "周长: " << shape->getPerimeter() << endl;
        cout << "---" << endl;
    }
}

// ================= 另一个例子：接口设计模式 =================
class Logger {
public:
    // 纯虚函数定义接口
    virtual void log(const string& message) = 0;
    virtual ~Logger() = default;
};

class FileLogger : public Logger {
public:
    void log(const string& message) override {
        cout << "[文件日志] " << message << endl;
    }
};

class ConsoleLogger : public Logger {
public:
    void log(const string& message) override {
        cout << "[控制台日志] " << message << endl;
    }
};

void demonstrateInterfacePattern() {
    cout << "\n=== 接口设计模式（纯虚函数的应用） ===" << endl;
    
    vector<unique_ptr<Logger>> loggers;
    loggers.push_back(make_unique<FileLogger>());
    loggers.push_back(make_unique<ConsoleLogger>());
    
    for (const auto& logger : loggers) {
        logger->log("这是一条测试日志");
    }
}

// ================= 主函数 =================
int main() {
    try {
        demonstrateVirtualFunctions();
        demonstrateInterfacePattern();
        
        cout << "\n=== 总结 ===" << endl;
        cout << "1. 虚函数可以有默认实现，子类可选择重写" << endl;
        cout << "2. 纯虚函数没有实现，子类必须重写" << endl;
        cout << "3. 包含纯虚函数的类是抽象类，不能实例化" << endl;
        cout << "4. 纯虚函数常用于定义接口" << endl;
        
    } catch (const exception& e) {
        cout << "错误: " << e.what() << endl;
    }
    
    return 0;
}
