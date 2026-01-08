#include <iostream>
#include <string>
#include <vector>
#include <memory>

/* 
=================================================================================
                            面向对象核心概念详解
=================================================================================
1. 类与对象 (Class & Object)
2. 封装 (Encapsulation) 
3. 继承 (Inheritance)
4. 多态 (Polymorphism)
=================================================================================
*/

// =================================================================================
// 1. 类与对象基础概念
// =================================================================================

class Student {
private:
    std::string name;
    int age;
    double score;
    
public:
    // 默认构造函数
    Student() : name("Unknown"), age(0), score(0.0) {
        std::cout << "Student默认构造函数被调用\n";
    }
    
    // 带参构造函数
    Student(const std::string& n, int a, double s) : name(n), age(a), score(s) {
        std::cout << "Student带参构造函数被调用: " << name << "\n";
    }
    
    // 拷贝构造函数
    Student(const Student& other) : name(other.name), age(other.age), score(other.score) {
        std::cout << "Student拷贝构造函数被调用: " << name << "\n";
    }
    
    // 析构函数
    ~Student() {
        std::cout << "Student析构函数被调用: " << name << "\n";
    }
    
    // 拷贝赋值运算符
    Student& operator=(const Student& other) {
        if (this != &other) {
            name = other.name;
            age = other.age;
            score = other.score;
            std::cout << "Student拷贝赋值运算符被调用: " << name << "\n";
        }
        return *this;
    }
    
    // 公共接口方法
    void display() const {
        std::cout << "姓名: " << name << ", 年龄: " << age << ", 分数: " << score << std::endl;
    }
    
    // Getter和Setter方法（体现封装）
    std::string getName() const { return name; }
    int getAge() const { return age; }
    double getScore() const { return score; }
    
    void setName(const std::string& n) { name = n; }
    void setAge(int a) { 
        if (a >= 0 && a <= 120) {  // 数据验证
            age = a; 
        } else {
            std::cout << "年龄设置无效: " << a << std::endl;
        }
    }
    void setScore(double s) { 
        if (s >= 0.0 && s <= 100.0) {  // 数据验证
            score = s; 
        } else {
            std::cout << "分数设置无效: " << s << std::endl;
        }
    }
};

// =================================================================================
// 2. 封装 (Encapsulation) - 访问控制和数据隐藏
// =================================================================================

class BankAccount {
private:
    std::string accountNumber;
    std::string ownerName;
    double balance;
    std::string pin;  // 私有数据，外部无法直接访问
    
    // 私有方法，内部验证逻辑
    bool validatePin(const std::string& inputPin) const {
        return pin == inputPin;
    }
    
public:
    BankAccount(const std::string& accNum, const std::string& owner, 
                double initialBalance, const std::string& accountPin) 
        : accountNumber(accNum), ownerName(owner), balance(initialBalance), pin(accountPin) {}
    
    // 受控的访问方法
    bool deposit(double amount, const std::string& inputPin) {
        if (!validatePin(inputPin)) {
            std::cout << "PIN验证失败！\n";
            return false;
        }
        if (amount > 0) {
            balance += amount;
            std::cout << "存款成功: +" << amount << ", 余额: " << balance << std::endl;
            return true;
        }
        std::cout << "存款金额必须大于0！\n";
        return false;
    }
    
    bool withdraw(double amount, const std::string& inputPin) {
        if (!validatePin(inputPin)) {
            std::cout << "PIN验证失败！\n";
            return false;
        }
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            std::cout << "取款成功: -" << amount << ", 余额: " << balance << std::endl;
            return true;
        }
        std::cout << "取款失败：金额无效或余额不足！\n";
        return false;
    }
    
    double getBalance(const std::string& inputPin) const {
        if (validatePin(inputPin)) {
            return balance;
        }
        std::cout << "PIN验证失败，无法查询余额！\n";
        return -1;
    }
    
    std::string getAccountInfo() const {
        return "账号: " + accountNumber + ", 户主: " + ownerName;
    }
};

// =================================================================================
// 3. 继承 (Inheritance) - 代码复用和is-a关系
// =================================================================================

// 基类 - Animal
class Animal {
protected:  // 受保护成员，派生类可以访问
    std::string name;
    int age;
    
public:
    Animal(const std::string& n, int a) : name(n), age(a) {
        std::cout << "Animal构造函数: " << name << std::endl;
    }
    
    virtual ~Animal() {  // 虚析构函数，支持多态销毁
        std::cout << "Animal析构函数: " << name << std::endl;
    }
    
    // 虚函数，可被派生类重写
    virtual void makeSound() const {
        std::cout << name << " makes some sound..." << std::endl;
    }
    
    virtual void move() const {
        std::cout << name << " is moving..." << std::endl;
    }
    
    // 普通成员函数
    void showInfo() const {
        std::cout << "动物信息 - 姓名: " << name << ", 年龄: " << age << std::endl;
    }
    
    // 纯虚函数，使Animal成为抽象类（如果有的话）
    // virtual void eat() const = 0;
};

// 派生类 - Dog
class Dog : public Animal {  // public继承
private:
    std::string breed;
    
public:
    Dog(const std::string& n, int a, const std::string& b) 
        : Animal(n, a), breed(b) {  // 调用基类构造函数
        std::cout << "Dog构造函数: " << name << " (" << breed << ")" << std::endl;
    }
    
    ~Dog() {
        std::cout << "Dog析构函数: " << name << std::endl;
    }
    
    // 重写基类虚函数
    void makeSound() const override {
        std::cout << name << " says: Woof! Woof!" << std::endl;
    }
    
    void move() const override {
        std::cout << name << " is running on four legs!" << std::endl;
    }
    
    // Dog特有的方法
    void wagTail() const {
        std::cout << name << " is wagging its tail happily!" << std::endl;
    }
    
    std::string getBreed() const { return breed; }
};

// 派生类 - Cat
class Cat : public Animal {
private:
    bool isIndoor;
    
public:
    Cat(const std::string& n, int a, bool indoor) 
        : Animal(n, a), isIndoor(indoor) {
        std::cout << "Cat构造函数: " << name << std::endl;
    }
    
    ~Cat() {
        std::cout << "Cat析构函数: " << name << std::endl;
    }
    
    void makeSound() const override {
        std::cout << name << " says: Meow! Meow!" << std::endl;
    }
    
    void move() const override {
        std::cout << name << " is stealthily moving like a cat!" << std::endl;
    }
    
    // Cat特有的方法
    void climb() const {
        std::cout << name << " is climbing!" << std::endl;
    }
    
    bool isIndoorCat() const { return isIndoor; }
};

// =================================================================================
// 4. 多态 (Polymorphism) - 动态绑定和虚函数机制
// =================================================================================

// 抽象基类 - Shape
class Shape {
protected:
    std::string color;
    
public:
    Shape(const std::string& c) : color(c) {}
    virtual ~Shape() = default;
    
    // 纯虚函数，必须被派生类实现
    virtual double getArea() const = 0;
    virtual double getPerimeter() const = 0;
    virtual void draw() const = 0;
    
    // 虚函数，可以被重写
    virtual void showInfo() const {
        std::cout << "Shape color: " << color << std::endl;
    }
    
    std::string getColor() const { return color; }
};

// 具体派生类 - Circle
class Circle : public Shape {
private:
    double radius;
    
public:
    Circle(const std::string& c, double r) : Shape(c), radius(r) {}
    
    double getArea() const override {
        return 3.14159 * radius * radius;
    }
    
    double getPerimeter() const override {
        return 2 * 3.14159 * radius;
    }
    
    void draw() const override {
        std::cout << "Drawing a " << color << " circle with radius " << radius << std::endl;
    }
    
    void showInfo() const override {
        std::cout << "Circle - Color: " << color << ", Radius: " << radius << std::endl;
    }
};

// 具体派生类 - Rectangle
class Rectangle : public Shape {
private:
    double width, height;
    
public:
    Rectangle(const std::string& c, double w, double h) 
        : Shape(c), width(w), height(h) {}
    
    double getArea() const override {
        return width * height;
    }
    
    double getPerimeter() const override {
        return 2 * (width + height);
    }
    
    void draw() const override {
        std::cout << "Drawing a " << color << " rectangle " 
                  << width << "x" << height << std::endl;
    }
    
    void showInfo() const override {
        std::cout << "Rectangle - Color: " << color 
                  << ", Size: " << width << "x" << height << std::endl;
    }
};

// =================================================================================
// 演示函数
// =================================================================================

void demonstrateClassAndObject() {
    std::cout << "\n=== 1. 类与对象演示 ===\n";
    
    // 创建对象
    Student student1;  // 默认构造
    Student student2("张三", 20, 85.5);  // 带参构造
    Student student3(student2);  // 拷贝构造
    
    // 使用对象
    student1.display();
    student2.display();
    student3.display();
    
    // 赋值操作
    student1 = student2;
    student1.display();
}

void demonstrateEncapsulation() {
    std::cout << "\n=== 2. 封装演示 ===\n";
    
    BankAccount account("12345", "李四", 1000.0, "1234");
    
    std::cout << account.getAccountInfo() << std::endl;
    
    // 正确的PIN
    account.deposit(500, "1234");
    std::cout << "当前余额: " << account.getBalance("1234") << std::endl;
    
    // 错误的PIN
    account.withdraw(200, "wrong_pin");
    
    // 正确的PIN
    account.withdraw(200, "1234");
}

void demonstrateInheritance() {
    std::cout << "\n=== 3. 继承演示 ===\n";
    
    Dog dog("旺财", 3, "金毛");
    Cat cat("咪咪", 2, true);
    
    // 基类功能
    dog.showInfo();
    cat.showInfo();
    
    // 重写的方法
    dog.makeSound();
    dog.move();
    
    cat.makeSound();
    cat.move();
    
    // 派生类特有功能
    dog.wagTail();
    cat.climb();
}

void demonstratePolymorphism() {
    std::cout << "\n=== 4. 多态演示 ===\n";
    
    // 使用多态容器
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Circle>("红色", 5.0));
    shapes.push_back(std::make_unique<Rectangle>("蓝色", 4.0, 6.0));
    shapes.push_back(std::make_unique<Circle>("绿色", 3.0));
    
    std::cout << "多态调用演示：\n";
    for (const auto& shape : shapes) {
        shape->showInfo();  // 虚函数调用
        shape->draw();      // 虚函数调用
        std::cout << "面积: " << shape->getArea() << std::endl;
        std::cout << "周长: " << shape->getPerimeter() << std::endl;
        std::cout << "-------------------\n";
    }
    
    // 运行时多态
    std::cout << "\n运行时多态演示：\n";
    std::vector<std::unique_ptr<Animal>> animals;
    animals.push_back(std::make_unique<Dog>("阿黄", 5, "拉布拉多"));
    animals.push_back(std::make_unique<Cat>("小白", 3, false));
    
    for (const auto& animal : animals) {
        animal->makeSound();  // 动态绑定
        animal->move();       // 动态绑定
    }
}

// 面试重点概念总结
void showInterviewKeyPoints() {
    std::cout << "\n=== 面试重点总结 ===\n";
    std::cout << "1. 类与对象：\n";
    std::cout << "   - 类是对象的蓝图，对象是类的实例\n";
    std::cout << "   - 构造函数、析构函数的调用时机\n";
    std::cout << "   - 拷贝构造与拷贝赋值的区别\n\n";
    
    std::cout << "2. 封装：\n";
    std::cout << "   - 数据隐藏和访问控制 (private/protected/public)\n";
    std::cout << "   - 通过方法控制数据访问，保证数据完整性\n\n";
    
    std::cout << "3. 继承：\n";
    std::cout << "   - is-a关系，代码复用\n";
    std::cout << "   - 构造和析构的顺序：构造从基类到派生类，析构相反\n\n";
    
    std::cout << "4. 多态：\n";
    std::cout << "   - 编译时多态：函数重载、模板\n";
    std::cout << "   - 运行时多态：虚函数、动态绑定\n";
    std::cout << "   - 虚函数表(vtable)的概念\n";
}

// =================================================================================
// 主函数
// =================================================================================

int main() {
    std::cout << "C++ 面向对象核心概念演示\n";
    std::cout << "========================\n";
    
    demonstrateClassAndObject();
    demonstrateEncapsulation();
    demonstrateInheritance();
    demonstratePolymorphism();
    showInterviewKeyPoints();
    
    return 0;
}
