#include <iostream>
#include <vector>
#include <memory>
using namespace std;

// ================= 1. 多态的基本概念演示 =================

// 多态 = "一个接口，多种实现"
// Polymorphism = "同一个函数调用，根据对象类型产生不同行为"

class Animal {
public:
    Animal(const string& name) : name_(name) {}
    
    // 虚函数 - 实现多态的关键
    virtual void makeSound() const {
        cout << name_ << " makes a generic animal sound" << endl;
    }
    
    virtual void move() const {
        cout << name_ << " moves in some way" << endl;
    }
    
    // 非虚函数 - 不支持多态
    void eat() const {
        cout << name_ << " is eating" << endl;
    }
    
    virtual ~Animal() = default;
    
    const string& getName() const { return name_; }

protected:
    string name_;
};

class Dog : public Animal {
public:
    Dog(const string& name) : Animal(name) {}
    
    // 重写虚函数 - 实现多态
    void makeSound() const override {
        cout << getName() << " barks: Woof! Woof!" << endl;
    }
    
    void move() const override {
        cout << getName() << " runs on four legs" << endl;
    }
    
    // 狗特有的方法
    void wagTail() const {
        cout << getName() << " wags tail happily!" << endl;
    }
};

class Cat : public Animal {
public:
    Cat(const string& name) : Animal(name) {}
    
    void makeSound() const override {
        cout << getName() << " meows: Meow~ Meow~" << endl;
    }
    
    void move() const override {
        cout << getName() << " walks silently like a ninja" << endl;
    }
    
    // 猫特有的方法
    void purr() const {
        cout << getName() << " purrs contentedly: Purr~" << endl;
    }
};

class Bird : public Animal {
public:
    Bird(const string& name) : Animal(name) {}
    
    void makeSound() const override {
        cout << getName() << " chirps: Tweet! Tweet!" << endl;
    }
    
    void move() const override {
        cout << getName() << " flies through the sky" << endl;
    }
};

// ================= 2. 对比：没有多态的情况 =================

class SimpleAnimal {
public:
    SimpleAnimal(const string& name) : name_(name) {}
    
    // 注意：没有virtual！
    void makeSound() const {
        cout << name_ << " makes a generic sound" << endl;
    }

protected:
    string name_;
};

class SimpleDog : public SimpleAnimal {
public:
    SimpleDog(const string& name) : SimpleAnimal(name) {}
    
    // 隐藏基类函数，但不是多态！
    void makeSound() const {
        cout << name_ << " barks: Woof!" << endl;
    }
};

// ================= 3. 演示函数 =================

void demonstratePolymorphism() {
    cout << "=== 🎭 多态演示 ===" << endl;
    
    // 创建不同类型的动物对象
    vector<unique_ptr<Animal>> zoo;
    zoo.push_back(make_unique<Dog>("Buddy"));
    zoo.push_back(make_unique<Cat>("Whiskers"));
    zoo.push_back(make_unique<Bird>("Tweety"));
    zoo.push_back(make_unique<Dog>("Rex"));
    
    cout << "\n--- 多态的神奇之处 ---" << endl;
    cout << "同一个函数调用，不同的行为：\n" << endl;
    
    // 🎯 关键：通过基类指针调用，但执行的是派生类的实现
    for (const auto& animal : zoo) {
        cout << "调用 animal->makeSound():" << endl;
        animal->makeSound();  // 多态！根据实际对象类型调用不同实现
        
        cout << "调用 animal->move():" << endl;
        animal->move();       // 多态！
        
        cout << "调用 animal->eat():" << endl;
        animal->eat();        // 非虚函数，总是调用基类版本
        
        cout << "---" << endl;
    }
}

void demonstrateNonPolymorphism() {
    cout << "\n=== ❌ 没有多态的情况 ===" << endl;
    
    SimpleAnimal* simple1 = new SimpleDog("SimpleRex");
    SimpleDog* simple2 = new SimpleDog("DirectRex");
    
    cout << "通过基类指针调用：" << endl;
    simple1->makeSound();  // 调用基类版本！不是多态
    
    cout << "直接使用派生类指针：" << endl;
    simple2->makeSound();  // 调用派生类版本
    
    delete simple1;
    delete simple2;
}

// ================= 4. 静态绑定 vs 动态绑定 =================

void demonstrateBindingTypes() {
    cout << "\n=== 🔗 静态绑定 vs 动态绑定 ===" << endl;
    
    Dog dog("StaticDog");
    Animal* animalPtr = &dog;
    
    cout << "\n--- 动态绑定（运行时决定） ---" << endl;
    cout << "通过基类指针调用虚函数：" << endl;
    animalPtr->makeSound();  // 动态绑定 - 运行时确定调用Dog::makeSound()
    animalPtr->move();       // 动态绑定 - 运行时确定调用Dog::move()
    
    cout << "\n--- 静态绑定（编译时决定） ---" << endl;
    cout << "通过基类指针调用非虚函数：" << endl;
    animalPtr->eat();        // 静态绑定 - 编译时确定调用Animal::eat()
    
    cout << "\n--- 直接调用（静态绑定） ---" << endl;
    cout << "直接通过对象调用：" << endl;
    dog.makeSound();         // 静态绑定 - 编译时确定调用Dog::makeSound()
    dog.wagTail();          // 只能通过Dog指针调用特有方法
}

// ================= 5. 多态的实际应用场景 =================

// 多态让我们可以写通用的算法
void feedAllAnimals(const vector<unique_ptr<Animal>>& animals) {
    cout << "\n=== 🍖 喂食时间 ===" << endl;
    for (const auto& animal : animals) {
        cout << "喂食 " << animal->getName() << ":" << endl;
        animal->eat();           // 统一接口
        animal->makeSound();     // 不同行为（多态）
    }
}

void makeAllAnimalsMove(const vector<unique_ptr<Animal>>& animals) {
    cout << "\n=== 🏃 运动时间 ===" << endl;
    for (const auto& animal : animals) {
        cout << animal->getName() << " 开始运动:" << endl;
        animal->move();          // 多态：每种动物不同的移动方式
    }
}

// ================= 6. 虚函数表机制简单演示 =================

void demonstrateVTableConcept() {
    cout << "\n=== 📋 虚函数表概念演示 ===" << endl;
    
    cout << "每个类都有自己的虚函数表：" << endl;
    cout << "Animal vtable: [Animal::makeSound, Animal::move]" << endl;
    cout << "Dog vtable:    [Dog::makeSound, Dog::move]" << endl;
    cout << "Cat vtable:    [Cat::makeSound, Cat::move]" << endl;
    
    cout << "\n每个对象都有一个指向虚函数表的指针(vptr)：" << endl;
    
    Dog dog("VTableDog");
    Cat cat("VTableCat");
    Animal* ptr1 = &dog;
    Animal* ptr2 = &cat;
    
    cout << "dog对象的vptr指向Dog vtable" << endl;
    cout << "cat对象的vptr指向Cat vtable" << endl;
    
    cout << "\n当调用 ptr1->makeSound() 时：" << endl;
    cout << "1. 通过ptr1找到dog对象" << endl;
    cout << "2. 通过dog对象的vptr找到Dog vtable" << endl;
    cout << "3. 在vtable中找到Dog::makeSound" << endl;
    cout << "4. 调用Dog::makeSound" << endl;
    
    ptr1->makeSound();  // 验证上述过程
    
    cout << "\n当调用 ptr2->makeSound() 时，过程类似但调用Cat::makeSound：" << endl;
    ptr2->makeSound();
}

// ================= 主函数 =================

int main() {
    cout << "=== 🎭 C++ 多态详解 ===" << endl;
    
    try {
        demonstratePolymorphism();
        demonstrateNonPolymorphism();
        demonstrateBindingTypes();
        
        // 创建动物园用于后续演示
        vector<unique_ptr<Animal>> zoo;
        zoo.push_back(make_unique<Dog>("Max"));
        zoo.push_back(make_unique<Cat>("Luna"));
        zoo.push_back(make_unique<Bird>("Rio"));
        
        feedAllAnimals(zoo);
        makeAllAnimalsMove(zoo);
        
        demonstrateVTableConcept();
        
        cout << "\n=== 📚 多态总结 ===" << endl;
        cout << "1. 多态 = 同一接口，不同实现" << endl;
        cout << "2. 实现条件：虚函数 + 继承 + 基类指针/引用" << endl;
        cout << "3. 动态绑定：运行时根据对象实际类型调用相应函数" << endl;
        cout << "4. 优势：代码复用，易扩展，符合开闭原则" << endl;
        cout << "5. 机制：虚函数表(vtable) + 虚函数指针(vptr)" << endl;
        
    } catch (const exception& e) {
        cout << "错误: " << e.what() << endl;
    }
    
    return 0;
}
