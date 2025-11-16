# Python类
## 创建类
基本语法
```python
class ClassName:
    # 类属性
    class_attribute = "defaut_value"
	
	# 构造函数
    def __init__(self, value):
        self.instance_variable = value  # 实例属性
    
    # 类方法（成员函数）
    def method_name(self, other_args):
        # 方法体
        pass
    # 析构函数
    def __del__(self):
	    pass
```
- 类属性：由这个类的所有实例所共享的变量
- 实例属性：每个实例会分别具有的变量
- 方法：定义在类中的函数被称作方法
- 构造函数：每一个实例在初始化时会被调用的函数
- 析构函数：每一个实例在销毁时会被调用的函数
- `self`变量：在方法定义中代表实例自身，可以用`self.instance_variable`访问实例属性或`self.method_name()`调用其他方法。实际上也可以把`self`换成别的名字，但是会降低代码可读性。在每一次调用类函数时，不必向`self`传参，程序会自动将实例作为参数传入。
示例：
```python
class Person:  
    count = 0  
    def __init__(self, name, age, country="China"):  
        self.name = name  
        self.age = age  
        self.country = country  
        Person.count += 1  
  
    def introduce(self):  
        print("Hello " + self.name + 
        ", I am " + str(self.age) + " years old, ", 
        "I am from " + str(self.country))
```
## 类的使用
### 根据类创建实例
创建实例时需要传入`__init__`函数所需的参数。如有默认值，则可以选择不传参，则实例将采用默认值。以上面的`Class Person`为例：
```python
person1 = Person("Alice", 30)  
person2 = Person("Bob", 27, "US")
```
### 访问实例属性
```python
person1 = Person("Alice", 30)
print(person1.name)
```
### 访问类属性
#### 通过类名访问
```python
print(Person.count)
```
#### 通过实例访问
```python
person1 = Person("Alice", 30)
print(person1.count)

# 如果实例创建了与类属性同名的属性，则优先访问实例属性
person1.count = 2
print(person1.count) # 打印的是新创建的实例属性
print(Person.count)  # 打印的是类属性，依然保持不变
```
### 修改实例属性或类属性
```python
person1 = Person("Alice", 30)
person1.name = "Bob"
person1.count = 2
```
### 调用方法
```python
my_dog = Dog('willie', 6)
my_dog.sit()
my_dog.roll_over()
```
### 修改或添加方法
添加新的方法
```python
class MyClass:  
    pass  
  
def new_method(self):  
    return "New Method"  
  
MyClass.new_method = new_method  # 添加新方法
obj = MyClass()  
print(obj.new_method()) # 输出"New Method"
```
替换原有方法
```python
class MyClass:
    def original_method(self):
        return "Original implementation"

# 创建实例
obj = MyClass()
print(obj.original_method())  # 输出: Original implementation

# 定义新实现
def new_implementation(self):
    return "Patched implementation"

# 猴子补丁：替换类的方法
MyClass.original_method = new_implementation

# 再次调用
print(obj.original_method())  # 输出: Patched implementation

# 新创建的实例也会受到影响
new_obj = MyClass()
print(new_obj.original_method())  # 输出: Patched implementation
```
## 继承
继承是一种创建新类的方式，新类（子类/派生类）可以继承现有类（父类/基类）的属性和方法，并可以添加自己的新特性或修改继承的行为
继承不能循环
### 基本语法
```python
class ParentClass:
    # 父类的属性和方法
    pass

class ChildClass(ParentClass):  # 继承自ParentClass
    # 子类的属性和方法
    pass
```
### 单继承
一个子类仅继承一个父类
```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        pass

class Dog(Animal):  # 单继承
    def speak(self):
        return "Woof!"

class Cat(Animal):  # 单继承
    def speak(self):
        return "Meow!"

dog = Dog("Buddy")
print(dog.name)     # 输出: Buddy
print(dog.speak())  # 输出: Woof!
```
### 多继承
一个子类可以继承多个父类。
```python
class Flyable:
    def fly(self):
        return "I can fly!"

class Swimmable:
    def swim(self):
        return "I can swim!"

class Duck(Flyable, Swimmable):  # 多继承
    def __init__(self, name):
        self.name = name

duck = Duck("Donald")
print(duck.fly())   # 输出: I can fly!
print(duck.swim())  # 输出: I can swim!
```
### 方法重写
子类可以重写父类的方法，以提供特定的实现，但子类的重写不影响父类原来的实现
```python
class Vehicle:
    def move(self):
        return "Moving..."

class Car(Vehicle):
    def move(self):  # 重写父类方法
        return "Driving on the road"

class Airplane(Vehicle):
    def move(self):  # 重写父类方法
        return "Flying in the sky"

car = Car()
print(car.move())  # 输出: Driving on the road

plane = Airplane()
print(plane.move())  # 输出: Flying in the sky
```
### super()函数
`super()` 函数可以用于调用父类的方法，特别是在重写方法时保留父类的功能
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        return f"I'm {self.name}, {self.age} years old"

class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)  # 调用父类的构造函数
        self.student_id = student_id
    
    def introduce(self):
        # 调用父类的方法并扩展
        return f"{super().introduce()} and my student ID is {self.student_id}"

student = Student("Alice", 20, "S12345")
print(student.introduce())  # 输出: I'm Alice, 20 years old and my student ID is S12345
```
在多重继承中，`super()` 遵循方法解析顺序（MRO, Method Resolution Order）向前取一个单元
python使用C3线性化算法计算方法解析顺序，其遵循三个重要原则：
- 子类优先于父类（类似于广搜）
- 多个同级父类按照其在继承列表中声明的顺序被检查
- 对每个父类递归应用相同的规则
菱形继承的情况：
```python
class A:
    def method(self):
        print("A method")

class B(A):
    def method(self):
        print("B method")
        super().method()

class C(A):
    def method(self):
        print("C method")
        super().method()

class D(B, C):
    def method(self):
        print("D method")
        super().method()

d = D()
d.method()
# 输出:
# D method
# B method
# C method
# A method

print(D.__mro__)  # 查看方法解析顺序
# 输出: (<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)
# 因此上面代码的执行顺序是：D method -> D下面的super()找到B -> B method -> B下面的super()找到C -> C method -> C下面的super()找到A -> A method  
```
在多继承的情况下，如果MRO无法合并，就会产生`TypeError`错误
```python
class A:
    pass

class B(A):
    pass

class C(A):
    pass

class D(B, C):
    pass

class E(C, B):
    pass

# 查看MRO（使用C3线性化）
print(D.__mro__)  # 输出: (D, B, C, A, object)
print(E.__mro__)  # 输出: (E, C, B, A, object)

# 尝试创建不一致的继承
try:
    class F(D, E):
        pass
except TypeError as e:
    print(f"错误: {e}")  # 输出: Cannot create a consistent method resolution order
# D->B->C-A->object和E->C->B-A->object无法合并，因此会报错
```
`object`是python中所有类的基类，用户自定义的所有类包括内置的`str`，`int`类都继承了这个类
### 检查继承关系
 `isinstance()` 函数检查对象是否是某个类或其子类的实例
 ```python
 class Animal:
    pass

class Dog(Animal):
    pass

dog = Dog()
print(isinstance(dog, Dog))    # 输出: True
print(isinstance(dog, Animal)) # 输出: True
print(isinstance(dog, object)) # 输出: True
 ```
  `issubclass()` 函数检查一个类是否是另一个类的子类
  ```python
  class Animal:
    pass

class Dog(Animal):
    pass

print(issubclass(Dog, Animal))  # 输出: True
print(issubclass(Dog, object))  # 输出: True
print(issubclass(Animal, Dog))  # 输出: False
  ```
