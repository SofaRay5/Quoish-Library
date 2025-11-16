# Python函数
## 定义函数
```python
def func(argument1, argument2):
	# 函数实现代码 
```
python中，函数像变量一样可以被储存，拷贝，传递
```python
def func(x, y):
	return x + y

func_copy = func
print(func_copy(2, 3))
```
## 传参
传递的参数可以是任意类型的变量，包括列表，元组，字典
### 位置传参
直接根据实参的填入顺序传递实参
```python
def describe_pet(animal_type, pet_name):
	"""显示宠物的信息""" 
	print("\nI have a " + animal_type + ".") 
	print("My " + animal_type + "'s name is " + pet_name.title() + ".")

describe_pet('hamster', 'harry')
describe_pet('dog', 'willie')
```
### 关键字传参
传递给函数形参-实参对，这样交换顺序函数也不会搞错
```python
def describe_pet(animal_type, pet_name):
	"""显示宠物的信息""" 
	print("\nI have a " + animal_type + ".") 
	print("My " + animal_type + "'s name is " + pet_name.title() + ".")

describe_pet(animal_type='hamster', pet_name='harry')
describe_pet(pet_name='harry', animal_type='hamster')
```
### 默认值
```python
def describe_pet(pet_name, animal_type='dog'):
# 注意：默认形参后禁止跟非默认形参，所以这里两个参数的位置相比之前调换了位置
	"""显示宠物的信息""" 
	print("\nI have a " + animal_type + ".") 
	print("My " + animal_type + "'s name is " + pet_name.title() + ".")

# 一条名为Willie的小狗 
describe_pet('willie') # 以位置传参的形式传递实参，这里是按顺序关联第一个参数，第二个参数使用默认值
describe_pet(pet_name='willie') # 以关键字传参的形式传递实参

# 一只名为Harry的仓鼠 
describe_pet('harry', 'hamster') 
describe_pet(pet_name='harry', animal_type='hamster') 
describe_pet(animal_type='hamster', pet_name='harry')
```
### 不定数量变量传参
```python
def make_pizza(*toppings): 
	"""打印顾客点的所有配料"""
	# 形参名*toppings中的星号让Python创建一个名为toppings的空元组，并将收到的所有值都封装到这个元组中 
	print(toppings) 

make_pizza('pepperoni') 
make_pizza('mushrooms', 'green peppers', 'extra cheese')
```
若要结合使用位置传参和不定量传参，则不定量传参必须写在最后
```python
def make_pizza(size, *toppings): 
	"""概述要制作的比萨""" 
	print("\nMaking a " + str(size) + "-inch pizza with the following toppings:") 
	for topping in toppings: 
		print("- " + topping) 

make_pizza(16, 'pepperoni') 
make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')
```
### 不定数量键值对传参
```python
def build_profile(first, last, **user_info): 
	"""创建一个字典，其中包含我们知道的有关用户的一切""" 
	profile = {}
	profile['first_name'] = first profile['last_name'] = last
	for key, value in user_info.items():
		profile[key] = value
	return profile

user_profile = build_profile('albert', 'einstein', location='princeton', field='physics') print(user_profile)
```
## 返回值
```python
def add_func(argument1, argument2):
	return argument1 + argument2
```
函数可以返回任意类型的值，包括列表，元组，字典
```python
def build_person(first_name, last_name):
	"""返回一个字典，其中包含有关一个人的信息"""
	# 形参**user_info中的两个星号让Python创建一个名为user_info的空字典，并将收到的所有名称—值对都封装到这个字典中
	person = {'first': first_name, 'last': last_name}
	return person 

musician = build_person('jimi', 'hendrix')
print(musician)
```
## 递归
```python
def func(argument1, argument2):
	# 函数实现代码…… 
	func(argument2, argument1) # 再次调用
```
## 导入模块
模块是扩展名为.py的文件，包含要导入到程序中的代码
### 导入整个模块
在`pizza.py`输入以下代码
```python
def make_pizza(size, *toppings): 
	"""概述要制作的比萨""" 
	print("\nMaking a " + str(size) + "-inch pizza with the following toppings:") 
	for topping in toppings: 
		print("- " + topping)
```
保存之后，在`making_pizzas.py`中使用`import`就能调用`pizza.py`中的函数
```python
import pizza

pizza.make_pizza(16, 'pepperoni') 
pizza.make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')
```
### 导入模块中特定的函数
使用以下格式语句导入模块中特定的函数
```python
from module_name import function_name
```
上例中，使用如下：
```python
from pizza import make_pizza 
make_pizza(16, 'pepperoni') 
make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')
```
### 使用as为函数指定别名
```python
from pizza import make_pizza as mp 
mp(16, 'pepperoni') 
mp(12, 'mushrooms', 'green peppers', 'extra cheese')
```
### 使用as为模块指定别名
```python
import pizza as p 
p.make_pizza(16, 'pepperoni') 
p.make_pizza(12, 'mushrooms', 'green peppers', 'extra cheese')
```
### 导入模块中所有函数
```python
from module_name import *
```
## 匿名函数
匿名函数（也称为lambda函数）是Python中不需要使用`def`关键字定义的函数。它们使用`lambda`关键字创建，适用于简单的、一次性的操作
其格式为：
```python
lambda arguments: expression
```
示例：
```python
# 普通函数定义
def add(x, y):
    return x + y

# 等效的匿名函数
add_lambda = lambda x, y: x + y

print(add(3, 5))        # 输出: 8
print(add_lambda(3, 5)) # 输出: 8
```
匿名函数常作为高阶函数的入参
## 高阶函数
高阶函数是指能够接受其他函数作为参数或返回值的函数
### 定义高阶函数
```python
def apply_function(func, value):
    """
    应用给定函数到值上
    :param func: 要应用的函数
    :param value: 要处理的值（也可以是其他数据类型如列表、字典）
    :return: 函数应用结果
    """
    # 可以有别的处理
    return func(value)

# 使用普通函数作为参数
def square(x):
    return x ** 2

result = apply_function(square, 4)
print(result)  # 输出: 16
```
### 返回函数的高阶函数
```python
def create_multiplier(factor):
    """
    创建一个乘法器函数
    :param factor: 乘数因子
    :return: 乘法器函数
    """
    def multiplier(x):
        return x * factor
    return multiplier

# 使用示例
double = create_multiplier(2)
triple = create_multiplier(3)

print(double(5))  # 输出: 10
print(triple(5))  # 输出: 15

# 也可以直接使用
result = create_multiplier(4)(5)
print(result)  # 输出: 20
```
### 复合函数
```python
def compose(func1, func2):
    """
    组合两个函数: func2(func1(x))
    :param func1: 第一个函数
    :param func2: 第二个函数
    :return: 组合后的函数
    """
    def composed_function(x):
        return func2(func1(x))
    return composed_function

# 使用示例
# 先加倍再平方
double_then_square = compose(lambda x: x * 2, lambda x: x ** 2)
result = double_then_square(3)
print(result)  # 输出: 36 (先3*2=6, 然后6^2=36)

# 先平方再加倍
square_then_double = compose(lambda x: x ** 2, lambda x: x * 2)
result = square_then_double(3)
print(result)  # 输出: 18 (先3^2=9, 然后9*2=18)
```

### 使用高级函数实现日志记录
```python
def with_logging(func):
    """
    为函数添加日志功能
    :param func: 要包装的函数
    :return: 带日志的函数
    """
    def logged_function(*args, **kwargs):
    """
    要记录日志的任意函数
    :param *args: 该函数的任意变量入参（储存在元组）
    :param func2: 该函数的任意键值对入参（储存在字典）
    :return: 原来函数的返回值保持不变
    """
        print(f"Calling function {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} returned: {result}")
        return result
    return logged_function

# 使用示例
logged_multiply = with_logging(lambda x, y: x * y)
result = logged_multiply(3, 4)
# 输出:
# Calling function <lambda> with args: (3, 4), kwargs: {}
# Function <lambda> returned: 12
```
### map()函数
```python
# 基本语法
map(function, iterable, ...)
```
- `function`: 要对每个元素执行的函数
- `iterable`: 一个或多个可迭代对象（如列表、元组等）
- 返回值: 一个map对象（迭代器），可以使用`list()`转换为列表
#### 使用自定义函数
```python
# 使用已定义的函数
numbers = [1, 2, 3, 4, 5]
def double(x):
    return x * 2

doubled = map(double, numbers)
print(list(doubled))  # 输出: [2, 4, 6, 8, 10]
```
#### 使用匿名函数
```python
# 将列表中的每个数字平方
numbers = [1, 2, 3, 4, 5]
squared = map(lambda x: x**2, numbers)
print(list(squared))  # 输出: [1, 4, 9, 16, 25]
```
#### 使用内置函数
```python
# 将字符串转换为整数
str_numbers = ['1', '2', '3', '4', '5']
int_numbers = map(int, str_numbers)
print(list(int_numbers))  # 输出: [1, 2, 3, 4, 5]

# 获取字符串长度
words = ['apple', 'banana', 'cherry']
lengths = map(len, words)
print(list(lengths))  # 输出: [5, 6, 6]
```
#### 多个可迭代对象的情况
```python
# 将两个列表的对应元素相加
a = [1, 2, 3]
b = [4, 5, 6]
result = map(lambda x, y: x + y, a, b)
print(list(result))  # 输出: [5, 7, 9]

# 三个列表的例子
a = [1, 2, 3]
b = [4, 5, 6]
c = [7, 8, 9]
result = map(lambda x, y, z: x + y + z, a, b, c)
print(list(result))  # 输出: [12, 15, 18]
```
### filter()函数
```python
# 基本语法
map(function, iterable, ...)
```
- `function`: 要对每个元素执行的函数
- `iterable`: 要进行过滤的一个或多个可迭代对象（如列表、元组等）
- 返回值: 一个filter对象（迭代器），可以使用`list()`转换为列表
#### 使用自定义函数
```python
# 使用已定义的函数
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
def is_even(x):
    return x % 2 == 0

even_numbers = filter(is_even, numbers)
print(list(even_numbers))  # 输出: [2, 4, 6, 8, 10]
```
#### 使用匿名函数
```python
# 使用匿名函数
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = filter(lambda x: x % 2 == 0, numbers)
print(list(even_numbers))  # 输出: [2, 4, 6, 8, 10]
```
#### 使用None作为过滤函数
```python
# 过滤掉假值（False, None, 0, "", [], {}等）
mixed_values = [0, 1, False, True, "", "hello", [], [1, 2, 3], None]

truthy_values = filter(None, mixed_values)
print(list(truthy_values))  # 输出: [1, True, 'hello', [1, 2, 3]]
```
#### 过滤字符串
```python
# 过滤出包含特定字符的字符串
words = ["apple", "banana", "cherry", "date", "elderberry"]

# 过滤出包含字母'a'的单词
a_words = filter(lambda word: 'a' in word, words)
print(list(a_words))  # 输出: ['apple', 'banana', 'date']

# 过滤出长度大于5的单词
long_words = filter(lambda word: len(word) > 5, words)
print(list(long_words))  # 输出: ['banana', 'cherry', 'elderberry']
```
## 装饰器
装饰器可以用于修改或增强函数或类的行为，而无需改变其本身的源代码
### 函数装饰器
```python
# 定义一个装饰器函数，它接受一个函数作为参数
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func() # 执行被装饰的函数
        print("Something is happening after the function is called.")
    return wrapper # 返回内部包装好的函数
    
# 第一种写法：直接用复合函数的方式：
# 1. 定义一个普通函数
def say_hello():
    print("Hello!")

# 2. 手动“装饰”这个函数：把原函数传递给装饰器，并用返回的新函数覆盖原函数
say_hello = my_decorator(say_hello)

# 3. 调用经过装饰的新函数
say_hello()

# 第二种写法：使用@语法糖
# 1. 通过@定义一个函数
@my_decorator
def say_hello():
    print("Hello!")
    
# 2. 直接调用，看起来和普通函数一样，但其实已经被增强了
say_hello()
```
如果被修饰的函数含参，可以定义类似下面的`arg`和`kwarg`
### 类装饰器
```python
def add_method_to_class(cls):
    def new_method(self):
        return f"I am a new method added to {self.__class__.__name__}!"
    cls.new_method = new_method # 动态地为类添加一个新方法
    return cls # 返回修改后的类

@add_method_to_class
class MyClass:
    def original_method(self):
        return "I am the original method."

obj = MyClass()
print(obj.original_method())
print(obj.new_method()) # 调用由装饰器添加的方法
```
