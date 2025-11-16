# 字符串
python中的字符串既可以是单引号引起来的，也可以是双引号
```python
string1 = "This is a string." 
string2 = 'This is also a string.'
print(string1)
print(string2)
# 注意：python的print自带换行
```
这种灵活性允许在字符串更加自由地添加"和'
## 修改字符串大小写
```python
# 将首字母大写
name = "quoish Comege"
print(name.title()) # 输出"Quoish Comege"
# 注意：name变量本身并没有发生变化，如果print(name)，得到的依然是"quoish Comege"

# 将字母全部大写
print(name.upper()) # 输出"QUOISH COMEGE"
print(name.lower()) # 输出"quoish comege"
# 同样，name变量本身并没有变化
```
## 拼接字符串
```python
first_name = "quoish"
last_name = "comge"
full_name = first_name + " " + last_name
print(full_name) # 输出"quoish comege"
```
## 使用转义字符格式化字符串
```python
print("Languages:\n\tPython\n\tC\n\tJavaScript")
#输出：
#Languages:
#	Python
#	C
#	JavaScript
```
## 删除空白
```python
string = "    python     "
string.rstrip() # 删除右边的空白，输出'    python'
string.lstrip() # 删除左边的空白，输出'python     '
string.strip()  # 删除两边的空白，输出'python'
```
# 数字
## 整数
```python
# python的四则运算
2 + 3 # 得到5
3 - 2 # 得到1
2 * 3 # 得到6
3 / 2 # 得到1.5
# 注意：python2中，3/2会得到1（整除除法）

# 乘方
3 ** 2 # 得到9

# 整除除法
5 // 2 # 得到2

# 求余
5 % 2 # 得到1
```
## 浮点数
```python
# 浮点数保留几位小数的方法

# 使用round()函数
number = 3.14159 
rounded_number = round(number, 2) 
print(rounded_number) # 输出: 3.14

# 使用format函数
number = 3.14159 
formatted_number = format(number, '.2f') 
print(formatted_number) # 输出: 3.14
```
## 数字转换为字符串
```python
age = 24 
message = "Happy " + str(age) + "rd Birthday!" # 将数字转为字符串以参与字符串拼接
print(message) # 输出"Happy 23rd Birthday!"
```
# 列表
列表由一系列按“特定顺序”排列的元素组成，这些元素可以是整数、浮点数、字符、字符串、组元、字典，也可以列表嵌套列表，混杂在一起也可以，同时，这些元素可以重复。概括列表的特性：
- 有序性
- 可重复性
- 可修改性
- 数据类型多元性
## 创建列表
```python
# 列表使用方括号包裹
inm = ["yajusenbai", 114514, 19.19, "x"]

# 列表兼容各种数据类型
mixed_list = (
    42,                          # 整数
    3.14,                        # 浮点数
    "Python",                    # 字符串
    [1, 2, 3],                   # 嵌套列表
    {"name": "Alice", "age": 30},# 字典
    (4, 5, 6),                   # 元组
    True,                        # 布尔值
    None,                        # None
    b"bytes",                    # 字节串
    complex(1,2),                # 复数
    range(10),                   # range对象
    frozenset([7,8,9]),          # 冻结集合
    print,                       # 函数
    object                       # 类
)

# 空列表创建
empty = []
```
## 访问列表元素
```python
# 访问指定位置元素
print(inm[0]) # 输出yajusenbai
print(inm[1]) # 输出114514
# 注意：索引从0开始而不是1

# 访问末尾元素
print(inm[-1]) # 输出x

# 遍历列表
for i in range(0,4):
	print(inm[i])
# 输出：
# yajusenbai
# 114514
# 19.19
# x
```
## 修改列表元素
```python
inm = ["yajusenbai", 114514, 19.19, "x"]
inm[0] = "mur"
# 此时列表变成了["mur", 114514, 19.19, "x"]
```
## 增加列表元素
```python
inm = ["yajusenbai", 114514, 19.19, "x"]
# 在列表末尾插入一个元素
inm.append("yarimasune")
# 此时列表变成了["yajusenbai", 114514, 19.19, "x", "yarimasune"]

# 在列表中指定位置插入一个元素
inm.insert(1, "black tea") # 在索引为1的位置插入该元素，右边的元素向右移动一格
# 此时列表变成了["yajusenbai", "black tea", 114514, 19.19, "x", "yarimasune"]
```
## 删除列表元素
```python
inm = ["yajusenbai", 114514, 19.19, "x"]

# 删除列表指定位置的元素
del inm[1]
# 此时列表变成了["yajusenbai", 19.19, "x"]


# 弹出列表末尾的元素
inm.pop() # 这个函数是有返回值的，返回的就是被弹出的元素，可以用变量接收
# 此时列表变成了["yajusenbai", 114514, 19.19]

poped_elm = inm.pop(1)
# 此时列表变成了["yajusenbai", 19.19]
# poped_elm的值被赋为114514


# 根据值删去列表元素
inm.remove("19.19") # 这里函数的入参也可以是变量
# 此时列表变成了["yajusenbai", 19.19]
# 注意：当列表中存在多个重复出现的元素时，要用remove()删去该元素，只会删去这个元素从左往右第一次出现的元素
```
## 匹配列表元素
```python
inm = ["yajusenbai", 114514, 19.19, "x"]

# 查找索引
inm.index("yajusenbai") # 返回列表中第一个匹配目标元素的索引值，这里是1
# 注意：若无匹配对象，则触发ValueError异常

# 计数元素
inm.count("yajusenbai") # 返回列表中该元素重复出现的次数，这里是1
```
## 组织列表
### 排列列表
```python
# 永久顺序排序
list = ['1919', '233', '514', 'a', 'b', 'faadc', 'xaw']
list.sort() # 此时列表变成['1919', '233', '514', 'a', 'b', 'faadc', 'xaw']
# 注意：sort对数字（整数和浮点数）时采用实数大小的顺序，
# 对字符串排序时采用字典序，按unicode编码顺序排列，B将排序在a的前面，
# 对组元采取每个组元先比较第一个，第一个相同再比较第二个的顺序
# 不同数据类型之间不支持相互比较，否则出现TypeError异常


# 非永久顺序排序
list = ['1919', '233', '514', 'a', 'b', 'faadc', 'xaw']
print(sorted(list))# 此时输出['1919', '233', '514', 'a', 'b', 'faadc', 'xaw']
# 注意：和sort直接改变list不同，使用完sorted之后，list并没有改变，仍然维持原样


# 自定义优先级进行不同数据类型的排列
def type_priority(item):
    type_order = {int: 0, float: 0, str: 1, list: 2, tuple: 2, dict: 3, set: 3}
    return (type_order.get(type(item), 4), item)

mixed_list.sort(key=type_priority)
# 示例: [{"a": 1}, "hello", 42, [1, 2]] → [42, "hello", [1, 2], {"a": 1}]
```
### 翻转列表
```python
list = ['1919', '233', '514', 'a', 'b', 'faadc', 'xaw']
list.reverse() # 此时列表变成了['xaw', 'faadc', 'b', 'a', '514', '233', '1919']
# 注意：reverse并不是对列表逆序排序，而是将列表的顺序颠倒
```
### 确定列表长度
```python
list = ['1919', '233', '514', 'a', 'b', 'faadc', 'xaw']
len(list) # 返回7
```
##  数值列表
### 创建数值列表
```python
# range()函数会返回一个range对象
range(1, 6) # 包含数字1,2,3,4,5
# 注意：range(1, 6)中不包含6

# 也可以只写一个参数，默认从0开始
range(10) # 包含数字0到9

# range对象可以通过list()转换成列表
list_a = list(range(1,6))


# range对象和列表不同，是一个惰性的序列生成器，并不直接存储这些数字，而只是按需生成
# 不转换 - 得到 range 对象
r = range(0, 1000000)  # 几乎不占内存

# 转换为列表 - 占用大量内存
lst = list(range(0, 1000000))  # 存储 100 万个整数
```
### 数值列表的简单统计
```python
digits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
min(digits) # 返回最小值，这里是0
max(digits) # 返回最大值，这里是9
sum(digits) # 返回总和，这里是45
```
### 列表解析
```python
squares = [value**2 for value in range(1,11)]
# 等价于下面的代码：
squares = []
for value in range(1, 11):
	square = value ** 2
	squares.append(square)

even_squares = [x**2 for x in range(10) if x % 2 == 0]
# 等价于下面的代码：
even_squares = []
for x in range(10):
	if x % 2 == 0:
		square = x ** 2 
		squares.append(square)
```
## 列表切片
```python
players = ['reimu', 'marisa', 'sakuya', 'sanae', 'youmu']

# 提取列表第2到第4个元素
print(players[1:4]) # 输出['marisa', 'sakuya', 'sanae']

# 没有起始索引时，将自动从开头数
print(players[:4]) # 输出['reimu', 'marisa', 'sakuya', 'sanae']

# 没有终止索引时，将自动数到结尾
print(players[2:]) # 输出['sakuya', 'sanae', 'youmu']

# 输出倒数前3个元素
print(players[-3:]) # 输出['sakuya', 'sanae', 'youmu']


# 复制列表
players2 = players[:]
# 这样是将players中的所有元素拷贝了一份储存在players2中，此时再修改players2中的元素不会影响players

players3 = players
# 这样是将players3关联到players（相当于指针赋值），此时再修改players3中的元素，players也会发生一样的变化
```
# 元组
元组是不可变的列表，和列表一样，可以储存多种混合类型的数据。其具有以下的特性：
- 有序性
- 可重复性
- 不可修改性
- 数据类型多元性
注意：虽然元组本身不可以改变，但元组内部的可修改元素依然可以改变，如若元组中包含了一个字典，可以索引找到该字典然后修改该字典
## 创建元组
```python
# 元组使用圆括号包裹
tuple1 = (100, 500)

# 元组兼容各种数据类型
mixed_tuple = (
    42,                          # 整数
    3.14,                        # 浮点数
    "Python",                    # 字符串
    [1, 2, 3],                   # 列表
    {"name": "Alice", "age": 30},# 字典
    (4, 5, 6),                   # 嵌套元组
    True,                        # 布尔值
    None,                        # None
    b"bytes",                    # 字节串
    complex(1,2),                # 复数
    range(10),                   # range对象
    frozenset([7,8,9]),          # 冻结集合
    print,                       # 函数
    object                       # 类
)

# 单元素元组
tuple2 = (1, ) # 不能少了逗号，少了逗号就会被认为是数字

# 空元组
empty = ()
```
## 访问元组
元组的访问、遍历、切片、`len`、`index`、`count`的通用函数与列表用法一致。
# 字典
字典是一系列键—值对，且各个键值对的顺序并非固定的。总的来说，字典有以下特征：
- 无序性（注意：python 3.7+版本字典并非完全无序，而是会按照插入顺序排列）
- 是键值对的双元组
- 键的不可重复性和值的可重复性
- 键的不可修改性和值的可修改性
- 键只能是不可变数据类型（字符串、数字、元组等），而值不限制数据类型。
## 创建字典
```python
# 空字典
empty_dict = {}

# 直接初始化
person = {"name": "Alice", "age": 30, "city": "New York"}

# dict() 构造函数
colors = dict(red="#FF0000", green="#00FF00", blue="#0000FF")

# 字典推导式
squares = {x: x*x for x in range(1, 6)}
# {1:1, 2:4, 3:9, 4:16, 5:25}
```
## 访问字典元素
```python
# 根据键访问值
person = {"name": "Alice", "age": 30, "city": "New York"}

# 直接访问
person['name'] # 返回'Alice'
# 使用get()函数访问（安全访问）
person.get('name') # 返回'Alice'

# 辨析：直接通过键访问字典值时，如果字典中不存在这样的键，会出现KeyError异常
# 如果采用get(key, defaut=None)函数访问字典值时，如果字典中不存在这样的键，不会出现异常，而只会返回defaut值（不填入这个参数时则是None）
person.get('name', '未知')
# 若以上面的方式调用get()函数，如果字典中不存在这样的键，就会返回默认值“未知”，可以避免出现异常
```
## 修改字典元素
```python
person = {"name": "Alice", "age": 30, "city": "New York"}
# 添加新键值对
person["job"] = "Engineer"

# 修改现有键
person["age"] = 31

# 使用 setdefault() (如果键不存在则添加)
person.setdefault("country", "USA")  # 键存在则不修改
```
## 更新字典元素
```python
# 更新字典和直接修改现有键类似，但在许多方面比直接修改现有键更方便

# 支持多种参数格式
person.update(job="Engineer")         # 关键字参数形式
person.update({"job": "Engineer"})    # 字典形式
person.update([("job", "Engineer")])  # 键值对元组列表

# 可以实现一次性更新多个键
person.update(age=32, city="London", job="Engineer")

# 可以使用字典参数批量更新
updates = {"age": 32, "city": "London", "job": "Engineer"}
person.update(updates)

# 添加原先没有的键值对
person.update({"first name": "Alice"})  # 使用字典入参（有效）
# person.update(first name="Alice")     # 语法错误！(空格无效)

# 处理数字键
person.update({42: "The Answer"})  # 有效
# person.update(42="The Answer")   # 语法错误
```
## 删除字典元素
```python
person = {"name": "Alice", "age": 30, "city": "New York"}
# 删除指定键值对
del person["city"]

# 删除键值对并返回值
age = person.pop("age")  # 返回31并从字典删除

# 删除最后添加的键值对 (Python 3.7+)
last_item = person.popitem()  # 返回 ("job", "Engineer")

# 清空字典
person.clear()
```
## 合并字典
```python
# Python 3.9+
# 运算形式
merged = dict1 | dict2
# 更新形式
merged = dict1
merged |= dict2
# 注意：以上两种合并字典的方式，若dict1中有和dict2中一样的键，则dict2的值会覆盖掉dict1的值

# 传统方法
merged = dict1.copy()
merged.update(dict2)
```
## 遍历字典
```python
person = {"name": "Alice", "age": 30, "city": "New York"}
# 遍历键
for key in person: # 默认遍历的是键
    print(key)

for key in person.keys():
    print(key)

# 遍历值
for value in person.values():
    print(value)

# 遍历键值对
for key, value in person.items():
    print(f"{key}: {value}")

# 使用 enumerate 获取索引，进行枚举遍历
for i, (key, value) in enumerate(person.items()):
    print(f"{i}. {key}={value}")

# 并行遍历多个字典
dict1 = {"a": 1, "b": 2}
dict2 = {"a": "x", "b": "y"}
for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
    print(f"{k1}={v1}, {k2}={v2}")

# 注意：由于python在3.7+版本字典依插入顺序有固定的顺序，所以多次遍历时，只要键值对不发生插入或删除，遍历各键值对的顺序都是固定的

# 按一定顺序遍历字典
for key in sorted(person.keys()): # 将字典的键排列后输出
    print(key)

for value in sorted(person.values()): # 将字典的值排列后输出
    print(value)
```
## 字典排序
```python
person = {"name": "Alice", "age": 30, "city": "New York"}

# 排序字典的键并输出为列表
sorted(person.keys())
# 排序字典的值并输出为列表
sorted(person.values())
```
