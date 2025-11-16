# 逻辑语句
## 循环语句
### for循环
#### 遍历基本数据类型
```python
# 基本语法
for 变量 in 可迭代对象:
	# 循环体代码

# 遍历列表
fruits = ['apple', 'banana', 'cherry'] 
for fruit in fruits: 
	print(fruit) # 按照列表的顺序依次输出每个元素

#遍历元组
dimensions = (200, 500, 100) 
for dimension in dimensions: 
	print(dimension) # 按照元组的顺序依次输出每个元素

#`` 遍历字符串
for char in "Hello": 
	print(char) # 输出 H, e, l, l, o
```
`for`循环遍历字典
`for`循环参与列表解析
#### 控制循环次数
使用`range()`控制循环次数
```python
# 遍历指定范围的整数序列
for i in range(5): 
	print(i) # 输出 0, 1, 2, 3, 4

# 指定遍历步长
for i in range(2, 10, 2): 
	print(i) # 输出 2, 4, 6, 8

# 按索引遍历列表
fruits = ['a', 'b', 'c'] 
for index in range(len(fruits)): 
	print(fruits[index]) # 这种迭代方式每次循环都要计算一次len(fruits)，会显得低效


# 注意：每次for循环结束时，遍历用的变量会停在最后一次循环时的情况
# 第一次运行
for i in range(5):
    print(i)  # 输出 0,1,2,3,4

print("循环结束后 i =", i)  # 输出 i = 4

# 第二次运行，range()函数会生成一个新的迭代器，然后i会被重新赋值，从0开始数
for i in range(5):  # 重新创建新的迭代器，i 被重置
    print(i)  # 再次输出 0,1,2,3,4
```
#### 循环控制语句
```python
# 使用break终止循环
for i in range(1, 4): 
	if i == 3: 
		break 
	print(i) # 输出 1, 2

# 使用continue跳过本次循环后面的代码
for i in range(1, 4): 
	if i % 2 == 0: 
		continue 
	print(i) # 输出 1, 3

# for-else联合语句，当for循环正常结束后（没有被break打断），会执行else部分的代码
for i in range(5):
	print(i) 
else: 
	print("循环结束") # 输出 0-4 后打印“循环结束”
```
#### 嵌套迭代
```python
# 二维列表
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

for row in matrix:
    for num in row:
        print(num, end=' ')
    print()  # 换行
```
#### 枚举迭代（同时获取索引和值）
使用`enumerate`进行枚举迭代
```python
fruits = ['apple', 'banana', 'cherry']
for index, fruit in enumerate(fruits):
    print(f"索引 {index}: {fruit}")

# 自定义起始索引
for i, fruit in enumerate(fruits, start=1):
    print(f"{i}. {fruit}")
```
#### 并行迭代
使用`zip`进行并行迭代
```python
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]

# 使用 zip
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# 处理不等长序列
from itertools import zip_longest
for a, b in zip_longest([1,2,3], ['a','b'], fillvalue='-'):
    print(a, b)  # 1a, 2b, 3-
```
### while循环
```python
while condition:
	# Loop Code
# 当condition为真时，Loop Code的内容会一直重复执行
```
#### break退出循环
```python
prompt = "\nPlease enter the name of a city you have visited:"
prompt += "\n(Enter 'quit' when you are finished.) "
while True: 
	city = input(prompt) 
	if city == 'quit': 
		break
	else: 
		print("I'd love to go to " + city.title() + "!")
```
#### continue跳过后续代码直接下一次循环
```python
current_number = 0 
while current_number < 10: 
	current_number += 1 
	if current_number % 2 == 0: 
		continue 
	print(current_number)
```
## 条件语句
### if语句
```python
if 条件1:
    # 语句块1
elif 条件2:
    # 语句块2
...
else:
    # 默认语句块
# 注意：在作为一个整体的if-elif-elif-...-else语句中，如果一个if或elif的条件测试通过了，就不会再进行下面的测试，直接跳过后面的语句
```
示例
```python
score = 85 
if score >= 90: 
	print("优秀") 
elif score >= 80: 
	print("良好") 
elif score >= 60: 
	print("及格") 
else: 
	print("不及格")
```
### 条件测试
#### 判断相等与不相等
```python
car = 'bwm' # 赋值，不是比较

car == 'bwm' # 返回True
car != 'bwm' # 返回False
```
#### 数字比较
```python
# 相等
a == b
# 大于
a > b
# 小于
a < b
# 大于等于
a >= b # 不能写成=>
# 小于等于
a <= b # 不能写成=<
```
#### 逻辑运算符
`and`：与
`or`：或
`not`：非
示例
```python
x, y = 10, 5 
if x > 0 and y > 0: 
	print("x 和 y 均为正数") 
elif x > 0 or y > 0: 
	print("x 或 y 为正数") 
else: 
	print("x 和 y 均非正数")
```
以下值在python中默认视为假值：
- `None`
- `False`
- 零值（0, 0.0, 0j）
- 空序列（`""`, `[]`, ``）
- 空字典 `{}`
其他值均视为真值
#### 检查元素是否在列表
```python
# 使用关键字in可以判断一个元素是否在容器对象中

# 字符串
print('a' in 'apple')  # True
# 列表
numbers = [1, 2, 3]
print(2 in numbers)    # True
# 元组
colors = ('red', 'green')
print('blue' in colors)  # False
# 集合
unique = {1, 2, 3}
print(4 not in unique)  # True（使用 not in）

# 字典（默认检查键，检查值需要加.values()
person = {'name': 'Alice', 'age': 30}
print('name' in person)    # True
print('Alice' in person)   # False
print('Alice' in person.values())  # True（检查值）
```

### match-case语句
python3.10+中存在，类似与`switch-case`语句
示例：
```python
status = 404
match status:
    case 200:
        print("成功")
    case 404:
        print("未找到")
    case _:
        print("未知状态码")
```