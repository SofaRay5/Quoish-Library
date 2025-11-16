# Python用户读写
## 终端输入
`input()`函数可以让程序暂停，让用户在终端输入信息
```python
message = input("Tell me something, and I will repeat it back to you: ") 
# 终端将显示引号内的提示词，并将输入的字符串赋值给message（也支持数字入参，同样可以以变量入参）
# 当提示词有多行的时候，可以用\n分隔
# 赋值给变量message的默认是字符串，如果要变成数字需要用int()、float()强转
print(message)
```
```python
# 排序函数
list.sort(key=None, reverse=False)

# 无参数调用######
# 数字排序
numbers = [3, 1, 4, 1, 5, 9, 2]
numbers.sort()
print(numbers)  # [1, 1, 2, 3, 4, 5, 9]

# 字符串排序
fruits = ["banana", "Apple", "cherry"]
fruits.sort()
print(fruits)  # ['Apple', 'banana', 'cherry'] (区分大小写)

# 降序排序#######
numbers.sort(reverse=True)
print(numbers)  # [9, 5, 4, 3, 2, 1, 1]

# key函数自定义函数排序
# 按字符串长度排序
words = ["apple", "kiwi", "banana", "pear"]
words.sort(key=len)
print(words)  # ['kiwi', 'pear', 'apple', 'banana']

# 按绝对值排序
values = [-5, 3, -1, 4, -2]
values.sort(key=abs)
print(values)  # [-1, -2, 3, 4, -5]

# 按小写字母排序（不区分大小写）
fruits.sort(key=str.lower)
print(fruits)  # ['Apple', 'banana', 'cherry']

# 使用匿名函数赋参key
lambda arguments: expression
# 按年龄排序
students.sort(key=lambda student: student["age"])
print(students)
# 输出: [{'name': 'Bob', 'age': 20}, {'name': 'Charlie', 'age': 22}, {'name': 'Alice', 'age': 25}]
# 按名字长度排序
students.sort(key=lambda s: len(s["name"]))
print(students)
```
## 打开文件
下面的代码实现了读取文本所有内容并打印出来
```python
with open('pi_digits.txt') as file_object: 
	contents = file_object.read() 
	print(contents)
```
## open()函数
基本语法
```python
open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, 
	closefd=True, opener=None)
```
相关重要参数如下
- 模式参数（`mode`）
模式参数决定了文件如何被打开

| 模式    | 描述                |
| ----- | ----------------- |
| `'r'` | 读取模式（默认）          |
| `'w'` | 写入模式，会覆盖已有文件      |
| `'x'` | 独占创建模式，如果文件已存在则失败 |
| `'a'` | 追加模式，在文件末尾添加内容    |
| `'b'` | 二进制模式             |
| `'t'` | 文本模式（默认）          |
| `'+'` | 更新模式（可读可写）        |
部分组合模式：
```python
# 读取文本文件（默认）
f = open("file.txt", "r")  # 或简写为 open("file.txt")

# 写入文本文件（覆盖）
f = open("file.txt", "w")

# 二进制读取
f = open("image.jpg", "rb")

# 二进制写入
f = open("data.bin", "wb")

# 读写模式（文件必须存在）
f = open("file.txt", "r+")

# 读写模式（创建新文件或覆盖已有文件）
f = open("file.txt", "w+")

# 追加模式
f = open("file.txt", "a")

# 追加和读取模式
f = open("file.txt", "a+")
```
- 编码参数（`encoding`）
指定文件的编码格式，当填入`None`时，就会选择系统默认编码，常用`'utf-8', 'gbk', 'latin-1', 'ascii'`等
- 换行参数（`newline`）
控制换行符的行为，当填入`None`时，就会选择通用换行符，可选`'\n', '\r', '\n\r'`
## read()方法
### 直接读取
可以使用`read()`方法读取`open()`返回的文件对象之中的内容
基本语法：
```python
file.read(size=-1)
```
`size`是可选参数，指定要读取的字节数，默认值为`-1`，可以一直读取到文件末尾
当文件对象是以文本模式打开时，返回字符串；当以二进制文件模式打开时，返回字节对象
```python
with open('example.txt', 'r') as file:
    # 读取前50个字符
    part1 = file.read(50)
    print(f"第一部分: {part1}")
    
    # 读取接下来的100个字符
    part2 = file.read(100)
    print(f"第二部分: {part2}")
    
    # 读取剩余所有内容
    rest = file.read()
    print(f"剩余内容: {rest}")
```
### 单行读取
可以使用`readline()`方法读取单行文本
```python
with open('example.txt', 'r') as file:
    # 读取第一行
    line1 = file.readline()
    print(f"第一行: {line1}")
    
    # 读取第二行
    line2 = file.readline()
    print(f"第二行: {line2}")
    
    # 读取第三行（限制最多读取50个字符）
    line3 = file.readline(50)
    print(f"第三行(前50字符): {line3}")
```
### 读取行到列表
可以使用`realines()`将文本中各行储存在列表中
```python
with open('example.txt', 'r') as file:
    # 读取所有行
    lines = file.readlines()
    for i, line in enumerate(lines, 1):
        print(f"第{i}行: {line.strip()}")
    
    # 读取大约100字符的行（可能包含多行）
    file.seek(0)  # 回到文件开头
    some_lines = file.readlines(100)
    print(f"大约100字符的内容: {some_lines}")
```
## seek()方法
用于移动读取文件时的指针
基本语法
```python
file.seek(offset, whence=0)
```
- `offset`：移动的字节数（在文本模式下可能是字符数，但行为可能因编码而异）
- `whence`：可选参数，指定参考位置，默认为 0
    - 0：从文件开头开始计算（默认）
    - 1：从当前位置开始计算
    - 2：从文件末尾开始计算
## write()方法
用于向文件写入数据，具体的写入模式将取决于打开的模式
基本语法
```python
file.write(string)
```
- `string`：要写入文件的字符串（文本模式）或字节对象（二进制模式）
- 返回值：写入的字符数（文本模式）或字节数（二进制模式）
文本模式写入
```python
# 写入文本文件
with open('example.txt', 'w', encoding='utf-8') as file:
    chars_written = file.write("Hello, World!\n")
    print(f"写入了 {chars_written} 个字符")
    
    # 继续写入更多内容
    chars_written2 = file.write("这是第二行内容。\n")
    print(f"又写入了 {chars_written2} 个字符")
```
二进制模式写入
```python
# 写入二进制数据
with open('data.bin', 'wb') as file:
    bytes_written = file.write(b'\x00\x01\x02\x03\x04\x05')
    print(f"写入了 {bytes_written} 个字节")
    
    # 写入字节数组
    byte_array = bytearray([10, 20, 30, 40, 50])
    bytes_written2 = file.write(byte_array)
    print(f"又写入了 {bytes_written2} 个字节")
```
