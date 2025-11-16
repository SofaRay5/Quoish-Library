在Python的数据处理和科学计算领域，Numpy库无疑是最核心的库之一。它提供了高性能的多维数组对象和一系列用于操作这些数组的工具。
使用之前，需要导入库
```python
import numpy as np
```
更多可以参考：
- [NumPy官方文档](https://numpy.org/doc/stable/user/index.html#user)
- [NumPy官方文档](https://numpy.org/doc/stable/user/index.html#user)
# NumPy数据类型
|      类型      |      描述       |
| :----------: | :-----------: |
|    `bool`    |    一位布尔类型     |
|    `inti`    | 由所在平台决定精度的整数  |
|    `int8`    |     8位整型      |
|   `int16`    |     16位整型     |
|   `int32`    |     32位整型     |
|   `int64`    |     64位整型     |
|   `uint8`    |    8位无符号整型    |
|   `uint16`   |   16位无符号整型    |
|   `uint32`   |   32位无符号整型    |
|   `uint64`   |   64位无符号整型    |
|  `float16`   |    16位浮点数     |
|  `float32`   |    32位浮点数     |
|  `float164`  |    64位浮点数     |
| `complex64`  | 两个32浮点数组成的复数  |
| `complex128` | 两个64位浮点数组成的复数 |
# NumPy数组
NumPy数组指的是由同一类型元素（一般是数字）组成的多维数组。
## 创建数组
有很多种方法创建数组。
### 从列表、元组创建

你可以通过使用数组函数从一个python的列表或元组创建。数组元素的类型由原数据的类型推断得到。
```python
# 从列表创建一维数组
arr1 = np.array([1, 2, 3, 4, 5])
print("一维数组:", arr1)

# 从嵌套列表创建二维数组
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("二维数组:")
print(arr2)

# 从元组创建数组
arr3 = np.array((1, 2, 3, 4))
print("从元组创建:", arr3)

# 指定数据类型
arr4 = np.array([1, 2, 3], dtype=float)
print("指定数据类型:", arr4, "类型:", arr4.dtype)
```
### 使用内置函数创建数组
我们经常遇到这样的情况，我们并不知道数组的元素，但是知道数组的大小。因此，NumPy提供了多个函数，用于创建有初始数值的占位数组，这样可以减少不必要的数组增长及运算成本。
- `zeros`函数：创建包含全是0的数组。
- `ones`函数：创建全是1的数组。
- `eye`函数：创建单位矩阵。
- `empty`函数：创建一个随机数值（未初始化）数组，其中的数值由当时的内存状态决定。
- `full`函数：创建填充特定值的数组。
这些函数创建的数组的数据类型都是默认的float64。
```python
# 创建一维全零数组
zeros_1d = np.zeros(5)
print("一维全零数组:", zeros_1d)

# 创建二维全零数组
zeros_2d = np.zeros((3, 4))
print("二维全零数组:")
print(zeros_2d)

# 创建三维全零数组
zeros_3d = np.zeros((2, 3, 4))
print("三维全零数组形状:", zeros_3d.shape)

# 指定数据类型
zeros_int = np.zeros(5, dtype=int)
print("整数零数组:", zeros_int)
'''''''''''''''''''''''''''''''''''''
# 创建一维全一数组
ones_1d = np.ones(5)
print("一维全一数组:", ones_1d)

# 创建二维全一数组
ones_2d = np.ones((2, 3))
print("二维全一数组:")
print(ones_2d)

# 指定数据类型
ones_float = np.ones(3, dtype=float)
print("浮点一数组:", ones_float)
'''''''''''''''''''''''''''''''''''''
# 创建 3x3 单位矩阵
eye = np.eye(3)
print("3x3 单位矩阵:")
print(eye)

# 创建非方阵的单位矩阵（对角线为1）
eye_rect = np.eye(3, 4)
print("3x4 单位矩阵:")
print(eye_rect)

# 创建偏移对角线的单位矩阵
eye_offset = np.eye(3, k=1)  # 对角线向上偏移1位
print("偏移对角线单位矩阵:")
print(eye_offset)
'''''''''''''''''''''''''''''''''''''
# 创建未初始化的一维数组
empty_1d = np.empty(5)
print("未初始化一维数组:", empty_1d)  # 值不确定，是内存中的随机值

# 创建未初始化的二维数组
empty_2d = np.empty((2, 3))
print("未初始化二维数组:")
print(empty_2d)
'''''''''''''''''''''''''''''''''''''
# 创建填充特定值的一维数组
full_1d = np.full(5, 7)
print("填充7的一维数组:", full_1d)

# 创建填充特定值的二维数组
full_2d = np.full((2, 3), 9)
print("填充9的二维数组:")
print(full_2d)
```
### 创建数值序列数组
- `arrage`函数：创建等差序列
- `linespace`函数：创建等间隔序列
- `logspace`函数：创建对数间隔序列
```python
# 创建 0 到 4 的数组
arr1 = np.arange(5)
print("0-4:", arr1)  # 输出: [0 1 2 3 4]

# 创建 5 到 9 的数组
arr2 = np.arange(5, 10)
print("5-9:", arr2)  # 输出: [5 6 7 8 9]

# 创建 0 到 10，步长为 2 的数组
arr3 = np.arange(0, 11, 2)
print("0-10步长2:", arr3)  # 输出: [0 2 4 6 8 10]

# 创建浮点数序列
arr4 = np.arange(0, 1, 0.1)
print("0-1步长0.1:", arr4)  # 输出: [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
'''''''''''''''''''''''''''''''''''''
# 创建 0 到 1 之间的 5 个等间隔数
arr1 = np.linspace(0, 1, 5)
print("0-1等分5份:", arr1)  # 输出: [0.   0.25 0.5  0.75 1.  ]

# 创建 0 到 2π 之间的 10 个等间隔数（用于三角函数）
arr2 = np.linspace(0, 2*np.pi, 10)
print("0-2π等分10份:", arr2)

# 不包括终点
arr3 = np.linspace(0, 1, 5, endpoint=False)
print("0-1等分5份(不包括终点):", arr3)  # 输出: [0.  0.2 0.4 0.6 0.8]
'''''''''''''''''''''''''''''''''''''
# 创建 10^0 到 10^2 之间的 5 个对数间隔数
arr1 = np.logspace(0, 2, 5)
print("10^0到10^2对数间隔:", arr1)  # 输出: [  1.      3.16227766  10.      31.6227766  100.  ]

# 创建 2^0 到 2^4 之间的 5 个对数间隔数（以2为底）
arr2 = np.logspace(0, 4, 5, base=2)
print("2^0到2^4对数间隔:", arr2)  # 输出: [ 1.  2.  4.  8. 16.]
```
### 创建随机数组
```python
# 设置随机种子以确保结果可重现
np.random.seed(42)

# 创建 0 到 1 之间的随机数
rand_1d = np.random.rand(5)
print("5个随机数:", rand_1d)

# 创建 2x3 的随机数组
rand_2d = np.random.rand(2, 3)
print("2x3随机数组:")
print(rand_2d)

# 创建标准正态分布的随机数
randn_1d = np.random.randn(5)
print("5个标准正态随机数:", randn_1d)

# 创建指定范围内的随机整数
randint_1d = np.random.randint(0, 10, 5)
print("0-10之间的5个随机整数:", randint_1d)

# 创建指定形状的随机整数数组
randint_2d = np.random.randint(0, 100, (3, 4))
print("3x4的随机整数数组:")
print(randint_2d)
```
### 从现有数组创建
使用`like`函数创建相似数组
```python
# 创建一个示例数组
example = np.array([[1, 2, 3], [4, 5, 6]])

# 创建与示例数组形状相同的全零数组
zeros_like = np.zeros_like(example)
print("与示例形状相同的全零数组:")
print(zeros_like)

# 创建与示例数组形状相同的全一数组
ones_like = np.ones_like(example)
print("与示例形状相同的全一数组:")
print(ones_like)

# 创建与示例数组形状相同的空数组
empty_like = np.empty_like(example)
print("与示例形状相同的空数组:")
print(empty_like)

# 创建与示例数组形状相同的填充数组
full_like = np.full_like(example, 7)
print("与示例形状相同的填充7的数组:")
print(full_like)
```
使用`copy()`创建数组副本
```python
# 创建原始数组
original = np.array([1, 2, 3, 4, 5])

# 创建副本
copy_arr = np.copy(original)
print("原始数组:", original)
print("副本数组:", copy_arr)

# 修改副本不会影响原始数组
copy_arr[0] = 99
print("修改后原始数组:", original)  # 不变
print("修改后副本数组:", copy_arr)  # 第一个元素变为99
```
### 创建特殊模式数组
使用 `tile()` 创建重复模式
```python
# 创建基础模式
base = np.array([1, 2, 3])

# 重复基础模式
tiled = np.tile(base, 3)
print("重复3次:", tiled)  # 输出: [1 2 3 1 2 3 1 2 3]

# 二维重复
base_2d = np.array([[1, 2], [3, 4]])
tiled_2d = np.tile(base_2d, (2, 3))
print("二维重复:")
print(tiled_2d)
```
使用 `repeat()` 创建重复元素
```python
# 创建数组
arr = np.array([1, 2, 3])

# 重复每个元素
repeated = np.repeat(arr, 3)
print("每个元素重复3次:", repeated)  # 输出: [1 1 1 2 2 2 3 3 3]

# 指定每个元素的重复次数
repeated_custom = np.repeat(arr, [2, 3, 1])
print("自定义重复次数:", repeated_custom)  # 输出: [1 1 2 2 2 3]
```
使用 `meshgrid()` 创建网格坐标
```python
# 创建一维坐标数组
x = np.linspace(-2, 2, 5)
y = np.linspace(-1, 1, 3)
print("x坐标:", x)
print("y坐标:", y)

# 创建网格坐标
X, Y = np.meshgrid(x, y)
print("X网格:")
print(X)
print("Y网格:")
print(Y)

# 可用于计算函数值
Z = X**2 + Y**2
print("函数值 Z = X^2 + Y^2:")
print(Z)
```
### 从文件加载数据创建数组
```python
# 从文本文件加载数据（假设文件存在）
data = np.loadtxt('data.txt')

# 从CSV文件加载数据
data_csv = np.genfromtxt('data.csv', delimiter=',')

# 从NPZ文件加载数据（NumPy专用格式）
data_npz = np.load('data.npz')

# 注意：在实际使用时，需要确保文件路径正确且文件存在
```
### 创建结构化数组
```python
# 定义数据类型
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f8')]

# 创建结构化数组
data = np.array([('Alice', 25, 165.5),
                 ('Bob', 30, 180.2),
                 ('Charlie', 35, 175.1)], dtype=dtype)
print("结构化数组:")
print(data)

# 访问特定字段
print("所有年龄:", data['age'])
print("所有姓名:", data['name'])
```
### 创建日期时间数组
```python
# 创建日期数组
dates = np.array(['2023-01-01', '2023-01-02', '2023-01-03'], dtype='datetime64[D]')
print("日期数组:", dates)

# 创建时间范围
date_range = np.arange('2023-01', '2023-02', dtype='datetime64[D]')
print("一月日期范围:")
print(date_range)

# 计算日期差
date1 = np.datetime64('2023-01-15')
date2 = np.datetime64('2023-01-20')
days_diff = date2 - date1
print("日期差:", days_diff)
```
### 性能优化技巧
```python
# 预分配数组比动态扩展更高效
# 不好的做法：不断追加到列表，然后转换为数组
bad_list = []
for i in range(10000):
    bad_list.append(i)
bad_arr = np.array(bad_list)

# 好的做法：预分配数组
good_arr = np.empty(10000)
for i in range(10000):
    good_arr[i] = i

# 更好的做法：使用向量化操作
best_arr = np.arange(10000)
```
## 数组的元素类型、存储大小
`np.dtype`：查看数组中元素的类型。创建数组时，可以指定dtype参数用于创建指定数据类型的数组
```python
>>>a.dtype
dtype('int64')
>>>d = np.array([1.2, 2.3, 3.4], dtype = np.float)
>>>d
array([1.2, 2.3, 3.4])
```
`np.itemsize`：用于查看数组中元素占用的字节，例如，一个元素类型是float64的数组，其中的元素占用的字节大小是8（也就是64bit/8）
```python
>>>d.itemsize
8
```
`np.data`：查看存储数组的真实内存地址
```python
>>>d.data
<memory at 0x10bf10ac8>
```
## 数组的形状、维数、大小
### 获取数组形状信息
`shape` 属性：获取数组形状
```python
# 创建数组
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("数组形状:", arr.shape)  # 输出: (2, 3)

# 获取特定维度的大小
print("行数:", arr.shape[0])  # 输出: 2
print("列数:", arr.shape[1])  # 输出: 3
```
`ndim` 属性：获取数组维数
```python
# 获取数组维度数
arr_1d = np.array([1, 2, 3])
arr_2d = np.array([[1, 2], [3, 4]])
arr_3d = np.ones((2, 3, 4))

print("一维数组维度:", arr_1d.ndim)  # 输出: 1
print("二维数组维度:", arr_2d.ndim)  # 输出: 2
print("三维数组维度:", arr_3d.ndim)  # 输出: 3
```
`size` 属性：获取数组总元素个数
```python
# 获取数组元素总数
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("数组元素总数:", arr.size)  # 输出: 6
```

### 改变数组形状
- `reshape()` 方法
```python
# 创建一维数组
arr = np.arange(12)
print("原始数组:", arr)
print("原始形状:", arr.shape)

# 重塑为二维数组
arr_2d = arr.reshape(3, 4)
print("重塑为 3x4:")
print(arr_2d)

# 重塑为三维数组
arr_3d = arr.reshape(2, 3, 2)
print("重塑为 2x3x2:")
print(arr_3d)

# 使用 -1 自动计算维度大小
arr_auto = arr.reshape(3, -1)  # -1 表示自动计算
print("使用 -1 自动计算:")
print(arr_auto)  # 形状为 (3, 4)
```
- `resize()` 方法
```python
# 创建数组
arr = np.array([[1, 2], [3, 4]])
print("原始数组:")
print(arr)

# 调整大小（可以改变元素总数）
arr.resize(3, 3)  # 新位置用0填充
print("调整大小后:")
print(arr)
```
`reshape`函数返回的是改变了形状数组，而`resize`方法改变的是数组本身的形状。
- `flatten()` 和`ravel()` 方法
```python
# 创建二维数组
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("原始数组:")
print(arr)

# flatten() 返回展平后的副本
flat_arr = arr.flatten()
print("flatten() 结果:", flat_arr)

# ravel() 返回展平后的视图（如果可能）
ravel_arr = arr.ravel()
print("ravel() 结果:", ravel_arr)

# 修改 ravel() 的结果会影响原数组
ravel_arr[0] = 99
print("修改后原数组:")
print(arr)  # 第一个元素变为99
```
NumPy数组默认按c风格存储（横向优先），因此`ravel()`函数的返回返回结果默认是先横后竖地枚举。也可以通过设置一个可选参数`ravel(order='F')`来按`FORTRAN`风格（纵向优先）返回。
### 增加和减少维度
使用 `np.newaxis` 增加维度
```python
# 创建一维数组
arr = np.array([1, 2, 3])
print("原始数组:", arr)
print("原始形状:", arr.shape)

# 增加行维度（变为列向量）
col_vector = arr[:, np.newaxis]
print("列向量:")
print(col_vector)
print("形状:", col_vector.shape)

# 增加列维度（变为行向量）
row_vector = arr[np.newaxis, :]
print("行向量:")
print(row_vector)
print("形状:", row_vector.shape)
```
`squeeze()` 方法减少维度
```python
# 创建有单维度的数组
arr = np.array([[[1], [2], [3]]])
print("原始数组:")
print(arr)
print("原始形状:", arr.shape)  # 输出: (1, 3, 1)

# 移除单维度
squeezed = np.squeeze(arr)
print("squeeze() 后:")
print(squeezed)
print("形状:", squeezed.shape)  # 输出: (3,)
```
`expand_dims()` 方法增加维度
```python
# 创建一维数组
arr = np.array([1, 2, 3])
print("原始数组:", arr)
print("原始形状:", arr.shape)

# 在指定位置增加维度
expanded = np.expand_dims(arr, axis=0)  # 在第0维增加
print("在第0维增加维度:")
print(expanded)
print("形状:", expanded.shape)  # 输出: (1, 3)

expanded2 = np.expand_dims(arr, axis=1)  # 在第1维增加
print("在第1维增加维度:")
print(expanded2)
print("形状:", expanded2.shape)  # 输出: (3, 1)
```
### 转置和轴交换
`T` 属性和 `transpose(`) 方法
```python
# 创建二维数组
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("原始数组:")
print(arr)

# 使用 T 属性转置
transposed = arr.T
print("转置后:")
print(transposed)

# 使用 transpose() 方法
transposed2 = arr.transpose()
print("transpose() 结果:")
print(transposed2)

# 对于高维数组，可以指定轴顺序
arr_3d = np.arange(24).reshape(2, 3, 4)
print("三维数组形状:", arr_3d.shape)

# 重新排列轴
rearranged = arr_3d.transpose(1, 0, 2)
print("重新排列轴后形状:", rearranged.shape)
```
`swapaxes()` 方法交换轴
```python
# 创建三维数组
arr = np.arange(24).reshape(2, 3, 4)
print("原始形状:", arr.shape)  # 输出: (2, 3, 4)

# 交换第0轴和第1轴
swapped = np.swapaxes(arr, 0, 1)
print("交换轴0和1后形状:", swapped.shape)  # 输出: (3, 2, 4)

# 交换第1轴和第2轴
swapped2 = np.swapaxes(arr, 1, 2)
print("交换轴1和2后形状:", swapped2.shape)  # 输出: (2, 4, 3)
```
`moveaxis()` 方法移动轴
```python
# 创建四维数组
arr = np.arange(24).reshape(2, 3, 1, 4)
print("原始形状:", arr.shape)  # 输出: (2, 3, 1, 4)

# 将第2轴移动到第0位置
moved = np.moveaxis(arr, 2, 0)
print("移动轴后形状:", moved.shape)  # 输出: (1, 2, 3, 4)

# 将多个轴移动到指定位置
moved2 = np.moveaxis(arr, [0, 1], [2, 3])
print("移动多个轴后形状:", moved2.shape)  # 输出: (1, 4, 2, 3)
```
### 连接和分割数组
`concatenate()` 方法连接数组
```python
# 创建两个数组
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# 沿行方向连接（增加行数）
concatenated_rows = np.concatenate((arr1, arr2), axis=0)
print("沿行方向连接:")
print(concatenated_rows)

# 沿列方向连接（增加列数）
concatenated_cols = np.concatenate((arr1, arr2), axis=1)
print("沿列方向连接:")
print(concatenated_cols)
```
`stack()` 方法堆叠数组
```python
# 创建两个一维数组
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 沿新轴堆叠
stacked = np.stack((a, b))
print("堆叠后:")
print(stacked)
print("形状:", stacked.shape)  # 输出: (2, 3)

# 指定堆叠轴
stacked_axis1 = np.stack((a, b), axis=1)
print("沿轴1堆叠:")
print(stacked_axis1)
print("形状:", stacked_axis1.shape)  # 输出: (3, 2)
```
`split()` 方法分割数组
```python
# 创建数组
arr = np.arange(12).reshape(3, 4)
print("原始数组:")
print(arr)

# 均等分割
splits = np.split(arr, 3, axis=0)  # 沿行方向分成3份
print("分割结果:")
for i, part in enumerate(splits):
    print(f"第{i}部分:")
    print(part)

# 按指定位置分割
splits_custom = np.split(arr, [1, 3], axis=1)  # 在第1列和第3列处分割
print("自定义分割:")
for i, part in enumerate(splits_custom):
    print(f"第{i}部分:")
    print(part)
```
## 打印数组
当你在屏幕打印一个数组时，NumPy显示这个数组的方式和嵌套的列表是相似的。但遵守以下布局：
- 最后一维由左至右打印
- 倒数第二维从上到下打印
- 其余维都是从上到下打印，且通过空行分隔
如下所示，一维数组输出为一行、二维为矩阵、三维为矩阵列表。
```python
>>>a = np.arange(6)  
>>>a
array([0, 1, 2, 3, 4, 5])
>>>b = np.arange(12).reshape(4,3) # reshape函数用于把数组重组为指定形状
>>>b
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])
>>>c = np.arange(24).reshape(2,3,4)
>>>c
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
```
如果数组太大了，NumPy自动跳过中间的部分不显示，只显示两边。
```python
>>>print(np.arange(10000))
[   0    1    2 ... 9997 9998 9999]
>>>print(np.arange(10000).reshape(100,100))
[[   0    1    2 ...   97   98   99]
 [ 100  101  102 ...  197  198  199]
 [ 200  201  202 ...  297  298  299]
 ...
 [9700 9701 9702 ... 9797 9798 9799]
 [9800 9801 9802 ... 9897 9898 9899]
 [9900 9901 9902 ... 9997 9998 9999]]
```
如果你想强制输出所有数据，可以设置`set_printoptions`参数。
```python
>>>np.set_printoptions(threshold=np.nan)
```
## 基本运算
数组中的算术运算一般是元素级的运算，运算结果会产生一个新的数组。
不同于很多矩阵语言，乘积运算操作`*`在NumPy中是元素级的运算。如果想要进行矩阵运算，可以使用`dot`函数或方法。
```python
>>>A = np.array( [[1,1],[0,1]] )
>>>B = np.array( [[2,0],[3,4]] )
>>>A*B# 元素乘积
array([[2, 0],
       [0, 4]])
>>>A.dot(B)# 矩阵运算
array([[5, 4],
       [3, 4]])
>>>np.dot(A, B)# 另一种方式矩阵运算
array([[5, 4],
       [3, 4]])
```
一些运算，例如`+=`和`*=`，会内在的改变一个数组的值，而不是生成一个新的数组。
```python
>>>a = np.ones((2,3), dtype=int)
>>>b = np.random.random((2,3))
>>>a *= 3
>>>a
array([[3, 3, 3],
       [3, 3, 3]])
>>>b += a
>>>b
array([[3.69902298, 3.1334804 , 3.62673199],
       [3.37038178, 3.74769131, 3.62235315]])
>>>a += b # b不会自动转换为整型
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
```
当操作不同数据类型的数组时，最后输出的数组类型一般会与更普遍或更精准的数组相同（这种行为叫做 Upcasting）。
```python
>>>a = np.ones(3, dtype=np.int32)
>>>b = np.linspace(0,pi,3)
>>>b.dtype.name
'float64'
>>>c = a+b
>>>c.dtype.name
'float64'
>>>d = np.exp(c*1j)
>>>d.dtype.name
'complex128'
```
许多一元运算，如计算数组中所有元素的总和，是属于 ndarray 类的方法。
```python
>>>a = np.random.random((2,3))
>>>a
array([[0.85827711, 0.5385761 , 0.0843277 ],
       [0.2609027 , 0.36414539, 0.12940627]])
>>>a.sum()
2.2356352707158513
>>>a.min()
0.08432769616897462
>>>a.max()
0.8582771053112916

# max(), min(), sum()都可以填入参数max(axis), min(axis), sum(axis)，可以按行列取最值或求和
```
默认状态下，这些运算会把数组视为一个数字列表而不关心它的shape。然而，可以指定axis参数针对哪一个维度进行运算。例如axis=0将针对每一个列进行运算。
```python
>>>b = np.arange(12).reshape(3,4)
>>>b
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>>b.sum(axis=0)   # 列相加
array([12, 15, 18, 21])
>>>b.min(axis=1)   # 行相加
array([0, 4, 8])
>>>b.cumsum(axis=1)# 行累加
array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38]])
```
### 矩阵乘法
### 矩阵除法
| **函数**                  | **行为**                | **示例**        |
| ----------------------- | --------------------- | ------------- |
| `np.divide(A, B)`       | 浮点除法（Python 3 等价 `/`） | `5 / 2 → 2.5` |
| `np.floor_divide(A, B)` | 向下取整除法（等价 `//`）       | `5 // 2 → 2`  |
| `np.true_divide(A, B)`  | 强制浮点结果（兼容 Python 2）   | `5 / 2 → 2.5` |
这三个都是元素除法。
## 通用函数
NumPy提供一些熟悉的数学函数，例如sin, cos,和exp等。在NumPy中，这些函数称为“通用函数”， 这些运算是元素级的，生成一个数组作为结果。
```python
>>>B = np.arange(3)
>>>B
array([0, 1, 2])
>>>np.exp(B)
array([1.        , 2.71828183, 7.3890561 ])
>>>np.sqrt(B)
array([0.        , 1.        , 1.41421356])
>>>C = np.array([2., -1., 4.])
>>>np.add(B, C)
array([2., 0., 6.])
```
## 索引、切片和迭代
### 索引访问
数字索引
```python
# 创建一维数组
arr = np.array([10, 20, 30, 40, 50])
print("原始数组:", arr)

# 正索引（从0开始）
print("arr[0]:", arr[0])    # 第一个元素 -> 10
print("arr[2]:", arr[2])    # 第三个元素 -> 30

# 负索引（从末尾开始）
print("arr[-1]:", arr[-1])   # 最后一个元素 -> 50
print("arr[-2]:", arr[-2])   # 倒数第二个元素 -> 40
''''''''''''''''''''''''''''''''
# 创建二维数组
arr_2d = np.array([[1, 2, 3], 
                   [4, 5, 6], 
                   [7, 8, 9]])
print("二维数组:")
print(arr_2d)

# 访问单个元素
print("arr_2d[0, 0]:", arr_2d[0, 0])   # 第一行第一列 -> 1
print("arr_2d[1, 2]:", arr_2d[1, 2])   # 第二行第三列 -> 6
print("arr_2d[2, 1]:", arr_2d[2, 1])   # 第三行第二列 -> 8

# 也可以使用逗号分隔的多个索引
print("arr_2d[2][1]:", arr_2d[2][1])   # 同上 -> 8
```
数组索引
```python
# 创建大数组
big_arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# 使用整数列表索引
indices = [0, 2, 4, 6, 8]
result = big_arr[indices]
print("使用列表索引:", result)  # 输出: [10 30 50 70 90]

# 使用整数数组索引
indices_arr = np.array([1, 3, 5, 7, 9])
result = big_arr[indices_arr]
print("使用数组索引:", result)  # 输出: [20 40 60 80 100]

# 使用负数索引
negative_indices = np.array([-1, -2, -3])
result = big_arr[negative_indices]
print("使用负数索引:", result)  # 输出: [100 90 80]
''''''''''''''''''''''''''''''''''''
# 创建二维大数组
big_2d = np.arange(25).reshape(5, 5)
print("二维大数组:")
print(big_2d)

# 使用一维索引数组选择行
row_indices = np.array([0, 2, 4])
selected_rows = big_2d[row_indices]
print("选择的行:")
print(selected_rows)

# 使用两个一维索引数组选择特定元素
row_idx = np.array([0, 1, 2])
col_idx = np.array([1, 2, 3])
selected_elements = big_2d[row_idx, col_idx]
print("选择的特定元素:", selected_elements)  # 输出: [1 7 13]
```
布尔索引
```python
# 创建数组
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print("原始数组:", arr)

# 创建布尔掩码
mask = arr > 5
print("布尔掩码:", mask)  # 输出: [False False False False False  True  True  True  True]

# 使用布尔索引
print("arr[mask]:", arr[mask])  # 输出: [6 7 8 9]

# 直接使用条件表达式
print("arr[arr > 5]:", arr[arr > 5])  # 输出: [6 7 8 9]

# 组合条件
print("arr[(arr > 3) & (arr < 7)]:", arr[(arr > 3) & (arr < 7)])  # 输出: [4 5 6]

# 二维数组的布尔索引
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mask_2d = arr_2d > 5
print("二维布尔索引:")
print(arr_2d[mask_2d])  # 输出: [6 7 8 9]
```
`np.ix_()` 函数用于创建开放网格，以便从多个维度进行索引。
```python
# 创建二维数组
arr = np.arange(20).reshape(4, 5)
print("原始数组:")
print(arr)

# 使用 ix_() 选择特定行和列
row_indices = np.array([0, 2])
col_indices = np.array([1, 3])
selected = arr[np.ix_(row_indices, col_indices)]
print("使用 ix_() 选择的结果:")
print(selected)
# 输出:
# [[ 1  3]
#  [11 13]]
```
### 切片
```python
# 创建一维数组
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print("原始数组:", arr)

# 基本切片
print("arr[2:5]:", arr[2:5])        # 索引2到5(不包括5) -> [2 3 4]
print("arr[:5]:", arr[:5])          # 从开始到索引5 -> [0 1 2 3 4]
print("arr[5:]:", arr[5:])          # 从索引5到末尾 -> [5 6 7 8 9]
print("arr[::2]:", arr[::2])        # 每隔一个元素 -> [0 2 4 6 8]
print("arr[1::2]:", arr[1::2])      # 从索引1开始，每隔一个元素 -> [1 3 5 7 9]
print("arr[::-1]:", arr[::-1])      # 反转数组 -> [9 8 7 6 5 4 3 2 1 0]
''''''''''''''''''''''''''''''''''''
# 创建二维数组
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])
print("原始二维数组:")
print(arr_2d)

# 行切片
print("arr_2d[1:3]:")          # 第2行到第3行(不包括第3行)
print(arr_2d[1:3])

# 列切片
print("arr_2d[:, 1:3]:")       # 所有行，第2列到第3列
print(arr_2d[:, 1:3])

# 行列组合切片
print("arr_2d[0:2, 1:3]:")     # 第1-2行，第2-3列
print(arr_2d[0:2, 1:3])

# 步长切片
print("arr_2d[::2, ::2]:")     # 每隔一行，每隔一列
print(arr_2d[::2, ::2])
''''''''''''''''''''''''''''''''''''''
# 创建三维数组
arr_3d = np.arange(24).reshape(2, 3, 4)
print("三维数组形状:", arr_3d.shape)
print("原始三维数组:")
print(arr_3d)

# 三维切片
print("arr_3d[0, :, :]:")      # 第一个块的所有行和列
print(arr_3d[0, :, :])

print("arr_3d[:, 1, :]:")      # 所有块的第二行的所有列
print(arr_3d[:, 1, :])

print("arr_3d[:, :, ::2]:")    # 所有块的所有行，每隔一列
print(arr_3d[:, :, ::2])
```
可以使用省略号(...)简化高维切片
```python
# 创建四维数组
arr_4d = np.arange(81).reshape(3, 3, 3, 3)

# 使用省略号简化切片
print("arr_4d[0, ..., 0]:")  # 等同于arr_4d[0, :, :, 0]
print(arr_4d[0, ..., 0])

print("arr_4d[..., 1]:")     # 等同于arr_4d[:, :, :, 1]
print(arr_4d[..., 1])
```
### 获取索引
`where()`函数：返回满足条件的点各个维的坐标，每个维的坐标用一个元组储存在一起。
```python
arr = np.array([1, 2, 0, 4, 0, 6, 0, 8])
# 获取满足条件的元素的索引
indices = np.where(arr != 0)
print(indices)  # 输出: (array([0, 1, 3, 5, 7]),)

# 获取满足条件的元素
values = arr[np.where(arr != 0)]
print(values)  # 输出: [1 2 4 6 8]
```
`argwhere()`函数：返回满足条件的点在各个维的坐标，每个点的坐标用元组储存在一起。
```python
arr = np.array([1, 2, 0, 4, 0, 6, 0, 8])
# 获取满足条件的元素的索引（返回二维数组）
indices = np.argwhere(arr != 0)
print(indices)
# 输出:
# [[0]
#  [1]
#  [3]
#  [5]
#  [7]]
```
获取布尔索引
```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# 创建布尔掩码
mask = arr > 5
print(mask)  # 输出: [False False False False False  True  True  True  True  True]

# 使用布尔索引获取元素
selected = arr[mask]
print(selected)  # 输出: [6 7 8 9 10]

# 直接使用条件表达式
selected = arr[arr > 5]
print(selected)  # 输出: [6 7 8 9 10]
```
`argsort()`函数：获取排序后的索引
```python
arr = np.array([3, 1, 4, 2, 5])
# 获取排序后的索引
indices = np.argsort(arr)
print(indices)  # 输出: [1 3 0 2 4]

# 使用索引获取排序后的数组
sorted_arr = arr[indices]
print(sorted_arr)  # 输出: [1 2 3 4 5]
```
`argmin()`和`argmax()`函数
```python
arr = np.array([3, 1, 4, 2, 5])
# 获取最大值和最小值的索引
max_index = np.argmax(arr)
min_index = np.argmin(arr)
print(f"最大值索引: {max_index}, 最小值索引: {min_index}")  # 输出: 最大值索引: 4, 最小值索引: 1
```
使用ndenumerate()获取索引
```python
# 创建二维数组
arr_2d = np.array([[1, 2], [3, 4]])
print("使用ndenumerate获取索引和值:")
for index, value in np.ndenumerate(arr_2d):
    print(f"索引 {index}: 值 {value}")
```
一维数组可以索引、切片和迭代，就像列表和其他python数据类型。
```python
>>>a = np.arange(10)**3
>>>a[2]
8
>>>a[2:5] # 从索引2到索引5切片
array([ 8, 27, 64])
>>>a[:6:2] = -1000 # 从0到6，每隔2个设为-1000
>>>a
array([-1000,     1, -1000,    27, -1000,   125,   216,   343,   512,
         729])
>>>a[ : :-1] # 翻转数组a
array([  729,   512,   343,   216,   125, -1000,    27, -1000,     1,
       -1000])
```
多维数组可以每个维度有一个索引值。这些索引值被逗号分开。
```python
>>>def f(x, y): # 定义一个函数用于生成数组
>>>    return 10 * x + y
>>>b = np.fromfunction(f, (5,4), dtype=int) # 从函数f生成数组，数组的形状是(5,4)，数组中(x,y)的元素值等于f(x,y)
>>>b
array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])
>>>b[2,3] #第2行，第3列
23
>>>b[0:5, 1] #等于b[ : ,1]  ，第0-5行，第1列
array([ 1, 11, 21, 31, 41])
>>>b[1:3, : ]  #1-3行，所有列
array([[10, 11, 12, 13],
       [20, 21, 22, 23]])
```
当索引值的数量少于数组的维度时，其他的索引值默认为分号`：`
```python
>>>b[-1] #相当于b[-1,:]
array([40, 41, 42, 43])
```
`b[i]`中的i代表i后面有足够多的：，用于表示其他维度的索引。你也可以使用点号`...`来表示。  
点号代表需要的足够多的列，用于使其他维度的索引值完整。例如，x是一个五维数组，那么
- x[1,2,...] 相当于 x[1,2,:,:,:]
- x[...,3] 相当于 x[:,:,:,:,3]
- x[4,...,5,:] 相当于 x[4,:,:,5,:]
```python
>>>c = np.array( [[[  0,  1,  2],  #构建一个三维数组
                [ 10, 12, 13]],
               [[100,101,102],
                [110,112,113]]])
>>>c.shape
(2, 2, 3)
>>>c[1,...]
array([[100, 101, 102],
       [110, 112, 113]])
>>>c[...,2]
array([[  2,  13],
       [102, 113]])
```
多维数组中的迭代：
```python
>>>for row in b:
>>>    print(row)
[0 1 2 3]
[10 11 12 13]
[20 21 22 23]
[30 31 32 33]
[40 41 42 43]
```
### 迭代
基本迭代方法
```python
# 创建一维数组
arr_1d = np.array([1, 2, 3, 4, 5])
print("一维数组:", arr_1d)

# 基本迭代
print("基本迭代:")
for element in arr_1d:
    print(element, end=" ")
# 输出: 1 2 3 4 5
''''''''''''''''''''''''''''''''''''
# 创建二维数组
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("二维数组:")
print(arr_2d)

# 默认迭代（按行迭代）
print("按行迭代:")
for row in arr_2d:
    print(row, end=" ")
# 输出: [1 2 3] [4 5 6] [7 8 9]

# 元素级迭代
print("\n元素级迭代:")
for row in arr_2d:
    for element in row:
        print(element, end=" ")
# 输出: 1 2 3 4 5 6 7 8 9
''''''''''''''''''''''''''''''''''''
# 创建三维数组
arr_3d = np.arange(24).reshape(2, 3, 4)
print("三维数组形状:", arr_3d.shape)

# 默认迭代（按第一个轴迭代）
print("按第一个轴迭代:")
for plane in arr_3d:
    print("平面:")
    print(plane)
```
`nditer()` 是 NumPy 提供的高效迭代器，可以处理多维数组并控制迭代顺序。
```python
# 创建二维数组
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("原始数组:")
print(arr)

# 使用 nditer 迭代所有元素
print("nditer 迭代:")
for x in np.nditer(arr):
    print(x, end=" ")
# 输出: 1 2 3 4 5 6
'''''''''`nditer()`控制迭代顺序'''''''''
# 创建数组
arr = np.arange(12).reshape(3, 4)
print("原始数组:")
print(arr)

# C顺序迭代（行优先）
print("C顺序迭代（行优先）:")
for x in np.nditer(arr, order='C'):
    print(x, end=" ")
# 输出: 0 1 2 3 4 5 6 7 8 9 10 11

# F顺序迭代（列优先）
print("\nF顺序迭代（列优先）:")
for x in np.nditer(arr, order='F'):
    print(x, end=" ")
# 输出: 0 4 8 1 5 9 2 6 10 3 7 11
'''''''''''''''修改迭代中的值'''''''''
# 创建数组
arr = np.array([1, 2, 3, 4, 5])
print("原始数组:", arr)

# 使用 nditer 修改值
for x in np.nditer(arr, op_flags=['readwrite']):
    x[...] = x * 2  # 将每个元素乘以2

print("修改后的数组:", arr)  # 输出: [2 4 6 8 10]
'''''''''''''''外部迭代模式''''''''''
# 创建数组
arr = np.arange(6).reshape(2, 3)
print("原始数组:")
print(arr)

# 使用外部迭代模式
it = np.nditer(arr, flags=['external_loop'])
for x in it:
    print("外部循环块:", x)
# 输出: 外部循环块: [0 1 2 3 4 5]
```
## 将数组拆分为更小的数组
使用`hsplit`，你可以沿着水平方向拆分数组，即可以通过指定平均切分的数量，也可以通过指定切分的列来实现。
```python
>>> a = np.floor(10*np.random.random((2,12)))
>>> a
array([[0., 2., 5., 6., 0., 2., 0., 2., 1., 5., 6., 2.],
       [3., 3., 2., 3., 6., 8., 8., 5., 6., 2., 0., 2.]])
>>> np.hsplit(a,3)   # 延水平方向切分为3个相同大小的数组
[array([[0., 2., 5., 6.],
        [3., 3., 2., 3.]]), array([[0., 2., 0., 2.],
        [6., 8., 8., 5.]]), array([[1., 5., 6., 2.],
        [6., 2., 0., 2.]])]
np.hsplit(a,(3,4))   # 从第3第4列切分a
[array([[0., 2., 5.],
        [3., 3., 2.]]), array([[6.],
        [3.]]), array([[0., 2., 0., 2., 1., 5., 6., 2.],
        [6., 8., 8., 5., 6., 2., 0., 2.]])]
```
`vsplit`沿着垂直方向切分，`array_split` 可以指定沿着哪个维度切分。
## View或浅复制
不同的数组对象能够共享相同的数据。`view`方法创建一个新的数组对象，这个对象与原始数组使用相同的数据。
```python
>>> c = a.view()     # c是a的view，或者说c是a的浅复制，c是另一个对象
>>> c is a
False
>>> c.base is a      # 确切的说，c是数值的view，数值的属于a
True
>>> c.shape = 2,6    # 如果改变c的形状，并不影响a
>>> a.shape
(3, 4)
>>> c[0,4] = 1234    # 但是，如果改变c的数值，a也会受影响
>>> a
array([[   0,    1,    2,    3],
       [1234,    5,    6,    7],
       [   8,    9,   10,   11]])
```
以上在python中称作view，或者叫浅复制，切片操作就会返回一个view：
```python
>>> s = a[ : , 1:3]     # s是a的切片
>>> s[:] = 10           # 改变s的值也会影响a
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```
## 深复制
`copy`方法可以生成一个数组的完整备份。
```python
>>> d = a.copy()  # 生成一个新的数组对象
>>> d is a
False
>>> d.base is a   # d与a不共享任何数值
False
>>> d[0,0] = 9999   # 改变d的数值，也不会影响a
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```
## np.tile()函数
`numpy.tile()` 函数用于将数组沿指定维度重复多次，创建更大的数组。这个函数在数据预处理、矩阵扩展和模式复制等场景中非常有用。
### 基本语法
`numpy.tile(A, reps)`
- `A`: 输入数组，可以是任意维度的数组
- `reps`: 重复的次数，可以是一个整数或一个表示各维度重复次数的元组
### 示例
```python
# 创建一维数组
>>>a = np.array([1, 2, 3])
# 沿行方向重复 2 次
>>>b = np.tile(a, 2)
>>>print(b)  
[1 2 3 1 2 3]
# 沿列方向重复 3 次
>>>c = np.tile(a, (3,1))
# 输出：
[[1 2 3]
 [1 2 3]
 [1 2 3]]

# 更复杂的重复操作
>>>d = np.tile(a, (2,3,1))
>>>print(d)
[[[1 2 3]
  [1 2 3]
  [1 2 3]]

 [[1 2 3]
  [1 2 3]
  [1 2 3]]]
>>>e=np.array([[1,2],[3,4]])
>>>f=np.tile(e, (3,2))
>>>print(f)
[[1 2 1 2]
 [3 4 3 4]
 [1 2 1 2]
 [3 4 3 4]
 [1 2 1 2]
 [3 4 3 4]]
```
## np.sum()函数
### 基本语法
```python
numpy.sum(a, axis=None, dtype=None, out=None, 
			keepdims=<no value>, initial=<no value>, where=<no value>)
```
- `a`: 输入数组
- `axis`: 沿着哪个轴求和（默认为 None，表示对所有元素求和）
- `dtype`: 计算结果的数据类型（默认为 None，使用输入数组的数据类型）
- `keepdims`: 布尔值，如果为 True，则保持求和后的维度与原始数组相同
- `initial`: 求和的起始值
- `where`: 布尔数组，指定哪些元素参与求和
### 示例
```python
# 对于二维数组
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
                   
# 计算所有元素的和
total = np.sum(arr_2d) # 45

# 按行求和（沿轴0）
sum_axis0 = np.sum(arr_2d, axis=0) # [12, 15, 18]

# 按列求和（沿轴1）
sum_axis1 = np.sum(arr_2d, axis=1) # [ 6, 15, 24]

# 对于三维数组

arr_3d = np.array([[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]]])
                   
# 沿第一个轴求和
sum_axis0 = np.sum(arr_3d, axis=0)  
# [[ 6  8]
# [10 12]]

# 沿第二个轴求和
sum_axis1 = np.sum(arr_3d, axis=1)  
#[[ 4  6]
# [12 14]]

# 沿第三个轴求和
sum_axis2 = np.sum(arr_3d, axis=2)  
#[[ 3  7]
# [11 15]]
```
轴序最小的轴是最新加的轴。
## argsort()函数
`numpy.argsort()` 返回的是数组排序后的索引值，而不是排序后的值本身。
### 基本语法
```python
numpy.argsort(a, axis=-1, kind=None, order=None)
```
- `a`: 要排序的输入数组
- `axis`: 沿着哪个轴进行排序（默认为 -1，表示最后一个轴）
- `kind`: 排序算法，可选 'quicksort', 'mergesort', 'heapsort', 'stable'（默认为 'quicksort'）
- `order`: 如果数组是结构化数组，可以指定按哪个字段排序
**返回值**：索引数组，表示排序后元素在原数组中的位置
### 一维数组
```python
# 创建一维数组
arr = np.array([3, 1, 4, 2, 5])
print("原始数组:", arr)

# 获取排序后的索引
indices = np.argsort(arr)
print("排序索引:", indices)  # 输出: [1 3 0 2 4]

# 使用索引获取排序后的数组
sorted_arr = arr[indices]
print("排序后的数组:", sorted_arr)  # 输出: [1 2 3 4 5]

# 获取降序排序的索引（两种方法）
# 方法1: 先升序排序，然后反转索引
asc_indices = np.argsort(arr)
desc_indices = asc_indices[::-1]
print("降序索引:", desc_indices)  # 输出: [4 2 0 3 1]

# 方法2: 对数组的负数进行argsort
desc_indices_alt = np.argsort(-arr)
print("降序索引(方法2):", desc_indices_alt)  # 输出: [4 2 0 3 1]

# 使用降序索引获取排序后的数组
sorted_desc = arr[desc_indices]
print("降序排序后的数组:", sorted_desc)  # 输出: [5 4 3 2 1]
```
### 多维数组
```python
# 创建二维数组
arr_2d = np.array([[3, 1, 4],
                   [2, 5, 0],
                   [7, 6, 8]])
print("原始二维数组:")
print(arr_2d)

# 沿最后一个轴排序（默认，即每行内部排序）
row_sorted_indices = np.argsort(arr_2d)
print("每行排序索引:")
print(row_sorted_indices)
# 输出:
# [[1 0 2]  # 第一行: [1, 0, 2] 表示最小值在索引1，中间值在索引0，最大值在索引2
#  [2 0 1]  # 第二行: [2, 0, 1] 表示最小值在索引2，中间值在索引0，最大值在索引1
#  [1 0 2]] # 第三行: [1, 0, 2] 表示最小值在索引1，中间值在索引0，最大值在索引2

# 沿第一个轴排序（即每列内部排序）
col_sorted_indices = np.argsort(arr_2d, axis=0)
print("每列排序索引:")
print(col_sorted_indices)
# 输出:
# [[1 0 1]  # 第一列: [1, 0, 1] 表示最小值在索引1，中间值在索引0，最大值在索引1
#  [0 2 0]  # 第二列: [0, 2, 0] 表示最小值在索引0，中间值在索引2，最大值在索引0
#  [2 1 2]] # 第三列: [2, 1, 2] 表示最小值在索引2，中间值在索引1，最大值在索引2

# 使用索引获取排序后的二维数组

# 按行排序
sorted_by_row = np.take_along_axis(arr_2d, row_sorted_indices, axis=1)
print("按行排序后的数组:")
print(sorted_by_row)
# 输出:
# [[1 3 4]  # 第一行排序后
#  [0 2 5]  # 第二行排序后
#  [6 7 8]] # 第三行排序后

# 按列排序
sorted_by_col = np.take_along_axis(arr_2d, col_sorted_indices, axis=0)
print("按列排序后的数组:")
print(sorted_by_col)
# 输出:
# [[2 1 0]  # 第一列排序后
#  [3 5 4]  # 第二列排序后
#  [7 6 8]] # 第三列排序后
```
# 数组计算
## 基本运算
### 基本算术运算
```python
import numpy as np

# 创建两个数组
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# 加法
print("a + b =", a + b)  # 输出: [6 8 10 12]

# 减法
print("a - b =", a - b)  # 输出: [-4 -4 -4 -4]

# 乘法（元素级，不是矩阵乘法）
print("a * b =", a * b)  # 输出: [5 12 21 32]

# 除法
print("a / b =", a / b)  # 输出: [0.2 0.33333333 0.42857143 0.5]

# 幂运算
print("a ** 2 =", a ** 2)  # 输出: [1 4 9 16]

# 取模运算
print("a % 2 =", a % 2)  # 输出: [1 0 1 0]

# 取整除法
print("a // 2 =", a // 2)  # 输出: [0 1 1 2]
```
### 标量与数组
```python
# 标量与数组的运算（广播机制）
a = np.array([1, 2, 3, 4])
print("a + 5 =", a + 5)  # 输出: [6 7 8 9]
print("a * 2 =", a * 2)  # 输出: [2 4 6 8]
print("2 ** a =", 2 ** a)  # 输出: [2 4 8 16]
```
### 就地运算
```python
# 就地运算（修改原数组）
a = np.array([1, 2, 3, 4])
a += 2
print("a after += 2:", a)  # 输出: [3 4 5 6]

a *= 2
print("a after *= 2:", a)  # 输出: [6 8 10 12]
```
## 比较运算
NumPy 支持元素级的比较运算，返回布尔数组。
```python
a = np.array([1, 2, 3, 4])
b = np.array([2, 2, 3, 5])

# 等于
print("a == b:", a == b)  # 输出: [False True True False]

# 不等于
print("a != b:", a != b)  # 输出: [True False False True]

# 大于
print("a > b:", a > b)  # 输出: [False False False False]

# 小于
print("a < b:", a < b)  # 输出: [True False False True]

# 大于等于
print("a >= b:", a >= b)  # 输出: [False True True False]

# 小于等于
print("a <= b:", a <= b)  # 输出: [True True True True]
```
## 逻辑运算
```python
a = np.array([True, False, True, False])
b = np.array([True, True, False, False])

# 逻辑与
print("np.logical_and(a, b):", np.logical_and(a, b))  # 输出: [True False False False]

# 逻辑或
print("np.logical_or(a, b):", np.logical_or(a, b))  # 输出: [True True True False]

# 逻辑非
print("np.logical_not(a):", np.logical_not(a))  # 输出: [False True False True]

# 逻辑异或
print("np.logical_xor(a, b):", np.logical_xor(a, b))  # 输出: [False True True False]
```
## 数学函数
### 基本数学函数
```python
a = np.array([1, 4, 9, 16, 25])

# 平方根
print("np.sqrt(a):", np.sqrt(a))  # 输出: [1. 2. 3. 4. 5.]

# 指数函数
print("np.exp(a):", np.exp(a))  # 输出: [2.71828183e+00 5.45981500e+01 8.10308393e+03 8.88611052e+06 7.20048993e+10]

# 对数函数
print("np.log(a):", np.log(a))  # 输出: [0. 1.38629436 2.19722458 2.77258872 3.21887582]
print("np.log10(a):", np.log10(a))  # 输出: [0. 0.60205999 0.95424251 1.20411998 1.39794001]

# 三角函数
angles = np.array([0, 30, 45, 60, 90]) * np.pi / 180  # 转换为弧度
print("np.sin(angles):", np.sin(angles))
print("np.cos(angles):", np.cos(angles))
print("np.tan(angles):", np.tan(angles))
```
### 舍入函数
```python
a = np.array([1.23, 4.56, 7.89, 10.11])

# 四舍五入
print("np.round(a):", np.round(a))  # 输出: [1. 5. 8. 10.]

# 向下取整
print("np.floor(a):", np.floor(a))  # 输出: [1. 4. 7. 10.]

# 向上取整
print("np.ceil(a):", np.ceil(a))  # 输出: [2. 5. 8. 11.]

# 截断小数部分
print("np.trunc(a):", np.trunc(a))  # 输出: [1. 4. 7. 10.]
```
## 统计运算
### 基本统计函数
```python
a = np.array([[1, 2, 3], [4, 5, 6]])

# 求和
print("np.sum(a):", np.sum(a))  # 输出: 21
print("np.sum(a, axis=0):", np.sum(a, axis=0))  # 输出: [5 7 9] (按列求和)
print("np.sum(a, axis=1):", np.sum(a, axis=1))  # 输出: [6 15] (按行求和)

# 求平均值
print("np.mean(a):", np.mean(a))  # 输出: 3.5
print("np.mean(a, axis=0):", np.mean(a, axis=0))  # 输出: [2.5 3.5 4.5]

# 求标准差
print("np.std(a):", np.std(a))  # 输出: 1.707825127659933

# 求方差
print("np.var(a):", np.var(a))  # 输出: 2.9166666666666665

# 求最小值
print("np.min(a):", np.min(a))  # 输出: 1
print("np.min(a, axis=0):", np.min(a, axis=0))  # 输出: [1 2 3]

# 求最大值
print("np.max(a):", np.max(a))  # 输出: 6
print("np.max(a, axis=0):", np.max(a, axis=0))  # 输出: [4 5 6]

# 求中位数
print("np.median(a):", np.median(a))  # 输出: 3.5
```
### 累积运算
```python
a = np.array([1, 2, 3, 4, 5])

# 累积和
print("np.cumsum(a):", np.cumsum(a))  # 输出: [1 3 6 10 15]

# 累积积
print("np.cumprod(a):", np.cumprod(a))  # 输出: [1 2 6 24 120]

# 差分
print("np.diff(a):", np.diff(a))  # 输出: [1 1 1 1]
```
## 集合运算
```python
a = np.array([1, 2, 3, 2, 1, 4, 5])
b = np.array([3, 4, 5, 6, 7])

# 找出唯一元素
print("np.unique(a):", np.unique(a))  # 输出: [1 2 3 4 5]

# 求交集
print("np.intersect1d(a, b):", np.intersect1d(a, b))  # 输出: [3 4 5]

# 求并集
print("np.union1d(a, b):", np.union1d(a, b))  # 输出: [1 2 3 4 5 6 7]

# 求差集（在a中但不在b中）
print("np.setdiff1d(a, b):", np.setdiff1d(a, b))  # 输出: [1 2]

# 求对称差集（在a或b中，但不同时在两者中）
print("np.setxor1d(a, b):", np.setxor1d(a, b))  # 输出: [1 2 6 7]
```
# 线性代数
在numpy文件夹的linalg.py查看更多内容。
## 矩阵操作
### 矩阵乘法
使用`numpy.dot(a, b)` 或 `a @ b`
```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# 矩阵乘法（使用 @ 运算符或 dot 函数）
print("a @ b:")
print(a @ b)  # 输出: [[19 22] [43 50]]

print("np.dot(a, b):")
print(np.dot(a, b))  # 输出: [[19 22] [43 50]]

# 向量点积
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print("np.dot(v1, v2):", np.dot(v1, v2))  # 输出: 32
```
### 矩阵转置
 使用`numpy.transpose(a)` 或 `a.T`
```python
A = np.array([[1, 2], [3, 4]])
A_transpose = np.transpose(A)  # 或 A.T
```
### 矩阵的逆
使用 `numpy.linalg.inv(a)`
```python
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)
```
### 矩阵的Moore-Penrose伪逆（广义逆）
使用`numpy.linalg.pinv(a)`
```python
A = np.array([[1, 2], [3, 4], [5, 6]])
A_pinv = np.linalg.pinv(A)
```
## 矩阵分解
### Cholesky分解
使用`numpy.linalg.cholesky(a)`
```python
A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
L = np.linalg.cholesky(A)  # A = L @ L.T
```
### QR分解
使用`numpy.linalg.qr(a)`
```python
A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
Q, R = np.linalg.qr(A)  # A = Q @ R
```
### 奇异值分解
使用`numpy.linalg.svd(a)`
```python
A = np.array([[1, 2], [3, 4], [5, 6]])
U, s, Vh = np.linalg.svd(A)  # A = U @ np.diag(s) @ Vh
```
## 特征值和特征向量
### 计算特征值和特征向量
使用`numpy.linalg.eig(a)`
```python
A = np.array([[1, 2], [2, 1]])
eigenvalues, eigenvectors = np.linalg.eig(A)
```
### 只计算特征值
使用`numpy.linalg.eigvals(a)`
```python
A = np.array([[1, 2], [2, 1]])
eigenvalues = np.linalg.eigvals(A)
```
## 矩阵范数和行列式
### 计算行列式
使用`numpy.linalg.det(a)`
```python
A = np.array([[1, 2], [3, 4]])
det_A = np.linalg.det(A)
```
### 计算范数
使用`numpy.linalg.norm(x, ord=None)`
- **参数**: `ord` 指定范数类型（如None=Frobenius范数, 1=1范数, 2=2范数, np.inf=无穷范数）
```python
v = np.array([3, 4])
norm_v = np.linalg.norm(v)  # 默认计算L2范数，结果为5.0
```
## 求解线性方程组
### 求解ax = b
使用`numpy.linalg.solve(a, b)`
要求a 必须是方阵且满秩
```python
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)  # 解为 [2., 3.]
```
### 超定方程最小二乘解
使用`numpy.linalg.lstsq(a, b, rcond=None)`
```python
A = np.array([[0, 1], [1, 1], [2, 1], [3, 1]])
b = np.array([-1, 0.2, 0.9, 2.1])
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
```
## 矩阵的秩和迹
### 计算秩
使用`numpy.linalg.matrix_rank(a)`
```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
rank_A = np.linalg.matrix_rank(A)  # 结果为2
```
### 计算迹
 使用`numpy.trace(a)`
 ```python
 A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
trace_A = np.trace(A)  # 结果为15 (1+5+9)
 ```
## 向量堆叠
我们如何从一系列同样大小的行向量创建一个二维数组？在matlab中，很简单，如果x和y是同样长度的两个向量，那么我们需要做的就是m=[x;y]。在NumPy中，我们使用`column_stack`, `dstack`, `hstack` 和`vstack`来实现，取决于你想要在哪个维度上堆叠你的向量。例如：
```python
x = np.arange(0,10,2)                     # x=([0,2,4,6,8])
y = np.arange(5)                          # y=([0,1,2,3,4])
m = np.vstack([x,y])                      # m=([[0,2,4,6,8],
                                          #     [0,1,2,3,4]])
xy = np.hstack([x,y])                     # xy =([0,2,4,6,8,0,1,2,3,4])
```
# 广播机制
NumPy 的广播机制是其最强大且独特的特性之一，它允许不同形状的数组进行算术运算，而无需显式地复制数据。
广播的核心思想是：**当两个数组的形状不同时，NumPy 会自动扩展较小的数组，使其形状与较大的数组匹配，以便进行元素级的运算**。
## 广播规则
NumPy 的广播遵循以下严格规则：
1. 如果两个数组的维度数不同，将为维度较少的数组的形状前面添加 1，直到两个数组的维度数相同。
2. 如果两个数组在某个维度上的大小相同，或者其中一个数组在该维度上的大小为 1，那么这两个数组在这个维度上是兼容的。
3. 如果两个数组在所有维度上都兼容，那么它们可以一起广播。
4. 广播之后，每个数组的行为就好像它的形状等于两个数组形状的元素最大值一样。
5. 在任何一个维度上，如果一个数组的大小为 1，而另一个数组的大小大于 1，那么首先数组会在该维度上复制数据，使得它们的大小相同。
## 广播示例
### 标量与数组的运算
```python
import numpy as np

# 创建一个数组
a = np.array([1, 2, 3])
print("a:", a)  # 输出: [1 2 3]

# 标量与数组相加（标量被广播到数组的每个元素）
b = a + 5
print("a + 5:", b)  # 输出: [6 7 8]

# 标量与数组相乘
c = a * 2
print("a * 2:", c)  # 输出: [2 4 6]
```
### 一维数组与二维数组的运算
```python
# 创建一个二维数组
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print("matrix:")
print(matrix)

# 创建一个一维数组
vector = np.array([10, 20, 30])
print("vector:", vector)  # 输出: [10 20 30]

# 一维数组与二维数组相加（向量被广播到每一行）
result = matrix + vector
print("matrix + vector:")
print(result)
# 输出:
# [[11 22 33]
#  [14 25 36]
#  [17 28 39]]
```
### 形状不匹配的数组运算
```python
# 创建一个 3x1 的数组
a = np.array([[1], [2], [3]])
print("a (3x1):")
print(a)

# 创建一个 1x3 的数组
b = np.array([[10, 20, 30]])
print("b (1x3):")
print(b)

# 两个数组都可以被广播到 3x3 的形状
result = a + b
print("a + b (广播到 3x3):")
print(result)
# 输出:
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]
```
### 无法广播的情况
```python
# 创建一个 3x2 的数组
a = np.array([[1, 2], [3, 4], [5, 6]])
print("a (3x2):")
print(a)

# 创建一个长度为 3 的向量
b = np.array([10, 20, 30])
print("b:", b)  # 输出: [10 20 30]

result = a + b
print(result)
# 尝试广播 - 这会失败，因为形状不兼容
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
```

