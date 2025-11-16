Pandas是一个基于[[NumPy库]]的分析结构化数据的工具集，[[NumPy库]]为其提供了高性能的数据处理能力。Pandas被普遍用于数据挖掘和数据分析，同时也提供数据清洗、数据I/O、数据可视化等辅助功能。
使用之前必须导入库
```python
from datetime import datetime, timedelta
import random
import pandas as pd
from matplotlib import pyplot as plt
```
# 数据结构
学习`Pandas`之前，一般都需要先了解下其对应的数据结构，方便后面理解和使用，`DataFrame(数据框)`和`Series(序列)`是`Pandas`库中两个最基本、最重要的数据结构。它们提供了灵活且高效的数据操作方法，使得数据分析和处理变得更加简单和可行。
- `Series(序列)`: 是一种类似于一维数组的数据结构，它可以存储任意类型的数据，并附带有标签（`label`），这些标签可以用于索引。`Series`可以看作是由两个数组组成，一个是数据值的数组，一个是与之相关的标签的数组。
- `DataFrame`: 是一个表格型的数据结构，包含有一组有序的列，每列可以是不同的值类型(数值、字符串、布尔型等)，`DataFrame`即有行索引也有列索引，可以被看做是由`Series`组成的字典。
## Series
`Series`是一个对象，其对应的构造函数所需参数如下：
```python
def __init__(
    self,
    data=None,
    index=None,
    dtype: Dtype | None = None,
    name=None,
    copy: bool | None = None,
    fastpath: bool = False,
)
```
### 参数说明
- `data`：指的是输入数据，当输入数据类型不同时,会有不同的操作,如下:  
	- 如果是一个数组（如列表或`NumPy` 数组），那么它将成为`Series` 的数据。  
	- 如果是一个字典，字典的键将成为 `Series` 的索引，字典的值将成为`Series` 的数据。  
	- 如果是一个标量值，它将被广播到整个`Series`。  
- `index`：指定`Series` 的索引;如果不提供，将创建默认的整数索引,从0开始。  
- `dtype`：指定`Series`的数据类型; 如果不提供，将自动推断数据类型。  
- `name`：为`Series` 指定一个名称。
- `copy`：如果为`True`，则复制输入数据。默认情况下不复制。
### 示例
> @注意: Series的所有数据都是同一种数据类型.
```python
if __name__ == '__main__':
    # -------------- 基于标量，创建Series --------------
    print("基于标量-默认索引-s_var: \n", pd.Series(18, name='s_var', dtype=np.float32))
    print("基于标量-指定索引-s_var2: \n", pd.Series(18, name='s_var2', dtype=np.float32, index=['小花']))

    print("------------------------分割线----------------------------")
    # -------------- 基于列表，创建Series --------------
    list_var = ["Go", "Python", "PHP", "Java"]
    s_list = pd.Series(list_var, name='s_list')
    print("基于列表-默认索引-s_list: \n", s_list)
    # 指定索引
    index_var = ["a", "b", "c", "d"]
    s_list2 = pd.Series(list_var, index=index_var, name='s_list2')
    print("基于列表-指定索引-s_list2: \n", s_list2)

    print("------------------------分割线----------------------------")
    # -------------- 基于字典，创建Series --------------
    dict_var = {'a': 'Go', 'b': 'Python', 'c': 'PHP', 'd': 'Java'}
    s_dict = pd.Series(dict_var, name='s_dict')
    print("基于字典-s_dict: \n", s_dict)
    print("------------------------分割线----------------------------")
    # -------------- 基于np，创建Series --------------
    s_np = pd.Series(np.arange(5, 10), name='s_np')
    print("基于numpy-s_np: \n", s_np)

"""
基于标量-默认索引-s_var: 
 0    18.0
Name: s_var, dtype: float32
基于标量-指定索引-s_var2: 
 小花    18.0
Name: s_var2, dtype: float32
------------------------分割线----------------------------
基于列表-默认索引-s_list: 
 0        Go
1    Python
2       PHP
3      Java
Name: s_list, dtype: object
基于列表-指定索引-s_list2: 
 a        Go
b    Python
c       PHP
d      Java
Name: s_list2, dtype: object
------------------------分割线----------------------------
基于字典-s_dict: 
 a        Go
b    Python
c       PHP
d      Java
Name: s_dict, dtype: object
------------------------分割线----------------------------
基于numpy-s_np: 
 0    5
1    6
2    7
3    8
4    9
Name: s_np, dtype: int64
"""
```
## DataFrame
`DataFrame`可以看作多个`Series`的集合，每个`Series`都可以拥有各自独立的数据类型，因此，`DataFrame`没有自身唯一的数据类型，自然也就没有`dtype`属性了。不过，`DataFrame`多了一个`dtypes`属性，这个属性的类型是`Series`类。除了`dtypes`属性，`DataFrame`的`values`属性、`index`属性、`columns`属性也都非常重要。
`DataFrame`对应的构造函数如下:
```python
def __init__(
    self,
    data=None,
    index: Axes | None = None,
    columns: Axes | None = None,
    dtype: Dtype | None = None,
    copy: bool | None = None,
)
```

### 参数说明
- `data`：指的是输入数据，当输入数据类型不同时,会有不同的操作,如下:  
	- 如果是一个字典，字典的值可以是列表、数组或其他字典，它们将成为 `DataFrame` 的列。  
	- 如果是一个数组或列表，它将被转换为二维数组，每一行将成为 `DataFrame` 的一行。
	- 如果是另一个`DataFrame`，则将创建一个副本。
- `index`：指定`DataFrame` 的索引;如果不提供，将创建默认的整数索引,从0开始。
- `columns`：指定 `DataFrame` 的列索引。如果不提供，将使用字典的键（如果数据是字典）或整数序列。
- `dtype`：指定`DataFrame`的数据类型; 如果不提供，将自动推断数据类型。
- `copy`：如果为`True`，则复制输入数据。默认情况下不复制。
### 示例
```python
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # 使用字典创建DataFrame
    data_dict = {'Name': ['Alice', 'Bob', 'Charlie'],
                 'Age': [25, 30, 35],
                 'City': ['New York', 'San Francisco', 'Los Angeles']}

    df_dict = pd.DataFrame(data_dict)
    print("1.使用字典创建DataFrame: \n", df_dict)

    # 使用NumPy数组创建DataFrame
    data_numpy = np.array([[1, 'Alice'], [2, 'Bob'], [3, 'Charlie']])
    df_numpy = pd.DataFrame(data_numpy, columns=['ID', 'Name'])
    print("2.使用NumPy数组创建DataFrame: \n", df_numpy)

    # 使用另一个DataFrame创建DataFrame
    data_existing_df = {'Name': ['Alice', 'Bob', 'Charlie'],
                        'Age': [25, 30, 35],
                        'City': ['New York', 'San Francisco', 'Los Angeles']}

    existing_df = pd.DataFrame(data_existing_df)

    # 创建新的DataFrame，复制已存在的DataFrame
    df_existing_df = pd.DataFrame(existing_df)
    print("3.使用DataFrame创建DataFrame:", df_existing_df)
    # 属性信息打印
    print("# ---------------- 属性信息打印 ----------------")
    print("DataFrame.dtypes:\n", df_existing_df.dtypes)
    print("DataFrame.values:", df_existing_df.values)
    print("DataFrame.index:", df_existing_df.index)
    print("DataFrame.columns:", df_existing_df.columns)

"""
1.使用字典创建DataFrame: 
       Name  Age           City
0    Alice   25       New York
1      Bob   30  San Francisco
2  Charlie   35    Los Angeles
2.使用NumPy数组创建DataFrame: 
   ID     Name
0  1    Alice
1  2      Bob
2  3  Charlie
3.使用DataFrame创建DataFrame:       Name  Age           City
0    Alice   25       New York
1      Bob   30  San Francisco
2  Charlie   35    Los Angeles

# ---------------- 属性信息打印 ----------------
DataFrame.dtypes:
 Name    object
Age      int64
City    object
dtype: object
DataFrame.values: [['Alice' 25 'New York']
 ['Bob' 30 'San Francisco']
 ['Charlie' 35 'Los Angeles']]
DataFrame.index: RangeIndex(start=0, stop=3, step=1)
DataFrame.columns: Index(['Name', 'Age', 'City'], dtype='object')
"""
```
# 基础用法
从数据结构构成上，我们知道`DataFrame`是由多个`Series`的组成, 在实际使用中也是使用`DataFrame`的场景居多，所以后续主要学习`DataFrame`数据结构的使用；
## 准备数据
```python
import pandas as pd

if __name__ == '__main__':
    # 使用字典创建DataFrame
    data_dict = {
        "成绩": [90, 88, 80, 95],
        "年龄": [23, 19, 20, 33],
        "身高": [175, 165, 170, 173],
        "体重": [71.5, 50.5, 66.5, 75.3],
        "城市": ["北京", "南京", "上海", "上海"],
    }
    names = ["小明", "小英", "李思", "王老五"]
    data_res = pd.DataFrame(data_dict, index=names)
    print("学员信息:\n", data_res)

"""
学员信息:
      成绩  年龄   身高    体重  城市
小明   90  23  175  71.5  北京
小英   88  19  165  50.5  南京
李思   80  20  170  66.5  上海
王老五  95  33  173  75.3  上海
"""
```
也可以从文件中读取数据
```python
# 读取 CSV 文件
df = pd.read_csv('data.csv')

# 读取 Excel 文件
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# 读取 JSON 文件
df = pd.read_json('data.json')

# 读取 SQL 数据库
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM table_name", conn)
```
## 访问数据
```python
if __name__ == '__main__':
    # data_res 参考准备数据，此处省略....
    # 使用head和tail获取头尾数据
    head_data = data_res.head(1)
    print("------ 第一行: ------\n", data_res.head(1))
    print("------ 最后一行: ------\n", data_res.tail(1))
    # 类似切片操作
    print("------ data_res[1:3]: ------ \n", data_res[1:3])
    # 返回某一行
    print("------ 返回某一行,小英数据: ------ \n", data_res["小英":"小英"])
    # 返回某一列
    print("------ 返回某一列，所有城市: ------ \n", data_res["城市"])
    # 同时选择行和列
    print("------ 同时选择行和列，小明、李思的成绩和年龄: ------ \n", data_res.loc[["小明", "李思"], ["成绩", "年龄"]])
    # 条件选择
    print("------ 条件选择| 年龄<20: ------ \n", data_res[data_res["年龄"] < 20])
    search_a = data_res[(data_res["城市"] == "上海") & (data_res["成绩"] > 80)]
    print("------ 条件选择| 城市=上海 & 成绩 > 80: ------ \n", search_a)
    # 访问具体值
    print("------ 根据标签,访问具体值-王老五的身高: ", data_res.at["王老五", "身高"])
    print("------ 根据位置,访问具体值-王老五的身高: ", data_res.iat[3, 2])


"""
------ 第一行: ------
     成绩  年龄   身高    体重  城市
小明  90  23  175  71.5  北京
------ 最后一行: ------
      成绩  年龄   身高    体重  城市
王老五  95  33  173  75.3  上海
------ data_res[1:3]: ------ 
     成绩  年龄   身高    体重  城市
小英  88  19  165  50.5  南京
李思  80  20  170  66.5  上海
------ 返回某一行,小英数据: ------ 
     成绩  年龄   身高    体重  城市
小英  88  19  165  50.5  南京
------ 返回某一列，所有城市: ------ 
 小明     北京
小英     南京
李思     上海
王老五    上海
Name: 城市, dtype: object
------ 同时选择行和列，小明、李思的成绩和年龄: ------ 
     成绩  年龄
小明  90  23
李思  80  20
------ 条件选择| 年龄<20: ------ 
     成绩  年龄   身高    体重  城市
小英  88  19  165  50.5  南京
------ 条件选择| 城市=上海 & 成绩 > 80: ------ 
      成绩  年龄   身高    体重  城市
王老五  95  33  173  75.3  上海
------ 根据标签,访问具体值-王老五的身高:  173
------ 根据位置,访问具体值-王老五的身高:  173
"""
```
## 编辑数据
```python
if __name__ == '__main__':
    # data_res 参考准备数据，此处省略....

    # 根据标签变更
    print("#-------------- 修改具体值 -------------------")
    print("标签赋值--变更前小英年龄:", data_res.at["小英", "年龄"])
    data_res.at["小英", "年龄"] = 18
    print("标签赋值--变更后小英年龄:", data_res.at["小英", "年龄"])
    # 根据位置变更
    print("位置赋值--变更前小英年龄:", data_res.iat[1, 1])
    data_res.iat[1, 1] = 24
    print("位置赋值--变更后小英年龄:", data_res.iat[1, 1])

    print("#-------------- 修改整列值 -------------------")
    print("修改整列值--变更前:", data_res["成绩"])
    data_res["成绩"] = 100
    print("修改整列值--变更后:", data_res["成绩"])

"""
#-------------- 修改具体值 -------------------
标签赋值--变更前小英年龄: 19
标签赋值--变更后小英年龄: 18
位置赋值--变更前小英年龄: 18
位置赋值--变更后小英年龄: 24
#-------------- 修改整列值 -------------------
修改整列值--变更前: 小明     90
小英     88
李思     80
王老五    95
Name: 成绩, dtype: int64
修改整列值--变更后: 小明     100
小英     100
李思     100
王老五    100
Name: 成绩, dtype: int64
"""
```
## 追加&删除
```python
import pandas as pd

if __name__ == '__main__':
    # data_res 参考准备数据，此处省略....

    # 增加一列
    data_res["爱好"] = ["旅游", "游戏", "篮球", "学习"]
    print("------------------ 增加一列: ------------------ \n ", data_res)
    # 删除一列
    new_data = data_res.drop(["城市", "成绩", "体重"], axis=1)
    print("------------------drop 删除多列: ------------------ \n ", new_data)

    # 使用pd.concat连接两个dataframe
    new_row = {"成绩": 100, "年龄": 30, "身高": 185, "体重": 80.5, "城市": "蜀国", "爱好": "练武"}
    new_dataframe = pd.DataFrame([new_row], index=["赵云"])
    data_res_new = pd.concat([data_res, new_dataframe])
    print("------------------ 增加行后: ------------------ \n ", data_res)

    # 使用drop也可以删除行
    new_data = data_res_new.drop(["小明", "王老五"], axis=0)
    print("------------------drop 删除多行: ------------------ \n ", new_data)

"""
------------------ 增加一列: ------------------ 
       成绩  年龄   身高    体重  城市  爱好
小明   90  23  175  71.5  北京  旅游
小英   88  19  165  50.5  南京  游戏
李思   80  20  170  66.5  上海  篮球
王老五  95  33  173  75.3  上海  学习
------------------drop 删除多列: ------------------ 
       年龄   身高  爱好
小明   23  175  旅游
小英   19  165  游戏
李思   20  170  篮球
王老五  33  173  学习
------------------ 增加行后: ------------------ 
       成绩  年龄   身高    体重  城市  爱好
小明   90  23  175  71.5  北京  旅游
小英   88  19  165  50.5  南京  游戏
李思   80  20  170  66.5  上海  篮球
王老五  95  33  173  75.3  上海  学习
------------------drop 删除多行: ------------------ 
       成绩  年龄   身高    体重  城市  爱好
小英   88  19  165  50.5  南京  游戏
李思   80  20  170  66.5  上海  篮球
赵云  100  30  185  80.5  蜀国  练武
"""
```
**函数说明:**
- `drop`: 用于删除行或列的函数。它可以在 `DataFrame` 或 `Series` 上使用，用于删除指定的行或列，并返回一个新的对象，原始对象保持不变。`axis` 参数用于指定是删除行还是列，其中 `axis=0` 表示行，`axis=1` 表示列。默认情况下，`axis=0`。
- `concat`: 将数据沿着某个轴进行拼接，可以按行拼接（垂直拼接）或按列拼接（水平拼接），使用参数`axis` 指定拼接的轴，`axis=0` 表示按行拼接（垂直拼接），`axis=1` 表示按列拼接（水平拼接），**默认是按照行拼接**。
## 排序数据
```python
if __name__ == '__main__':
    # data_res 参考准备数据，此处省略....
    # 根据年龄排序
    by_age_asc = data_res.sort_values(by="年龄", ascending=True)
    print("-------------------- 根据年龄升序排序(原始数据不变) --------------------:\n", by_age_asc)
    # 根据成绩降序排序，inplace=True会修改原始数据
    data_res.sort_values(by="成绩", ascending=False, inplace=True)
    print("-------------------- 根据成绩降序排序(修改原始数据) --------------------:\n", data_res)
    # 根据多列排序,先按照年龄，再根据成绩
    by_age_score = data_res.sort_values(by=["年龄", "成绩"], ascending=False)
    print("-------------------- 根据多列排序(先按年龄，再根据成绩) --------------------:\n", by_age_score)
    # 使用rank排序，返回当前数据在所属列的排名名次
    rank_data = data_res.rank(ascending=False)
    print("------------- 使用rank排序,返回当前数据在所属列的排名名次 : -------------\n", rank_data)

"""
-------------------- 根据年龄升序排序(原始数据不变) --------------------:
      成绩  年龄   身高    体重  城市
小英   88  19  165  50.5  南京
李思   80  20  170  66.5  上海
小明   90  23  175  71.5  北京
王老五  95  33  173  75.3  上海
-------------------- 根据成绩降序排序(修改原始数据) --------------------:
      成绩  年龄   身高    体重  城市
王老五  95  33  173  75.3  上海
小明   90  23  175  71.5  北京
小英   88  19  165  50.5  南京
李思   80  20  170  66.5  上海
-------------------- 根据多列排序(先按年龄，再根据成绩) --------------------:
      成绩  年龄   身高    体重  城市
王老五  95  33  173  75.3  上海
小明   90  23  175  71.5  北京
李思   80  20  170  66.5  上海
小英   88  19  165  50.5  南京
------------- 使用rank排序,返回当前数据在所属列的排名名次 : -------------
       成绩   年龄   身高   体重   城市
王老五  1.0  1.0  2.0  1.0  3.5
小明   2.0  2.0  1.0  2.0  2.0
小英   3.0  4.0  4.0  4.0  1.0
李思   4.0  3.0  3.0  3.0  3.5
"""
```
**函数说明:**
`sort_values` 函数用于排序 `DataFrame` 或 `Series`,具体参数如下:
```python
def sort_values(
    self,
    by: IndexLabel,
    *,
    axis: Axis = 0,
    ascending: bool | list[bool] | tuple[bool, ...] = True,
    inplace: bool = False,
    kind: str = "quicksort",
    na_position: str = "last",
    ignore_index: bool = False,
    key: ValueKeyFunc = None,
) -> DataFrame | None:
```
- **by:** 指定排序的列名或列名的列表。如果是多个列，可以传递一个包含多个列名的列表。按照列表中的列的顺序进行排序。
- **axis:** 指定排序的轴，`axis=0` 表示按行排序，`axis=1` 表示按列排序。默认为 0。
- **ascending:** 指定排序的顺序，`True` 表示升序，`False` 表示降序。可以是一个[布尔值](https://zhida.zhihu.com/search?content_id=235773534&content_type=Article&match_order=1&q=%E5%B8%83%E5%B0%94%E5%80%BC&zhida_source=entity)或布尔值的列表，用于指定每个列的排序顺序。默认为 `True`。
- **inplace:** 如果为 `True`，则在原地修改对象而不返回新对象；如果为 `False`（默认），则返回一个新对象，原对象保持不变。
- **kind:** 指定排序算法的种类，可选值有 `quicksort`(快速排序)、`mergesort`(归并排序)、`heapsort`(堆排序); 默认为 `quicksort`。
- **na_position:** 指定缺失值的位置，可选值有 `first`（在前）和 `last`（在后）。默认为 `last`。
- **ignore_index:** 如果为 `True`，则重新设置索引，忽略现有索引。默认为 `False`。
- **key:** 用于排序的函数，可以是函数、类实例或类的方法。如果指定，将用该函数的返回值进行排序。
# 数据运算
## 数据摘要
```python
if __name__ == '__main__':
    # 创建DataFrame
    data_dict = {
        "水果": ["香蕉", "苹果", "葡萄", "橘子"],
        "进货价": [1.25, 0.56, 3.5, 1.15],
        "最低价": [1.55, 0.80, 4.0, 2.00],
        "最高价": [3.45, 1.5, 6.5, 4.15],
        "数量": [600, 500, 400, 500],
    }
    fruit_data = pd.DataFrame(data_dict)
    print("--------------- 原始数据 -----------------")
    print(fruit_data)
    print("--------------- 查看数据的统计摘要 -----------------")
    print(fruit_data.describe())

"""
--------------- 原始数据 -----------------
   水果   进货价   最低价   最高价   数量
0  香蕉  1.25  1.55  3.45  600
1  苹果  0.56  0.80  1.50  500
2  葡萄  3.50  4.00  6.50  400
3  橘子  1.15  2.00  4.15  500
--------------- 查看数据的统计摘要 -----------------
           进货价       最低价      最高价          数量
count  4.00000  4.000000  4.00000    4.000000
mean   1.61500  2.087500  3.90000  500.000000
std    1.29302  1.367708  2.06438   81.649658
min    0.56000  0.800000  1.50000  400.000000
25%    1.00250  1.362500  2.96250  475.000000
50%    1.20000  1.775000  3.80000  500.000000
75%    1.81250  2.500000  4.73750  525.000000
max    3.50000  4.000000  6.50000  600.000000
"""
```
**输出结果说明:**
- `count`：非缺失值的数量。
- `mean`：平均值。
- `std`：标准差，衡量数据的离散程度。
- `min`：最小值。
- `25%`：第一四分位数，数据中的 25% 的值小于此值。
- `50%`：中位数（第二四分位数），数据中的中间值。
- `75%`：第三四分位数，数据中的 75% 的值小于此值。
- `max`：最大值。
## 统计运算
```python
if __name__ == '__main__':
    # 创建DataFrame
    data_dict = {
        "水果": ["香蕉", "苹果", "葡萄", "橘子"],
        "进货价": [1.25, 0.56, 3.5, 1.15],
        "最低价": [1.55, 0.80, 4.0, 2.00],
        "最高价": [3.45, 1.5, 6.5, 4.15],
        "数量": [600, 500, 400, 500],
    }
    fruit_data = pd.DataFrame(data_dict)
    print("--------------- 原始数据 -----------------")
    print(fruit_data)
    print("--------------- 均值运算 -----------------")
    print("[数量]这一列均值: ", fruit_data["数量"].mean())
    print("[进货价]这一列均值: ", fruit_data["进货价"].mean())
    print("--------------- 极值运算 -----------------")
    print("[进货价]这一列最小值: ", fruit_data["进货价"].min())
    print("[进货价]这一列最大值: ", fruit_data["进货价"].max())
    print("--------------- 累和运算 -----------------")
    # 数量和进货价这两列，累和运算
    print(fruit_data[["数量", "进货价"]].cumsum())
    print("--------------- 广播运算-数量列减半 -----------------")
    fruit_data["数量"] = fruit_data["数量"] / 2
    print(fruit_data)

"""
--------------- 原始数据 -----------------
   水果   进货价   最低价   最高价   数量
0  香蕉  1.25  1.55  3.45  600
1  苹果  0.56  0.80  1.50  500
2  葡萄  3.50  4.00  6.50  400
3  橘子  1.15  2.00  4.15  500
--------------- 均值运算 -----------------
[数量]这一列均值:  500.0
[进货价]这一列均值:  1.6150000000000002
--------------- 极值运算 -----------------
[进货价]这一列最小值:  0.56
[进货价]这一列最大值:  3.5
--------------- 累和运算 -----------------
     数量   进货价
0   600  1.25
1  1100  1.81
2  1500  5.31
3  2000  6.46
--------------- 广播运算-数量列减半 -----------------
   水果   进货价   最低价   最高价     数量
0  香蕉  1.25  1.55  3.45  300.0
1  苹果  0.56  0.80  1.50  250.0
2  葡萄  3.50  4.00  6.50  200.0
3  橘子  1.15  2.00  4.15  250.0
"""
```
## 自定义函数
`apply()` 函数的存在，可以让我们更灵活的处理数据，它可以接收一个我们实现的函数，然后对数据进行自定义处理，具体参数如下:
```python
def apply(
    self,
    func: AggFuncType,
    axis: Axis = 0,
    raw: bool = False,
    result_type: Literal["expand", "reduce", "broadcast"] | None = None,
    args=(),
    **kwargs,
):
```
- `func`：要应用的函数。可以是函数、字符串函数名称、`NumPy` 函数或字典。
- `axis`：指定应用函数的轴，`axis=0` 表示按列（默认），`axis=1` 表示按行。
- `raw`：如果为 `True`，则将每一行或列作为一维数组传递给函数。如果为 `False`（默认），则将每一行或列作为 `Series`传递给函数。
- `result_type`：指定返回结果的数据类型，可以是 `'expand'、'reduce'、'broadcast'` 或`None`。默认为 `None`。
- `args`：传递给函数的位置参数。
- `kwargs`：传递给函数的关键字参数。
```python
import random
import pandas as pd


def custom_compute(x: pd.Series):
    """
    变更指定列信息
    :param x:
    :return:
    """
    # 这里接受的是Series
    if x.name == "最低价":
        return x + random.randint(1, 100) / 100
    elif x.name == "数量":
        return x * 1.5
    return x


def total_cost(x: pd.Series):
    """
    计算总成本
    :param x:
    :return:
    """
    money = x["进货价"] * x["数量"]
    print("{} 进货价:{} 数量:{} 成本:{}".format(x["水果"], x["进货价"], x["数量"], money))
    return money


if __name__ == '__main__':
    # 创建DataFrame
    data_dict = {
        "水果": ["香蕉", "苹果", "葡萄", "橘子"],
        "进货价": [1.25, 0.56, 3.5, 1.15],
        "最低价": [1.55, 0.80, 4.0, 2.00],
        "最高价": [3.45, 1.5, 6.5, 4.15],
        "数量": [600, 500, 400, 500],
    }
    fruit_data = pd.DataFrame(data_dict)
    print("--------------- 原始数据 -----------------")
    print(fruit_data)
    print("--------------- 自定义函数运算:最低价加上随机数，数量*1.5 -----------------")
    new_data = fruit_data.apply(custom_compute)
    print(new_data)
    print("--------------- 自定义函数运算:计算变更后的总成本 -----------------")
    total_money = new_data.apply(total_cost, axis=1).sum()
    print("变更后的总成本:", total_money)


"""
--------------- 原始数据 -----------------
   水果   进货价   最低价   最高价   数量
0  香蕉  1.25  1.55  3.45  600
1  苹果  0.56  0.80  1.50  500
2  葡萄  3.50  4.00  6.50  400
3  橘子  1.15  2.00  4.15  500
--------------- 自定义函数运算:最低价加上随机数，数量*1.5 -----------------
   水果   进货价   最低价   最高价     数量
0  香蕉  1.25  1.86  3.45  900.0
1  苹果  0.56  1.11  1.50  750.0
2  葡萄  3.50  4.31  6.50  600.0
3  橘子  1.15  2.31  4.15  750.0
--------------- 自定义函数运算:计算变更后的总成本 -----------------
香蕉 进货价:1.25 数量:900.0 成本:1125.0
苹果 进货价:0.56 数量:750.0 成本:420.00000000000006
葡萄 进货价:3.5 数量:600.0 成本:2100.0
橘子 进货价:1.15 数量:750.0 成本:862.4999999999999
变更后的总成本: 4507.5
"""
```
## 分组运算
通过函数 `groupby`,可以按照某一列或多列的值将数据集分成多个组，并在这些组上应用各种操作。
```python
if __name__ == '__main__':
    # 创建DataFrame 科目
    names = ["小明", "小丽", "小龙", "小花"]
    dates = ["2023-05", "2023-06", "2023-07", "2023-08", "2023-09", "2023-10"]
    subjects = ["语文", "数学", "英语"]
    rows = 5
    data_dict = {
        "姓名": [random.choice(names) for _ in range(rows)],
        "日期": [random.choice(dates) for _ in range(rows)],
        "学科": [random.choice(subjects) for _ in range(rows)],
        "成绩": [random.randint(60, 100) for _ in range(rows)],
    }
    data = pd.DataFrame(data_dict)
    print("-------------- 原始数据 ------------------")
    print(data)
    print("-------------- 根据姓名分组 ------------------")
    for name, group in data.groupby("姓名"):
        print("====== 姓名:{} ====== \n".format(name))
        print(group)
    print("-------------- 分组后统计总分数 ------------------")
    sum_data = data.groupby("姓名")["成绩"].sum()
    print(sum_data)
    print("-------------- 分组后,针对多列执行不同的统计 ------------------")
    agg_data = data.groupby("姓名").agg({"成绩": ["max", "mean", "min"], "学科": "count"})
    print(agg_data)

"""
-------------- 原始数据 ------------------
   姓名       日期  学科  成绩
0  小丽  2023-05  数学  76
1  小龙  2023-06  数学  74
2  小花  2023-09  英语  62
3  小丽  2023-05  英语  68
4  小明  2023-10  语文  87
-------------- 根据姓名分组 ------------------
====== 姓名:小丽 ====== 

   姓名       日期  学科  成绩
0  小丽  2023-05  数学  76
3  小丽  2023-05  英语  68
====== 姓名:小明 ====== 

   姓名       日期  学科  成绩
4  小明  2023-10  语文  87
====== 姓名:小花 ====== 

   姓名       日期  学科  成绩
2  小花  2023-09  英语  62
====== 姓名:小龙 ====== 

   姓名       日期  学科  成绩
1  小龙  2023-06  数学  74
-------------- 分组后统计总分数 ------------------
姓名
小丽    144
小明     87
小花     62
小龙     74
Name: 成绩, dtype: int64
-------------- 分组后,针对多列执行不同的统计 ------------------
    成绩              学科
   max  mean min count
姓名                    
小丽  76  72.0  68     2
小明  87  87.0  87     1
小花  62  62.0  62     1
小龙  74  74.0  74     1
"""
```
# 数据过滤

在数据处理中，我们经常会对数据进行过滤，为此`Pandas`中提供`mask()`和`where()`两个函数；
- `mask()`: 在满足条件的情况下替换数据，而不满足条件的部分则保留原始数据；
- `where()`: 在不满足条件的情况下替换数据，而满足条件的部分则保留原始数据;
```python
from datetime import datetime, timedelta
import random
import pandas as pd

def getDate() -> str:
    """
    用来生成日期
    :return: 
    """
    # 随机减去天数
    tmp = datetime.now() - timedelta(days=random.randint(1, 100))
    # 格式化时间
    return tmp.strftime("%Y-%m-%d")

if __name__ == '__main__':
    # 准备数据 水果
    fruits = ["苹果", "香蕉", "橘子", "榴莲", "葡萄"]
    rows = 3
    today = datetime.now()
    data_dict = {
        "fruit": [random.choice(fruits) for _ in range(rows)],
        "date": [getDate() for _ in range(rows)],
        "price": [round(random.uniform(1, 5), 2) for _ in range(rows)],  # 随机生成售价,并保留两位小数
    }
    data = pd.DataFrame(data_dict)
    # 复制数据，方便后续演示
    data_tmp = data.copy()
    print("-------------- 生成数据预览 ------------------")
    print(data)
    print("-------------- 普通条件:列出价格大于3.0的水果 ------------------")
    condition = data["price"] > 3.0
    # 列出价格大于3.0的水果
    fruit_res = data[condition]["fruit"]
    print("价格大于3.0的水果:", fruit_res.to_numpy())

    print("-------------- 使用mask:把价格大于3.0设置成0元 ------------------")
    # 把价格大于3.0设置成0元
    data["price"] = data["price"].mask(data["price"] > 3.0, other=0)
    print(data)
    print("-------------- 使用where:把价格不大于3.0设置成0元 ------------------")
    # 把价格不大于3.0设置成0元
    data_tmp["price"] = data_tmp["price"].where(data_tmp["price"] > 3.0, other=0)
    print(data_tmp)

"""
-------------- 生成数据预览 ------------------
  fruit        date  price
0    橘子  2023-10-16   3.52
1    苹果  2023-09-08   1.07
2    葡萄  2023-09-27   2.69
-------------- 普通条件:列出价格大于3.0的水果 ------------------
价格大于3.0的水果: ['橘子']
-------------- 使用mask:把价格大于3.0设置成0元 ------------------
  fruit        date  price
0    橘子  2023-10-16   0.00
1    苹果  2023-09-08   1.07
2    葡萄  2023-09-27   2.69
-------------- 使用where:把价格不大于3.0设置成0元 ------------------
  fruit        date  price
0    橘子  2023-10-16   3.52
1    苹果  2023-09-08   0.00
2    葡萄  2023-09-27   0.00
"""
```

> @注:从功能上可以看出,mask()和where()是正好两个相反的函数
# 数据遍历
```python
if __name__ == '__main__':
    # 数据生成参考上面
    print("-------------- 生成数据预览 ------------------")
    print(data)
    print("-------------- 遍历dataframe数据 ------------------")
    for index, row in data.iterrows():
        print("index:{} 水果:{} 日期:{} 售价:{}".format(index, row["fruit"], row["date"], row["price"]))

    print("-------------- 遍历Series数据 ------------------")
    series_data = pd.Series({"name": "张三", "age": 20, "height": 185})
    for k, v in series_data.items():
        print("key:{} value:{}".format(k, v))

"""
-------------- 生成数据预览 ------------------
  fruit        date  price
0    橘子  2023-10-14   3.71
1    橘子  2023-10-03   3.74
2    香蕉  2023-09-06   1.17
3    葡萄  2023-08-30   1.16
4    榴莲  2023-10-21   1.47
-------------- 遍历dataframe数据 ------------------
index:0 水果:橘子 日期:2023-10-14 售价:3.71
index:1 水果:橘子 日期:2023-10-03 售价:3.74
index:2 水果:香蕉 日期:2023-09-06 售价:1.17
index:3 水果:葡萄 日期:2023-08-30 售价:1.16
index:4 水果:榴莲 日期:2023-10-21 售价:1.47
-------------- 遍历Series数据 ------------------
key:name value:张三
key:age value:20
key:height value:185
"""
```

# 分层索引
分层索引（`MultiIndex`）是`Pandas` 中一种允许在一个轴上拥有多个（两个或更多）级别的索引方式。这种索引方式适用于多维数据和具有多个层次结构的数据。
## 使用set_index
```python
from datetime import datetime, timedelta
import random
import pandas as pd

if __name__ == '__main__':
    # 创建一个示例 DataFrame
    fruits = ["苹果", "苹果", "橘子", "橘子", "橘子", "百香果"]
    rows = len(fruits)
    today = datetime.now()
    dict_var = {
        'fruit': fruits,
        'date': [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(rows)],
        'price': [round(random.uniform(1, 5), 2) for _ in range(rows)],
        'num': [round(random.uniform(10, 500), 2) for _ in range(rows)]
    }
    sale_data = pd.DataFrame(dict_var)

    # 设置多层次索引
    sale_data.set_index(['fruit', 'date'], inplace=True)
    print("----------------------------- 创建多层次索引-----------------------------------")
    print(sale_data)
    print("----------------------------- 打印索引信息-----------------------------------")
    index_info = sale_data.index
    print(index_info)
    print("----------------------------- 使用loc 访问多层次索引-----------------------------------")
    search_price = sale_data.loc[('苹果', '2023-11-02'), 'price']
    print(search_price)
    print("----------------------------- 使用xs 访问多层次索引-----------------------------------")
    search_xs = sale_data.xs(key=('苹果', '2023-11-02'), level=['fruit', 'date'])
    print(search_xs)


"""
----------------------------- 创建多层次索引-----------------------------------
                    price     num
fruit    date                     
苹果    2023-11-02   1.08  211.31
        2023-11-01   1.35  308.87
橘子    2023-10-31   3.25  180.84
        2023-10-30   2.53  115.14
        2023-10-29   2.61  146.49
百香果   2023-10-28   1.36  246.01
----------------------------- 打印索引信息-----------------------------------
MultiIndex([( '苹果', '2023-11-02'),
            ( '苹果', '2023-11-01'),
            ( '橘子', '2023-10-31'),
            ( '橘子', '2023-10-30'),
            ( '橘子', '2023-10-29'),
            ('百香果', '2023-10-28')],
           names=['fruit', 'date'])
----------------------------- 使用loc 访问多层次索引-----------------------------------
1.08
----------------------------- 使用xs 访问多层次索引-----------------------------------
                  price     num
fruit date                     
苹果    2023-11-02   1.08  211.31
"""
```

## 使用`MultiIndex`
```python
if __name__ == '__main__':
    fruits = ["苹果", "香蕉", "橘子", "榴莲", "葡萄", "雪花梨", "百香果"]
    date_list = ['2023-03-11', '2023-03-13', '2023-03-15']
    cols = pd.MultiIndex.from_product([date_list, ["售卖价", "成交量"]], names=["日期", "水果"])
    list_var = []
    for i in range(len(fruits)):
        tmp = [
            round(random.uniform(1, 5), 2), round(random.uniform(1, 100), 2),
            round(random.uniform(1, 5), 2), round(random.uniform(1, 100), 2),
            round(random.uniform(1, 5), 2), round(random.uniform(1, 100), 2),
        ]
        list_var.append(tmp)
    print("--------------------------------创建多层次索引--------------------------------")
    multi_data = pd.DataFrame(list_var, index=fruits, columns=cols)
    print(multi_data)
    print("--------------------------------打印多层次索引--------------------------------")
    print(multi_data.index)
    print(multi_data.columns)
    # 搜行
    print("----------------------------- 使用filter-- 行搜索-----------------------------------")
    print(multi_data.filter(like='苹果', axis=0))    
    print("----------------------------- 使用filter-- 列搜索-----------------------------------")
    # 搜列
    print(multi_data.filter(like='2023-03-11', axis=1))

"""
--------------------------------创建多层次索引--------------------------------
日期  2023-03-11        2023-03-13        2023-03-15       
水果         售卖价    成交量        售卖价    成交量        售卖价    成交量
苹果        2.54  69.40       3.27  18.89       1.93   3.37
香蕉        1.99  53.33       1.88  92.77       3.64  26.60
橘子        2.48  27.81       3.20   8.71       2.58  85.44
榴莲        3.15  47.89       1.09  93.15       2.51  85.30
葡萄        4.59  35.58       4.88  77.02       3.08  64.96
雪花梨       3.17   9.58       4.48  44.17       4.15  88.94
百香果       3.05   7.65       3.51  82.03       3.97  52.06
--------------------------------打印多层次索引--------------------------------
Index(['苹果', '香蕉', '橘子', '榴莲', '葡萄', '雪花梨', '百香果'], dtype='object')
MultiIndex([('2023-03-11', '售卖价'),
            ('2023-03-11', '成交量'),
            ('2023-03-13', '售卖价'),
            ('2023-03-13', '成交量'),
            ('2023-03-15', '售卖价'),
            ('2023-03-15', '成交量')],
           names=['日期', '水果'])
----------------------------- 使用filter-- 行搜索-----------------------------------
日期 2023-03-11       2023-03-13        2023-03-15      
水果        售卖价   成交量        售卖价    成交量        售卖价   成交量
苹果       2.54  69.4       3.27  18.89       1.93  3.37
----------------------------- 使用filter-- 列搜索-----------------------------------
日期  2023-03-11       
水果         售卖价    成交量
苹果        2.54  69.40
香蕉        1.99  53.33
橘子        2.48  27.81
榴莲        3.15  47.89
葡萄        4.59  35.58
雪花梨       3.17   9.58
百香果       3.05   7.65
"""
```
# 数据读写
## 写入表格
```python
from datetime import datetime, timedelta
import random
import pandas as pd

if __name__ == '__main__':
    fruits = ["苹果", "香蕉", "橘子", "榴莲", "葡萄", "雪花梨", "百香果"]
    rows = 20
    today = datetime.now()
    print(today.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
    dict_var = {
        '水果': [random.choice(fruits) for _ in range(rows)],
        '进价': [round(random.uniform(1, 5), 4) for _ in range(rows)],
        '售价': [round(random.uniform(1, 5), 4) for _ in range(rows)],
        '日期': [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(rows)],
        '销量': [round(random.uniform(10, 500), 4) for _ in range(rows)]
    }
    sale_data = pd.DataFrame(dict_var)
    print(sale_data)
    # 保存,浮点数保留两位小数
    sale_data.to_excel("./test.xlsx", float_format="%.2f")
```
**a.主要参数说明:**
- `excel_writer`: `Excel` 文件名或 `ExcelWriter` 对象。如果是文件名，将创建一个 `ExcelWriter` 对象，并在退出时自动关闭文件。
- `sheet_name`: 字符串，工作表的名称，默认为 `Sheet1`。
- `na_rep`: 用于表示缺失值的字符串，默认为空字符串。
- `float_format`: 用于设置浮点数列的数据格式。默认为 `None`，表示使用 `Excel` 默认的格式，当设置`%.2f`表示保留两位。
- `columns`: 要写入的列的列表，默认为 `None`。如果设置为 `None`，将写入所有列；如果指定列名列表，将只写入指定的列。
- `header`: 是否包含列名，默认为 `True`。如果设置为 `False`，将不写入列名。
- `index`: 是否包含行索引，默认为 `True`。如果设置为 `False`，将不写入行索引。
- `index_label`: 用于指定行索引列的名称。默认为 `None`。
- `startrow`: 数据写入的起始行，默认为 `0`。
- `startcol`: 数据写入的起始列，默认为 `0`。
- `freeze_panes`: 值是一个元组，用于指定要冻结的行和列的位置。例如，`(2, 3)` 表示冻结第 2 行和第 3 列。默认为 `None`，表示不冻结任何行或列。

## 读取表格
```python
import pandas as pd

if __name__ == '__main__':
    # ------------------------------ 读取表格 ----------------------------------
    print("----------------------------- 读取全部数据 -----------------------")
    # 读取全部数据
    read_all_data = pd.read_excel("./test.xlsx")
    print(read_all_data)
    print("----------------------------- 只读取第1、2列数据 -----------------------")
    # 只读取第1、2列数据
    read_column_data = pd.read_excel("./test.xlsx", usecols=[1, 2])
    print(read_column_data)
    print("----------------------------- 只读取列名为:日期、销量 的数据 -----------------------")
    # 读取
    read_column_data2 = pd.read_excel("./test.xlsx", usecols=['日期', '销量'])
    print(read_column_data2)

"""
----------------------------- 读取全部数据 -----------------------
   Unnamed: 0   水果    进价    售价          日期      销量
0           0   榴莲  3.74  2.35  2023-11-03  217.03
1           1  百香果  2.08  3.64  2023-11-02  311.40
2           2  百香果  2.17  4.94  2023-11-01  404.55
3           3   橘子  2.41  2.71  2023-10-31  431.20
4           4   葡萄  2.78  3.99  2023-10-30  323.01
5           5   苹果  4.79  1.68  2023-10-29  161.26
6           6  百香果  1.61  2.78  2023-10-28  407.27
7           7   榴莲  1.56  4.08  2023-10-27   44.74
8           8  雪花梨  1.60  3.02  2023-10-26  119.13
9           9   葡萄  3.03  1.08  2023-10-25  152.87
----------------------------- 只读取第1、2列数据 -----------------------
    水果    进价
0   榴莲  3.74
1  百香果  2.08
2  百香果  2.17
3   橘子  2.41
4   葡萄  2.78
5   苹果  4.79
6  百香果  1.61
7   榴莲  1.56
8  雪花梨  1.60
9   葡萄  3.03
----------------------------- 只读取列名为:日期、销量 的数据 -----------------------
           日期      销量
0  2023-11-03  217.03
1  2023-11-02  311.40
2  2023-11-01  404.55
3  2023-10-31  431.20
4  2023-10-30  323.01
5  2023-10-29  161.26
6  2023-10-28  407.27
7  2023-10-27   44.74
8  2023-10-26  119.13
9  2023-10-25  152.87
"""
```
参数说明：
- `io`: 文件路径、`ExcelWriter` 对象或者类似文件对象的路径/对象。
- `sheet_name`: 表示要读取的工作表的名称或索引。默认为 0，表示读取第一个工作表。
- `header`: 用作列名的行的行号。默认为 0，表示使用第一行作为列名。
- `names`: 覆盖 header 的结果，即指定列名。
- `index_col`: 用作行索引的列的列号或列名。
- `usecols`: 要读取的列的列表，可以是列名或列的索引。
## 更多方法
除了上面的表格读取，还有更多类型的读取方式,方法简单整理如下：

|        类型        |                        描述                         |
| :--------------: | :-----------------------------------------------: |
|    `read_csv`    |          从文件、URL或文件型对象读取分隔好的数据，逗号是默认分隔符           |
|   `read_table`   | 由所在平台决定精度的整数从文件、URL或文件型对象读取分隔好的数据，制表符（`\t`）是默认分隔符 |
|    `read_fwf`    |               从特定宽度格式的文件中读取数据（无分隔符）               |
| `read_clipboard` |      `read_table`的剪贴板版本，在将表格从Web页面上转换成数据时有用       |
|   `read_excel`   |             从Excel 的XLS或XLSX文件中读取表格数据             |
|    `read_hdf`    |               读取用 pandas存储的 HDF5 文件               |
|   `read_html`    |                 从HTML文件中读取所有表格数据                  |
|   `read_json`    |     从JSON(JavaScript Object Notation)字符串中读取数据     |
|  `read_msgpack`  |           读取MessagePack二进制格式的 pandas数据            |
|  `read_pickle`   |            读取以 Python pickle格式存储的任意对象             |
|    `read_sas`    |             读取存储在SAS系统中定制存储格式的SAS数据集              |
|    `read_sql`    |  将 SQL查询的结果(使用 SQLAlchemy)读取为 pandas 的 DataFrame  |
|   `read_stata`   |                   读取Stata格式的数据集                   |
|  `read_feather`  |                 读取 Feather 二进制格式                  |
## 数据可视化
`Pandas`底层对`Matplotlib`进行了封装，所以可以直接使用`Matplotlib`的绘图方法；
### 折线图
```python
from datetime import datetime, timedelta
import random
import pandas as pd
from matplotlib import pyplot as plt

# 设置字体以便正确显示中文
plt.rcParams['font.sans-serif'] = ['FangSong']
# 正确显示连字符
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    # ------------------------------ 生成数据 ----------------------------------
    rows = 30
    beginDate = datetime(2023, 4, 10)
    print("beginDate:", beginDate.strftime("%Y-%m-%d"))
    dict_var = {
        '日期': [(beginDate + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(rows)],
        '进价': [round(random.uniform(1, 4), 2) for _ in range(rows)],
        '售价': [round(random.uniform(2, 6), 2) for _ in range(rows)],
    }
    apple_data = pd.DataFrame(dict_var)
    apple_data.plot(x='日期', y=['进价', '售价'], title='苹果销售数据')
    plt.show()
```
![[折线图.png]]
## 散点图
```python
from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 设置字体以便正确显示中文
plt.rcParams['font.sans-serif'] = ['FangSong']
# 正确显示连字符
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    # ------------------------------ 生成数据 ----------------------------------
    rows = 30
    beginDate = datetime(2023, 4, 1)
    print("beginDate:", beginDate.strftime("%Y-%m-%d"))
    dict_var = {
        '日期': [(beginDate + timedelta(days=i)).strftime("%d") for i in range(rows)],
        '进价': [round(random.uniform(1, 4), 2) for _ in range(rows)],
        '售价': [round(random.uniform(2, 10), 2) for _ in range(rows)],
        '销量': [round(random.uniform(10, 500), 4) for _ in range(rows)]
    }
    apple_data = pd.DataFrame(dict_var)
    # 设置颜色
    colorList = 10 * np.random.rand(rows)
    # 设置
    apple_data.plot(x='日期', y='售价', kind='scatter', title='苹果销售数据', color=colorList, s=dict_var['销量'])
    plt.show()
```
![[散点图.png]]
## 柱形图
```python
import random

import pandas as pd
from matplotlib import pyplot as plt

# 设置字体以便正确显示中文
plt.rcParams['font.sans-serif'] = ['FangSong']
# 正确显示连字符
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    # ------------------------------ 生成数据 ----------------------------------
    fruits = ["苹果", "香蕉", "橘子", "榴莲", "葡萄", "雪花梨", "百香果"]
    rows = 7
    beginDate = datetime(2023, 4, 1)
    print("beginDate:", beginDate.strftime("%Y-%m-%d"))
    dict_var = {
        '水果': ["苹果", "香蕉", "橘子", "榴莲", "葡萄", "雪花梨", "百香果"],
        '销量': [round(random.uniform(10, 1000), 2) for _ in range(rows)]
    }
    apple_data = pd.DataFrame(dict_var)
    apple_data.plot(x='水果', y='销量', kind='bar', title='水果销售数据')
    plt.show()
```
![[柱形图.png]]