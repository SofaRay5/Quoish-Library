Matplotlib是Python中最经典的2D/3D数据可视化库，由John D. Hunter于2003年开发（灵感来源于MATLAB的绘图功能）。
使用时需要导入库，数据处理方面常使用`numpy`库。
```python
import matplotlib as mpl
import matplotlib.pyplot as plt # 约定俗成的别名 
import numpy as np # 配合生成数据
```
# 结构组件
Matplotlib的图表采用“双层结构”：
- **Figure（画布）**：最外层容器，相当于“画纸”，可以包含多个子图；
- **Axes（子图）**：Figure中的一个具体图表（如折线图、柱状图），每个Axes有独立的坐标轴（x轴、y轴）、标题、图例等。
```
Figure（画布）
├─ Axes 1（子图1：折线图）
│  ├─ xaxis（x轴）
│  └─ yaxis（y轴）
└─ Axes 2（子图2：柱状图）
   ├─ xaxis（x轴）
   └─ yaxis（y轴）
```
# 全局配置
`matplotlib`库默认字体为`DejaVu Sans`，其不支持`Unicode`字符，显示中文时会出现乱码。
Matplotlib 默认使用`Unicode`长减号（`−`，`Unicode` 码位`U+2212`）表示负号，而非 `ASCII` 普通减号（`-`，`U+002D`），因为长减号在数学排版中更规范（如论文图表），但需字体支持；普通减号兼容性更好，尤其在中文字体环境下。
针对以上两个问题，需要对`matplotlib`进行全局配置工作。
```python
# 设置字体族
plt.rcParams['font.family'] = 'sans-serif' # 设置为无衬线字体
# 这段代码也可以填入诸如'serif'（衬线字体）, 'monospace'（单间隔字体）

# 指定无衬线字体族优先使用 Arial
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] # Arial 不可用时，使用 DejaVu Sans
# 其他Windows 常用中文字体：'SimHei' (黑体), 'Microsoft YaHei' (微软雅黑), 'KaiTi' (楷体), 'FangSong' (仿宋)

# 减号支持问题
plt.rcParams['axes.unicode_minus'] = False # 默认是True，这里需要关闭
```
# 折线图
**适用场景**：时间序列数据（如销售额月变化、温度日变化）、函数曲线（如sin(x)）
**核心函数**：`plt.plot(x, y, [参数])`
示例
```python
# 准备数据（x轴：月份，y轴：温度）
months = np.arange(1, 13)  # [1,2,...,12]
temperatures = [3, 5, 8, 15, 20, 25, 28, 27, 22, 16, 10, 5]

# 绘制折线图
plt.plot(months, temperatures, 
         color='r',       # 线条颜色（红色）
         linestyle='--',  # 线条样式（虚线）
         linewidth=2,     # 线条粗细
         marker='o')      # 数据点标记（圆圈）

# 添加图表标题和坐标轴标签
plt.title("2023年月平均气温变化", fontsize=14)
plt.xlabel("月份", fontsize=12)
plt.ylabel("温度（℃）", fontsize=12)

# 显示图表
plt.show()
```
# 柱状图
**适用场景**：不同类别数据的对比（如各部门销售额、各省人口）。
**核心函数**：`plt.bar(x, height, [参数])`（垂直柱状图）或`plt.barh(y, width)`（水平柱状图）
```python
# 准备数据（部门名称、销售额）
departments = ['技术部', '销售部', '运营部', '财务部']
sales = [80, 150, 120, 60]

# 绘制垂直柱状图
plt.bar(departments, sales, 
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],  # 自定义颜色
        edgecolor='black',  # 柱子边框颜色
        width=0.6)          # 柱子宽度

# 添加数据标签（柱顶显示具体数值）
for i, val in enumerate(sales):
    plt.text(i, val + 3, f'{val}万', ha='center', fontsize=10)

# 添加标题和标签
plt.title("2023年各部门销售额对比", fontsize=14)
plt.xlabel("部门", fontsize=12)
plt.ylabel("销售额（万元）", fontsize=12)

plt.show()
```
# 散点图
**适用场景**：分析两个变量的相关性（如身高与体重、广告投入与销量）。
**核心函数**：`plt.scatter(x, y, [参数])`
```python
# 生成模拟数据（广告投入x，销量y）
np.random.seed(42)  # 固定随机数，保证结果可复现
x = np.random.randint(10, 100, size=50)  # 广告投入（10-100万）
y = 0.8 * x + np.random.normal(0, 10, 50)  # 销量=0.8*投入+随机误差

# 绘制散点图
plt.scatter(x, y, 
            color='green',    # 点颜色
            s=50,             # 点大小
            alpha=0.7,        # 透明度（0-1）
            edgecolor='black')# 点边框颜色

# 添加趋势线（用NumPy拟合直线）
m, b = np.polyfit(x, y, 1)  # 一次多项式拟合（斜率m，截距b）
plt.plot(x, m*x + b, color='red', linestyle='--', linewidth=2)  # 绘制趋势线

# 添加标题和标签
plt.title("广告投入与销量的相关性", fontsize=14)
plt.xlabel("广告投入（万元）", fontsize=12)
plt.ylabel("销量（千件）", fontsize=12)

plt.show()
```
# 拓展功能
## 分辨率调节
默认图表较小，通过`plt.figure(figsize=(宽, 高), dpi=分辨率)`调整：
```python
plt.figure(figsize=(10, 6), dpi=100)  # 宽10英寸，高6英寸，分辨率100dpi
```
## 添加图例（Legend）
当图表中有多条曲线或多个类别时，图例能明确标识每个元素的含义：
```python
# 绘制两条折线（2022 vs 2023年气温）
plt.plot(months, temperatures_2022, label='2022年')
plt.plot(months, temperatures_2023, label='2023年')
plt.legend(loc='upper left', fontsize=10)  # 图例位置（左上）、字体大小
```
## 定制坐标轴刻度
通过`plt.xticks()`和`plt.yticks()`调整刻度标签和间隔：
```python
# 设置x轴刻度为1-12月，字体旋转45度（避免重叠）
plt.xticks(months, labels=[f'{m}月' for m in months], rotation=45)

# 设置y轴刻度范围（0-30℃），间隔5℃
plt.yticks(np.arange(0, 31, 5))
```
## 添加网格线（Grid）
```python
plt.grid(axis='y', linestyle='--', color='gray', alpha=0.3)  # 仅显示y轴网格
```
## 保存图表到文件
通过plt.savefig()将图表保存为图片（支持PNG、PDF、SVG等）：
```python
plt.savefig('temperature_trend.png', dpi=300, bbox_inches='tight')  # 高清保存，自动裁剪白边
```

