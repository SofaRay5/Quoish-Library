`Pillow` 是第三方开源的 `Python` 图像处理库，它支持多种图片格式，包括 `BMP、GIF、JPEG、PNG、TIFF`等。`Pillow` 库包含了大量的图片处理函数和方法，可以进行图片的读取、显示、旋转、缩放、裁剪、转换等操作。
更多可以参考：[Pillow库官方文档](https://pillow.readthedocs.io/en/stable/handbook/index.html)
# Pillow库常用子模块
- `Image`: 该模块是`Pillow`中最重要的模块之一，用于处理图像文件。它提供了打开、保存、调整大小、旋转、裁剪、滤镜等功能，是图像处理的核心。
- `ImageDraw`: 该模块提供了在图像上绘制各种形状（如线条、矩形、圆形）和文本的功能。可以使用不同的颜色和宽度绘制，创建自定义的标记或绘制图表等。
- `ImageFont`: 该模块用于加载和使用`TrueType`字体文件，以便在图像上绘制文本时设置字体样式、大小和颜色。
- `ImageFilter`: 该模块提供了各种滤镜效果，如模糊、锐化、边缘增强等。这些滤镜可以用于图像增强、特效处理和图像识别等应用。
- `ImageEnhance`: 该模块用于调整图像的亮度、对比度、颜色饱和度等参数，使得图像更加清晰、明亮或具有特定的调色效果。
- `ImageChops`: 该模块用于执行图像的逻辑和算术操作，如合并、比较、掩蔽等。可以进行图像合成、混合和提取等操作。
- `ImageOps`: 该模块提供了各种图像处理操作，如镜像、翻转、自动对比度调整等。可以方便地进行图像变换和增强。
- `ImageStat`: 该模块用于计算图像的统计信息，如均值、中位数、直方图等。可用于图像质量评估、颜色分析和特征提取等任务。
使用之前，需要导入库
我们使用`"C:\Users\Quoish\Desktop\可可.png"`作为使用案例。
```python
from PIL import Image

import os
imgPath = os.path.join(os.environ['USERPROFILE'], 'Desktop', '可可.png')
```
# 读取图片
使用`Image.open()`来打开图像后，可以直接访问其属性信息，属性信息如下：
```python
# 读取图像  
img = Image.open(imgPath)  
# 打印图像属性  
print("读取对象img:", img)  
print("图像文件名:", img.filename)  
print("图像扩展名:", img.format)  
print("图像描述:", img.format_description)  
print("图像尺寸:", img.size)  
print("色彩模式:", img.mode)  
print("图像宽度(像素):", img.width)  
print("图像高度(像素):", img.height)  
print("图象有关的数据的字典:", img.info)
```
输出
```bash
读取对象img: <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1080x1080 at 0x1E8C2B0FA10>
图像文件名: C:\Users\Quoish\Desktop\可可.png
图像扩展名: PNG
图像描述: Portable network graphics
图像尺寸: (1080, 1080)
色彩模式: RGB
图像宽度(像素): 1080
图像高度(像素): 1080
图象有关的数据的字典: {'dpi': (599.9988, 599.9988)}
```
# 另存图片
使用`Image`对象的`.save()`方法可以保存图像。
```python
savePath = os.path.join(os.environ['USERPROFILE'], 'Desktop')

img.save(os.path.join(savePath, '可可_副本.png'))  # 保存为副本  
img.save(os.path.join(savePath, '可可_副本.jpg'))  # 保存为其他文件拓展名  
img.save(os.path.join(savePath, '可可_副本_lq.jpg'), quality=1)  # 可选保存图像质量（1~100），默认为75
```
# 调整图片
`Pillow`模块还提供对图片进行大小调整、逆时针方向旋转、上下翻转、左右翻转等方法
```python
savePath = os.path.join(os.environ['USERPROFILE'], 'Desktop')

# 更改图片大小为500 * 500  
img.resize((500, 500), resample=Image.LANCZOS).save(os.path.join(savePath, "可可_small.png"))  
# ------------------------ 图片逆时针旋转 ----------------------# 图片逆时针旋转90  
img.rotate(90).save(os.path.join(savePath, "可可_90.jpg"))  
# 图片逆时针旋转120  
img.rotate(120).save(os.path.join(savePath, "可可_120.jpg"))  
# 图片逆时针旋转180  
img.rotate(180).save(os.path.join(savePath, "可可_180.jpg"))  
# ------------------------ 图片翻转 ----------------------# 图片左右翻转  
img.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(savePath, "可可_flip_left_right.jpg"))  
# 图片上下翻转  
img.transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(savePath, "可可_flip_top_bottom.jpg"))
```
`resizer`中的参数`resample`是一个可选的重采样过滤器，常用的常量含义如下：
- `Image.NEAREST`: 从输入图像中选取一个最近的像素。忽略所有其他输入像素
- `Image.BILINEAR`： 双线采样法
- `Image.LANCZOS`: 力求输出最高质量像素的过滤器，只可用于 `resize()` 和 `thumbnail()`方法。
# 编辑图片
```python
savePath = os.path.join(os.environ['USERPROFILE'], 'Desktop')

# 复制原对象  
imgCopy = img.copy()  

# 创建缩略图  
img.thumbnail((500, 500))  
img.save(os.path.join(savePath,"可可_thumb.png"))  
  
# 裁剪图片  
cropImg = imgCopy.crop((0, 253, 1080, 749))  
cropImg.save(os.path.join(savePath,"可可_crop.jpg"))  
  
# 粘贴图片  
imgPaste = imgCopy.copy()  
for x in range(0, 1080, 270):  
    for y in range(0, 1080, 124):  
        imgPaste.paste(cropImg.resize((270, 124)), (x, y))  
  
imgPaste.save(os.path.join(savePath, "可可_paste_spread_out.jpg"))
```
图像裁剪函数`crop`，接受的元组四个数字代表含义如下：(左上角的x轴坐标，左上角的y轴坐标，左上角的y轴坐标，左上角的y轴坐标）
# 绘制图片
图像绘制的功能基本都在`ImageDraw`包内
```python
from PIL import ImageDraw

savePath = os.path.join(os.environ['USERPROFILE'], 'Desktop')  
  
# 创建一个图像  
img = Image.new("RGBA", (1000, 1000), "Cyan")  
# 获取绘制对象  
draw = ImageDraw.Draw(img)  
# 画点 黑色  
for x in range(0, 200, 5):  
    for y in range(0, 200, 5):  
        draw.point([(x, y)], fill="black")  
# 划线(十字架)  
draw.line([(500, 0), (500, 1000)], fill="red")  
draw.line([(0, 500), (1000, 500)], fill="blue")  
  
# 画圆  
draw.ellipse((180, 180, 480, 480), fill="green")  
# 画椭圆  
draw.ellipse((710, 100, 900, 410), fill="red")  
# 画矩形(蓝色底层、红色边框线)  
draw.rectangle((100, 520, 400, 800), fill="blue", outline="black")  
# 画多变形  
draw.polygon([(650, 650), (700, 510), (930, 880), (508, 711), (666, 999)], fill="Purple")  
img.save(os.path.join(savePath, "draw.png"))
```
# 填充图片
