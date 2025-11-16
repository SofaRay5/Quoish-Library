```matlab
% 创建一个图形窗口，并获取图形句柄
f = figure;
福
创建新的图形窗口f
创建坐标轴并获取坐标轴句柄
ax = axes;
创建新的坐标轴 ax
生成数据并绘制曲线，获取曲线句柄
X = 0:0.1:10;
生成x数据
y=sin(x);
8生成y数据
h= plot(x,
绘制曲线并获取曲线句柄h
修改曲线属性
set(h, 'LineWidth', 2);
8设置曲线的线宽为2
set(h, 'Color','r');
设置曲线的颜色为红色
set(h, 'LineStyle',
O
8设置曲线为虚线
修改坐标轴的属性
set(ax,
'XLim',[010], "YLim', [-1.51.5]);
设置坐标轴的X范围和Y范围
set(ax,'FontSize',12);
设置坐标轴刻度字体大小
给图形添加标题
title('Sine Wave
8设置标题
给图形添加坐标轴标签
xlabel('X Axis');
设置X轴标签
ylabel('Y Axis');
8设置Y轴标签
```
