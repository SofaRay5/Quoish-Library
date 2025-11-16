PyTorch是一个基于python的科学计算包，服务于两个广泛的目的：
- 取代NumPy使用gpu和其他加速器的能力。
- 用于实现神经网络的自动微分库。
[官方文档](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
# 张量
张量是一种特殊的数据结构，与数组和矩阵非常相似。在PyTorch中，我们使用张量来编码模型的输入和输出，以及模型的参数。
张量类似于[[NumPy库]]的nd-array，除了张量可以在gpu或其他专用硬件上运行以加速计算。
```python
import torch
import numpy as np
```
## 创建张量
张量可以用不同的方式初始化。看看下面的例子：
### 直接从数组创建
张量可以直接从数据中创建。自动推断数据类型。
```python
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
```
### 从Numpy数组创建
张量可以从NumPy数组中创建（反之亦然）。
```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```
### 从另一个张量创建
新张量保留参数张量的属性（形状，数据类型），除非显式覆盖。
```python
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
'''''''''''''''''''输出''''
Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.1176, 0.2952],
        [0.1362, 0.2888]])

```
### 使用随机或固定值
形状是张量维的元组。在下面的函数中，它决定了输出张量的维数。
```python
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
'''''''''''''输出''''
Random Tensor:
 tensor([[0.9887, 0.6382, 0.5760],
        [0.1588, 0.7824, 0.9119]])

Ones Tensor:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```
## 张量的属性
张量属性描述它们的形状、数据类型和存储它们的设备。
```python
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
'''''''''''''输出''''
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```
## 张量操作
超过100张量操作，包括转置，索引，切片，数学运算，线性代数，随机抽样，以及更多的全面描述在[这里](https://docs.pytorch.org/docs/stable/torch.html)。
### 转移到GPU
它们中的每一个都可以在GPU上运行（通常比在CPU上运行的速度更快）。如果你使用Colab，通过编辑>笔记本设置来分配一个GPU。
```python
# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")
'''''''''''''输出''''
Device tensor is stored on: cuda:0
```
### 类似Numpy的索引和切片
```python
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)
'''''''''''''输出''''
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```
您可以使用torch.cat沿着给定的维度连接一系列张量。也可以参见torch.Stack，这是另一个连接张量的操作符，与torch.cat略有不同。
```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
'''''''''''''输出''''
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```
### 张量相乘
```python
# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")
tensor.mul(tensor)
'''''''''''''输出''''
 tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor * tensor
 tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```
下面这个计算两个张量之间的矩阵乘法
```python
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")
'''''''''''''输出''''
tensor.matmul(tensor.T)
 tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])

tensor @ tensor.T
 tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
`````
以`_`为后缀的操作就是就地操作。例如：`x.copy_(y)`， `x.t_()`，这些操作将改变x。
```python
print(tensor, "\n")
tensor.add_(5)
print(tensor)
'''''''''''''输出''''
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
```
> 就地操作可以节省一些内存，但是在计算派生时可能会出现问题，因为会立即丢失历史记录。因此，不鼓励使用它们。
## 与Numpy数组的关系
CPU上的张量和NumPy数组可以共享它们的底层内存位置，更改其中一个将更改另一个。
### 张量转Numpy数组
```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
'''''''''''''输出''''
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
```
张量的变化将反映在NumPy数组中。
```python
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
'''''''''''''输出''''
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
```
### Numpy数组转张量
```python
n = np.ones(5)
t = torch.from_numpy(n)
```
NumPy数组的变化将反映在张量中。
```python
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
'''''''''''''输出''''
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
```
# 自动梯度计算（Autograd）
`torch.autograd`是PyTorch的自动微分引擎，为神经网络训练提供动力。在本节中，您将从概念上了解`autograd`如何帮助神经网络训练。
神经网络（NNs）是一组嵌套函数的集合，这些函数在一些输入数据上执行。这些函数由参数（由权重和偏置组成）定义，这些参数在PyTorch中存储在张量中。
训练一个神经网络分为两个步骤：
前向传播：在前向传播中，神经网络对正确的输出做出最好的猜测。它通过每个函数运行输入数据来进行猜测。
反向传播：在反向传播中，神经网络根据其猜测中的误差调整其参数。它通过从输出向后遍历，收集误差相对于函数参数（梯度）的导数，并使用梯度下降优化参数来实现这一点。
让我们看一下单个训练步骤。对于这个例子，我们从`torchvision`加载一个预训练的`resnet18`模型。我们创建了一个随机数据张量来表示具有3个通道的单个图像，高度和宽度为64，并将其相应的标签初始化为一些随机值。预训练模型中的标签具有形状（1,1000）。
>本教程只适用于CPU，不适用于GPU设备（即使张量被移动到CUDA）。

```python
import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
```
接下来，我们通过模型的每一层运行输入数据来进行预测。这是前向传播。
```python
prediction = model(data) # forward pass
```
我们使用模型的预测和相应的标签来计算误差（`loss`）。下一步是通过网络反向传播这个误差。当我们调用误差张量上的`.backward()`时，反向传播被启动。然后Autograd计算并存储参数中每个模型参数的梯度到`.grad`属性。
```python
loss = (prediction - labels).sum()
loss.backward() # backward pass
```
接下来，我们加载一个优化器，在本例中是学习率（learn rate即`lr`）为0.01，转动惯量（`momentum`）为0.9的SGD。我们在优化器中给出模型的所有参数。
```python
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
```
最后，我们调用`.step()`启动梯度下降。优化器根据存储在`.grad`中的梯度调整每个参数。
```python
optim.step() # gradient descent
```
至此，你已经拥有了训练神经网络所需的一切。下面的部分详细介绍了autograd的工作原理——可以随意跳过。
# 神经网络
神经网络可以使用库`torch.nn`来构建。
现在您已经了解了`autograd`， `nn`依赖于`autograd`来定义模型并给它们做微分。一个`nn.Module`包含若干个层（layers）和一个返回输出`output`的`forward(input)`方法。
例如，看看这个对数字图像进行分类的网络：
![[convnet.png]]
这是一个简单的前馈网络。它接受输入，将其通过几个层一个接一个地传递，然后最后给出输出。
神经网络的典型训练过程如下：
- 定义具有一些可学习参数（或权重）的神经网络
- 迭代输入数据集
- 通过网络处理输入
- 计算损失（输出距离正确有多远）
- 将梯度传播回网络的参数
- 更新网络的权重，通常使用简单的更新规则：`weight = weight - learning_rate * gradient`
## 定义神经网络
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 卷积层定义
        self.conv1 = nn.Conv2d(1, 6, 5)  # 输入1通道，输出6通道，5x5卷积核
        self.conv2 = nn.Conv2d(6, 16, 5) # 输入6通道，输出16通道，5x5卷积核
        
        # 全连接层定义
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 输入维度16*5*5，输出120维
        self.fc2 = nn.Linear(120, 84)          # 输入120维，输出84维
        self.fc3 = nn.Linear(84, 10)           # 输入84维，输出10维（对应10个类别）

	def forward(self, input):
		# 假设输入是形状为 (N, 1, 32, 32) 的批处理图像（N是批量大小）
	
	    # 第一卷积层 + ReLU激活（因为5×5卷积会减少4个像素）
	    c1 = F.relu(self.conv1(input))        # 输出形状: (N, 6, 28, 28)
	    
	    # 第一池化层（下采样）（2×2池化减半）
	    s2 = F.max_pool2d(c1, (2, 2))         # 输出形状: (N, 6, 14, 14)
	    
	    # 第二卷积层 + ReLU激活
	    c3 = F.relu(self.conv2(s2))           # 输出形状: (N, 16, 10, 10)
	    
	    # 第二池化层（下采样）
	    s4 = F.max_pool2d(c3, 2)              # 输出形状: (N, 16, 5, 5)
	    
	    # 展平操作，将多维张量转换为一维
	    s4 = torch.flatten(s4, 1)             # 输出形状: (N, 400)
	    
	    # 全连接层 + ReLU激活
	    f5 = F.relu(self.fc1(s4))             # 输出形状: (N, 120)
	    f6 = F.relu(self.fc2(f5))             # 输出形状: (N, 84)
	    
	    # 输出层（无激活函数）
	    output = self.fc3(f6)                 # 输出形状: (N, 10)
	    return output


net = Net()
print(net)
'''''''''''''输出''''
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```
你只需要定义前向函数，后向函数（计算梯度）会使用autograd自动为你定义。你可以在正向函数中使用任何张量操作。
模型的可学习参数由`net.parameters()`返回。
```python
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
'''''''''''''输出''''
10
torch.Size([6, 1, 5, 5])
```
让我们尝试一个随机的32x32输入。
注：此网（LeNet）的预期输入大小为32x32。要在MNIST数据集上使用此网络，请将数据集中的图像大小调整为32x32。
```python
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
'''''''''''''输出''''
tensor([[ 0.0727,  0.0602,  0.0593, -0.0491,  0.1452, -0.0927,  0.0315, -0.0786,
          0.0077, -0.0741]], grad_fn=<AddmmBackward0>)
```
将所有参数和随机梯度backprops的梯度缓冲区归零：
```python
net.zero_grad()
out.backward(torch.randn(1, 10))
```
>`torch.nn` 模块仅支持小批量（mini-batches） 输入。所有 `torch.nn` 层（如 `nn.Conv2d`、`nn.Linear` 等）均要求输入数据包含批量维度（batch dimension），即使批量大小为1。
>如：
> - `nn.Conv2d`：需输入 **4D 张量**，形状为 `nSamples x nChannels x Height x Width`。
> - `nn.Linear`：需输入 **2D 张量**，形状为 `nSamples x nChannels`。
> 如果只有一个样本，就使用`input.unsqueeze(0)`以添加一个假的批量维度。

在继续之前，让我们回顾一下到目前为止看到的所有类。
回顾：
- `torch.Tensor`：一个多维数组，支持像`backward()`这样的autograd操作。同时储存着对该张量的梯度。
- `nn.Module`：神经网络模块。方便的方式封装参数，与帮助移动到GPU，导出，加载等。
- `nn.Parameter`：一种张量，当作为属性分配给模块时，它会自动作为参数给出。
- `autograd.Function`：实现Autograd操作的前向、反向传播的定义。每个张量操作至少创建一个Function节点，该节点连接到创建张量的函数并编码其历史。
至此，我们已经定义了神经网络、输入处理、正向、反向传播，还需要计算损失、更新网络参数。
## 损失函数
损失函数接受`(输出,目标)`对作为输入，并计算一个值来估计输出与目标的距离。
在`nn`库下有几种不同的损失函数。一个简单的损失函数是：`nn.MSELoss`，用于计算输出和目标之间的均方误差。
例如：
```python
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
'''''''''''''输出''''
tensor(1.5713, grad_fn=<MseLossBackward0>)
```
现在，如果你在反向传播的过程中跟踪`loss`，用它的`.grad_fn`属性，你会看到一个计算图，看起来像这样：
```
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> flatten -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
```
所以，当我们调用`loss.backward()`时，整个图会对神经网络的参数求导，并且图中所有具有`requires_grad=True`的张量将使它们的`.grad`张量随着梯度累积。
为了说明这一点，让我们跟踪几步反向传播：
```python
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
'''''''''''''输出''''
<MseLossBackward0 object at 0x7f3f2364d5a0>
<AddmmBackward0 object at 0x7f3f3b64ef80>
<AccumulateGrad object at 0x7f3f3b64ee60>
```
## 反向传播
把误差反向传播，我们所要做的就是调用`loss.backward()`。你需要清除现有的梯度，否则梯度会累积到现有的梯度上。
现在我们调用`loss.backward()`，看看在反向传播前后`conv1`的偏置梯度。
梯度将保存在网络各权重向量的`grad`属性中。
```python
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
'''''''''''''输出''''
conv1.bias.grad before backward
None
conv1.bias.grad after backward
tensor([-0.0279, -0.0083, -0.0096, -0.0049, -0.0144,  0.0053])
```
现在，我们已经知道了如何使用损失函数。
神经网络包包含各种模块和损失函数，这些模块和损失函数构成了深度神经网络的构建块。完整的列表和文档在[这里](https://docs.pytorch.org/docs/stable/nn.html)。
## 更新网络权值
在实践中使用的最简单的更新规则是随机梯度下降（SGD）：
```
weight = weight - learning_rate * gradient
```
我们可以用简单的Python代码来实现：
```python
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```
然而，当你使用神经网络时，你想要使用各种不同的更新规则，如SGD、Nesterov-SGD、Adam、RMSProp等。为了实现这一点，我们构建了一个小库：`torch.optim`实现了所有这些方法。使用它很简单：
```python
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
```
# 训练一个分类器
到这里。你已经看到了如何定义神经网络，计算损失和更新网络的权重。
这个时候你可能会想
## 如何处理数据
通常，当必须处理图像、文本、音频或视频数据时，可以使用将数据加载到`numpy`数组中的标准python包。然后你可以把这个数组转换成一个`torch.*Tensor`。
- 对于图像，像Pillow， OpenCV这样的包是有用的
- 对于音频，使用scipy和librosa等软件包
- 对于文本，无论是原始的Python或基于Cython的加载，还是NLTK和SpaCy都是有用的
特别是对于视觉，我们已经创建了一个名为`torchvision`的包，其中有用于常见数据集（如`ImageNet`， `CIFAR10`， `MNIST`等）的数据加载器和用于图像的数据转换器，即`torchvision.datasets`和`torch.utils.data.DataLoader`。
这提供了极大的便利，避免了编写样板代码。
对于本教程，我们将使用CIFAR10数据集。它有“飞机”、“汽车”、“鸟”、“猫”、“鹿”、“狗”、“青蛙”、“马”、“船”、“卡车”等类。CIFAR-10中的图像尺寸为3x32x32，即尺寸为32x32像素的3通道彩色图像。
## 训练一个图像分类器
我们将依次执行以下步骤：
- 使用`torchvision`加载和规范化CIFAR10训练和测试数据集
- 定义卷积神经网络
- 定义一个损失函数
- 在训练数据上训练网络
- 测试网络上的测试数据
### 加载并标准化CIFAR10数据库
使用`torchvision`可以非常容易加载CIFAR10。
```python
import torch
import torchvision
import torchvision.transforms as transforms
```
`torchvision`数据集的输出是范围为$[0,1]$的PILImage图像。我们把它们变换成归一化范围$[- 1,1]$的张量。
>如果在Windows上运行并出现一个`BrokenPipeError`，尝试将`torch.utils.data.DataLoader()`的`num_worker`设置为0。
```python
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
'''''''''''''输出''''
  0%|          | 0.00/170M [00:00<?, ?B/s]
  0%|          | 328k/170M [00:00<00:57, 2.95MB/s]
  1%|          | 885k/170M [00:00<00:38, 4.41MB/s]
  1%|          | 1.61M/170M [00:00<00:30, 5.55MB/s]
  1%|▏         | 2.46M/170M [00:00<00:25, 6.67MB/s]
  2%|▏         | 3.51M/170M [00:00<00:20, 7.98MB/s]
  3%|▎         | 4.78M/170M [00:00<00:17, 9.55MB/s]
  4%|▎         | 6.32M/170M [00:00<00:14, 11.4MB/s]
  5%|▍         | 8.26M/170M [00:00<00:11, 13.9MB/s]
  6%|▌         | 10.6M/170M [00:00<00:09, 17.0MB/s]
  8%|▊         | 13.6M/170M [00:01<00:07, 20.8MB/s]
 10%|▉         | 16.8M/170M [00:01<00:06, 24.2MB/s]
 12%|█▏        | 20.0M/170M [00:01<00:05, 26.6MB/s]
 14%|█▍        | 24.0M/170M [00:01<00:04, 30.5MB/s]
 17%|█▋        | 29.4M/170M [00:01<00:03, 37.3MB/s]
 21%|██        | 35.9M/170M [00:01<00:02, 45.7MB/s]
 26%|██▌       | 44.1M/170M [00:01<00:02, 56.7MB/s]
 30%|███       | 51.4M/170M [00:01<00:01, 61.5MB/s]
 34%|███▍      | 58.0M/170M [00:01<00:01, 62.7MB/s]
 38%|███▊      | 65.3M/170M [00:01<00:01, 65.4MB/s]
 42%|████▏     | 72.4M/170M [00:02<00:01, 67.0MB/s]
 46%|████▋     | 79.1M/170M [00:02<00:01, 66.6MB/s]
 51%|█████     | 86.4M/170M [00:02<00:01, 68.4MB/s]
 55%|█████▍    | 93.2M/170M [00:02<00:01, 68.2MB/s]
 59%|█████▉    | 101M/170M [00:02<00:00, 70.2MB/s]
 63%|██████▎   | 108M/170M [00:02<00:00, 67.5MB/s]
 67%|██████▋   | 115M/170M [00:02<00:00, 68.8MB/s]
 72%|███████▏  | 122M/170M [00:02<00:00, 65.1MB/s]
 76%|███████▌  | 129M/170M [00:02<00:00, 67.0MB/s]
 80%|███████▉  | 136M/170M [00:02<00:00, 64.6MB/s]
 84%|████████▍ | 143M/170M [00:03<00:00, 66.1MB/s]
 88%|████████▊ | 150M/170M [00:03<00:00, 64.3MB/s]
 92%|█████████▏| 157M/170M [00:03<00:00, 65.9MB/s]
 96%|█████████▌| 163M/170M [00:03<00:00, 65.1MB/s]
100%|█████████▉| 170M/170M [00:03<00:00, 66.5MB/s]
100%|██████████| 170M/170M [00:03<00:00, 48.7MB/s]
```
为了好玩，让我们展示一些训练图像。
```python
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
```
### 定义卷积神经网络
从前面的神经网络的章节复制神经网络，并修改它以获取3通道图像（而不是定义的1通道图像）。
```python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
```
### 定义损失函数和优化器
让我们使用分类交叉熵损失和SGD。
```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```
### 训练网络
这就是事情开始变得有趣的时候。我们只需要循环遍历数据迭代器，并将输入输入提供给网络并进行优化。
```python
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
'''''''''''''输出''''
[1,  2000] loss: 2.224
[1,  4000] loss: 1.861
[1,  6000] loss: 1.681
[1,  8000] loss: 1.571
[1, 10000] loss: 1.518
[1, 12000] loss: 1.472
[2,  2000] loss: 1.386
[2,  4000] loss: 1.356
[2,  6000] loss: 1.312
[2,  8000] loss: 1.321
[2, 10000] loss: 1.279
[2, 12000] loss: 1.260
Finished Training
```
让我们快速保存训练好的模型：
```python
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
```
有关保存PyTorch模型的更多细节，请参阅[此处](https://docs.pytorch.org/docs/stable/notes/serialization.html)。
### 测试网络上的测试数据
我们已经在训练数据集上训练了2次网络。但我们需要检查网络是否学到了什么。
我们将通过预测神经网络输出的类标签来检查这一点，并根据基本事实进行检查。如果预测是正确的，我们将样本添加到正确预测列表中。
好的，第一步。让我们显示测试集中的一个图像来熟悉一下。
```python
dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
```
接下来，让我们重新加载我们保存的模型（注意：这里不需要保存和重新加载模型，我们这样做只是为了说明如何这样做）：
```python
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))
```
好了，现在让我们看看神经网络是如何看待上面这些例子的：
```python
outputs = net(images)
```
输出是10个类所得的分数。一个类的得分越高，网络就越认为图像属于这个类。那么，让我们得到最高能量的指数：
```python
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))
```
结果似乎相当不错。  
让我们看看网络在整个数据集上的表现。
```python
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
'''''''''''''输出''''
Accuracy of the network on the 10000 test images: 56 %
```
这看起来比运气好得多，运气是10%的准确率(从10个类别中随机选择一个类别)。看来网络学到了一些东西。
嗯，哪些类型表现良好，哪些类型表现不佳：
```python
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
'''''''''''''输出''''
Accuracy for class: plane is 50.4 %
Accuracy for class: car   is 71.0 %
Accuracy for class: bird  is 44.3 %
Accuracy for class: cat   is 47.0 %
Accuracy for class: deer  is 62.6 %
Accuracy for class: dog   is 33.4 %
Accuracy for class: frog  is 62.7 %
Accuracy for class: horse is 49.5 %
Accuracy for class: ship  is 73.9 %
Accuracy for class: truck is 69.7 %
```
好吧，接下来呢？  
我们如何在GPU上运行这些神经网络？
### 在GPU上训练
就像你把张量（Tensor）转移到GPU上一样，你把神经网络转移到GPU上。
让我们首先将我们的设备定义为第一个可见的cuda设备，如果我们有cuda可用：
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
'''''''''''''输出''''
cuda:0
```
本节的其余部分假设`device`是CUDA设备。
然后这些方法将递归地遍历所有模块并将其参数和缓冲区转换为CUDA张量：
```python
net.to(device)
```
记住，你也必须在每一步向GPU发送输入和目标：
```python
inputs, labels = data[0].to(device), data[1].to(device)
```
为什么我没有感觉到与CPU相比的巨大加速？因为你的网络真的很小。
# 重要函数
## nn.Conv2d
用于对二维数据做卷积
```python
class torch.nn.Conv2d(in_channels, 
					out_channels, 
					kernel_size, # 卷积核大小为kernel_size*kernel_size
					stride=1, # 每轮卷积过后卷积窗移动的像素
					padding=0, # 在特征图四周填充一圈像素，用于控制张量尺寸
					dilation=1, 
					groups=1, 
					bias=True, 
					padding_mode='zeros', 
					device=None, 
					dtype=None)
```
一个2维卷积层中：
- 输入张量$(N,C_{in},H_{in},W_{in})$
	- **N**：`batch_size`（批大小）
	- **C_in**：输入通道数
	- **H_in, W_in**：输入特征图的高度和宽度
- 输出张量$(N,C_{out},H_{out},W_{out})$
	- **C_out**：输出通道数（由 `out_channels` 参数决定）
	- **H_out, W_out**：由卷积核大小、步长、填充等参数决定
卷积层前后张量尺寸公式
设：
* $H_\text{in}, W_\text{in}$：输入高宽
* $K$：卷积核大小（如果是方形卷积核，写成 (K \times K)）
* $S$：stride（步长）
* $P$：padding（填充）
* $D$：dilation（膨胀卷积的扩张率，默认=1）
则
$$H_\text{out} = \Bigg\lfloor \frac{H_\text{in} + 2P - D,(K-1) - 1}{S} + 1 \Bigg\rfloor$$
$$W_\text{out} = \Bigg\lfloor \frac{W_\text{in} + 2P - D,(K-1) - 1}{S} + 1 \Bigg\rfloor$$

## nn.BatchNorm2d
```python
_class_ torch.nn.BatchNorm2d(_num_features_, 
						_eps=1e05_, 
						_momentum=0.1_, 
						_affine=True_, 
						_track_running_stats=True_, 
						_device=None_, 
						_dtype=None_)
```
对卷积后的每个通道做标准化（均值 0 方差 1）
可以加快收敛，并且有一定的正则化效果，防止过拟合，同时缓解梯度爆炸