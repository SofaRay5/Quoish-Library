一、安装显卡驱
1、右击鼠标桌面，点击NVIDIA 控制面板，可以查看是否安装了显卡驱动，以及驱动的版本号，以及显卡的名称，比如我的就是GeForce 940MX，驱动版本如果太旧的话，也需要执行下面的步骤，进行驱动的安装
2、进入NVIDIA官网（[官网](https://www.nvidia.cn/drivers/lookup/)），搜索驱动程序，如下图所示，然后随便选一个下载就行了
3、双击下载好的安装程序
4、点击ok
5、等待
6、点击同意并继续
7、选择精简，点击下一步
8、安装完成点击关闭
二、安装CUDA
1、在cmd输入命令`nvidia-smi`可以查看CUDA版本号，如我的是13.0，就是说我安装的CUDA版本号不能高于13.0
2、进入网站（[官网](https://developer.nvidia.com/cuda-toolkit-archive)）下载CUDA，我安装的是11.8
3、选择自定义，点击下一步
4、只选第一个组件（我在实际操作的时候全选上了）
5、选择下一步
6、点击Next
7、安装中
8、点击下一步
9、安装完成选择关闭
10、cmd输入nvcc -V查看安装的cuda版本，如下所示，安装成功
三、下载cudnn
[下载地址](https://developer.nvidia.com/rdp/cudnn-archive)，需要注册nvidia账号
复制下载好的三个文件夹bin lib include 进cuda安装地址一般默认为：
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
四、安装anaconda
anaconda是一个python环境管理软件，它可以创建一个独立隔绝的虚拟环境，便于各种调试
[下载地址](https://www.anaconda.com)，下载双击安装windows版本安装到数据盘
安装的过程中记得将conda添加到环境变量path，否则无法在命令行界面调用conda
```bash
conda create -n name python==3.10.0   ## 创建名字为name的环境，指定python版本
conda activate name ## 进入激活

```
进入环境之后，命令行前面会有个括号，里面是当前环境的名字
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia #安装pytorch所需的各库
conda deactivate # 退出环境
conda list ##看此环境的细节
```
注意：有的时候，因为pytorch-cuda的版本不兼容，输入上面conda install的命令时，有可能自动安装成cpu的版本而非gpu的版本，可以使用命令`conda list | findstr torch`检查
```bash
# 错误情况
pytorch                      2.2.2               py3.10_cpu_0           pytorch
pytorch-cuda                 11.7                h16d0643_5             pytorch
pytorch-msssim               1.0.0               pypi_0                 pypi
pytorch-mutex                1.0                 cpu                    pytorch
torchaudio                   2.2.2               py310_cpu              pytorch
torchvision                  0.17.2              py310_cpu              pytorch
# 正确情况
pytorch                      2.2.2               py3.10_cuda11.8_cudnn8_0  pytorch
pytorch-cuda                 11.8                h24eeafa_6                pytorch
pytorch-msssim               1.0.0               pypi_0                    pypi
pytorch-mutex                1.0                 cuda                      pytorch
torchaudio                   2.2.2               pypi_0                    pypi
torchvision                  0.17.2              pypi_0                    pypi
```
conda可以做到像pip一样安装、更新库，但它们是库的资源是来自不同的地方的，有些库conda没有但是pip有，就只能用pip安装
conda还可以使用conda-forge，conda直接安装的库是来自conda官方，但用conda-forge安装的包来自开源社区，可以用如下命令选择从开源社区安装库
```bash
conda install -c conda-forge numpy=1.20 # 安装特定版本
```
一个兼容的库版本列表
```bash
python=3.10.0
pytorch-cuda=11.8
numpy==1.23.5
opencv-python==4.7.0
scipy==1.10.1
scikit-image==0.20.0
matplotlib==3.7.1
PyWavelets==1.4.1
```

安装CUDA 和Cudnn（复制）（GPU调用相关）
1.    装显卡驱动
2.    在cmd输入命令nvidia-smi可以查看CUDA版本号，我的是11.7，就是说我安装的CUDA版本号不能高于11.7
3.    CUDA Toolkit Archive | NVIDIA Developer下载相应cuda版本（选择相应系统）双击开始安装
4.    cmd输入nvcc -V查看安装的cuda版本
5.    下载cudnn cuDNN Archive | NVIDIA Developer
复制下载好的三个文件夹bin lib include 进cuda安装地址一般默认为：
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7

删除环境
```bash
conda env remove --name 环境名
```
查看已经创建的环境
```bash
conda env list
```