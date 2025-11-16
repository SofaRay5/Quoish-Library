# 下载和卸载Python
官网下载安装包之后，需要注意勾选添加环境变量到Path
卸载的时候也需要注意把注册表、环境变量删干净，不然如果这样重装，不同的版本可能会出现冲突
可以在cmd窗口查看python版本
```bash
>python --version
```
# 修改Pycharm的解释器
刚安装Pycharm的时候Jetbrain会在Pycharm自己的安装路径搞一套独有的Python解释器
为了统一用同一个Python解释器，避免出现如安装了库这边没有显示的问题，更新的问题，需要在`文件→设置→项目：python→Python解释器`中修改当前使用的解释器
# 使用pip管理python库
从官网安装python之后默认会安装pip
可以使用cmd窗口使用pip
## 查看pip版本
```bash
>pip --version
```
## 更新pip
```bash
>python -m pip install --upgrade pip
```
## 使用pip安装库
以`requests`库为例
```bash
>pip install requests
```
安装指定版本
```bash
>pip install django==4.2
>pip install "flask>=2.0,<3.0"
```
## 升级库
```bash
>pip install --upgrade pandas
```
## 卸载库
```bash
>pip uninstall numpy
```
## 查看已安装库
```bash
>pip list
```
## 查看已安装库详情
```bash
>pip show request
```