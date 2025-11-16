`os`库主要干两件事：
1. **管理文件和目录**（创建、删除、重命名、查看信息等）。
2. **管理进程和环境**（运行系统命令、获取环境变量等）。
使用之前，需要导入库
```python
import os
```
某些需要管理员权限的命令，需要“以管理员身份运行”IDE或者exe。
`os`库没有复制文件的直接支持，也不可以直接删除非空目录，[[Shutil库]]可以弥补这两个缺点。
# 路径操作
假设 `path = ‘/home/user/file.txt’`

| 方法                      | 说明            | 示例                                          | 输出示例                          |
| ----------------------- | ------------- | ------------------------------------------- | ----------------------------- |
| `os.path.join()`        | **智能地拼接路径**   | `os.path.join(‘folder’, ‘sub’, ‘file.txt’)` | `‘folder/sub/file.txt’`       |
| `os.path.abspath()`     | 获取绝对路径        | `os.path.abspath(‘file.txt’)`               | `‘/home/user/code/file.txt’`  |
| `os.path.dirname()`     | 获取目录名         | `os.path.dirname(path)`                     | `‘/home/user’`                |
| `os.path.basename()`    | 获取文件名         | `os.path.basename(path)`                    | `‘file.txt’`                  |
| `os.path.split()`       | 分割目录和文件名      | `os.path.split(path)`                       | `(‘/home/user’, ‘file.txt’)`  |
| `os.path.exists()`      | 判断路径是否存在      | `os.path.exists(‘file.txt’)`                | `True / False`                |
| `os.path.isfile()`      | 判断是否是文件       | `os.path.isfile(path)`                      | `True`                        |
| `os.path.isdir()`       | 判断是否是目录       | `os.path.isdir(‘/home/user’)`               | `True`                        |
| `os.path.splitext()`    | **分割文件名和扩展名** | `os.path.splitext(path)`                    | `(‘/home/user/file’, ‘.txt’)` |
| `os.path.getsize(path)` | 获取文件大小（字节）    | `os.path.getsize("test.txt")`               | `1024`                        |
处理路径时，**永远不要**用手拼接字符串（比如 `'folder' + ‘/’ + ‘subfolder’`），而要用 `os.path` 子模块，它能自动处理不同操作系统（Windows/ macOS / Linux）的路径差异。
# 目录操作
| 方法                    | 说明                                  | 示例                                    | 注意                         |
| --------------------- | ----------------------------------- | ------------------------------------- | -------------------------- |
| `os.getcwd()`         | 获取当前工作目录（Current Working Directory） | `print(os.getcwd())`                  | 返回绝对路径                     |
| `os.chdir()`          | 改变当前工作目录（Change Directory）          | `os.chdir(‘/tmp’)`                    | 失败抛出 `FileNotFoundError`   |
| `os.listdir()`        | 列出指定目录下的所有文件和子目录名                   | `items = os.listdir(‘.’) # 列出当前目录`    | 返回文件名列表（不递归）               |
| `os.mkdir()`          | **创建单个目录**                          | `os.mkdir(‘new_folder’)`              | 目录已存在时抛出 `FileExistsError` |
| `os.makedirs()`       | **递归创建多层目录**                        | `os.makedirs(‘level1/level2/level3’)` | 类似 `mkdir -p`              |
| `os.rmdir()`          | 删除**空**目录                           | `os.rmdir(‘empty_folder’)`            | 目录非空时抛出 `OSError`          |
| `os.removedirs(path)` | 递归删除空目录                             | `os.removedirs("a/b/c")`              | 从最深层开始向上删除                 |
os跟cmd一样，拥有一个光标用于标定当前工作的目录，当写的不是绝对路径时，就是从该目录下面找。
**注意**：`os.mkdir(‘a/b/c’)` 如果 `a/b` 不存在会报错，而 `os.makedirs(‘a/b/c’)` 会连父目录一起创建。
# 文件操作
| 方法                      | 说明          | 示例                                             | 注意                           |
| ----------------------- | ----------- | ---------------------------------------------- | ---------------------------- |
| `os.remove(path)`       | 删除文件        | `os.remove("temp.txt")`                        | 文件不存在时抛出 `FileNotFoundError` |
| `os.rename(src, dst)`   | 重命名/移动文件或目录 | `os.rename("old.txt", "new.txt")`              | 跨磁盘移动可能失败                    |
| `os.stat(path)`         | 获取文件状态信息    | ``st = os.stat("file.txt")`  <br>`st.st_size`` | 返回包含大小、时间等的对象                |
| `os.utime(path, times)` | 修改文件访问/修改时间 | `os.utime("file.txt", (atime, mtime))`         | 时间戳为秒数                       |
# 进程管理
| 方法                     | 说明                  | 示例                             | 注意                           |
| ---------------------- | ------------------- | ------------------------------ | ---------------------------- |
| `os.system(command)`   | 执行系统命令（cmd或Shell命令） | `os.system("ls -l")`           | 文件不存在时抛出 `FileNotFoundError` |
| `os.kill(pid, signal)` | 向进程发送信号             | `os.kill(pid, signal.SIGTERM)` | Windows 支持有限                 |
# 环境变量
| 方法                        | 说明          | 示例                                 | 注意                  |
| ------------------------- | ----------- | ---------------------------------- | ------------------- |
| `os.environ`              | 字典形式的环境变量   | `os.environ["PATH"]`               | 修改后仅影响当前进程          |
| `os.getenv(key, default)` | 获取环境变量值     | `os.getenv("USERPROFILE", "/tmp")` | 键不存在时返回 `default`   |
| `os.putenv(key, value)`   | 设置环境变量（不推荐） | `os.putenv("DEBUG", "1")`          | 建议直接操作 `os.environ` |
# 系统信息
| 方法               | 说明         | 示例               | 输出示例                                    |
| ---------------- | ---------- | ---------------- | --------------------------------------- |
| `os.name`        | 操作系统名称     | `print(os.name)` | `posix`（Linux/macOS）  <br>`nt`（Windows） |
| `os.sep`         | 路径分隔符      | `os.sep`         | `/` 或 `\`                               |
| `os.linesep`     | 行终止符       | `os.linesep`     | `\n`（Linux）  <br>`\r\n`（Windows）        |
| `os.cpu_count()` | 返回 CPU 核心数 | `os.cpu_count()` | 16                                      |
