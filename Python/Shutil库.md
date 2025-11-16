`shutil`（Shell Utilities 的缩写）是 Python 标准库中一个用于高级文件操作的模块。
它建立在 `os` 模块之上，提供了对文件和文件集合进行复制、移动、删除等更高级别的操作，是文件管理自动化的利器。
# 复制文件
| 方法                              | 说明                                                     |
| ------------------------------- | ------------------------------------------------------ |
| **`shutil.copy(src, dst)`**     | **复制文件内容**和**权限**。`dst` 可以是目标目录或目标文件路径。                |
| **`shutil.copy2(src, dst)`**    | 比 `copy()` 更强，**额外复制所有元数据**（如创建时间、修改时间）。**（通常是更好的选择）** |
| **`shutil.copytree(src, dst)`** | **递归复制整个目录树**（包括所有子目录和文件）。                             
这是 `shutil` 最常用的功能。
示例：复制文件夹
```python
import shutil

src_project = ‘my_awesome_project‘
dst_backup = ‘my_awesome_project_backup_20231027‘

# 如果备份目录已存在，copytree 会报错 FileExistsError
try:
    shutil.copytree(src_project, dst_backup)
    print(f"项目已成功备份至: {dst_backup}")
except FileExistsError:
    print("备份目录已存在，请删除或重命名后再试。")
```
# 移动/重命名文件/目录
| 方法                          | 说明                         |
| --------------------------- | -------------------------- |
| **`shutil.move(src, dst)`** | **递归移动文件或目录**到新位置，也可用于重命名。 |
`os.rename()` 功能类似，但在跨文件系统移动时可能会失败，而 `shutil.move()` 会处理得更好。
示例：移动和重命名
```python
import shutil

# 移动文件到新目录
shutil.move(‘downloads/temp_file.iso‘, ‘software/‘)

# 移动并重命名
shutil.move(‘old_name.txt‘, ‘documents/new_name.txt‘)

# 移动整个目录
shutil.move(‘old_folder‘, ‘new_location/renamed_folder‘)
```
# 删除目录树
| 方法                        | 说明                         |
| ------------------------- | -------------------------- |
| **`shutil.rmtree(path)`** | **递归删除整个目录树**（包括所有子目录和文件）。 |
相比之下，`os.rmdir()` 只能删除**空目录**，功能非常有限。
**警告：此操作是永久性的，无法从回收站恢复！**
示例：安全删除构建产物
```python
import shutil
import os

# 常见的构建输出目录
build_dirs = [‘build‘, ‘dist‘, ‘__pycache__‘, ‘.pytest_cache‘]

for dir_name in build_dirs:
    if os.path.exists(dir_name):
        print(f"正在删除目录: {dir_name}")
        shutil.rmtree(dir_name)
print("清理完成！")
```
# 磁盘使用情况
| 方法                            | 说明                                               |
| ----------------------------- | ------------------------------------------------ |
| **`shutil.disk_usage(path)`** | 返回一个命名元组 `(total, used, free)`，显示路径所在磁盘的字节数使用情况。 |
可以在执行大文件操作前检查磁盘空间。
示例：检查C盘剩余空间
```python
import shutil

# 检查C盘空间
total, used, free = shutil.disk_usage(‘C:/‘)

# 转换为GB并打印
gib = 2**30 # GiB (Gibibyte)
print(f"总空间: {total / gib:.2f} GiB")
print(f"已用空间: {used / gib:.2f} GiB")
print(f"剩余空间: {free / gib:.2f} GiB")

if free / gib < 10: # 如果剩余空间小于10GB
    print("警告：磁盘空间不足！")
```
# 压缩功能
| 方法                                                     | 说明                |
| ------------------------------------------------------ | ----------------- |
| **`shutil.make_archive(base_name, format, root_dir)`** | 创建归档文件（如ZIP或TAR）。 |
| **`shutil.unpack_archive(filename, extract_dir)`**     | 解开归档文件。           |
`shutil` 提供了创建和解开归档文件（如 .zip, .tar）的高级接口。
示例：压缩和解压
```python
import shutil

# 1. 将 ‘my_project‘ 目录打包成 ZIP 文件
# 生成的文件名为 ‘project_backup.zip‘
shutil.make_archive(‘project_backup‘, ‘zip‘, ‘my_project‘)

# 2. 将 ‘data.zip‘ 解压到 ‘extracted_data‘ 目录
shutil.unpack_archive(‘data.zip‘, ‘extracted_data‘)

# 它支持多种格式：'zip', 'tar', 'gztar' (.tar.gz), 'bztar' (.tar.bz2), 'xztar' (.tar.xz)
```