# 常用代码命名规则

|  类型   | 推荐风格         | 示例                                               | 注意事项                      |
| :---: | ------------ | ------------------------------------------------ | ------------------------- |
| 类/结构体 | PascalCase   | `class MyClass;`                                 | 含模板：`class HashMap<T>;`   |
| 函数/方法 | camelCase    | `getSize()` / `get\_size()`                      | 动词开头（动作类函数）               |
|  变量   | lower\_snake | `int total\_count;`                              | 全局变量前面加`g\_`，静态变量前面加`s\_` |
| 类成员变量 | 后缀下划线        | `int count\_;`                                   | 或前缀 `m\_count`（较少用）       |
|  常量   | UPPER\_SNAKE | `kMaxSize` / `MAX\_SIZE`                         | 全局常量需显式标记 `const`         |
|  枚举值  | PascalCase   | `enum Color { Red, Green };`                     | 全大写（传统风格）                 |
| 命名空间  | snake\_case  | `namespace my\_project;`                         | 避免缩写                      |
|   宏   | UPPER\_SNAKE | `#define DEBUG\_MODE`                            | 尽量少用宏                     |
| 类型别名  | PascalCase   | `using StringVector = std::vector<std::string>;` | 模板别名同理                    |

# C语言中的关键字
## `auto`
用于声明该变量是局部变量，在该代码块执行结束后自动销毁
通常不用写`auto`也默认函数里的变量是局部变量
```c
void func {
  int a = 10;  // 等价于 auto int a = 10;
}
```
## `const`
用于声明该变量（或指针）是不可被修改的
```c
const int MAX = 100;  // 值不可变
const int\* ptr = \&x;  // 通过指针不可修改数据


const int \*p;         //常量指针，指针指向的变量不可更改
int \*const p = \&x;    //指针常量，指针的地址不可更改


//const只是只读变量，可以通过指针间接修改（编译器可能会警告），这种行为在C++中严格禁止
const int x = 5;  
int \*p = (int\*)\&x;  
\*p = 10;


// 函数参数保护功能
void print(const char \*str) {  
    // str内容不可修改 
}


// 阻止通过返回值修改底层数据
const int\* getConstValue {  
    static int val = 42;  
    return \&val;  // 需用const指针接收 
} 
```
## `extern`
用于声明外部文件中定义的全局变量
```c
extern int g\_Var;  // 变量定义在另一个文件中
```
## `register`
用于声明建议编译器将该变量存入寄存器中，以便更快速的存取
```c
register int i;  // 用于频繁访问的变量
```
## `sizeof`
用于计算类型、变量、表达式所占用的字节数大小
```c
sizeof(int);      // 类型名，这里返回4
int var = 0;
sizeof(var);      // 变量名，这里返回4
sizeof(var++);    // 表达式，这里返回4，但是并不执行var++表达式，var依然会是0


char string = "Hello";
sizeof(string);   //返回的大小包括"\\0"，这里返回6
strlen(string);   //返回的大小不包括\\0"，这里返回5
```
## `static`
用于将变量存储在静态工作区，这样定义在函数里面的局部变量，只会在第一次调用该函数的时候初始化，即使函数执行结束后，也不会释放内存，变量的生命长度能贯穿程序始终
但是与全局变量不同，`static`修饰下的局部变量依然是局部变量，只能在那个函数内部进行访问，无法在外部访问
`static`也能修饰函数名、全局变量，其作用域局限在定义它的源文件里，在其他文件即使用`extern`也不能访问
```c
void counter {
    static int count = 0; // 静态局部变量，首次调用初始化，后续保留值
    count++;
    printf("Count: %d\\n", count);
}


// file1.c
static int hiddenVar = 42; // 仅file1.c可见
// file2.c
extern int hiddenVar;      // 错误！无法访问
```

| 场景               | 实现方式                     | 优势                         |
| ------------------ | ---------------------------- | ---------------------------- |
| 模块私有函数       | 静态函数                     | 隐藏实现细节，防止外部误调用 |
| 状态保持           | 静态局部变量                 | 如计数器、缓存               |
| 文件级数据封装     | 静态全局变量                 | 避免全局变量污染命名空间     |
| 多文件同名符号共存 | 不同文件定义同名`static`符号 | 编译通过，无冲突             |
## `volatile`
用于告知编译器每次访问变量时直接从内存读取或写入，而非使用寄存器缓存的值
当变量的值可能被程序之外的因素意外修改（如硬件、中断或并发线程）时，使用`volatile`可以禁止编译器优化该变量的访问操作
在多核环境下，`volatile`变量的修改会直接同步到主存，使其他核心可见（但需注意其原子性限制）
```c
volatile bool data\_ready = false;
void IRQ\_Handler(){
    data\_ready = true;
}                               // 中断触发时立即修改


volatile int \*p;                // 指针指向的值可能被意外修改
int \*volatile p;                // 指针本身的地址可能变化


const volatile int i;           // 表示变量可能被外部修，但是程序内不可修改
```

| 场景           | 未使用`volatile`           | 使用`volatile`         |
| -------------- | -------------------------- | ---------------------- |
| 多中断异步写入 | 寄存器缓存导致变量无法同步 | 强制内存读取，循环终止 |
| 硬件异步写入   | 编译器优化掉多次写操作     | 每次写入均生效         |
| 多线程共享变量 | 线程读取旧值               | 立即感知最新值         |
## `typedef`
用于为现有数据类型重命名
```c
typedef unsigned char BYTE;  // 定义单字节别名
BYTE b1, b2;                 // 等价于 unsigned char b1, b2;


typedef struct {             // 与struct联合使用
    int x, y;
} Point; 
Point p = {1, 2};

// 函数指针封装
int sum(int a, int b) {
    return a + b;
}
typedef int (\*MathFunc)(int, int);  // 定义新类型MathFunc，它表示"指向int func(int, int)形式函数的指针"
MathFunc add = \&sum;                // 指向符合签名的函数

// 嵌套类型定义（如：链表）
typedef struct Node {
    int data;
    struct Node\* next;       // 直接使用struct标签，避免未定义别名的依赖
} Node; 
```

