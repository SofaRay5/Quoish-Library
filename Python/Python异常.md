# Python异常
异常是指在程序执行过程中发生的一个事件，该事件会中断正常的指令流
## 部分python内置异常
- 打开一个不存在的文件 (`FileNotFoundError`)
- 用零除一个数 (`ZeroDivisionError`)
- 访问列表末尾之后的位置 (`IndexError`)
- 将字符串和整数相加 (`TypeError`)
- 网络请求超时 (`requests.exceptions.Timeout`)
## 异常处理
### try-except
捕获并处理异常
```python
try:
    # 可能抛出异常的代码
    risky_operation()
except SomeSpecificError: # 捕获特定异常
    # 处理该异常
    handle_specific_error()
except (AnotherError, YetAnotherError) as e: # 捕获多个异常，并获取异常对象
    # e 是异常对象的实例，可以访问它的信息
    print(f"发生了错误：{e}")
    handle_other_errors()
except Exception as e: # 捕获所有异常（谨慎使用！）
    # 这是一个宽泛的捕获，可能会隐藏你未预料到的错误
    print(f"发生了未知错误：{e}")
```
### else
当没有异常发生时执行
```python
try:
    result = calculate_something()
except CalculationError:
    print("计算出错了！")
else:
    # 只有在 try 成功时才使用 result
    print(f"计算成功，结果是：{result}")
    save_result(result) # 如果放在 try 里，save_result出错也会被except捕获，这不合理
```
### finally
无论如何都会执行
```python
file = open('file.txt')
try:
    process_file(file)
except IOError:
    print('处理文件时出错')
finally:
    file.close() # 无论是否出错，文件都会被关闭
```
### LBYL和EAFP
- **LBYL (Look Before You Leap)**：事先检查。“在跳之前先看。”
```python
if key in my_dict:
    value = my_dict[key]
else:
    handle_missing_key()
```
- **EAFP (Easier to Ask for Forgiveness than Permission)**：事后处理。“先做，出错再请求原谅。” 这是更Pythonic的风格。
```python
try:
    value = my_dict[key]
except KeyError:
    handle_missing_key()
```
EAFP通常更可取，因为它避免了竞争条件（尤其在多线程环境中），并且代码通常更简洁、可读。
## 自定义异常
可以通过继承`Exception`类自定义异常
```python
class MyCustomError(Exception):
    """我的自定义异常"""
    pass
```
示例
```python
class ECommerceError(Exception):
    """电商平台基础异常"""
    pass

class PaymentError(ECommerceError):
    """支付处理异常"""
    
    def __init__(self, message, order_id, payment_method, original_error=None):
        super().__init__(message)
        self.order_id = order_id
        self.payment_method = payment_method
        self.original_error = original_error

class InventoryError(ECommerceError):
    """库存不足异常"""
    
    def __init__(self, product_id, requested, available):
        message = f"产品 {product_id} 库存不足 (请求: {requested}, 可用: {available})"
        super().__init__(message)
        self.product_id = product_id
        self.requested = requested
        self.available = available

# 使用示例
def process_order(order):
    try:
        check_inventory(order)
        process_payment(order)
        # 其他处理逻辑...
    except InventoryError as e:
        # 通知用户库存不足，建议替代产品
        notify_user_of_inventory_issue(e.product_id, e.requested, e.available)
        raise
    except PaymentError as e:
        # 记录支付失败，可能需要重试或使用备用支付方式
        log_payment_failure(e.order_id, e.payment_method, e.original_error)
        raise
    except ECommerceError as e:
        # 处理其他电商相关错误
        handle_other_ecommerce_errors(e)
        raise

def check_inventory(order):
    for item in order.items:
        if item.quantity > get_available_quantity(item.product_id):
            raise InventoryError(
                item.product_id, 
                item.quantity, 
                get_available_quantity(item.product_id)
            )

def process_payment(order):
    try:
        # 调用支付网关API
        result = payment_gateway.charge(order.total, order.payment_method)
        if not result.success:
            raise PaymentError(
                f"支付被拒绝: {result.reason}",
                order.id,
                order.payment_method
            )
    except PaymentGatewayError as e:
        # 将第三方支付网关异常包装为我们的自定义异常
        raise PaymentError(
            "支付处理过程中发生错误",
            order.id,
            order.payment_method,
            e
        ) from e
```