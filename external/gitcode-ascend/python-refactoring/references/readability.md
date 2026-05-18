# Skill: Python 代码可读性与可维护性改进

## 触发条件

当用户要求"提升代码可读性"、"改善可维护性"、"clean code review"、"代码整洁度优化"，或在重构任务中需要改善代码质量时触发。

## 目标

在不改变代码行为的前提下，通过命名、结构、类型标注、文档等手段提升代码的可读性和可维护性。

## 核心原则

- 代码是写给人读的，顺便让机器执行
- 好的命名胜过注释，好的结构胜过文档
- 每次修改只做一件事，保持原子性

---

## 1. 命名改进

### 1.1 变量与函数命名

**规则：**
- 变量名表达"是什么"，函数名表达"做什么"
- 布尔变量用 `is_`/`has_`/`can_`/`should_` 前缀
- 避免单字母变量（循环索引和 lambda 除外）
- 避免缩写，除非是领域内公认缩写（如 `url`、`http`、`db`）

```python
# Bad
d = get_data()
tmp = process(d)
flag = check(tmp)

# Good
user_records = fetch_user_records()
validated_records = validate_records(user_records)
is_all_valid = all(r.is_valid for r in validated_records)
```

### 1.2 函数命名约定

| 操作类型 | 前缀 | 示例 |
|----------|------|------|
| 获取数据 | `get_` / `fetch_` | `get_user_by_id()` |
| 计算/派生 | `calculate_` / `compute_` | `calculate_total_price()` |
| 检查/验证 | `is_` / `has_` / `validate_` | `is_valid_email()` |
| 转换 | `to_` / `convert_` / `parse_` | `to_json()`, `parse_config()` |
| 创建 | `create_` / `build_` | `create_connection()` |
| 副作用操作 | 动词开头 | `send_email()`, `save_report()` |

### 1.3 类命名

- 类名用名词或名词短语：`UserRepository`，不是 `ManageUsers`
- 避免无意义后缀：`Manager`、`Handler`、`Helper`、`Utils`（除非职责确实如此）
- Mixin 类加 `Mixin` 后缀：`LoggingMixin`
- Protocol 类描述能力：`Serializable`、`Comparable`

---

## 2. 结构改进

### 2.1 函数结构

**单一抽象层级原则：** 一个函数内的所有语句应处于同一抽象层级。

```python
# Bad: 混合抽象层级
def process_order(order):
    # 高层：业务逻辑
    if not order.items:
        raise ValueError("Empty order")
    # 低层：数据库细节
    conn = sqlite3.connect("orders.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO orders ...")
    conn.commit()
    conn.close()

# Good: 统一抽象层级
def process_order(order: Order) -> None:
    validate_order(order)
    total = calculate_total(order)
    save_order(order, total)
    notify_customer(order)
```

**函数长度指导：** ⚙ 阈值见 [`refactoring-config.json`](refactoring-config.json)
- 理想：≤ `function.ideal_lines`（默认 15）行
- 可接受：≤ `function.acceptable_lines`（默认 30）行
- 需要拆分：超过 `function.acceptable_lines` 行

### 2.2 条件表达式简化

```python
# Bad: 冗余布尔判断
if is_valid == True:
    return True
else:
    return False

# Good
return is_valid

# Bad: 嵌套三元
x = a if condition1 else (b if condition2 else c)

# Good: 明确的 if-elif
if condition1:
    x = a
elif condition2:
    x = b
else:
    x = c
```

### 2.3 用推导式替代循环构建

```python
# Bad
result = []
for item in items:
    if item.is_active:
        result.append(item.name)

# Good
result = [item.name for item in items if item.is_active]

# 注意：推导式超过一行时，改回显式循环更清晰
# Bad: 过度复杂的推导式
result = [
    transform(item.value)
    for group in groups
    for item in group.items
    if item.is_active and item.value > threshold
]

# Good: 回到显式循环
result = []
for group in groups:
    for item in group.items:
        if item.is_active and item.value > threshold:
            result.append(transform(item.value))
```

### 2.4 异常处理规范

```python
# Bad: 裸 except
try:
    data = fetch_data()
except:
    pass

# Bad: 过宽的异常捕获
try:
    data = json.loads(raw)
    user = User(**data)
    save(user)
except Exception as e:
    logger.error(e)

# Good: 精确捕获 + 合理处理
try:
    data = json.loads(raw)
except json.JSONDecodeError as e:
    raise InvalidInputError(f"Malformed JSON: {e}") from e

try:
    user = User(**data)
except TypeError as e:
    raise InvalidInputError(f"Missing required fields: {e}") from e

save(user)
```

---

## 3. 类型标注

### 3.1 何时添加类型标注

**必须标注：**
- 公共 API（模块级函数、类的公共方法）
- 函数参数和返回值
- 复杂数据结构（嵌套 dict、Union 类型）

**可以省略：**
- 局部变量（类型可推断时）
- 测试代码（除非提升可读性）
- 简单的单行 lambda

### 3.2 类型标注最佳实践

```python
# 使用 modern syntax（Python 3.10+）
def process(items: list[str]) -> dict[str, int]:
    ...

# Union 类型用 |
def find_user(id: int) -> User | None:
    ...

# 复杂类型用 TypeAlias
type UserMap = dict[str, list[User]]

# 回调函数类型
from collections.abc import Callable
Processor = Callable[[str, int], bool]

# 使用 Protocol 替代 ABC 做结构化类型
from typing import Protocol

class Readable(Protocol):
    def read(self, size: int = -1) -> bytes: ...
```

### 3.3 渐进式类型标注策略

对遗留代码，按优先级逐步添加：
1. 公共 API 的参数和返回值
2. 容易出错的地方（`Any`、`dict`、`Optional`）
3. 核心业务逻辑
4. 工具函数和内部方法

---

## 4. 注释与文档

### 4.1 注释原则

- 注释解释"为什么"，不解释"是什么"
- 如果需要注释解释代码在做什么，先考虑重命名或重构
- 删除过时注释比保留错误注释好

```python
# Bad: 解释"是什么"
# 遍历用户列表
for user in users:
    ...

# Bad: 过时注释
# 返回用户名（实际已改为返回 User 对象）
def get_user(id: int) -> User:
    ...

# Good: 解释"为什么"
# 倒序遍历以避免删除元素时索引偏移
for i in range(len(items) - 1, -1, -1):
    if items[i].is_expired:
        del items[i]
```

### 4.2 Docstring 规范

对公共 API 使用 Google 风格 docstring：

```python
def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> T:
    """Execute a function with exponential backoff retry.

    Retries on any exception, with delay doubling after each attempt.
    Useful for unreliable network calls or external service interactions.

    Args:
        func: The callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds between retries.

    Returns:
        The return value of the successfully executed function.

    Raises:
        Exception: The last exception if all retries are exhausted.
    """
```

### 4.3 TODO/FIXME 规范

```python
# TODO(username): 简短描述需要做什么 - #issue_number
# FIXME(username): 简短描述已知问题 - #issue_number
# HACK: 说明为什么需要这个临时方案以及何时可以移除
```

---

## 5. 模块组织

### 5.1 导入顺序（遵循 isort / PEP 8）

```python
# 1. 标准库
import os
import sys
from pathlib import Path

# 2. 第三方库
import numpy as np
from fastapi import FastAPI

# 3. 本地模块
from .models import User
from .utils import validate_email
```

### 5.2 模块内代码顺序

```python
"""Module docstring."""

# 导入

# 常量
MAX_RETRY = 3
DEFAULT_TIMEOUT = 30

# 类型别名
type UserID = int

# 公共函数/类（按依赖顺序或字母序）

# 私有函数/类（_ 前缀）

# 模块级执行代码（尽量避免）
```

### 5.3 `__all__` 显式导出

对公共模块定义 `__all__` 明确公共 API：

```python
__all__ = ["User", "create_user", "UserNotFoundError"]
```

---

## 6. Python 惯用写法（Pythonic Idioms）

| 非惯用写法 | Pythonic 写法 |
|------------|---------------|
| `if len(lst) == 0:` | `if not lst:` |
| `for i in range(len(lst)):` | `for i, item in enumerate(lst):` |
| `d.has_key(k)` | `k in d` |
| `f = open(...); f.read(); f.close()` | `with open(...) as f: f.read()` |
| `if x == None:` | `if x is None:` |
| `lst = list(filter(fn, items))` | `lst = [x for x in items if fn(x)]` |
| `try: d[k] except KeyError:` | `d.get(k, default)` |
| `a = a + [item]` | `a.append(item)` |
| 手动字符串拼接 | f-string: `f"{name}: {value}"` |
| `dict(a=1, b=2)` | `{"a": 1, "b": 2}`（字面量更快更清晰） |

---

## 执行流程

1. 通读目标代码，标记可读性和可维护性问题
2. 按影响范围排序：命名 > 结构 > 类型标注 > 注释 > 风格
3. 对每个问题给出：位置、当前写法、改进写法、改进理由
4. 优先修复影响理解的问题（命名、结构），再处理规范性问题
5. 确保所有修改不改变代码行为
6. 修复后运行已有测试和 linter 验证
