# Skill: Python 代码坏味道识别与修复

## 触发条件

当用户要求"识别代码坏味道"、"代码异味检查"、"code smell review"，或在重构任务中需要先定位问题代码时触发。

## 目标

系统性地识别 Python 代码中的坏味道（Code Smells），并给出具体的修复方案。

> **约束阈值可配置：** 本文档中的数值阈值（如函数行数、参数个数等）均从 [`refactoring-config.json`](refactoring-config.json) 读取。用户可在该文件中自定义阈值或禁用特定检查。下文用 `⚙` 标记可配置项。

---

## 1. 函数/方法级坏味道

### 1.1 过长函数（Long Method）

**识别标准：** ⚙ 函数超过 `function.max_lines`（默认 80）行逻辑代码（不含注释和空行）。

**修复策略：**
- 提取子函数（Extract Method）：将独立逻辑块抽为命名函数
- 用生成器替代累积变量：`yield` 替代 `result.append()`
- 用早返回（Guard Clause）减少嵌套

```python
# Bad
def process_order(order):
    if order is not None:
        if order.status == "pending":
            # ... 80 行处理逻辑
            pass

# Good
def process_order(order):
    if order is None:
        return
    if order.status != "pending":
        return
    _validate_order(order)
    _calculate_total(order)
    _apply_discounts(order)
```

### 1.2 过长参数列表（Long Parameter List）

**识别标准：** ⚙ 参数超过 `function.max_parameters`（默认 4）个。

**修复策略：**
- 引入参数对象（dataclass / TypedDict / NamedTuple）
- 使用 `**kwargs` 配合明确的类型注解
- 复杂配置场景参考 → [design-patterns.md](design-patterns.md) 建造者模式

```python
# Bad
def create_user(name, email, age, address, phone, role, department):
    ...

# Good
@dataclass
class UserProfile:
    name: str
    email: str
    age: int
    address: str
    phone: str
    role: str
    department: str

def create_user(profile: UserProfile):
    ...
```

### 1.3 重复代码（Duplicated Code）

**识别标准：** ⚙ `duplication.min_duplicate_lines`（默认 3）行以上相似逻辑出现 `duplication.min_duplicate_occurrences`（默认 2）次及以上。

**修复策略：**
- 提取公共函数
- 横切关注点（日志、重试、权限）→ [design-patterns.md](design-patterns.md) 装饰器模式
- 继承场景下的流程复用 → [design-patterns.md](design-patterns.md) 模板方法模式

---

## 2. 类级坏味道

### 2.1 上帝类（God Class）

**识别标准：** ⚙ 类超过 `class.max_lines`（默认 300）行，或承担 `class.max_responsibilities`（默认 3）个以上不相关职责。

**修复策略：**
- 按职责拆分为多个类
- 使用组合（Composition）替代继承
- 引入 Mixin 分离可复用行为

### 2.2 数据类滥用（Data Class Smell）

**识别标准：** 类只有属性和 getter/setter，没有行为方法。

**修复策略：**
- 将操作数据的外部函数移入类中（Move Method）
- 使用 `dataclass` 或 `NamedTuple` 明确意图
- 如果确实只是数据载体，用 `dataclass(frozen=True)` 标记不可变

### 2.3 过度继承（Deep Inheritance）

**识别标准：** ⚙ 继承层级超过 `class.max_inheritance_depth`（默认 3）层。

**修复策略：**
- 优先使用组合替代继承
- 使用 Protocol（结构化子类型）替代 ABC
- 扁平化继承树，合并中间层

```python
# Bad: 深层继承
class Animal: ...
class Mammal(Animal): ...
class DomesticMammal(Mammal): ...
class Dog(DomesticMammal): ...

# Good: 组合 + Protocol
class Domesticable(Protocol):
    def interact_with_human(self) -> None: ...

@dataclass
class Dog:
    movement: WalkBehavior
    sound: BarkBehavior

    def interact_with_human(self) -> None:
        ...
```

---

## 3. 逻辑级坏味道

### 3.1 布尔参数（Boolean Parameter / Flag Argument）

**修复：** 拆分为两个语义明确的函数。

```python
# Bad
def get_users(include_inactive=False): ...

# Good
def get_active_users(): ...
def get_all_users(): ...
```

### 3.2 嵌套条件地狱（Nested Conditionals）

**修复策略：**
- Guard Clause 提前返回
- 用字典映射替代 if-elif 链
- 用多态替代类型判断

```python
# Bad
def calculate_price(product_type, quantity):
    if product_type == "book":
        if quantity > 10:
            return quantity * 8
        else:
            return quantity * 10
    elif product_type == "electronics":
        ...

# Good: 字典映射
PRICING = {
    "book": lambda q: q * (8 if q > 10 else 10),
    "electronics": lambda q: q * (45 if q > 5 else 50),
}

def calculate_price(product_type: str, quantity: int) -> float:
    calculator = PRICING.get(product_type)
    if calculator is None:
        raise ValueError(f"Unknown product type: {product_type}")
    return calculator(quantity)
```

### 3.3 魔法数字/字符串（Magic Numbers/Strings）

**修复：** 提取为命名常量或枚举。

```python
# Bad
if user.role == 3:
    ...
if status == "act":
    ...

# Good
class Role(IntEnum):
    ADMIN = 3

class Status(str, Enum):
    ACTIVE = "act"
```

### 3.4 类型分发滥用（isinstance Chain）

**识别标准：** ⚙ `logic.max_isinstance_chain`（默认 3）个以上 `isinstance` / `type()` 判断组成的 if-elif 链。

**修复策略：**
- 用多态替代：让每个类型实现自己的处理方法
- 用 `functools.singledispatch` 做函数级分发
- 用字典映射 `{type: handler}` 做数据驱动分发

```python
# Bad: isinstance 链
def process(obj):
    if isinstance(obj, Image):
        return resize(obj)
    elif isinstance(obj, Video):
        return transcode(obj)
    elif isinstance(obj, Audio):
        return normalize(obj)

# Good: singledispatch
from functools import singledispatch

@singledispatch
def process(obj):
    raise TypeError(f"Unsupported type: {type(obj)}")

@process.register
def _(obj: Image):
    return resize(obj)

@process.register
def _(obj: Video):
    return transcode(obj)

@process.register
def _(obj: Audio):
    return normalize(obj)
```

更多分发模式参考 → [design-patterns.md](design-patterns.md) 策略模式

### 3.5 可变默认参数（Mutable Default Argument）

**识别标准：** 函数参数默认值为 `[]`、`{}`、`set()` 等可变对象。

**修复：** 使用 `None` 作为哨兵值。

```python
# Bad
def add_item(item, items=[]):
    items.append(item)
    return items

# Good
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```

---

## 4. 模块级坏味道

### 4.1 循环导入（Circular Import）

**修复策略：**
- 延迟导入（在函数内部 import）
- 提取公共模块打破循环
- 使用 `TYPE_CHECKING` 守卫仅在类型检查时导入

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .other_module import OtherClass
```

### 4.2 通配符导入（Wildcard Import）

**修复：** 显式导入所需名称，或定义 `__all__`。

### 4.3 过大模块（Large Module）

**识别标准：** ⚙ 单文件超过 `module.max_lines`（默认 500）行。

**修复：** 按职责拆分为子模块（package），保持 `__init__.py` 的公共 API 不变。

---

## 执行流程

1. 通读目标代码，按上述分类逐项检查
2. 按严重程度排序：逻辑错误 > 可维护性 > 可读性 > 风格
3. 对每个坏味道给出：位置、问题描述、修复方案、修复后代码
4. 修复时确保不改变外部行为（保持接口兼容）
5. 修复后运行已有测试验证无回归
