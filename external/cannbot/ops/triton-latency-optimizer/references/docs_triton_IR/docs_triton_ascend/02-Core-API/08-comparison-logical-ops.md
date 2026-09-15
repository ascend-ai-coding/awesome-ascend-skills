# 比较与逻辑操作 API

## 概述

本文档详细描述 Triton-Ascend 中的比较操作、逻辑操作、移位操作和 `where` 操作 API。比较操作包括 `eq`/`ne`/`lt`/`le`/`gt`/`ge`，逻辑操作包括 `and`/`or`/`xor`/`not`/`logical_and`/`logical_or`/`invert`/`neg`，移位操作包括 `lshift`/`rshift`，条件选择操作为 `where`。这些操作是构建条件逻辑和控制流的基础，在 Ascend NPU 上由 Vector 计算单元执行，行为与 GPU 基本一致，但 bfloat16 的比较操作需要先提升为 float32。

关键词：比较, 逻辑, 移位, where, eq, ne, lt, le, gt, ge, and, or, xor, not, logical_and, logical_or, invert, neg, lshift, rshift, NPU, bfloat16

---

## API 参考

### 比较操作

所有比较操作返回 `int1`（布尔）类型的张量，形状与输入广播后的形状相同。

#### tl.eq / `==`

```python
triton.language.eq(x, y) -> tensor  # 或 x == y
```

逐元素相等比较。浮点使用 `fcmpOEQ`（有序相等），整数使用 `icmpEQ`。

#### tl.ne / `!=`

```python
triton.language.ne(x, y) -> tensor  # 或 x != y
```

逐元素不等比较。浮点使用 `fcmpUNE`（无序不等），整数使用 `icmpNE`。

#### tl.lt / `<`

```python
triton.language.lt(x, y) -> tensor  # 或 x < y
```

逐元素小于比较。浮点使用 `fcmpOLT`，有符号整数使用 `icmpSLT`，无符号整数使用 `icmpULT`。

#### tl.le / `<=`

```python
triton.language.le(x, y) -> tensor  # 或 x <= y
```

逐元素小于等于比较。浮点使用 `fcmpOLE`，有符号整数使用 `icmpSLE`，无符号整数使用 `icmpULE`。

#### tl.gt / `>`

```python
triton.language.gt(x, y) -> tensor  # 或 x > y
```

逐元素大于比较。浮点使用 `fcmpOGT`，有符号整数使用 `icmpSGT`，无符号整数使用 `icmpUGT`。

#### tl.ge / `>=`

```python
triton.language.ge(x, y) -> tensor  # 或 x >= y
```

逐元素大于等于比较。浮点使用 `fcmpOGE`，有符号整数使用 `icmpSGE`，无符号整数使用 `icmpUGE`。

**比较操作汇总**：

| 操作 | 浮点 IR | 有符号整数 IR | 无符号整数 IR |
|------|---------|-------------|-------------|
| `==` | `fcmpOEQ` | `icmpEQ` | `icmpEQ` |
| `!=` | `fcmpUNE` | `icmpNE` | `icmpNE` |
| `<` | `fcmpOLT` | `icmpSLT` | `icmpULT` |
| `<=` | `fcmpOLE` | `icmpSLE` | `icmpULE` |
| `>` | `fcmpOGT` | `icmpSGT` | `icmpUGT` |
| `>=` | `fcmpOGE` | `icmpSGE` | `icmpUGE` |

**注意**：浮点比较使用**有序**（ordered）比较，即 NaN 与任何值的比较结果均为 False。

### 逻辑操作

#### 按位操作（整数专用）

| 操作 | 运算符 | 函数 | IR 指令 | 说明 |
|------|--------|------|---------|------|
| 按位与 | `x & y` | `and_(x, y)` | `create_and` | 两个操作数必须为整数类型 |
| 按位或 | `x \| y` | `or_(x, y)` | `create_or` | 两个操作数必须为整数类型 |
| 按位异或 | `x ^ y` | `xor_(x, y)` | `create_xor` | 两个操作数必须为整数类型 |
| 按位取反 | `~x` | `invert(x)` | `xor(x, all_ones)` | 操作数必须为整数类型，等价于与全 1 异或 |

**类型提升**：按位操作的两个操作数会先进行整数类型提升（`integer_promote_impl`），统一到更宽的类型。

#### 逻辑操作（布尔语义）

| 操作 | 运算符 | 函数 | 说明 |
|------|--------|------|------|
| 逻辑与 | `x and y` | `logical_and(x, y)` | 返回 `int1` 类型。非布尔输入先转换为布尔（`!= 0`） |
| 逻辑或 | `x or y` | `logical_or(x, y)` | 返回 `int1` 类型。非布尔输入先转换为布尔（`!= 0`） |
| 逻辑非 | `not x` | `not_(x)` | 返回 `int1` 类型。浮点类型不支持 |

**逻辑与 vs 按位与的区别**：
- `logical_and`：返回布尔类型，非布尔输入会先转换为布尔
- `&`（按位与）：返回整数类型，对每一位执行与操作

```python
# logical_and: 非零为 True
a = tl.full([4], 3, tl.int32)  # [3, 3, 3, 3]
b = tl.full([4], 5, tl.int32)  # [5, 5, 5, 5]
c = tl.logical_and(a, b)        # [True, True, True, True] (int1)

# 按位与: 逐位操作
d = a & b                       # [1, 1, 1, 1] (int32, 3 & 5 = 1)
```

#### 取负操作

| 操作 | 运算符 | 函数 | 说明 |
|------|--------|------|------|
| 算术取负 | `-x` | `minus(x)` | 等价于 `0 - x`。不支持指针和布尔类型 |
| 按位取反 | `~x` | `invert(x)` | 等价于 `x ^ all_ones`。不支持浮点和指针类型 |

### 移位操作

| 操作 | 运算符 | 函数 | IR 指令 | 说明 |
|------|--------|------|---------|------|
| 左移 | `x << y` | `shl(x, y)` | `create_shl` | 两个操作数必须为整数类型 |
| 逻辑右移 | `x >> y`（无符号） | `lshr(x, y)` | `create_lshr` | 无符号整数右移，高位填 0 |
| 算术右移 | `x >> y`（有符号） | `ashr(x, y)` | `create_ashr` | 有符号整数右移，高位填符号位 |

**注意**：
- 移位量超过操作数位宽时行为未定义（会触发警告）
- 有符号整数使用算术右移（`ashr`），无符号整数使用逻辑右移（`lshr`）
- 运算符 `>>` 根据左操作数的符号性自动选择 `ashr` 或 `lshr`

### tl.where

条件选择操作。

```python
triton.language.where(condition, x, y, _builder=None) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `condition` | `tensor<int1>` | 条件张量。为 True 时选择 x，为 False 时选择 y |
| `x` | `tensor` 或 `scalar` | 条件为 True 时选择的值 |
| `y` | `tensor` 或 `scalar` | 条件为 False 时选择的值 |

**约束**：
- `x` 和 `y` 必须具有相同的数据类型
- `x`、`y` 和 `condition` 的形状会自动广播
- `x` 和 `y` 总是会被求值（不是短路求值）

**重要**：`where` 不是短路求值。如果需要避免不必要的内存操作，应使用 `tl.load`/`tl.store` 的 `mask` 参数。

---

## 代码示例

### 基础用法：比较与 where

```python
import triton
import triton.language as tl

@triton.jit
def compare_where_kernel(
    x_ptr, y_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    cond = x > y
    result = tl.where(cond, x, y)

    tl.store(out_ptr + offsets, result, mask=mask)
```

### 进阶用法：逻辑组合与掩码构建

```python
@triton.jit
def masked_relu_kernel(
    x_ptr, mask_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    boundary_mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=boundary_mask, other=0.0)
    custom_mask = tl.load(mask_ptr + offsets, mask=boundary_mask, other=0)

    positive = x > 0
    active = positive & (custom_mask != 0)
    result = tl.where(active, x, 0.0)

    tl.store(out_ptr + offsets, result, mask=boundary_mask)

@triton.jit
def clamp_with_where_kernel(
    x_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    result = tl.where(x < min_val, min_val, tl.where(x > max_val, max_val, x))

    tl.store(out_ptr + offsets, result, mask=mask)
```

---

## NPU 适配要点

### 1. bfloat16 比较操作的精度提升

NPU 的 Vector 计算单元**不支持** bfloat16 的浮点比较指令（`fcmpOGT` 等）。因此，当 bfloat16 张量参与比较操作时，会自动提升为 float32 执行。这由 `_promote_bfloat16_to_float32` 函数处理，影响 `maximum`、`minimum` 以及间接影响 `max`/`min` 归约操作。

### 2. 浮点比较的有序性

Triton 的浮点比较使用**有序比较**（ordered comparison）：
- `NaN == anything` → False
- `NaN != anything` → True
- `NaN < anything` → False
- `NaN > anything` → False

这与 Python 的 NaN 行为一致。如果需要检测 NaN，使用 `x != x`（NaN 不等于自身）。

### 3. 逻辑与 vs 按位与的语义差异

在 Triton 中，`logical_and` 和 `&` 有不同的语义：

| 特性 | `logical_and` | `&` (按位与) |
|------|-------------|------------|
| 输入类型 | 任意数值（非布尔先转布尔） | 仅整数 |
| 输出类型 | `int1`（布尔） | 提升后的整数类型 |
| 语义 | 布尔逻辑与 | 逐位与 |

**常见陷阱**：对 `int1` 张量使用 `&` 和 `logical_and` 结果相同，但对 `int32` 张量结果不同。

### 4. 移位操作的位宽检查

移位操作会检查移位量是否超过操作数的位宽：

```python
def check_bit_width(value, shift_value):
    if isinstance(value, tensor) and isinstance(shift_value, constexpr):
        bitwidth = value.type.scalar.primitive_bitwidth
        if shift_value.value >= bitwidth:
            warn(f"Value {shift_value.value} exceeds the maximum bitwidth ({bitwidth})...")
```

超过位宽的移位量会产生未定义行为，Triton 会发出警告但不会阻止编译。

### 5. where 操作的广播行为

`where(condition, x, y)` 中三个参数的形状会自动广播：
- `condition` 的形状决定输出形状
- `x` 和 `y` 广播到 `condition` 的形状
- `x` 和 `y` 必须具有相同的 dtype

### 6. 布尔类型的特殊处理

Triton 中 `int1`（布尔）类型在部分操作中有特殊处理：
- `tl.load` 加载布尔指针时，实际加载为 `int8`，并标记 `was_bool_to_int8`
- `logical_and`/`logical_or` 检测 `was_bool_to_int8` 标记，先将 `int8` 转回 `int1`
- `invert` 同样检测 `was_bool_to_int8` 标记

---

## 常见问题

**Q1: 为什么 bf16 的比较结果和 fp32 不完全一致？**

A: bf16 比较会先提升为 fp32，比较结果本身是精确的。但如果 bf16 值在提升为 fp32 时有精度损失（bf16 只有 7 位尾数），可能导致比较结果与直接用 fp32 计算的结果不同。

**Q2: 如何判断一个值是否为 NaN？**

A: 使用 `x != x`。因为 NaN 不等于任何值（包括自身），所以 `x != x` 为 True 当且仅当 x 是 NaN。

**Q3: where 和 mask + load 有什么区别？**

A: `where` 总是会对 x 和 y 都求值，而 `mask + load` 不会读取 mask 为 False 的内存位置。如果 x 或 y 的计算涉及内存访问且可能越界，应使用 mask 而非 where。

**Q4: 有符号和无符号右移有什么区别？**

A: 有符号右移（`ashr`）保留符号位，高位填符号位；无符号右移（`lshr`）高位填 0。Triton 根据左操作数的类型自动选择：有符号整数用 `ashr`，无符号整数用 `lshr`。

**Q5: 可以对浮点数使用按位操作吗？**

A: 不可以。按位与/或/异或/取反仅支持整数类型。如需对浮点数的位模式进行操作，需要先 `bitcast` 为整数类型。

---

## 相关文档

- [02-math-ops.md](./02-math-ops.md) - 数学运算 API
- [03-reduction-ops.md](./03-reduction-ops.md) - 归约操作
- [05-atomic-ops.md](./05-atomic-ops.md) - 原子操作

## 源码参考

- [semantic.py - 比较操作实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L552-L643)
- [semantic.py - 按位操作实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L412-L516)
- [semantic.py - 逻辑操作实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L443-L501)
- [semantic.py - 移位操作实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L504-L516)
- [semantic.py - where 实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L1738)
- [core.py - where 函数定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1886-L1905)
- [core.py - tensor 运算符重载](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L888-L951)
