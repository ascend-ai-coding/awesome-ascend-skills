# 内存操作 API - tl.load / tl.store 及 Ascend 扩展参数

## 概述

本文档详细描述 Triton-Ascend 中的内存操作 API，包括 `tl.load`、`tl.store`、`tl.make_block_ptr` 和 `tl.advance`。这些操作是 Triton kernel 与 Global Memory 交互的核心接口，负责从 NPU 全局内存读取数据到 Unified Buffer（UB）以及将计算结果从 UB 写回全局内存。在 Ascend NPU 上，内存操作的语义与 GPU 有显著差异，包括 `care_padding` 等 Ascend 扩展参数、离散访存与连续访存的性能差异、以及 mask 处理方式的不同。

关键词：tl.load, tl.store, make_block_ptr, advance, mask, other, care_padding, boundary_check, padding_option, 离散访存, 连续访存, NPU, Ascend, 内存操作

---

## API 参考

### tl.load

从 Global Memory 中 `pointer` 指向的位置加载数据到 Unified Buffer。

```python
triton.language.load(
    pointer,
    mask=None,
    other=None,
    boundary_check=(),
    padding_option="",
    cache_modifier="",
    eviction_policy="",
    volatile=False,
    care_padding=True,
    _builder=None
) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `pointer` | `PointerType` 或 `tensor<PointerType>` 或 `PointerType<tensor>`（来自 `make_block_ptr`） | 指向 GM 上待读取数据的指针 |
| `mask` | `int1` 或 `tensor<int1>`，可选 | 当 `mask[idx]` 为 False 时，不读取 `pointer[idx]` 处的数据。仅当 `pointer` 不来自 `make_block_ptr` 时可传入 |
| `other` | `tensor` 或 `scalar`，可选 | 当 `mask[idx]` 为 False 时，返回值的第 `i` 个位置设置为 `other[i]`。仅在 `mask != None` 时可传入 |
| `boundary_check` | `tuple(int)`，可选 | 整数元组，指示需要做边界检查的维度。仅当 `pointer` 来自 `make_block_ptr` 时可传入 |
| `padding_option` | `""` / `"zero"` / `"nan"` | 越界时填充的值。`""` 表示未定义值，`"zero"` 填充零，`"nan"` 填充 NaN（仅浮点类型）。仅当 `boundary_check` 不为空时可传入 |
| `cache_modifier` | `""` / `".ca"` / `".cg"` | 控制 cache 选项，**对 Ascend 硬件无效** |
| `eviction_policy` | `""` / `"evict_first"` / `"evict_last"` | 控制 eviction 策略，**对 Ascend 硬件无效** |
| `volatile` | `bool` | 控制 volatile 选项，**对 Ascend 硬件无效** |
| `care_padding` | `bool`，默认 `True` | **Ascend 扩展参数**。控制 mask 为 False 时的填充行为：(1) 若 `other` 不为 None，`care_padding` 不生效；(2) 若 `other` 为 None 且 `care_padding=True`，masked 位置填充零；(3) 若 `other` 为 None 且 `care_padding=False`，masked 位置为随机值，但 `tl.load` 可能有更好的性能 |

**返回值**：`tl.tensor`，加载的数据张量

**约束**：
- 若 `pointer` 是单指针：`mask` 和 `other` 必须是标量，不允许 `boundary_check` 和 `padding_option`
- 若 `pointer` 是 N 维指针张量：`mask` 和 `other` 会隐式广播到 `pointer.shape`，不允许 `boundary_check` 和 `padding_option`
- 若 `pointer` 来自 `make_block_ptr`：`mask` 和 `other` 必须为 None，可指定 `boundary_check` 和 `padding_option`
- `mask` 为 None 而 `other` 不为 None 是非法的

**数据类型支持（Ascend）**：

| | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | fp16 | fp32 | fp64 | bf16 | bool |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ascend | √ | √ | √ | √ | × | × | × | × | √ | √ | × | √ | √ |

### tl.store

将数据从 Unified Buffer 存储到 Global Memory。

```python
triton.language.store(
    pointer,
    value,
    mask=None,
    boundary_check=(),
    cache_modifier="",
    eviction_policy="",
    _builder=None
) -> None
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `pointer` | `PointerType` 或 `tensor<PointerType>` 或 `PointerType<tensor>`（来自 `make_block_ptr`） | 指向 GM 上待存储地址的指针 |
| `value` | `tensor` 或 `scalar` | 要存储的值，支持隐式广播和隐式类型转换 |
| `mask` | `int1` 或 `tensor<int1>`，可选 | 当 `mask[idx]` 为 False 时，不存储 `value[idx]`。仅当 `pointer` 不来自 `make_block_ptr` 时可传入 |
| `boundary_check` | `tuple(int)`，可选 | 整数元组，指示需要做边界检查的维度。仅当 `pointer` 来自 `make_block_ptr` 时可传入 |
| `cache_modifier` | `""` / `".wb"` / `".cg"` / `".cs"` / `".wt"` | 控制 cache 选项，**对 Ascend 硬件无效** |
| `eviction_policy` | `""` / `"evict_first"` / `"evict_last"` | 控制 eviction 策略，**对 Ascend 硬件无效** |

**返回值**：无

**约束**：
- 不允许向 `const` 指针 store
- 若 `pointer` 是单指针：`value` 和 `mask` 必须是标量，不允许 `boundary_check`
- 若 `pointer` 来自 `make_block_ptr`：`mask` 必须为 None，可指定 `boundary_check`
- `value` 会隐式广播到 `pointer.shape` 并类型转换为 `pointer.dtype.element_ty`

**数据类型支持（Ascend）**：与 `tl.load` 相同

### tl.make_block_ptr

创建一个指向父张量中某个块的指针。

```python
triton.language.make_block_ptr(
    base,
    shape,
    strides,
    offsets,
    block_shape,
    order,
    _builder=None
) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `base` | `tensor` | 父张量的基地址指针 |
| `shape` | `tuple(int)` | 父张量的形状 |
| `strides` | `tuple(int)` | 父张量的步幅 |
| `offsets` | `tuple(int)` | 块的偏移量 |
| `block_shape` | `tuple(int)` | 块的形状 |
| `order` | `tuple(int)` | 原始数据格式的顺序 |

### tl.advance

推进块指针的偏移量。

```python
triton.language.advance(
    base,
    offsets,
    _builder=None
) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `base` | `tensor` | 要推进的块指针 |
| `offsets` | `tuple` | 按维度推进的偏移量 |

---

## 代码示例

### 基础用法：指针张量加载与存储

```python
import triton
import triton.language as tl

@triton.jit
def load_store_kernel(
    x_ptr, y_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=mask)
```

### 进阶用法：care_padding 参数与 block pointer

```python
@triton.jit
def load_with_care_padding_kernel(
    x_ptr, y_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, care_padding=False)
    y = x + 1.0
    tl.store(y_ptr + offsets, y, mask=mask)

@triton.jit
def block_ptr_kernel(
    x_ptr, y_ptr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    x_block = tl.make_block_ptr(
        base=x_ptr,
        shape=(N, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    x = tl.load(x_block, boundary_check=(0, 1), padding_option="zero")
    y = x + 1.0
    y_block = tl.make_block_ptr(
        base=y_ptr,
        shape=(N, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(y_block, y, boundary_check=(0, 1))
    x_block = tl.advance(x_block, (BLOCK_M, 0))
```

---

## NPU 适配要点

### 1. care_padding 扩展参数

`care_padding` 是 Triton-Ascend 特有的扩展参数，用于控制 mask 为 False 时的填充行为。在 GPU 上，masked load 的越界位置值是未定义的；在 NPU 上，通过 `care_padding` 参数可以明确控制：

- `care_padding=True`（默认）：masked 位置填充零值（浮点类型填 0.0，整数类型填 0）
- `care_padding=False`：masked 位置为随机值，但可能获得更好的性能

当 `other` 参数已提供时，`care_padding` 不生效，因为 `other` 已经指定了填充值。

### 2. mask 和 other 在 NPU 上的处理方式

在 NPU 上，当 `mask` 不为 None 而 `other` 为 None 时：
- 若 `care_padding=True`，系统自动根据元素类型生成默认填充值（浮点填 0.0，整数填 0）
- 离散 mask（非连续的 True/False 模式）在 store 操作中会被拆解为 atomic {load, select, store} 组合，可能存在泛化性问题

### 3. 离散访存 vs 连续访存的性能差异

NPU 的 Vector 计算单元对连续内存访问有更好的优化。离散访存（如 gather/scatter 模式）会导致：
- 更多的内存事务
- 更高的延迟
- 可能触发 atomic 操作拆解（store 场景）

建议优先使用连续访存模式，即指针偏移量是连续递增的。使用 `make_block_ptr` + `boundary_check` 的方式通常比指针张量 + mask 的方式更高效。

### 4. cache_modifier / eviction_policy / volatile 参数

这些参数在 Ascend NPU 上均无效，因为 NPU 的内存层次结构与 GPU 不同，没有对应的 L1/L2 cache 控制语义。传入这些参数不会报错，但不会产生任何效果。

### 5. 数据类型限制

Ascend NPU 不支持 uint8、uint16、uint32、uint64、fp64 类型的 load/store 操作，这是硬件限制。如需处理这些类型，需要先转换为支持的类型。

### 6. padding_option 的支持状态

当前 `padding_option` 参数在部分场景下可能不完全支持。建议使用 `mask` + `other` 的方式替代。

---

## 常见问题

**Q1: care_padding=False 时 masked 位置的值是什么？**

A: 是未定义的随机值。仅在后续计算中不依赖这些位置的值时（例如 mask 后续会过滤掉这些位置），才应使用 `care_padding=False` 以获得更好的性能。

**Q2: 为什么在 NPU 上离散 mask 的 store 会有问题？**

A: NPU 上离散 mask 的 store 会被拆解为 atomic {load, select, store} 组合，在 corner case 中可能存在精度或正确性问题。建议尽量使用连续 mask 或 block pointer 方式。

**Q3: make_block_ptr 在 NPU 上的行为与 GPU 有何不同？**

A: `make_block_ptr` 的语义在 NPU 和 GPU 上基本一致，但底层数据布局可能不同。NPU 上可能需要 ND2NZ 等数据格式转换，这由编译器自动处理，但可能影响性能。

**Q4: 能否在 NPU 上使用 cache_modifier 优化访存？**

A: 不能。NPU 没有 GPU 的 L1/L2 cache 层次结构，`cache_modifier`、`eviction_policy` 和 `volatile` 参数在 NPU 上均无效。

**Q5: 如何处理不支持的数据类型（如 uint64）？**

A: 需要在 host 端将数据转换为支持的类型（如 int64），或在 kernel 中使用 `tl.cast` 进行类型转换。

---

## 相关文档

- [02-math-ops.md](./02-math-ops.md) - 数学运算 API
- [04-linear-algebra-ops.md](./04-linear-algebra-ops.md) - tl.dot 及 Cube 管线
- [05-atomic-ops.md](./05-atomic-ops.md) - 原子操作

## 源码参考

- [core.py - load 函数定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1589-L1652)
- [core.py - store 函数定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1678-L1726)
- [core.py - make_block_ptr 函数定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1729-L1741)
- [core.py - advance 函数定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1744-L1753)
- [semantic.py - load 实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L1100-L1213)
- [semantic.py - store 实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L1253-L1340)
