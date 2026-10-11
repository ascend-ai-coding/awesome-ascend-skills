# 原子操作 API 及 NPU 限制

## 概述

本文档详细描述 Triton-Ascend 中的原子操作 API，包括 `atomic_add`、`atomic_and`、`atomic_or`、`atomic_xor`、`atomic_xchg`、`atomic_cas`、`atomic_max` 和 `atomic_min`。原子操作保证在多线程并发访问同一内存地址时的数据一致性，是并行计算中实现同步和数据聚合的关键机制。在 Ascend NPU 上，原子操作的支持范围、数据类型限制和内存序保证与 GPU 有显著差异，需要特别注意。

关键词：原子操作, atomic_add, atomic_and, atomic_or, atomic_xor, atomic_xchg, atomic_cas, atomic_max, atomic_min, NPU, 内存序, sem, scope, 数据竞争

---

## API 参考

### 通用参数说明

所有原子操作共享以下参数模式：

```python
tl.atomic_xxx(pointer, val, mask=None, sem=None, scope=None)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `pointer` | `PointerDType` | 要操作的内存位置 |
| `val` | `pointer.dtype.element_ty` | 操作值（右操作数） |
| `mask` | `int1` 或 `tensor<int1>`，可选 | 指定操作范围，防止越界访问 |
| `sem` | `str`，可选 | 内存语义。可选值：`"acquire"` / `"release"` / `"acq_rel"`（默认）/ `"relaxed"` |
| `scope` | `str`，可选 | 同步范围。可选值：`"gpu"`（默认）/ `"cta"` / `"sys"` |

**返回值**：操作前 `pointer` 指向的旧值

### tl.atomic_add

原子加法操作：`*pointer = *pointer + val`

```python
triton.language.atomic_add(pointer, val, mask=None, sem=None, scope=None) -> tensor
```

**数据类型支持（Ascend）**：

| | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | fp16 | fp32 | fp64 | bf16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPU | × | × | √ | √ | × | × | × | √ | √ | √ | √ |
| Ascend | √ | √ | √ | × | √ | √ | √ | √ | √ | × | √ |

**NPU 注意**：浮点类型使用 `FADD` 操作码，整数类型使用 `ADD` 操作码。

### tl.atomic_max

原子最大值操作：`*pointer = max(*pointer, val)`

```python
triton.language.atomic_max(pointer, val, mask=None, sem=None, scope=None) -> tensor
```

**NPU 特殊行为**：
- 有符号整数：使用 `MAX` 操作码
- 无符号整数：使用 `UMAX` 操作码
- 浮点数：**NPU 直接使用 `MAX` 操作码**（与 GPU 不同，GPU 需要通过 bitcast 实现浮点 atomic_max）

### tl.atomic_min

原子最小值操作：`*pointer = min(*pointer, val)`

```python
triton.language.atomic_min(pointer, val, mask=None, scope=None) -> tensor
```

**NPU 特殊行为**：与 `atomic_max` 类似，NPU 直接使用硬件 `MIN`/`UMIN` 操作码支持浮点 atomic_min，无需 bitcast。

### tl.atomic_and

原子按位与操作：`*pointer = *pointer & val`

```python
triton.language.atomic_and(pointer, val, mask=None, sem=None, scope=None) -> tensor
```

仅支持整数类型。

### tl.atomic_or

原子按位或操作：`*pointer = *pointer | val`

```python
triton.language.atomic_or(pointer, val, mask=None, sem=None, scope=None) -> tensor
```

仅支持整数类型。

### tl.atomic_xor

原子按位异或操作：`*pointer = *pointer ^ val`

```python
triton.language.atomic_xor(pointer, val, mask=None, sem=None, scope=None) -> tensor
```

仅支持整数类型。

### tl.atomic_xchg

原子交换操作：`*pointer = val`，返回旧值

```python
triton.language.atomic_xchg(pointer, val, mask=None, sem=None, scope=None) -> tensor
```

### tl.atomic_cas

原子比较并交换操作（Compare-And-Swap）：

```
old = *pointer
if old == cmp:
    *pointer = val
return old
```

```python
triton.language.atomic_cas(pointer, cmp, val, sem=None, scope=None) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `pointer` | `PointerDType` | 要操作的内存位置 |
| `cmp` | `pointer.dtype.element_ty` | 期望的旧值 |
| `val` | `pointer.dtype.element_ty` | 要写入的新值 |

**数据类型支持（Ascend）**：

| 平台 | 支持的数据类型 |
|------|-------------|
| 910B | int8, uint8, int16, int32, int64, fp16, bf16, fp32 |
| 910-95 | int8, uint8, int16, uint16, int32, uint32, int64, uint64, fp8e4m3, fp8e5m2, fp16, bf16, fp32 |

---

## 代码示例

### 基础用法：原子加法实现直方图

```python
import triton
import triton.language as tl

@triton.jit
def atomic_histogram_kernel(
    x_ptr, hist_ptr,
    N: tl.constexpr,
    NUM_BINS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    x = x % NUM_BINS
    ones = tl.where(mask, 1, 0)
    tl.atomic_add(hist_ptr + x, ones, mask=mask)
```

### 进阶用法：atomic_cas 实现自旋锁

```python
@triton.jit
def atomic_cas_kernel(
    lock_ptr, data_ptr, result_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    old = tl.atomic_cas(lock_ptr, 0, 1)
    if old == 0:
        data = tl.load(data_ptr + offsets, mask=mask, other=0.0)
        result = data * 2.0
        tl.store(result_ptr + offsets, result, mask=mask)
        tl.atomic_xchg(lock_ptr, 0)
```

---

## NPU 适配要点

### 1. 原子操作的支持状态和限制

| 操作 | NPU 支持状态 | 限制 |
|------|-------------|------|
| `atomic_add` | 支持 | 不支持 int64/fp64 |
| `atomic_max` | 支持（含浮点） | NPU 直接硬件支持浮点 max，无需 bitcast |
| `atomic_min` | 支持（含浮点） | NPU 直接硬件支持浮点 min，无需 bitcast |
| `atomic_and` | 支持 | 仅整数 |
| `atomic_or` | 支持 | 仅整数 |
| `atomic_xor` | 支持 | 仅整数 |
| `atomic_xchg` | 支持 | - |
| `atomic_cas` | 支持 | 910B 和 910-95 支持的数据类型不同 |

### 2. 内存序保证

NPU 上的原子操作内存序支持有限：

| sem 值 | GPU | NPU |
|--------|-----|-----|
| `"acq_rel"` | 支持 | **支持（唯一支持）** |
| `"acquire"` | 支持 | 不支持 |
| `"release"` | 支持 | 不支持 |
| `"relaxed"` | 支持 | 不支持 |

| scope 值 | GPU | NPU |
|----------|-----|-----|
| `"gpu"` | 支持 | **支持（唯一支持）** |
| `"cta"` | 支持 | 不支持 |
| `"sys"` | 支持 | 不支持 |

**重要**：NPU 上 `sem` 和 `scope` 参数仅支持默认值（`"acq_rel"` 和 `"gpu"`），传入其他值可能导致未定义行为。

### 3. atomic_max/atomic_min 的浮点支持

在 GPU 上，浮点类型的 `atomic_max`/`atomic_min` 需要通过 bitcast 将浮点值转为整数，然后利用整数比较的语义实现（正数用有符号比较，负数用无符号比较）。这是一个复杂的软件实现。

在 NPU 上，**硬件直接支持浮点 atomic_max/atomic_min**，无需 bitcast，实现更简洁高效：

```python
# NPU 上的 atomic_max 实现（简化）
def atomic_max(ptr, val, mask, sem, scope, builder):
    if sca_ty.is_int():
        if sca_ty.is_int_signed():
            return builder.create_atomic_rmw(ATOMIC_OP.MAX, ...)
        else:
            return builder.create_atomic_rmw(ATOMIC_OP.UMAX, ...)
    # Design for NPU: 直接使用 MAX 操作码
    return builder.create_atomic_rmw(ATOMIC_OP.MAX, ...)
```

### 4. mask 参数的隐式处理

当 `mask=None` 时，原子操作会自动创建全 True 的 mask：

```python
if not mask:
    mask_ir = builder.get_int1(True)
    if ptr.type.is_block():
        mask_ir = builder.create_splat(mask_ir, ptr.type.get_block_shapes())
    mask = tl.tensor(mask_ir, tl.int1)
```

### 5. 数据类型差异总结

与 GPU 相比，NPU 上的原子操作：
- **额外支持**：int8, int16, uint8, uint16, uint32 的 atomic_add
- **不支持**：int64, fp64 的 atomic_add
- **atomic_cas**：910B 和 910-95 支持的类型列表不同

---

## 常见问题

**Q1: NPU 上可以使用 relaxed 内存序吗？**

A: 不可以。NPU 仅支持 `"acq_rel"` 内存序和 `"gpu"` 同步范围。传入其他值可能导致未定义行为。

**Q2: 为什么 NPU 不支持 int64 的 atomic_add？**

A: 这是 NPU 硬件限制。NPU 的原子操作单元不支持 64 位整数的原子加法。如需 64 位原子操作，可以考虑使用 `atomic_cas` 实现自旋锁保护的读写。

**Q3: atomic_max 对 NaN 的行为是什么？**

A: NPU 硬件直接执行浮点 max 操作，NaN 的处理遵循 IEEE 754 标准：`max(NaN, x)` 和 `max(x, NaN)` 的结果可能是 NaN 或 x，取决于具体实现。

**Q4: 多个 program 同时对同一地址执行 atomic_add 是否安全？**

A: 是的，原子操作保证对同一地址的操作是串行化的，不会出现数据竞争。但注意 NPU 仅支持 `"acq_rel"` 内存序，不保证不同地址间的操作顺序。

**Q5: atomic_cas 可以实现锁吗？**

A: 可以。`atomic_cas(lock_ptr, 0, 1)` 是典型的锁获取操作：如果旧值为 0（未锁定），则设为 1（已锁定）并返回 0 表示成功。释放锁使用 `atomic_xchg(lock_ptr, 0)`。

---

## 相关文档

- [01-memory-ops.md](./01-memory-ops.md) - 内存操作 API
- [03-reduction-ops.md](./03-reduction-ops.md) - 归约操作
- [08-comparison-logical-ops.md](./08-comparison-logical-ops.md) - 比较与逻辑操作

## 源码参考

- [core.py - 原子操作函数定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1760-L1877)
- [semantic.py - 原子操作实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L1348-L1542)
- [semantic.py - atom_red_typechecking_impl](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L1365-L1384)
