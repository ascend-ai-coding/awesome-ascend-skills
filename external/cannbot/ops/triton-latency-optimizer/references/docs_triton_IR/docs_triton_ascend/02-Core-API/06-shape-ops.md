# 张量形状操作 API

## 概述

本文档详细描述 Triton-Ascend 中的张量形状操作 API，包括 `broadcast`、`cat`、`reshape`、`view`、`trans`、`permute`、`expand_dims`、`join`、`split`、`ravel`、`flip` 和 `interleave`。这些操作用于改变张量的维度、形状和元素排列方式，是构建复杂计算图的基础。在 Ascend NPU 上，形状操作的行为与 GPU 基本一致，但 `reshape` 的 `can_reorder` 参数和数据布局转换可能影响性能。

关键词：形状操作, broadcast, cat, reshape, view, trans, permute, expand_dims, join, split, ravel, flip, interleave, NPU, 数据布局, can_reorder

---

## API 参考

### tl.broadcast / tl.broadcast_to

将两个张量广播到兼容形状，或将张量广播到指定形状。

```python
# 双张量广播
triton.language.broadcast(input, other, _builder=None) -> (tensor, tensor)

# 广播到指定形状
triton.language.broadcast_to(input, *shape, _builder=None) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量 |
| `other` | `tensor` | 第二个输入张量（仅 `broadcast`） |
| `shape` | `tuple(int)` 或可变参数 | 目标形状（仅 `broadcast_to`） |

**广播规则**：
- 维度从右向左对齐
- 大小为 1 的维度可以扩展到任意大小
- 非单一维度必须匹配
- 标量可以广播到任意形状

### tl.cat

沿第一维拼接两个张量。

```python
triton.language.cat(input, other, can_reorder=False, _builder=None) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 第一个输入张量（必须为 1D） |
| `other` | `tensor` | 第二个输入张量（必须为 1D） |
| `can_reorder` | `bool`，默认 `False` | 编译器提示：若为 True，允许在拼接时重排元素。当前实现**仅支持 `can_reorder=True`** |

**约束**：当前实现要求 `can_reorder=True`，且输入必须为 1D 张量。

### tl.reshape

改变张量形状，不改变元素数量。

```python
triton.language.reshape(input, *shape, can_reorder=False, _builder=None) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量 |
| `shape` | `tuple(int)` 或可变参数 | 目标形状 |
| `can_reorder` | `bool`，默认 `False` | 若为 True，允许编译器重排元素以优化数据布局 |

**约束**：新形状的元素总数必须与原形状相同。

### tl.view（已弃用）

```python
triton.language.view(input, *shape, _builder=None) -> tensor
```

**已弃用**，请使用 `reshape`。`view` 实际调用 `reshape(input, shape, can_reorder=False)`。

### tl.trans / tl.permute

置换张量的维度顺序。

```python
# 转置（默认交换最后两维）
triton.language.trans(input, *dims, _builder=None) -> tensor

# 置换维度
triton.language.permute(input, *dims, _builder=None) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量 |
| `dims` | `tuple(int)` 或可变参数 | 维度排列顺序 |

**区别**：
- `trans`：若不传 `dims`，默认执行 `(1, 0)` 置换（2D 转置）
- `permute`：必须显式指定 `dims`

**约束**：`dims` 必须是 `0, 1, ..., n-1` 的一个排列。

### tl.expand_dims

在指定位置插入长度为 1 的新维度。

```python
triton.language.expand_dims(input, axis, _builder=None) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `input` | `tensor` | 输入张量 |
| `axis` | `int` 或 `Sequence(int)` | 插入新维度的位置索引（相对于结果张量） |

**约束**：不允许重复的 axis 值。

### tl.join

沿新的最内层维度连接两个张量。

```python
triton.language.join(a, b, _builder=None) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `a` | `tensor` | 第一个输入张量 |
| `b` | `tensor` | 第二个输入张量 |

两个输入会先广播到相同形状。结果形状为 `a.shape + [2]`。

`join` 是 `split` 的逆操作。

### tl.split

沿最后一维（大小必须为 2）将张量拆分为两个。

```python
triton.language.split(a, _builder=None, _generator=None) -> (tensor, tensor)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `a` | `tensor` | 输入张量，最后一维大小必须为 2 |

`split` 是 `join` 的逆操作。

### tl.ravel

返回张量的连续扁平化视图。

```python
triton.language.ravel(x) -> tensor
```

等价于 `reshape(x, [x.numel], can_reorder=False)`。

### tl.flip

沿指定维度翻转张量。

```python
triton.language.flip(x, dim=None) -> tensor
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `tensor` | 输入张量 |
| `dim` | `int`，可选 | 翻转维度。当前仅支持最后一维 |

**约束**：翻转维度的大小和 `x.numel` 必须都是 2 的幂。

### tl.interleave

沿最后一维交错两个张量的值。

```python
triton.language.interleave(a, b) -> tensor
```

等价于 `join(a, b).reshape(a.shape[:-1] + [2 * a.shape[-1]])`。

---

## 代码示例

### 基础用法：reshape / trans / broadcast

```python
import triton
import triton.language as tl

@triton.jit
def shape_ops_kernel(
    x_ptr, y_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :])
    y = tl.load(y_ptr + offs_m[:, None] * N + offs_n[None, :])

    x_t = tl.trans(x)
    result = x_t * y

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], result)
```

### 进阶用法：join/split 与 ravel

```python
@triton.jit
def join_split_kernel(
    a_ptr, b_ptr, out_a_ptr, out_b_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    joined = tl.join(a, b)
    processed = joined * 2.0

    out_a, out_b = tl.split(processed)
    tl.store(out_a_ptr + offsets, out_a, mask=mask)
    tl.store(out_b_ptr + offsets, out_b, mask=mask)

@triton.jit
def ravel_kernel(
    x_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_SIZE // N
    offs_n = pid * BLOCK_SIZE % N
    x = tl.load(x_ptr + offs_m * N + offs_n)
    flat = tl.ravel(x)
    tl.store(out_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), flat)
```

---

## NPU 适配要点

### 1. reshape vs view 的区别

| 特性 | `reshape` | `view` |
|------|----------|--------|
| 状态 | 推荐使用 | **已弃用** |
| `can_reorder` | 支持，默认 `False` | 不支持（内部使用 `can_reorder=False`） |
| 语义 | 可控制是否允许重排 | 不允许重排 |

`view` 已弃用，所有新代码应使用 `reshape`。

### 2. can_reorder 参数的影响

`can_reorder` 参数是 Triton 特有的编译器提示：

- `can_reorder=False`（默认）：reshape 必须保持元素的逻辑顺序。编译器可能需要插入额外的数据搬运指令。
- `can_reorder=True`：允许编译器重排元素以优化数据布局。适用于后续操作不依赖元素顺序的场景（如归约操作前的 reshape）。

在 NPU 上，`can_reorder=True` 可以避免不必要的 UB 数据搬运，提升性能。但需确保重排不影响计算正确性。

### 3. trans/permute 在 NPU 上的行为

`trans` 和 `permute` 在 NPU 上是元数据操作（仅改变形状描述，不移动数据），与 GPU 行为一致。但后续对转置后张量的访问模式会影响性能：

- 连续访问转置后的张量 = 对原张量的跨步访问
- NPU 对跨步访问的效率低于连续访问

建议在矩阵乘法前使用 `trans` 调整布局，让 Cube 单元获得连续输入。

### 4. cat 操作的限制

当前 `cat` 操作的实现**仅支持 `can_reorder=True`**，且输入必须为 1D 张量。这意味着：
- 拼接后的元素顺序可能与输入不一致
- 仅适用于后续操作不依赖元素顺序的场景

### 5. flip 和 interleave 的约束

- `flip`：翻转维度的大小和 `x.numel` 必须是 2 的幂（因为实现基于 bitonic merge）
- `interleave`：两个输入张量必须形状相同

---

## 常见问题

**Q1: reshape 和 view 有什么区别？**

A: `view` 已弃用，等价于 `reshape(input, shape, can_reorder=False)`。建议统一使用 `reshape`，并通过 `can_reorder` 参数控制是否允许重排。

**Q2: can_reorder=True 什么时候可以安全使用？**

A: 当 reshape 后的操作不依赖元素的逻辑顺序时，可以安全使用。典型场景包括：reshape 后立即进行归约操作（sum/max/min），或 reshape 后的元素会被原子写入不同位置。

**Q3: 为什么 cat 只支持 can_reorder=True？**

A: 这是当前实现的限制。`cat` 的底层实现通过合并两个张量的句柄完成，编译器可能会重排元素以优化数据布局。

**Q4: trans 和 permute 会复制数据吗？**

A: 不会。`trans` 和 `permute` 是元数据操作，仅改变张量的形状和步幅描述，不移动实际数据。但后续对转置张量的访问可能触发实际的数据搬运。

**Q5: 如何实现多维 cat？**

A: Triton 的 `cat` 仅支持 1D。对于多维拼接，需要先 reshape 为 1D，cat 后再 reshape 回目标形状。或者使用 `join` 沿新维度连接。

---

## 相关文档

- [01-memory-ops.md](./01-memory-ops.md) - 内存操作 API
- [04-linear-algebra-ops.md](./04-linear-algebra-ops.md) - 线性代数操作
- [08-comparison-logical-ops.md](./08-comparison-logical-ops.md) - 比较与逻辑操作

## 源码参考

- [core.py - broadcast/broadcast_to 定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1240-L1271)
- [core.py - trans/permute 定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1274-L1323)
- [core.py - reshape/view 定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1407-L1447)
- [core.py - expand_dims 定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1457-L1484)
- [core.py - join/split/cat 定义](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1326-L1404)
- [semantic.py - broadcast_impl_value](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L796-L846)
- [semantic.py - permute 实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L768-L775)
- [standard.py - ravel/flip/interleave 实现](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/standard.py#L60-L561)
