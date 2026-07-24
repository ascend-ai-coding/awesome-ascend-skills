# Where 条件优化：单位置 False 替换为 get_element + insert_slice

## 触发条件

当 Agent 发现 Triton kernel 中存在 `tl.where` 语句，且其 condition 满足以下所有条件时，应考虑应用此优化：

1. **condition 形式为 `X != y`**：其中 `X` 是关于地址偏移的向量（如 `base + tl.arange(0, BLOCK_SIZE)`），`y` 是标量或常量
2. **仅有一个位置为 False**：`X == y` 仅在一个索引处成立（即 `y` 在 `X` 的取值范围内仅出现一次）
3. **A、B 和 X 的 shape 相同**：向量维度一致
4. **替换逻辑清晰**：原 `tl.where(condition, A, B)` 表示当 `X != y` 时选 A，否则选 B

**直观理解**：当 `y` 在 `[base, base + BLOCK_SIZE)` 范围内时，`X == y` 仅在偏移量 `_offs = y - base` 处成立，因此 `where` 只在一个位置选择 B，其他位置都选 A。

---

## 核心知识

### 为什么需要优化

`tl.where` 在 NPU 上会导致**离散访存**，退化为 scalar 操作。当 condition 是逐元素比较时，Vector 单元需要对每个元素分别判断条件并选择结果，无法充分利用向量化内存访问，严重影响性能。

当 condition 仅在**一个位置**为 False 时，`where` 的语义等价于"在 A 的某个位置替换为 B 的对应值"。这个操作可以用更高效的 `get_element` + `insert_slice` 组合来实现：

- `get_element`：从 B 中提取单个元素（标量操作，但只执行一次）
- `insert_slice`：将单个元素写入 A 的指定位置（向量化操作，只修改一个位置）

### 关键前提

| 前提 | 说明 |
|------|------|
| condition 仅一个位置为 False | 这是此优化的核心前提，多个位置为 False 时不适用 |
| y 可能在范围外 | 当 `y` 不在 `[base, base + BLOCK_SIZE)` 时，condition 恒为 True，不需要任何操作 |
| dtype 一致 | `tl.full` 创建的 tensor 必须与 A 的数据类型一致 |

### extension 回退机制

`get_element` 和 `insert_slice` 来自 `triton.language.extra.cann.extension`。如果该模块不可用，需要回退到 `tl`（Triton 核心库也支持这两个操作）：

```python
try:
    import triton.language.extra.cann.extension as extension
except Exception:
    extension = tl
```

---

## 代码模式

### 优化前：tl.where 退化为 scalar 计算

```python
X = base + tl.arange(0, BLOCK_SIZE)
A = tl.load(A_ptr + X)
B = tl.load(B_ptr + X)

condition = (X != y)
A = tl.where(condition, A, B)
```

**问题**：`tl.where(condition, A, B)` 中 condition 是逐元素比较，导致离散访存，退化为 scalar 操作。

### 优化后：get_element + insert_slice

```python
try:
    import triton.language.extra.cann.extension as extension
except Exception:
    extension = tl

X = base + tl.arange(0, BLOCK_SIZE)
A = tl.load(A_ptr + X)
B = tl.load(B_ptr + X)

if base <= y < base + BLOCK_SIZE:
    _offs = y - base
    _val = extension.get_element(B, (_offs,))
    _tensor = tl.full((1,), _val, dtype=A.dtype)
    A = extension.insert_slice(
        A,
        _tensor,
        offsets=(_offs,),
        sizes=(1,),
        strides=(1,)
    )
```

**改进**：消除了 `tl.where` 的离散访存，改为单次 `get_element` 提取 + 单次 `insert_slice` 写入，保持向量化访存。

### 关键代码解析

| 步骤 | 代码 | 作用 |
|------|------|------|
| 1 | `if base <= y < base + BLOCK_SIZE:` | 判断 y 是否在当前 block 范围内 |
| 2 | `_offs = y - base` | 计算 y 在 block 内的偏移 |
| 3 | `extension.get_element(B, (_offs,))` | 从 B 中提取 `_offs` 位置的值 |
| 4 | `tl.full((1,), _val, dtype=A.dtype)` | 创建形状为 (1,) 的 tensor，用于 insert_slice |
| 5 | `extension.insert_slice(A, _tensor, ...)` | 将 _tensor 插入 A 的 `_offs` 位置 |

### if 条件不满足时

当 `y` 不在 `[base, base + BLOCK_SIZE)` 范围内时，`X != y` 恒为 True，`tl.where` 的结果就是 A 本身，不需要任何操作，也不用保留原有的 `tl.where` 逻辑。

---

## 910_95 特别注意

### Reg-based 架构下的 where 行为

910_95 属于 Reg-based 架构，支持 SIMT VF 模式。在 SIMT 模式下，`tl.where` 的退化行为与 910B（Mem-based 架构）不同：

| 架构 | where 退化行为 |
|------|---------------|
| 910B (Mem-based) | where 退化为标量循环，性能损失 10x-100x |
| 910_95 (Reg-based, SIMD VF) | where 同样退化为标量操作，性能损失严重 |
| 910_95 (Reg-based, SIMT VF) | 通用算术操作在 SIMT VF 模式下会降级，where 场景更复杂 |

**结论**：无论哪种架构，单位置 False 的 where 都应替换为 `get_element` + `insert_slice`。

### get_element / insert_slice 的数据类型支持

910_95 上 `get_element` 和 `insert_slice` 支持的数据类型：

| int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
|------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|
| Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | N |

**注意**：`bool` 类型不支持这两个操作。如果 A/B 是 bool 类型，不能使用此优化。

### insert_slice 参数约束

`insert_slice` 的参数必须满足以下约束：

| 参数 | 约束 |
|------|------|
| `offsets` | 长度必须与张量维度数相同 |
| `sizes` | 长度必须与张量维度数相同，且 `sub` 的 shape 必须与 `sizes` 匹配 |
| `strides` | 长度必须与张量维度数相同 |
| 插入区域 | 不能超出 `ful` 张量的边界 |

对于 1D 场景（本优化最常见的场景），参数为：
- `offsets=(_offs,)` - 1D 偏移
- `sizes=(1,)` - 插入 1 个元素
- `strides=(1,)` - 步长为 1

### 多维场景扩展

当 A/B 是多维 tensor 时，`get_element` 和 `insert_slice` 的索引需要对应每个维度：

```python
if base_m <= y_m < base_m + BLOCK_M and base_n <= y_n < base_n + BLOCK_N:
    _offs_m = y_m - base_m
    _offs_n = y_n - base_n
    _val = extension.get_element(B, (_offs_m, _offs_n))
    _tensor = tl.full((1, 1), _val, dtype=A.dtype)
    A = extension.insert_slice(
        A,
        _tensor,
        offsets=(_offs_m, _offs_n),
        sizes=(1, 1),
        strides=(1, 1)
    )
```

---

## 相关文档

- [标量降级规避指南](../docs_for_triton_agent/06-scalar-degradation-avoidance.md) - where 退化与标量降级的关系
- [内存访问模式优化指南](../docs_for_triton_agent/04-memory-access-patterns.md) - 离散访存 vs 连续访存
- [硬件速查手册](../docs_for_triton_agent/00-hardware-quick-ref.md) - 910_95 Reg-based 架构差异
- [get_element API 文档](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/triton_api/Extension_Ops/get_element.md) - get_element 参数规格与数据类型支持
- [insert_slice API 文档](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/triton_api/Extension_Ops/insert_slice.md) - insert_slice 参数规格与约束
- [vec_ops.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/vec_ops.py) - get_element / insert_slice 实现源码
