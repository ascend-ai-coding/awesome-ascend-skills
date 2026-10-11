# sub_vec_id / sub_vec_num

## 概述

在昇腾 NPU 的 Cube-Vector 架构中，一个 AI Core 包含 1 个 Cube 核心和 2 个 Vector 核心，即 Cube:Vector 的硬件比例为 1:2。为了在 Cube-Vector 融合算子中区分和利用多个 Vector 核心，Triton-Ascend 提供了 `sub_vec_id()` 和 `sub_vec_num()` 两个函数。

`sub_vec_id()` 返回当前 Vector 核心在 AI Core 内的编号（0 或 1），`sub_vec_num()` 返回每个 AI Core 中 Vector 核心的数量（通常为 2）。这两个函数是实现多 Vector 核心并行计算的基础。

## 关键概念

### 1:2 Cube:Vector 硬件比例

```
┌─────────────────────────────────────────┐
│              AI Core                     │
│                                          │
│  ┌──────────┐  ┌──────────┐             │
│  │  Cube    │  │ Vector 0 │  (sub_vec_id = 0) │
│  │  (1个)   │  │          │             │
│  └──────────┘  ├──────────┤             │
│                │ Vector 1 │  (sub_vec_id = 1) │
│                │          │             │
│                └──────────┘             │
│                                          │
│  sub_vec_num() = 2                       │
└─────────────────────────────────────────┘
```

- **Cube**：矩阵计算单元，执行 `tl.dot` 等矩阵乘法操作
- **Vector 0/1**：向量计算单元，执行逐元素运算、数据搬运等操作
- **sub_vec_num()**：返回 Vector 核心数量，由硬件配置决定（`aivector_core_num / aicore_num`）

### CV 融合算子

CV（Cube-Vector）融合算子是指在一个 kernel 中同时使用 Cube 和 Vector 核心的算子。在 CV 融合算子中：

1. Cube 负责矩阵乘法计算
2. 多个 Vector 核心并行负责后处理（如 Softmax、量化、激活等）
3. 通过 `sub_vec_id()` 区分不同 Vector 核心的工作负载
4. 通过 `sync_block_set/wait` 协调 Cube 和 Vector 的执行

## API 参考

### sub_vec_id

获取当前 Vector 核心在 AI Core 内的编号。

```python
@builtin
def sub_vec_id(_builder=None) -> tl.tensor:
    """
    Get the Vector Core index on the AI Core.
    """
```

**返回值：**

| 类型 | 说明 |
|------|------|
| `tl.tensor` (int64) | 当前 Vector 核心的编号，0 或 1 |

### sub_vec_num

获取每个 AI Core 中 Vector 核心的数量。

```python
@builtin
def sub_vec_num(_builder=None) -> tl.constexpr:
    """
    Get the Vector Core Num on one AI Core.
    """
```

**返回值：**

| 类型 | 说明 |
|------|------|
| `tl.constexpr` (int) | Vector 核心数量，通常为 2 |

**实现细节：** `sub_vec_num()` 通过 `NPUUtils` 获取硬件信息，计算公式为 `aivector_core_num / aicore_num`。

## 代码示例

### 示例 1：基本使用

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def sub_vec_demo_kernel(out_ptr):
    vec_id = al.sub_vec_id()
    vec_num = al.sub_vec_num()

    if vec_id == 0:
        tl.store(out_ptr, 100)
    else:
        tl.store(out_ptr + 1, 200)
```

### 示例 2：CV 融合算子中的工作负载分配

```python
@triton.jit
def cv_fusion_kernel(a_ptr, b_ptr, c_ptr, bias_ptr, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    with al.scope(core_mode="cube"):
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptr + ...)
            b = tl.load(b_ptr + ...)
            acc = tl.dot(a, b, acc)

        ub_buf = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], al.ascend_address_space.UB)
        al.fixpipe(acc, ub_buf)
        al.sync_block_set("cube", "vector", 0)

    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0)

        vec_id = al.sub_vec_id()
        vec_num = al.sub_vec_num()

        half_m = BLOCK_M // vec_num
        start_row = vec_id * half_m
        end_row = start_row + half_m

        acc_tensor = bl.to_tensor(ub_buf)
        my_slice = acc_tensor[start_row:end_row, :]
        bias_slice = tl.load(bias_ptr + tl.arange(0, BLOCK_N))
        result = my_slice + bias_slice

        tl.store(c_ptr + ..., result)
```

### 示例 3：parallel 循环中的 bind_sub_block

```python
@triton.jit
def parallel_sub_vec_kernel(data_ptr, out_ptr, N,
                            BLOCK_N: tl.constexpr):
    with al.scope(core_mode="vector"):
        vec_num = al.sub_vec_num()

        for i in al.parallel(0, N, BLOCK_N, bind_sub_block=True):
            vec_id = al.sub_vec_id()
            offset = i + vec_id * BLOCK_N
            data = tl.load(data_ptr + offset + tl.arange(0, BLOCK_N))
            result = data * 2.0
            tl.store(out_ptr + offset + tl.arange(0, BLOCK_N), result)
```

### 示例 4：动态工作负载分配

```python
@triton.jit
def dynamic_workload_kernel(input_ptr, output_ptr, total_elements,
                            BLOCK_SIZE: tl.constexpr):
    with al.scope(core_mode="vector"):
        vec_id = al.sub_vec_id()
        vec_num = al.sub_vec_num()

        per_vec = total_elements // vec_num
        my_start = vec_id * per_vec
        my_end = my_start + per_vec

        for offset in range(my_start, my_end, BLOCK_SIZE):
            idx = offset + tl.arange(0, BLOCK_SIZE)
            mask = idx < my_end
            data = tl.load(input_ptr + idx, mask=mask)
            result = tl.sigmoid(data)
            tl.store(output_ptr + idx, result, mask=mask)
```

## NPU 适配要点

1. **仅限 Vector 核心上下文**：`sub_vec_id()` 和 `sub_vec_num()` 应在 `al.scope(core_mode="vector")` 上下文中使用。在 Cube 上下文中调用 `sub_vec_id()` 的行为未定义。

2. **sub_vec_id 返回 tensor**：`sub_vec_id()` 返回的是 `tl.tensor`（int64 类型），可以在 kernel 中参与计算和条件判断。`sub_vec_num()` 返回 `tl.constexpr`，是编译期常量。

3. **bind_sub_block 与 parallel 配合**：在 `al.parallel` 循环中设置 `bind_sub_block=True`，可以让多个 Vector 核心参与循环迭代。此时循环的迭代次数决定了参与的 Vector 核心数量（910B 上最多 2 个）。

4. **硬件比例因平台而异**：虽然当前主流平台是 1:2 的 Cube:Vector 比例，但 `sub_vec_num()` 是通过运行时查询硬件信息获取的，可以适应不同平台。

5. **工作负载划分**：在多 Vector 核心场景下，需要合理划分工作负载，确保每个 Vector 核心处理的数据量均衡，避免负载不均导致的性能浪费。

## 常见问题

**Q: sub_vec_id() 在 Cube 核心上调用会返回什么？**
A: `sub_vec_id()` 应仅在 Vector 核心上下文中调用。在 Cube 上下文中调用可能导致未定义行为。

**Q: sub_vec_num() 的值是编译期常量还是运行时值？**
A: `sub_vec_num()` 返回 `tl.constexpr`，是编译期常量。它通过 `NPUUtils` 在编译时查询硬件信息获得。

**Q: 如何确保两个 Vector 核心不会写冲突？**
A: 需要通过 `sub_vec_id()` 区分不同 Vector 核心的工作范围，确保它们写入不同的内存区域。同时可以使用 `sync_block_all("all_sub_vector", event_id)` 同步所有子 Vector 核心。

**Q: bind_sub_block=True 的 parallel 循环如何工作？**
A: 设置 `bind_sub_block=True` 后，编译器会将循环迭代分配给不同的子 Vector 核心。每个子 Vector 核心执行循环的一个子集，实现并行计算。

## 相关文档

- [04-sync-operations.md](./04-sync-operations.md) - 同步操作（多 Vector 核心同步）
- [08-aux-ops.md](./08-aux-ops.md) - parallel 迭代器（bind_sub_block 参数）
- [03-fixpipe.md](./03-fixpipe.md) - fixpipe 操作（CV 融合算子中的数据搬运）

## 源码参考

- [core.py: sub_vec_id](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py#L166-L171) - sub_vec_id 函数定义
- [core.py: sub_vec_num](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py#L359-L368) - sub_vec_num 函数定义
- [semantic.py: sub_vec_id](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/semantic.py#L90-L91) - sub_vec_id 的 IR 生成
