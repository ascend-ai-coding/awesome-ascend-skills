# index_put / gather_out_to_ub / scatter_ub_to_out / index_select_simd

## 概述

Triton-Ascend 提供了一组高效的内存操作（mem_ops），用于在 Global Memory（GM）和 Unified Buffer（UB）之间进行索引读写。这些操作利用昇腾 NPU 的 SIMD/SIMT 硬件能力，实现了高效的 gather（收集）和 scatter（散列）操作。

与标准 Triton 的 `tl.load/store` + 指针算术方式不同，这些操作直接支持多维索引、边界检查和 UB 零拷贝语义，是注意力机制、嵌入查找等场景的关键组件。

## 关键概念

### 操作对比

| 操作 | 方向 | 模式 | 说明 |
|------|------|------|------|
| `index_put` | UB -> GM | SIMD | 按索引将值写入 GM |
| `gather_out_to_ub` | GM -> UB | SIMD | 按索引从 GM 收集到 UB |
| `scatter_ub_to_out` | UB -> GM | SIMD | 按索引从 UB 散列到 GM |
| `index_select_simd` | GM -> UB | SIMD | 并行索引选择（零拷贝） |

### 数据流向

```
Global Memory (GM)                          Unified Buffer (UB)
┌─────────────────┐                        ┌─────────────────┐
│                 │  gather_out_to_ub       │                 │
│   source data   │ ──────────────────────> │  gathered data  │
│                 │  index_select_simd      │                 │
│                 │ ──────────────────────> │                 │
│                 │                         │                 │
│   dest data     │  index_put              │  value data     │
│                 │ <────────────────────── │                 │
│                 │  scatter_ub_to_out      │                 │
│                 │ <────────────────────── │  value data     │
└─────────────────┘                        └─────────────────┘
```

### 通用参数说明

| 参数 | 说明 |
|------|------|
| `dim` | 操作沿哪个维度进行索引 |
| `index_boundary` / `bound` | 索引上界，用于边界检查 |
| `src_stride` / `dst_stride` | 源/目标张量各维步长 |
| `start_offset` | 各维起始偏移 |
| `end_offset` | 各维结束偏移 |

## API 参考

### index_put

按索引将值从 UB 写入 GM 目标张量。

```python
@_tensor_member_fn
@builtin
def index_put(
    ptr: tensor,
    index: tensor,
    value: tensor,
    dim: int,
    index_boundary: int,
    end_offset: tuple,
    start_offset: tuple,
    dst_stride: tuple,
    _builder=None,
):
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `ptr` | `tensor`（指针） | 目标张量指针（GM） |
| `index` | `tensor` | 索引张量（UB，必须为整数类型） |
| `value` | `tensor` | 值张量（UB） |
| `dim` | `int` | 散列维度（0 <= dim < rank(value) - 1） |
| `index_boundary` | `int` | 索引上界 |
| `end_offset` | `tuple[int]` | 各维结束偏移 |
| `start_offset` | `tuple[int]` | 各维起始偏移 |
| `dst_stride` | `tuple[int]` | 目标张量各维步长 |

**约束：**
- `ptr` 和 `value` 的 rank 必须相同
- `ptr.dtype` 仅支持 float16/bfloat16/float32
- `index` 必须为整数张量，若 rank != 1 会被 reshape 为 1D
- `index.numel` 必须等于 `value.shape[dim]`
- `value` 支持 2~5D 张量
- `dim` 必须满足 0 <= dim < rank(value) - 1
- `end_offset`/`start_offset`/`dst_stride` 长度必须等于 value 的 rank

**操作语义（2D 示例，dim=0）：**
```
out[index[i]][start_offset[1]:end_offset[1]] = value[i][0:end_offset[1]-start_offset[1]]
```

### gather_out_to_ub

按索引从 GM 源张量收集数据到 UB。

```python
@_tensor_member_fn
@builtin
def gather_out_to_ub(
    src: tensor,
    index: tensor,
    index_boundary: int,
    dim: int,
    src_stride: tuple,
    end_offset: tuple,
    start_offset: tuple,
    other=None,
    _builder=None,
):
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | `tensor`（指针） | 源张量指针（GM） |
| `index` | `tensor` | 索引张量（UB，必须为整数类型，1~5D） |
| `index_boundary` | `int` | 索引上界 |
| `dim` | `int` | 收集维度（0 <= dim < rank(index)） |
| `src_stride` | `tuple[int64]` | 源张量各维步长 |
| `end_offset` | `tuple[int32]` | 各维结束偏移 |
| `start_offset` | `tuple[int32]` | 各维起始偏移 |
| `other` | `scalar` | 索引越界时的默认值（可选） |

**约束：**
- `src` 和 `index` 的 rank 必须相同
- `src.dtype` 仅支持 float16/bfloat16/float32
- `index` 必须为整数张量，rank 1~5
- `dim` 必须满足 0 <= dim < rank(index)
- `other` 必须为标量值

**返回值：** 形状与 `index.shape` 相同的张量（在 UB 中）。

**操作语义（2D 示例，dim=0）：**
```
out[i][j] = src[start_offset[0] + index[i][j]][start_offset[1] + j]
```

### scatter_ub_to_out

按索引将 UB 中的值散列写入 GM 目标张量。

```python
@_tensor_member_fn
@builtin
def scatter_ub_to_out(
    ptr: tensor,
    value: tensor,
    index: tensor,
    index_boundary: int,
    dim: int,
    dst_stride: tuple,
    end_offset: tuple,
    start_offset: tuple,
    _builder=None,
):
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `ptr` | `tensor`（指针） | 目标张量指针（GM） |
| `value` | `tensor` 或 `scalar` | 值张量（UB）或标量值 |
| `index` | `tensor` | 索引张量（UB，必须为整数类型，1~5D） |
| `index_boundary` | `int` | 索引上界 |
| `dim` | `int` | 散列维度（0 <= dim < rank(index)） |
| `dst_stride` | `tuple[int64]` | 目标张量各维步长 |
| `end_offset` | `tuple[int32]` | 各维结束偏移 |
| `start_offset` | `tuple[int32]` | 各维起始偏移 |

**约束：**
- `ptr`、`index`、`value` 的 rank 必须相同
- `ptr.dtype` 仅支持 float16/bfloat16/float32
- `index` 必须为整数张量，rank 1~5
- `dim` 必须满足 0 <= dim < rank(index)
- 如果 `value` 不是 ranked tensor，会使用 `index.shape` 和 `ptr.dtype` 创建全值张量

**操作语义（2D 示例，dim=0）：**
```
out[start_offset[0] + index[i][j]][start_offset[1] + j] = value[i][j]
```

### index_select_simd

并行索引选择操作，从 GM 直接加载到 UB，支持零拷贝语义。

```python
@_tensor_member_fn
@builtin
def index_select_simd(
    src,
    dim,
    index,
    src_shape,
    src_offset,
    read_shape,
    _builder=None,
) -> tensor:
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | `tensor`（指针） | 源张量指针（GM） |
| `dim` | `int` 或 `constexpr` | 索引选择维度 |
| `index` | `tensor` | 1D 索引张量（UB） |
| `src_shape` | `List[Union[int, tensor]]` | 源张量完整形状 |
| `src_offset` | `List[Union[int, tensor]]` | 各维读取起始偏移 |
| `read_shape` | `List[Union[int, tensor]]` | 各维读取大小 |

**约束：**
- `read_shape[dim]` 必须为 `-1`（表示该维度由 index 长度决定）
- `src_offset[dim]` 可以为 `-1`（将被忽略）
- `dim` 不能是最后一个维度（不支持 trailing dimension）
- `index` 必须为 1D 张量
- 边界处理：`src_offset + read_shape > src_shape` 时自动截断到 `src_shape` 边界
- 不检查 `index` 是否包含越界值

**返回值：** 在 UB 中的张量，形状中 `dim` 维度被替换为 `index` 的长度。

## 代码示例

### 示例 1：index_put 基本使用

```python
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def simple_index_put_kernel(value_ptr, index_ptr, dst_ptr):
    index_local = tl.arange(0, 2)
    x1_local = tl.arange(0, 2)[None, :]

    index_tile = tl.load(index_ptr + index_local)
    value_tile = tl.load(value_ptr + index_local[:, None] * 2 + x1_local)

    al.index_put(
        ptr=dst_ptr,
        index=index_tile,
        value=value_tile,
        dim=0,
        index_boundary=4,
        end_offset=(2, 2),
        start_offset=(0, 0),
        dst_stride=(2, 1)
    )

dst = torch.zeros((4, 2), device='npu', dtype=torch.float32)
value = torch.tensor([[1., 2.], [3., 4.]], device='npu')
index = torch.tensor([2, 0], device='npu')

simple_index_put_kernel[(1,)](value, index, dst)
```

### 示例 2：gather_out_to_ub 基本使用

```python
@triton.jit
def simple_gather_kernel(src_ptr, index_ptr, out_ptr):
    y0_local = tl.arange(0, 2)[:, None]
    x1_local = tl.arange(0, 2)[None, :]
    mask = (y0_local < 2) & (x1_local < 2)

    index = tl.load(index_ptr + y0_local * 2 + x1_local, mask)

    gathered = al.gather_out_to_ub(
        src=src_ptr,
        index=index,
        index_boundary=4,
        dim=0,
        src_stride=(2, 1),
        end_offset=(2, 2),
        start_offset=(0, 0)
    )

    tl.store(out_ptr + y0_local * 2 + x1_local, gathered, mask)

src = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], device='npu')
index = torch.tensor([[0, 1], [2, 3]], device='npu')
out = torch.empty((2, 2), device='npu', dtype=torch.float32)

simple_gather_kernel[(1,)](src, index, out)
```

### 示例 3：scatter_ub_to_out 基本使用

```python
@triton.jit
def simple_scatter_kernel(value_ptr, index_ptr, dst_ptr):
    y0_local = tl.arange(0, 2)[:, None]
    x1_local = tl.arange(0, 2)[None, :]
    mask = (y0_local < 2) & (x1_local < 2)

    value = tl.load(value_ptr + y0_local * 2 + x1_local, mask)
    index = tl.load(index_ptr + y0_local * 2 + x1_local, mask)

    al.scatter_ub_to_out(
        ptr=dst_ptr,
        value=value,
        index=index,
        index_boundary=4,
        dim=0,
        dst_stride=(2, 1),
        end_offset=(2, 2),
        start_offset=(0, 0)
    )

dst = torch.zeros((4, 2), device='npu', dtype=torch.float32)
value = torch.tensor([[1., 2.], [3., 4.]], device='npu')
index = torch.tensor([[1, 2], [3, 0]], device='npu')

simple_scatter_kernel[(1,)](value, index, dst)
```

### 示例 4：index_select_simd 静态形状

```python
@triton.jit
def index_select_static_kernel(src_ptr, output_ptr, indices_ptr):
    indices = tl.load(indices_ptr + tl.arange(0, 4))

    result = al.index_select_simd(
        src_ptr,
        dim=1,
        index=indices,
        src_shape=[8, 100, 256],
        src_offset=[4, -1, 128],
        read_shape=[4, -1, 128]
    )

    tl.store(output_ptr + tl.arange(0, 4)[:, None] * 128 + tl.arange(0, 128)[None, :], result)
```

### 示例 5：index_select_simd 动态形状

```python
@triton.jit
def index_select_dynamic_kernel(src_ptr, output_ptr, indices_ptr, M, N, D):
    indices = tl.load(indices_ptr + tl.arange(0, 4))

    result = al.index_select_simd(
        src_ptr,
        dim=1,
        index=indices,
        src_shape=[M, N, D],
        src_offset=[4, -1, 128],
        read_shape=[4, -1, 128]
    )

    tl.store(output_ptr + ..., result)
```

### 示例 6：gather_out_to_ub 带 other 默认值

```python
@triton.jit
def gather_with_default_kernel(src_ptr, index_ptr, out_ptr):
    index = tl.load(index_ptr + tl.arange(0, 4))

    gathered = al.gather_out_to_ub(
        src=src_ptr,
        index=index,
        index_boundary=10,
        dim=0,
        src_stride=(16,),
        end_offset=(4,),
        start_offset=(0,),
        other=0.0
    )

    tl.store(out_ptr + tl.arange(0, 4), gathered)
```

## NPU 适配要点

1. **GM 与 UB 的区分**：这些操作的核心价值在于直接在 GM 和 UB 之间搬运数据，避免中间转储。`ptr`/`src` 参数是 GM 指针，`index`/`value` 参数在 UB 中。

2. **stride/offset 的整数类型**：
   - `src_stride` 和 `dst_stride` 使用 int64（i64）
   - `end_offset` 和 `start_offset` 使用 int32（i32）
   - 这与昇腾 NPU 硬件指令的参数类型要求一致

3. **index_boundary 的作用**：`index_boundary` 用于硬件边界检查，当索引值超过此边界时，硬件会进行特殊处理（如使用默认值或跳过）。

4. **index_select_simd 的 dim 限制**：不支持最后一个维度作为 `dim`，因为硬件实现需要 trailing dimension 连续存储。

5. **数据类型限制**：当前 `ptr.dtype` 仅支持 float16/bfloat16/float32，不支持整数类型的直接 scatter/gather。

6. **index 的 reshape**：`index_put` 中如果 `index` 的 rank != 1，会自动 reshape 为 1D（flatten），`index.numel` 必须等于 `value.shape[dim]`。

7. **scatter_ub_to_out 的标量值**：`value` 参数支持标量值，当传入标量时，会自动创建一个形状与 `index` 相同、所有元素为该标量值的张量。

## 常见问题

**Q: gather_out_to_ub 和 index_select_simd 有什么区别？**
A: `gather_out_to_ub` 是通用的索引收集操作，支持多维索引和边界检查；`index_select_simd` 是更高效的并行索引选择，支持零拷贝语义，但限制更多（index 必须为 1D，dim 不能是最后一维）。

**Q: index_put 和 scatter_ub_to_out 有什么区别？**
A: `index_put` 的 `value` 和 `index` 分开传入，`dim` 范围是 `0 <= dim < rank(value) - 1`；`scatter_ub_to_out` 的 `value` 可以是标量，`dim` 范围是 `0 <= dim < rank(index)`。两者在语义上类似，但参数约束不同。

**Q: 如何处理索引越界？**
A: `gather_out_to_ub` 支持通过 `other` 参数指定越界默认值；`index_select_simd` 不检查索引越界，需要用户确保索引值合法。

**Q: src_shape/src_offset/read_shape 什么时候用动态值？**
A: 当源张量的形状在编译期不确定时（如由 kernel 参数传入），使用动态值（tensor 类型）。编译期已知的形状使用静态常量即可。

**Q: 为什么 index_put 的 dim 范围是 0 <= dim < rank(value) - 1？**
A: 因为 `index_put` 不支持沿最后一个维度进行散列，这是硬件实现的限制。如果需要沿最后一维操作，可以考虑转置后操作。

## 相关文档

- [01-extension-overview.md](./01-extension-overview.md) - 扩展 API 总览
- [06-custom-op.md](./06-custom-op.md) - 自定义算子（内置 index_select/index_put/gather_load/scatter_store）
- [07-buffer-model.md](./07-buffer-model.md) - Buffer 编程模型（UB 内存操作）

## 源码参考

- [mem_ops.py: index_put](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/mem_ops.py#L40-L177) - index_put 函数定义
- [mem_ops.py: gather_out_to_ub](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/mem_ops.py#L180-L329) - gather_out_to_ub 函数定义
- [mem_ops.py: scatter_ub_to_out](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/mem_ops.py#L332-L482) - scatter_ub_to_out 函数定义
- [mem_ops.py: index_select_simd](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/mem_ops.py#L485-L636) - index_select_simd 函数定义
- [builtin_custom_ops.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/builtin_custom_ops.py) - 内置自定义算子（SIMT 版本的 index_select/index_put/gather_load/scatter_store）
