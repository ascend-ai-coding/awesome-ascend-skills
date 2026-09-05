# 自定义算子注册与使用

## 概述

Triton-Ascend 提供了自定义算子机制，允许开发者通过 `@register_custom_op` 装饰器注册自定义算子类，然后通过 `al.custom()` 在 kernel 中调用。自定义算子机制是 Triton-Ascend 扩展 CANN 算子生态的核心接口，支持将 CANN 自定义算子（通过 bitcode 提供）集成到 Triton kernel 中。

此外，Triton-Ascend 还内置了一组自定义算子（`__builtin_*`），包括 `index_select`、`index_put`、`gather_load`、`scatter_store` 等，这些算子无需用户注册即可直接使用。

## 关键概念

### 自定义算子注册流程

```
1. 定义算子类（设置 name/core/pipe/mode 等属性）
2. 使用 @register_custom_op 装饰器注册
3. 在 kernel 中通过 al.custom(name, *args, **kwargs) 调用
```

### 自定义算子类必需属性

| 属性 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `name` | `str` | 是 | 算子名称，全局唯一标识符 |
| `core` | `CORE` | 是 | 执行核心类型（CORE_VECTOR/CORE_CUBE 等） |
| `pipe` | `PIPE` | 是 | 执行流水线类型（PIPE_V/PIPE_M 等） |
| `mode` | `MODE` | 是 | 执行模式（MODE.SIMD/MODE.SIMT/MODE.MIX） |
| `symbol` | `str` | 非内置必需 | CANN bitcode 中的符号名 |
| `bitcode` | `str` | 非内置必需 | CANN bitcode 文件路径 |
| `__init__` | 方法 | 推荐 | 参数校验和初始化 |
| `iterator_types` | `List[IteratorType]` | 可选 | 迭代器类型（用于 indexing_map） |
| `indexing_map` | `AffineMap` | 可选 | 仿射映射（描述循环嵌套与张量维度的关系） |
| `extra_buffers` | `tuple` | 可选 | 额外缓冲区规格（类型, 大小） |
| `align_dim` | `dict` | 可选 | 参数对齐维度约束 |
| `extra_attr` | `str` | 可选 | 额外属性字符串 |

### 内置自定义算子列表

| 算子名 | 类名 | core | pipe | mode | 说明 |
|--------|------|------|------|------|------|
| `__builtin_index_select` | `_index_select` | VECTOR | PIPE_V | SIMT | 索引选择（GM -> UB，2D-5D） |
| `__builtin_index_put` | `_index_put` | VECTOR | PIPE_V | SIMT | 索引写入（UB -> GM，2D-5D） |
| `__builtin_gather_load` | `_gather_load` | VECTOR | PIPE_V | SIMT | 索引收集加载（GM -> UB，1D-5D） |
| `__builtin_scatter_store` | `_scatter_store` | VECTOR | PIPE_V | SIMT | 索引散列存储（UB -> GM，2D-5D） |

### 内置自定义算子与 SIMD 版本的区别

| 维度 | 内置自定义算子（SIMT） | SIMD 版本 |
|------|----------------------|-----------|
| 调用方式 | `al.custom('__builtin_index_select', ...)` | `al.index_select_simd(...)` |
| 执行模式 | SIMT（线程级并行） | SIMD（数据级并行） |
| 输出张量 | 需要通过 `out` 参数预分配 | 自动返回结果张量 |
| 适用场景 | 控制流密集、动态索引 | 数据并行、静态形状 |
| 参数风格 | `bound`/`src_stride`/`dst_stride` 等底层参数 | `src_shape`/`src_offset`/`read_shape` 等高层参数 |

## API 参考

### @register_custom_op 装饰器

```python
def register_custom_op(op) -> type:
    """Register a custom operation so that we can invoke it using al.custom()."""
```

**校验规则：**

1. 被装饰对象必须是类（`inspect.isclass(op)` 为 True）
2. 如果未设置 `name`，使用类名作为算子名
3. 算子名不能重复（不能与已注册的算子同名）
4. 必须设置 `core`（CORE 类型）、`pipe`（PIPE 类型）、`mode`（MODE 类型）
5. 非内置算子（名称不以 `__builtin_` 开头）必须设置 `symbol` 和 `bitcode`

### custom 函数

```python
@builtin
def custom(name: str, *args, _builder=None, **kwargs) -> tl.tensor | tuple[tl.tensor, ...] | None:
    """Invoke a custom operation with the given name and arguments."""
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 算子名称 |
| `*args` | - | 算子输入参数 |
| `out` | `tl.tensor` 或 `list[tl.tensor]` | 输出张量（通过 kwargs 传入） |
| `**kwargs` | - | 其他命名参数 |

### custom_semantic 函数

```python
def custom_semantic(name: str, *args, _builder=None, **kwargs):
    """自定义算子的语义实现，负责 IR 生成。"""
```

### int64 类型包装

```python
class int64(int):
    """For custom op, python int argument will be converted to int32 by default,
    if a device-side int64 is required, you can pass an al.int64(x) to it."""
```

**使用场景：** 自定义算子的 Python int 参数默认转换为 int32。如果设备端需要 int64，使用 `al.int64(x)` 包装。

### __builtin_index_select - SIMT 索引选择

从 GM 张量中按索引选择数据到 UB，支持 2D-5D 源张量。

```python
al.custom('__builtin_index_select',
    src,           # GM 源张量指针
    index,         # UB 索引张量（整数类型，1D 或 2D）
    dim,           # 选择维度（0 <= dim < src_rank）
    bound,         # 索引上界（al.int64）
    end_offset,    # 索引张量各维结束偏移（tuple）
    start_offset,  # 源张量各维起始偏移（tuple）
    src_stride,    # 源张量各维步长（tuple）
    other=None,    # 越界时的默认值（可选）
    out=output     # 输出张量（必需，UB）
)
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | `tensor`（指针） | GM 源张量指针 |
| `index` | `tensor`（整数） | UB 索引张量，1D 或 2D |
| `dim` | `int` | 选择维度，0 <= dim < src_rank |
| `bound` | `al.int64` | 索引上界 |
| `end_offset` | `tuple[int]` | 索引张量各维结束偏移 |
| `start_offset` | `tuple[int]` | 源张量各维起始偏移 |
| `src_stride` | `tuple[int]` | 源张量各维步长 |
| `other` | `scalar` | 越界时的默认值（可选） |
| `out` | `tensor` | 输出张量（必需），dtype 与 src 元素类型一致 |

**约束：**
- `src` 必须是指针类型
- `index` 必须是整数张量，rank 为 1 或 2
- `src` rank 范围：2-5
- `len(start_offset) == len(src_stride)`
- `len(end_offset) == index_rank + len(start_offset) - 1`
- `out.dtype == src.dtype.element_ty`

**参考公式（2D 示例）：**

```
dim=0, index_rank=1:
    out[i][0:B_end-B_begin] = src[index[i]][B_begin:B_end]

dim=0, index_rank=2:
    out[i][j][0:B_end-B_begin] = src[index[i][j]][B_begin:B_end]
```

### __builtin_index_put - SIMT 索引写入

将 UB 中的值按索引写入 GM 目标张量，支持 2D-5D 值张量。

```python
al.custom('__builtin_index_put',
    dst,           # GM 目标张量指针
    index,         # UB 索引张量（整数类型）
    value,         # UB 值张量
    dim,           # 散列维度（0 <= dim < value_rank - 1）
    bound,         # 索引上界（al.int64）
    dst_shape,     # 目标张量形状（tuple[int]）
    dst_offset,    # 目标张量各维偏移（tuple[int]）
    dst_stride     # 目标张量各维步长（tuple[int]）
)
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `dst` | `tensor`（指针） | GM 目标张量指针 |
| `index` | `tensor`（整数） | UB 索引张量 |
| `value` | `tensor` | UB 值张量，rank 2-5 |
| `dim` | `int` | 散列维度，0 <= dim < value_rank - 1 |
| `bound` | `al.int64` | 索引上界 |
| `dst_shape` | `tuple[int]` | 目标张量形状 |
| `dst_offset` | `tuple[int]` | 目标张量各维偏移 |
| `dst_stride` | `tuple[int]` | 目标张量各维步长 |

**约束：**
- `dst` 必须是指针类型
- `index` 必须是整数张量
- `value` rank 范围：2-5
- `len(dst_shape) == len(dst_offset) == len(dst_stride)`

### __builtin_gather_load - SIMT 索引收集加载

从 GM 源缓冲区按索引收集数据到 UB，支持 1D-5D 索引张量。

```python
al.custom('__builtin_gather_load',
    src,           # GM 源缓冲区指针
    index,         # UB 索引张量（整数类型，1D-5D）
    bound,         # 索引上界（al.int64）
    dim,           # 收集维度
    src_stride,    # 源张量各维步长（al.int64）
    index_shape,   # 索引张量形状（tuple[int]）
    offsets,       # 索引张量各维偏移（tuple[int]）
    out=output     # 输出张量（必需，形状与 index 相同）
)
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | `tensor`（指针） | GM 源缓冲区指针 |
| `index` | `tensor`（整数） | UB 索引张量，rank 1-5 |
| `bound` | `al.int64` | 索引上界 |
| `dim` | `int` | 收集维度，0 <= dim < index_rank |
| `src_stride` | `al.int64` | 源张量各维步长 |
| `index_shape` | `tuple[int]` | 索引张量形状 |
| `offsets` | `tuple[int]` | 索引张量各维偏移 |
| `out` | `tensor` | 输出张量（必需），形状与 index 相同 |

**约束：**
- `src` 必须是指针类型
- `index` 必须是整数张量，rank 1-5
- `len(src_stride) == len(index_shape) == len(offsets) == index_rank`
- `out.shape == index.shape`

### __builtin_scatter_store - SIMT 索引散列存储

将 UB 中的值按索引散列存储到 GM 目标缓冲区，支持 2D-5D 索引张量。

```python
al.custom('__builtin_scatter_store',
    dst,           # GM 目标缓冲区指针
    value,         # UB 值张量
    index,         # UB 索引张量（整数类型，1D-5D）
    bound,         # 索引上界（al.int64）
    dim,           # 散列维度
    dst_stride,    # 目标各维步长（al.int64）
    index_shape,   # 索引张量形状（tuple[int]）
    offsets        # 索引各维偏移（tuple[int]）
)
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `dst` | `tensor`（指针） | GM 目标缓冲区指针 |
| `value` | `tensor` | UB 值张量 |
| `index` | `tensor`（整数） | UB 索引张量，rank 1-5 |
| `bound` | `al.int64` | 索引上界 |
| `dim` | `int` | 散列维度，0 <= dim < index_rank |
| `dst_stride` | `al.int64` | 目标各维步长 |
| `index_shape` | `tuple[int]` | 索引张量形状 |
| `offsets` | `tuple[int]` | 索引各维偏移 |

**约束：**
- `dst` 必须是指针类型
- `index` 必须是整数张量，rank 1-5
- `len(dst_stride) == len(index_shape) == len(offsets) == index_rank`

### dtype.cname 属性

注册自定义算子后，`tl.dtype` 对象会获得 `cname` 属性，返回对应的 C 类型名：

| Triton dtype | C name |
|-------------|--------|
| int1 | bool |
| int8 | int8_t |
| int16 | int16_t |
| int32 | int32_t |
| int64 | int64_t |
| uint8 | uint8_t |
| uint16 | uint16_t |
| uint32 | uint32_t |
| uint64 | uint64_t |
| fp16 | half |
| bf16 | bfloat16_t |
| fp32 | float |
| fp64 | double |
| fp8e5 | float8_e5m2_t |
| fp8e4nv | float8_e4m3_t |

## 代码示例

### 示例 1：注册并使用自定义算子

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@al.register_custom_op
class my_elementwise_op:
    name = 'my_elementwise_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = 'my_elementwise_kernel'
    bitcode = '/path/to/my_op.bc'

    def __init__(self, input, output):
        assert input.dtype == output.dtype, "input and output must have same dtype"

@triton.jit
def my_kernel(in_ptr, out_ptr, N):
    x = tl.load(in_ptr + tl.arange(0, 128))
    result = al.custom('my_elementwise_op', x, out=tl.zeros([128], dtype=tl.float32))
    tl.store(out_ptr + tl.arange(0, 128), result)
```

### 示例 2：使用内置 index_select 自定义算子

```python
@triton.jit
def index_select_kernel(src_ptr, index_ptr, out_ptr, N, dim,
                        BLOCK: tl.constexpr):
    index = tl.load(index_ptr + tl.arange(0, BLOCK))
    out = tl.zeros([BLOCK, 16], dtype=tl.float32)

    result = al.custom(
        '__builtin_index_select',
        src_ptr,
        index,
        dim=0,
        bound=al.int64(N),
        end_offset=(BLOCK, 16),
        start_offset=(0, 0),
        src_stride=(N, 1),
        out=out
    )
    tl.store(out_ptr + tl.arange(0, BLOCK)[:, None] * 16 + tl.arange(0, 16)[None, :], result)
```

### 示例 3：带 indexing_map 的自定义算子

```python
@al.register_custom_op
class my_matmul_op:
    name = 'my_matmul_op'
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_M
    mode = al.MODE.SIMD
    symbol = 'my_matmul_kernel'
    bitcode = '/path/to/matmul.bc'

    iterator_types = [
        al.IteratorType.Parallel,
        al.IteratorType.Parallel,
        al.IteratorType.Reduction,
    ]

    def __init__(self, a, b, out):
        assert a.dtype == b.dtype
        assert out.dtype == a.dtype
```

### 示例 4：带 extra_buffers 的自定义算子

```python
@al.register_custom_op
class my_op_with_workspace:
    name = 'my_op_with_workspace'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = 'my_op_kernel'
    bitcode = '/path/to/op.bc'

    extra_buffers = (tl.float32, 1024)

    def __init__(self, input, out):
        pass
```

### 示例 5：带 align_dim 的自定义算子

```python
@al.register_custom_op
class my_aligned_op:
    name = 'my_aligned_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = 'my_aligned_kernel'
    bitcode = '/path/to/aligned_op.bc'
    align_dim = {'input': 16}

    def __init__(self, input, out):
        pass
```

## NPU 适配要点

1. **bitcode 路径必须是绝对路径**：注册自定义算子时，`bitcode` 属性指定的文件路径会被转换为绝对路径，确保文件存在。

2. **内置算子无需 bitcode**：以 `__builtin_` 开头的内置算子不需要 `symbol` 和 `bitcode` 属性，它们的实现由编译器内置提供。

3. **arg_type 动态类型指定**：通过 `self.arg_type` 字典可以为参数指定动态类型，覆盖默认的类型推断。例如 `self.arg_type['src_stride'] = index.dtype`。

4. **输出张量必须通过 out 参数传入**：自定义算子的输出张量通过 `kwargs` 中的 `out` 参数传入，算子的返回类型与输出张量类型一致。

5. **参数类型转换**：`custom_semantic` 会自动处理参数类型转换：
   - `tl.tensor` 直接使用 handle
   - Python `int` 默认转为 int32，使用 `al.int64()` 可转为 int64
   - Python `float` 默认转为 float32
   - `tl.constexpr` 自动解包为值

6. **dtype.cname 用于 bitcode 生成**：`tl.dtype.cname` 属性返回 C 类型名，方便在生成 CANN bitcode 时使用正确的类型声明。

## 常见问题

**Q: 自定义算子的 bitcode 如何生成？**
A: bitcode 是 CANN 自定义算子的编译产物，需要使用 CANN 的算子开发工具链编译 C/C++ 算子代码生成。具体流程请参考 CANN 自定义算子开发文档。

**Q: 内置自定义算子和 Python 层的 mem_ops 有什么区别？**
A: 内置自定义算子（如 `__builtin_index_select`）通过 `al.custom()` 调用，使用 SIMT 模式；Python 层的 `index_select_simd` 等操作使用 SIMD 模式，提供更高级的 API 封装。两者在底层实现和性能特征上有所不同。

**Q: 如何调试自定义算子？**
A: 可以通过设置 `source` 属性指定源码路径，设置 `compile` 属性指定编译选项，帮助调试。此外，`extra_attr` 可以传递额外的编译器属性。

**Q: 自定义算子支持多输出吗？**
A: 支持。通过 `out` 参数传入多个输出张量的列表，返回值是对应的 tuple。

## 相关文档

- [02-pipe-and-core.md](./02-pipe-and-core.md) - PIPE/CORE/MODE 枚举
- [09-vec-ops.md](./09-vec-ops.md) - 向量操作
- [10-mem-ops.md](./10-mem-ops.md) - 内存操作（内置自定义算子的 Python 封装）

## 源码参考

- [custom_op.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/custom_op.py) - 自定义算子注册与调用机制
- [builtin_custom_ops.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/builtin_custom_ops.py) - 内置自定义算子定义
- [core.py: int64](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py#L93-L101) - int64 类型包装
