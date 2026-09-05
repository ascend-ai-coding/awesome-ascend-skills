# 自定义算子示例（Custom Op Example）

## 概述

Triton-Ascend 提供了自定义算子（Custom Op）机制，允许开发者将 NPU 原生 CCE 算子集成到 Triton kernel 中。通过 `triton.language.extra.cann.extension` 模块，可以注册自定义算子、声明编译属性、指定 bitcode 来源，并在 JIT kernel 中调用。

| 关键概念 | 说明 |
|---------|------|
| `@al.register_custom_op` | 注册自定义算子类的装饰器 |
| `al.custom()` | 在 kernel 中调用自定义算子 |
| `@al.builtin` | 将自定义算子包装为内置操作的装饰器 |
| `core / pipe / mode` | 算子的计算核心类型、流水线阶段、执行模式 |
| `symbol` | CCE 实现函数的符号名 |
| `bitcode / source / compile` | 算子实现的提供方式 |
| `indexing_map` | 自定义算子的仿射映射，描述数据访问模式 |
| `extra_buffers` | 声明算子需要的额外设备缓冲区（scratch buffer） |

## 自定义算子完整示例

### 基本自定义算子注册与调用

```python
import os
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@al.register_custom_op
class min_custom_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_MTE2
    mode = al.MODE.SIMD

    symbol = 'min_custom_op_impl'
    bitcode = os.path.abspath(__file__)


@al.register_custom_op
class simple_custom_op:
    name = 'simple_custom_op'

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    symbol = 'simple_custom_op_impl'
    bitcode = os.path.abspath(__file__)

    def __init__(self, x, y, dim=0, out=None):
        assert x.shape == y.shape, "x and y should have same shape"
        assert isinstance(dim, int), "dim should be const integer"
        assert out, "out is required"


@al.register_custom_op
class _example_custom_op:
    name = 'example_custom_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    symbol = 'example_custom_op_impl'
    bitcode = os.path.abspath(__file__)

    def __init__(self, src, index, offset: al.int64, axis, out=None):
        assert isinstance(src, tl.tensor), "src should be tensor"
        assert index.dtype.is_int(), "index should be integer tensor"
        assert isinstance(offset, int), "offset should be integer"
        assert isinstance(axis, int), "axis should be integer"
        assert isinstance(out, tuple) and len(out) == 2, "out should be tuple of 2 items"

        rank = len(index.shape)
        self.symbol = f"{self.name}_{rank}d_{src.dtype.cname}_{index.dtype.cname}"

        self.source = f"workspace/example_custom_op_impl.cce"
        self.compile = "bisheng -O2 -std=c++17 -o $@ -c $<"

        self.arg_type['axis'] = index.dtype


@al.builtin
def example_op(src, index, offset, axis, _builder=None):
    x = tl.semantic.full(src.shape, 0, tl.float32, _builder)
    y = tl.semantic.full(index.shape, 0, tl.float32, _builder)
    return al.custom_semantic(_example_custom_op.name,
        src, index, offset, axis, out=(x, y), _builder=_builder)


@triton.jit
def my_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)
    y = tl.load(y_ptr + i, mask=i < n)
    y = al.custom("min_custom_op", x, x_ptr, y_ptr + i, al.int64(0), (1, 2, 3), [4.1, 5.2], out=y)
    y = al.custom("simple_custom_op", x, y, dim=1, out=y)
    index = tl.full((2, 3), 0, tl.int64)
    x, y = al.custom("example_custom_op", x, index, offset=1, axis=0, out=(x, y))
    result, _ = example_op(x, index, offset=2, axis=1)
    tl.store(out_ptr + i, result, mask=i < n)
```

## Bitcode 提供方式

自定义算子的实现可以通过三种方式提供：

### 方式一：bitcode 文件

```python
@al.register_custom_op
class my_op:
    symbol = 'my_op_impl'
    bitcode = os.path.abspath(__file__)
```

`bitcode` 指向包含 CCE 实现的目标文件路径。编译器会从该文件中查找 `symbol` 对应的函数实现。

### 方式二：源码编译

```python
@al.register_custom_op
class my_op:
    symbol = 'my_op_impl'
    source = "workspace/my_op_impl.cce"
    compile = "bisheng -O2 -std=c++17 -o $@ -c $<"
```

- `source`：CCE 源码文件路径
- `compile`：编译命令模板，`$@` 为输出文件，`$<` 为输入文件

### 方式三：动态 symbol

在 `__init__` 中根据参数动态设置 symbol：

```python
def __init__(self, src, index, offset, axis, out=None):
    rank = len(index.shape)
    self.symbol = f"{self.name}_{rank}d_{src.dtype.cname}_{index.dtype.cname}"
```

## 自定义算子注册流程

### 必需属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `core` | `al.CORE` | 计算核心类型：`VECTOR` 或 `CUBE` |
| `pipe` | `al.PIPE` | 流水线阶段 |
| `mode` | `al.MODE` | 执行模式：`SIMD` 或 `SIMT` |
| `symbol` | str | CCE 实现函数的符号名 |
| `bitcode` | str | 实现文件路径 |

### 可选属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `name` | str | 算子名称，默认使用类名 |
| `indexing_map` | list | 仿射映射列表，描述输入/输出数据访问模式 |
| `extra_buffers` | list | 额外设备缓冲区声明 |
| `source` | str | CCE 源码文件路径 |
| `compile` | str | 编译命令模板 |
| `arg_type` | dict | 动态参数类型映射 |

### CORE 类型

| 值 | 说明 |
|----|------|
| `al.CORE.VECTOR` | Vector 计算核心 |
| `al.CORE.CUBE` | Cube 矩阵计算核心 |

### PIPE 类型

| 值 | 说明 |
|----|------|
| `al.PIPE.PIPE_MTE2` | 内存到 Vector 核心数据搬运 |
| `al.PIPE.PIPE_V` | Vector 计算流水线 |
| `al.PIPE.PIPE_M` | Cube 矩阵乘流水线 |
| `al.PIPE.PIPE_FIX` | Fixpipe 后处理流水线 |

### MODE 类型

| 值 | 说明 |
|----|------|
| `al.MODE.SIMD` | 单指令多数据模式 |
| `al.MODE.SIMT` | 单指令多线程模式 |

## 自定义算子的 Indexing Map

Indexing Map 使用仿射映射（Affine Map）描述自定义算子的数据访问模式：

```python
from triton._C.libtriton.ascend import ir as ascend_ir


def _make_indexing_maps():
    d0 = ascend_ir.affine_expr.get_dim(0)
    d1 = ascend_ir.affine_expr.get_dim(1)
    c8 = ascend_ir.affine_expr.get_constant(8)

    in0 = ascend_ir.affine_map.get(2, 0, [d1, d0])
    in1 = ascend_ir.affine_map.get(2, 0, [d0, d1])
    out = ascend_ir.affine_map.get(2, 0, [d0.floordiv(c8), d1.mod(c8)])
    return [in0, in1, out]


@al.register_custom_op
class complex_indexing_map_custom_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = "complex_indexing_map_custom"
    bitcode = os.path.abspath(__file__)

    def __init__(self, x, y, out=None):
        assert out is not None
        self.indexing_map = _make_indexing_maps()
```

### Affine Map 组合

```python
def _compose_indexing_maps():
    d0 = ascend_ir.affine_expr.get_dim(0)
    d1 = ascend_ir.affine_expr.get_dim(1)
    c4 = ascend_ir.affine_expr.get_constant(4)

    perm = ascend_ir.affine_map.get_permutation([1, 0])
    tile = ascend_ir.affine_map.get(2, 0, [d0.floordiv(c4), d1.mod(c4)])
    out = tile.compose(perm)

    in0 = ascend_ir.affine_map.get_identity(2)
    in1 = perm
    return [in0, in1, out]
```

## 额外缓冲区（Extra Buffers）

自定义算子可以声明需要的额外设备缓冲区（scratch buffer）：

```python
SCRATCH_SPEC = [
    (tl.float32, 1024),
    (tl.bfloat16, 512),
    (tl.int32, 256),
]


@al.register_custom_op
class demo_extra_buffer_op:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT
    symbol = "demo_extra_buffer_op_impl"
    bitcode = os.path.abspath(__file__)

    def __init__(self, x, out=None):
        self.indexing_map = [al.affine_map.get_identity(1)]
        self.extra_buffers = list(SCRATCH_SPEC)
```

`extra_buffers` 列表中的每个元素为 `(dtype, element_count)` 元组，声明了算子运行时需要的临时缓冲区大小和类型。这些信息会被编码到 HIVM MLIR 的 `extra_buffers_sizes` 属性中。

## 内置自定义算子（Builtin Custom Ops）

Triton-Ascend 提供了一系列内置自定义算子，通过 `al.custom()` 调用：

```python
@triton.jit
def builtin_ops_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)
    y = tl.load(y_ptr + i, mask=i < n)
    index = tl.full([8], 0, tl.int32)
    value = tl.full([8, 64], 0, tl.float32)
    tmp = tl.full([8], 0, tl.float32)

    x = al.custom("__builtin_index_select",
                   x_ptr, index,
                   dim=0, bound=100,
                   end_offset=(2, 2), start_offset=(0, 0),
                   src_stride=(4, 1), out=x)

    al.custom("__builtin_index_put",
              x_ptr, index, value,
              dim=0, bound=12,
              dst_shape=(1, 2, 3),
              dst_offset=(4, 5, 6),
              dst_stride=(8, 4, 1))

    tmp = al.custom("__builtin_gather_load",
              y_ptr, index,
              bound=100, dim=0,
              src_stride=(1,),
              index_shape=(3,),
              offsets=(0,),
              out=tmp)

    al.custom("__builtin_scatter_store",
              out_ptr, value, index,
              1, 0, (1, ), (2, ), (1, ))

    y = al.custom("__builtin_indirect_load", x_ptr, index, mask=i < n, other=y, out=y)
    al.custom("__builtin_indirect_store", out_ptr, index, value)
    tl.store(out_ptr + i, y, mask=i < n)
```

### 内置算子列表

| 算子名 | 功能 |
|--------|------|
| `__builtin_index_select` | 按索引选择元素 |
| `__builtin_index_put` | 按索引放置元素 |
| `__builtin_gather_load` | 聚合加载 |
| `__builtin_scatter_store` | 散射存储 |
| `__builtin_indirect_load` | 间接加载 |
| `__builtin_indirect_store` | 间接存储 |

## 与标准 Triton 操作的组合使用

自定义算子可以与标准 Triton 操作自由组合：

```python
@triton.jit
def combined_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)
    y = tl.load(y_ptr + i, mask=i < n)

    result = x + y
    result = al.custom("my_custom_op", result, out=result)
    result = tl.where(result > 0, result, 0)

    tl.store(out_ptr + i, result, mask=i < n)
```

## 常见问题（Q&A）

**Q: 自定义算子编译报错 "symbol not found"？**

A: 检查 `bitcode` 路径是否正确，以及 `symbol` 名称是否与 CCE 实现中的函数名一致。

**Q: 如何调试自定义算子的 MLIR 输出？**

A: 使用 `ASTSource` 和 `ttir_to_linalg` 手动编译并查看 IR：

```python
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions, ttir_to_linalg

src = ASTSource(my_kernel, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}, {"BLOCK": 256})
context = ir.context()
ir.load_dialects(context)
ascend_ir.load_dialects(context)
options = NPUOptions()
ttir = ast_to_ttir(my_kernel, src, context, options, {}, {})
print(ttir)
linalg = ttir_to_linalg(ttir, {**options.__dict__}, options, named_ops=True)
print(linalg)
```

**Q: extra_buffers 的缓冲区大小如何确定？**

A: 缓冲区大小取决于算子实现的需求。在 `__init__` 中通过 `self.extra_buffers` 声明，运行时由编译器分配。大小信息会编码到 HIVM MLIR 的 `extra_buffers_sizes` 属性中。

**Q: 自定义算子能否在循环中使用？**

A: 可以，但需要注意 NPU 上某些原子操作（`atomic_or/atomic_xor/atomic_and/atomic_xchg/atomic_cas`）暂不支持在 loop 中使用。

## 相关文档

- [03-matmul.md](./03-matmul.md) - 矩阵乘法模式（含 compile_hint）
- [01-api-support-matrix.md](../09-Reference/01-api-support-matrix.md) - API 支持矩阵
- 源码参考：[custom_op_demo.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/examples/custom_op/custom_op_demo.py)
- 源码参考：[builtin_ops_demo.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/examples/custom_op/builtin_ops_demo.py)
- 源码参考：[custom_op_extra_buffer_demo.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/examples/custom_op/custom_op_extra_buffer_demo.py)
- 源码参考：[custom_op_indexing_map_complex_demo.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/examples/custom_op/custom_op_indexing_map_complex_demo.py)
