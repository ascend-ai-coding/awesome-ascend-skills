# PIPE 枚举与 CORE 枚举

## 概述

昇腾 NPU 采用独特的 Cube-Vector 分离架构，每个 AI Core 包含一个 Cube 核心（矩阵计算单元）和多个 Vector 核心（向量计算单元）。不同的计算操作在不同的硬件流水线（PIPE）上执行，而 PIPE 和 CORE 枚举正是用来标注操作所属的硬件单元和数据通路。

这些枚举在自定义算子注册（`register_custom_op`）、同步操作（`sync_block_set/wait`）和编译器代码生成中起关键作用。

## 关键概念

### PIPE 枚举

PIPE 枚举定义了昇腾 NPU 上的硬件流水线类型，每种流水线对应不同的硬件执行单元和数据通路。

| 枚举值 | 硬件单元 | 数据通路 | 说明 |
|--------|----------|----------|------|
| `PIPE_S` | Scalar 标量单元 | 标量寄存器 | 标量运算，如整数加减、比较、分支 |
| `PIPE_V` | Vector 向量单元 | Vector 计算流水线 | 向量运算，如逐元素加减乘除、激活函数 |
| `PIPE_M` | Cube 矩阵单元 | Cube 计算流水线 | 矩阵乘法（MatMul） |
| `PIPE_MTE1` | MTE1 搬运引擎 | GM -> L1 | 从 Global Memory 搬运数据到 L1 缓存 |
| `PIPE_MTE2` | MTE2 搬运引擎 | GM -> UB | 从 Global Memory 搬运数据到 Unified Buffer |
| `PIPE_MTE3` | MTE3 搬运引擎 | UB -> GM | 从 Unified Buffer 搬运数据到 Global Memory |
| `PIPE_FIX` | Fixpipe 流水线 | L0C -> GM/L1/UB | Cube 结果从 L0C 搬运到 GM/L1，或 UB（仅 910_95） |
| `PIPE_ALL` | 全部流水线 | - | 表示操作涉及所有流水线 |

### CORE 枚举

CORE 枚举定义了操作执行的核心类型。

| 枚举值 | 说明 | 使用场景 |
|--------|------|----------|
| `CORE_VECTOR` | 在 Vector 核心上执行 | 向量运算、逐元素操作 |
| `CORE_CUBE` | 在 Cube 核心上执行 | 矩阵乘法 |
| `CORE_CUBE_OR_VECTOR` | 可在 Cube 或 Vector 上执行 | 编译器自动选择 |
| `CORE_CUBE_AND_VECTOR` | 需要同时使用 Cube 和 Vector | Cube-Vector 融合算子 |

### MODE 枚举

MODE 枚举定义了操作的执行模式。

| 枚举值 | 说明 | 特点 |
|--------|------|------|
| `MODE.SIMD` | 单指令多数据模式 | 向量化并行，适合数据并行操作 |
| `MODE.SIMT` | 单指令多线程模式 | 线程级并行，适合控制流密集操作 |
| `MODE.MIX` | 混合模式 | SIMD 与 SIMT 混合 |

### IteratorType 枚举

IteratorType 枚举定义了循环迭代器的语义类型，用于自定义算子的 `indexing_map` 中标注各维度的迭代方式。

| 枚举值 | 说明 |
|--------|------|
| `IteratorType.Parallel` | 并行迭代维度 |
| `IteratorType.Broadcast` | 广播维度 |
| `IteratorType.Transpose` | 转置维度 |
| `IteratorType.Reduction` | 归约维度 |
| `IteratorType.Interleave` | 交错维度 |
| `IteratorType.Deinterleave` | 解交错维度 |
| `IteratorType.Inverse` | 逆序维度 |
| `IteratorType.Pad` | 填充维度 |
| `IteratorType.Concat` | 拼接维度 |
| `IteratorType.Gather` | 收集维度 |
| `IteratorType.Cumulative` | 累积维度 |
| `IteratorType.Opaque` | 不透明维度 |

## API 参考

### PIPE

```python
class PIPE(enum.Enum):
    PIPE_S = ...      # 标量流水线
    PIPE_V = ...      # 向量流水线
    PIPE_M = ...      # 矩阵流水线
    PIPE_MTE1 = ...   # GM -> L1 搬运
    PIPE_MTE2 = ...   # GM -> UB 搬运
    PIPE_MTE3 = ...   # UB -> GM 搬运
    PIPE_FIX = ...    # L0C -> GM/L1/UB 搬运（fixpipe，L0C->UB 仅 910_95）
    PIPE_ALL = ...    # 全部流水线
```

### CORE

```python
class CORE(enum.Enum):
    VECTOR = ...           # Vector 核心
    CUBE = ...             # Cube 核心
    CUBE_OR_VECTOR = ...   # Cube 或 Vector
    CUBE_AND_VECTOR = ...  # Cube 和 Vector
```

### MODE

```python
class MODE(enum.Enum):
    SIMD = ...   # 单指令多数据
    SIMT = ...   # 单指令多线程
    MIX = ...    # 混合模式
```

### IteratorType

```python
class IteratorType(enum.Enum):
    Parallel = ...      # 并行
    Broadcast = ...     # 广播
    Transpose = ...     # 转置
    Reduction = ...     # 归约
    Interleave = ...    # 交错
    Deinterleave = ...  # 解交错
    Inverse = ...       # 逆序
    Pad = ...           # 填充
    Concat = ...        # 拼接
    Gather = ...        # 收集
    Cumulative = ...    # 累积
    Opaque = ...        # 不透明
```

## 代码示例

### 示例 1：在自定义算子中使用 PIPE/CORE/MODE

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@al.register_custom_op
class my_vector_op:
    name = 'my_vector_op'
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMT

    def __init__(self, input, output):
        pass

@triton.jit
def my_kernel(in_ptr, out_ptr, N):
    x = tl.load(in_ptr + tl.arange(0, 128))
    result = al.custom('my_vector_op', x, out=tl.zeros([128], dtype=tl.float32))
    tl.store(out_ptr + tl.arange(0, 128), result)
```

### 示例 2：在同步操作中使用 PIPE

```python
@triton.jit
def cube_vector_sync_kernel():
    with al.scope(core_mode="cube"):
        result = tl.dot(a, b)
        al.sync_block_set("cube", "vector", 0,
                          sender_pipe=al.PIPE.PIPE_FIX,
                          receiver_pipe=al.PIPE.PIPE_MTE2)

    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0,
                           sender_pipe=al.PIPE.PIPE_FIX,
                           receiver_pipe=al.PIPE.PIPE_MTE2)
        processed = result * 2.0
```

### 示例 3：IteratorType 在自定义算子中的使用

```python
@al.register_custom_op
class my_matmul_reduce:
    name = 'my_matmul_reduce'
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_M
    mode = al.MODE.SIMD
    iterator_types = [al.IteratorType.Parallel, al.IteratorType.Parallel, al.IteratorType.Reduction]
    indexing_map = al.affine_map(
        ...
    )
```

## NPU 适配要点

1. **PIPE 选择对性能至关重要**：错误的 PIPE 标注会导致编译器生成低效代码或编译失败。向量操作必须标注 `PIPE_V`，矩阵操作必须标注 `PIPE_M`。

2. **数据通路理解**：理解昇腾 NPU 的数据通路是正确使用 PIPE 枚举的基础：
   - GM（Global Memory）是片外存储，容量大但延迟高
   - L1 是片上缓存，Cube 专用
   - UB（Unified Buffer）是片上存储，Vector 专用
   - L0C 是 Cube 的输出缓冲区

3. **同步操作中的 PIPE 默认值**：在 `sync_block_set/wait` 中，如果不指定 `sender_pipe` 和 `receiver_pipe`，系统会根据 sender/receiver 类型自动选择默认值：
   - cube -> vector：sender_pipe=PIPE_FIX, receiver_pipe=PIPE_MTE2
   - vector -> cube：sender_pipe=PIPE_MTE3, receiver_pipe=PIPE_MTE2

4. **CORE_CUBE_AND_VECTOR 用于融合算子**：当自定义算子需要同时使用 Cube 和 Vector 时，使用 `CORE_CUBE_AND_VECTOR`，此时需要配合 `scope` 和同步操作使用。

## 常见问题

**Q: PIPE_M 和 PIPE_FIX 有什么区别？**
A: `PIPE_M` 是 Cube 矩阵计算流水线，执行矩阵乘法运算；`PIPE_FIX` 是 fixpipe 流水线，负责将 Cube 计算结果从 L0C 搬运到 GM/L1/UB。在 A2/A3 上 fixpipe 仅支持 L0C -> GM/L1，910_95 上额外支持 L0C -> UB。两者是 Cube 侧的不同阶段。

**Q: 什么时候使用 MODE.SIMT 而不是 MODE.SIMD？**
A: 当操作涉及复杂的控制流（如条件分支、动态索引）时使用 SIMT 模式；当操作是纯数据并行（如逐元素运算）时使用 SIMD 模式。SIMD 模式通常性能更高。

**Q: CORE_CUBE_OR_VECTOR 和 CORE_CUBE_AND_VECTOR 的区别？**
A: `CUBE_OR_VECTOR` 表示操作可以在 Cube 或 Vector 上执行，编译器会自动选择；`CUBE_AND_VECTOR` 表示操作需要同时使用两种核心，通常用于 Cube-Vector 融合算子。

## 相关文档

- [01-extension-overview.md](./01-extension-overview.md) - 扩展 API 总览
- [03-fixpipe.md](./03-fixpipe.md) - fixpipe 操作（PIPE_FIX 详解）
- [04-sync-operations.md](./04-sync-operations.md) - 同步操作（PIPE 在同步中的使用）
- [06-custom-op.md](./06-custom-op.md) - 自定义算子（CORE/PIPE/MODE 在注册中的使用）

## 源码参考

- [core.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py) - PIPE/CORE/MODE/IteratorType 枚举定义
- [semantic.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/semantic.py) - PIPE 枚举的语义层定义
