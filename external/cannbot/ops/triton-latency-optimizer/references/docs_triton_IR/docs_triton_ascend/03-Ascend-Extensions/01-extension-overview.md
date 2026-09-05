# Ascend 扩展 API 总览与导入方式

## 概述

Triton-Ascend 在标准 OpenAI Triton 语言的基础上，提供了一套 Ascend 扩展 API，用于暴露华为昇腾 NPU 的硬件特性和高级功能。这些扩展 API 是 triton-ascend 最重要的特有功能，涵盖了硬件枚举定义、Cube-Vector 协同计算、同步操作、自定义算子、Buffer 编程模型、向量操作和内存操作等领域。

标准 Triton API 提供了通用的 GPU 编程抽象，而 Ascend 扩展 API 则针对昇腾 NPU 的独特架构（如 Cube/Vector 双核、L0C/L1/UB 多级存储、NZ 数据排布等）提供了底层控制能力。当开发者需要利用这些硬件特性来获得极致性能时，就需要使用 Ascend 扩展 API。

## 导入方式

Ascend 扩展 API 通过以下方式导入：

```python
from triton.language.extra.cann.extension import *
```

或按需导入特定符号：

```python
from triton.language.extra.cann.extension import (
    PIPE, CORE, MODE, IteratorType,
    fixpipe, sync_block_set, sync_block_wait, sync_block_all,
    sub_vec_id, sub_vec_num,
    custom, register_custom_op,
    parallel, compile_hint, multibuffer,
    insert_slice, extract_slice, get_element, sort, flip, cast,
    index_put, gather_out_to_ub, scatter_ub_to_out, index_select_simd,
)
```

Buffer 编程模型相关 API 通过以下方式导入：

```python
import triton.extension.buffer.language as bl
```

## 扩展 API 与标准 Triton API 的关系

| 维度 | 标准 Triton API | Ascend 扩展 API |
|------|----------------|----------------|
| 编程模型 | 基于 Tensor 的 SPMD 模型 | Tensor + Buffer 双模型 |
| 硬件抽象 | 通用 GPU（SM、Shared Memory） | 昇腾 NPU（Cube/Vector、UB/L1/L0C） |
| 同步机制 | `tl.debug_barrier` | `sync_block_set/wait/all`（Cube-Vector 间） |
| 数据搬运 | `tl.load/store` | `fixpipe`、`copy`、Buffer `alloc/to_buffer/to_tensor` |
| 类型转换 | `tl.cast` | 扩展 `cast`（支持 `overflow_mode`、`fp_downcast_rounding`） |
| 索引操作 | `tl.load` + 指针算术 | `index_put`、`gather_out_to_ub`、`scatter_ub_to_out`、`index_select_simd` |
| 自定义算子 | 不支持 | `@register_custom_op` + `al.custom()` |

## 何时使用扩展 API

1. **Cube-Vector 协同计算**：当需要同时利用 Cube（矩阵计算单元）和 Vector（向量计算单元）时，必须使用 `scope`、`sync_block_set/wait` 等扩展 API。
2. **矩阵乘法后处理**：使用 `fixpipe` 将 Cube 计算结果从 L0C 搬运到 UB（仅 910_95），实现零拷贝的量化/ReLU 融合。
3. **精细内存控制**：使用 Buffer 模型直接操作 UB/L1 等片上存储，避免不必要的 GM 访问。
4. **SIMD gather/scatter**：使用 `gather_out_to_ub`/`scatter_ub_to_out` 实现高效的索引读写。
5. **编译器优化提示**：使用 `compile_hint`、`multibuffer` 指导编译器进行流水线优化。
6. **自定义算子集成**：使用 `register_custom_op` 注册 CANN 自定义算子。

## 完整 API 列表

### 核心枚举与常量

| API | 类别 | 说明 |
|-----|------|------|
| `PIPE` | 枚举 | 流水线类型（PIPE_S/PIPE_V/PIPE_M/PIPE_MTE1/PIPE_MTE2/PIPE_MTE3/PIPE_FIX/PIPE_ALL） |
| `CORE` | 枚举 | 核心类型（CORE_VECTOR/CORE_CUBE/CUBE_OR_VECTOR/CUBE_AND_VECTOR） |
| `MODE` | 枚举 | 执行模式（MODE.SIMD/MODE.SIMT/MODE.MIX） |
| `IteratorType` | 枚举 | 迭代器类型（Parallel/Broadcast/Transpose/Reduction/Interleave/Deinterleave/Inverse/Pad/Concat/Gather/Cumulative/Opaque） |
| `SYNC_IN_VF` | 枚举 | Vector 内部同步模式（VV_ALL/VST_VLD/VLD_VST/VST_VST/VS_ALL/VST_LD/VLD_ST/VST_ST/SV_ALL/ST_VLD/LD_VST/ST_VST） |
| `int64` | 类型包装 | 用于自定义算子中指定 int64 参数 |
| `ascend_address_space` | 地址空间组 | NPU 缓冲区地址空间（UB/L1/L0C/GM 等） |
| `builtin` | 装饰器 | 标记函数为 Ascend 内建函数（`@builtin`），确保 `_builder` 参数正确传递 |
| `is_builtin` | 函数 | 检查函数是否为已注册的 Ascend 内建函数 |

### 同步操作

| API | 类别 | 说明 |
|-----|------|------|
| `sync_block_set` | 函数 | 生产者核心发送同步信号 |
| `sync_block_wait` | 函数 | 消费者核心等待同步信号 |
| `sync_block_all` | 函数 | 全局屏障同步 |
| `debug_barrier` | 函数 | Vector 内部流水线同步（SYNC_IN_VF） |
| `SYNC_IN_VF` | 枚举 | Vector 内部同步模式 |

### Cube-Vector 协同

| API | 类别 | 说明 |
|-----|------|------|
| `scope` | 上下文管理器 | 指定代码块运行在 Cube 或 Vector 核心 |
| `sub_vec_id` | 函数 | 获取子向量核 ID |
| `sub_vec_num` | 函数 | 获取子向量核数量 |

### fixpipe 操作

| API | 类别 | 说明 |
|-----|------|------|
| `fixpipe` | 函数 | L0C 到 UB 的数据搬运（支持 NZ 格式转换，仅 910_95） |
| `FixpipeDMAMode` | 枚举 | DMA 传输模式（NZ2ND/NZ2DN/NZ2NZ） |
| `FixpipeDualDstMode` | 枚举 | 双目的模式（NO_DUAL/COLUMN_SPLIT/ROW_SPLIT） |
| `FixpipePreQuantMode` | 枚举 | 预量化模式（NO_QUANT/F322BF16/F322F16/S322I8） |
| `FixpipePreReluMode` | 枚举 | 预 ReLU 模式（NO_RELU/NORMAL_RELU/LEAKY_RELU/P_RELU） |

### 内存拷贝

| API | 类别 | 说明 |
|-----|------|------|
| `copy` | 函数 | UB 到 UB/L1 的数据拷贝 |
| `copy_from_ub_to_l1` | 函数 | UB 到 L1 的数据拷贝（已废弃，使用 copy 替代） |

### 自定义算子

| API | 类别 | 说明 |
|-----|------|------|
| `custom` | 函数 | 调用自定义算子 |
| `register_custom_op` | 装饰器 | 注册自定义算子类 |
| `custom_semantic` | 函数 | 自定义算子语义实现 |

### 辅助操作

| API | 类别 | 说明 |
|-----|------|------|
| `parallel` | 迭代器 | 并行范围声明（支持 `bind_sub_block`） |
| `compile_hint` | 函数 | 编译器提示标注 |
| `multibuffer` | 函数 | 多缓冲设置（ping-pong 流水线） |

### 向量操作

| API | 类别 | 说明 |
|-----|------|------|
| `insert_slice` | 函数 | 将子张量插入到张量中 |
| `extract_slice` | 函数 | 从张量中提取子张量 |
| `get_element` | 函数 | 获取张量中的单个元素 |
| `sort` | 函数 | 沿指定维度排序 |
| `flip` | 函数 | 沿指定维度翻转 |
| `cast` | 函数 | 类型转换（扩展版，支持 overflow_mode） |

### 内存操作

| API | 类别 | 说明 |
|-----|------|------|
| `index_put` | 函数 | 索引写入（UB 到 GM） |
| `gather_out_to_ub` | 函数 | 索引读取（GM 到 UB） |
| `scatter_ub_to_out` | 函数 | 索引散列写入（UB 到 GM） |
| `index_select_simd` | 函数 | SIMD 并行索引选择（GM 到 UB） |

### 数学操作

| API | 类别 | 说明 |
|-----|------|------|
| `atan2` | 函数 | 反正切（双参数） |
| `isfinited` | 函数 | 有限值检测（fp64/fp32/fp16/bf16） |
| `finitef` | 函数 | 有限值检测（仅 fp32） |

### libdevice 数学函数库

`libdevice` 是独立的数学函数库，通过 `from triton.language.extra.cann import libdevice` 导入，提供 HMF 硬件加速的高精度数学函数。详见 [11-libdevice.md](./11-libdevice.md)。

| API | 类别 | 说明 |
|-----|------|------|
| `libdevice.pow` | 函数 | 幂运算（HMF 硬件加速，高精度） |
| `libdevice.tanh` | 函数 | 双曲正切（覆盖 tl.math.tanh） |
| `libdevice.tan` | 函数 | 正切 |
| `libdevice.atan` | 函数 | 反正切 |
| `libdevice.atan2` | 函数 | 双参数反正切 |
| `libdevice.log1p` | 函数 | log(1+x)，x 接近 0 时数值稳定 |
| `libdevice.expm1` | 函数 | e^x-1，x 接近 0 时数值稳定 |
| `libdevice.acos` | 函数 | 反余弦 |
| `libdevice.asin` | 函数 | 反正弦 |
| `libdevice.sinh` | 函数 | 双曲正弦 |
| `libdevice.cosh` | 函数 | 双曲余弦 |
| `libdevice.erfinv` | 函数 | 逆误差函数 |
| `libdevice.relu` | 函数 | ReLU 激活函数 |
| `libdevice.reciprocal` | 函数 | 倒数 1/x |
| `libdevice.fast_dividef` | 函数 | 快速浮点除法 |
| `libdevice.fast_expf` | 函数 | 快速指数运算 |
| `libdevice.div_rz` | 函数 | 向零截断除法 |

### Buffer 编程模型

| API | 类别 | 说明 |
|-----|------|------|
| `bl.address_space` | 基类 | 缓冲区地址空间基类 |
| `bl.buffer_type` | 类型 | 缓冲区类型（元素类型+形状+地址空间+步长） |
| `bl.buffer` | 数据结构 | 缓冲区对象 |
| `bl.alloc` | 函数 | 分配缓冲区 |
| `bl.to_buffer` | 函数 | Tensor 转 Buffer |
| `bl.to_tensor` | 函数 | Buffer 转 Tensor |
| `bl.subview` | 函数 | 创建缓冲区子视图 |

### MLIR Affine 工具

| API | 类别 | 说明 |
|-----|------|------|
| `affine_expr` | 类 | Affine 表达式基类 |
| `affine_constant_expr` | 类 | Affine 常量表达式 |
| `affine_dim_expr` | 类 | Affine 维度表达式 |
| `affine_symbol_expr` | 类 | Affine 符号表达式 |
| `affine_binary_op_expr` | 类 | Affine 二元操作表达式 |
| `affine_map` | 类 | Affine 映射 |

## 相关文档

- [02-pipe-and-core.md](./02-pipe-and-core.md) - PIPE 枚举与 CORE 枚举详解
- [03-fixpipe.md](./03-fixpipe.md) - fixpipe 操作详解
- [04-sync-operations.md](./04-sync-operations.md) - 同步操作详解
- [07-buffer-model.md](./07-buffer-model.md) - Buffer 编程模型详解
- [11-libdevice.md](./11-libdevice.md) - libdevice 数学函数库详解

## 源码参考

- [__init__.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/__init__.py) - 扩展模块入口，所有 API 的导出定义
- [core.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py) - 核心枚举与基础操作
- [buffer/core.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/extension/buffer/language/core.py) - Buffer 编程模型核心定义
