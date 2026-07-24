# fixpipe 操作

## 概述

fixpipe 是昇腾 NPU 上的一个关键硬件流水线，负责将 Cube（矩阵计算单元）的计算结果从 L0C 缓冲区搬运到其他存储层级，同时可选地执行格式转换、量化和 ReLU 激活等后处理操作。在 910_95 系列上，fixpipe 支持将数据直接从 L0C 搬运到 UB（Unified Buffer），实现 Cube 计算到 Vector 后处理的零拷贝流水线，是高性能矩阵乘法算子的核心组件。在 A2/A3 系列上，fixpipe 仅支持 L0C 到 GM/L1 的搬运。

在 Triton-Ascend 中，`fixpipe` 函数将 L0C 上的 tensor 数据直接搬运到 UB 上的 buffer（仅 910_95 系列支持），支持 NZ 到 ND 的布局转换、双目的输出、预量化和预 ReLU 等功能。

## 关键概念

### 数据通路

```
Cube 计算 (PIPE_M) → L0C → fixpipe (PIPE_FIX) → UB → Vector 后处理 (PIPE_V)
```

- **L0C**：Cube 的输出缓冲区，数据以 NZ（Narrow Z）格式存储
- **UB**：Unified Buffer，Vector 的输入/输出缓冲区
- **fixpipe**：连接 L0C 和 UB 的硬件流水线，支持格式转换和融合后处理

### FixpipeDMAMode - DMA 传输模式

| 枚举值 | 说明 | 源格式 | 目标格式 |
|--------|------|--------|----------|
| `FixpipeDMAMode.NZ2ND` | NZ 到 ND 转换（默认） | NZ（分块 Z 格式） | ND（密集格式） |
| `FixpipeDMAMode.NZ2DN` | NZ 到 DN 转换 | NZ | DN 格式 |
| `FixpipeDMAMode.NZ2NZ` | NZ 到 NZ（无格式转换） | NZ | NZ |

### FixpipeDualDstMode - 双目的输出模式

| 枚举值 | 说明 |
|--------|------|
| `FixpipeDualDstMode.NO_DUAL` | 单目的输出（默认） |
| `FixpipeDualDstMode.COLUMN_SPLIT` | 列分割双输出 |
| `FixpipeDualDstMode.ROW_SPLIT` | 行分割双输出 |

### FixpipePreQuantMode - 预量化模式

| 枚举值 | 说明 |
|--------|------|
| `FixpipePreQuantMode.NO_QUANT` | 不做量化（默认） |
| `FixpipePreQuantMode.F322BF16` | float32 转 bfloat16 |
| `FixpipePreQuantMode.F322F16` | float32 转 float16 |
| `FixpipePreQuantMode.S322I8` | int32 转 int8 |

### FixpipePreReluMode - 预 ReLU 模式

| 枚举值 | 说明 |
|--------|------|
| `FixpipePreReluMode.NO_RELU` | 不做 ReLU（默认） |
| `FixpipePreReluMode.NORMAL_RELU` | 标准 ReLU |
| `FixpipePreReluMode.LEAKY_RELU` | Leaky ReLU |
| `FixpipePreReluMode.P_RELU` | P-ReLU |

## API 参考

### fixpipe

```python
@builtin
def fixpipe(
    src: tl.tensor,
    dst: bl.buffer,
    dma_mode: FixpipeDMAMode = FixpipeDMAMode.NZ2ND,
    dual_dst_mode: FixpipeDualDstMode = FixpipeDualDstMode.NO_DUAL,
    _builder=None,
) -> None
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `src` | `tl.tensor` | 必需 | 源张量，必须位于 L0C 内存区域 |
| `dst` | `bl.buffer` | 必需 | 目标缓冲区，必须位于 UB 内存区域 |
| `dma_mode` | `FixpipeDMAMode` | `NZ2ND` | DMA 传输模式 |
| `dual_dst_mode` | `FixpipeDualDstMode` | `NO_DUAL` | 双目的输出模式 |

**对齐约束：**

| 数据位宽 | 条件 | 对齐要求 |
|----------|------|----------|
| 32 位（float32/int32） | 最后一维 | 对齐到 8 |
| 32 位（非 NZ2ND 模式） | 最后一维 | 对齐到 16 |
| 32 位（COLUMN_SPLIT 模式） | 最后一维 | 对齐到 32 |
| 32 位（NZ2DN 模式） | 第一维 | 对齐到 8 |
| 16 位（float16/int16/bfloat16） | 最后一维 | 对齐到 16 |
| 16 位（NZ2DN 模式） | 第一维 | 对齐到 16 |

**平台限制：** fixpipe 仅在 Ascend 910_95 系列上支持。

## 使用场景

### 场景 1：矩阵乘法后处理

矩阵乘法（`tl.dot`）的结果存储在 L0C 中，需要通过 fixpipe 搬运到 UB 才能进行后续的向量操作（如加 bias、激活函数等）。

### 场景 2：量化融合

通过 `FixpipePreQuantMode`，可以在数据搬运的同时完成 float32 到 bfloat16/float16/int8 的量化，避免额外的转换开销。

### 场景 3：ReLU 融合

通过 `FixpipePreReluMode`，可以在数据搬运的同时完成 ReLU 激活，实现计算与搬运的流水线重叠。

## 代码示例

### 示例 1：基本 fixpipe 使用

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl

@triton.jit
def matmul_fixpipe_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    ub_buf = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], al.ascend_address_space.UB)
    al.fixpipe(acc, ub_buf, dma_mode=al.FixpipeDMAMode.NZ2ND)

    acc_tensor = bl.to_tensor(ub_buf)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc_tensor)
```

### 示例 2：fixpipe 带量化融合

```python
@triton.jit
def matmul_quant_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + ...)
        b = tl.load(b_ptr + ...)
        acc = tl.dot(a, b, acc)

    ub_buf = bl.alloc(tl.bfloat16, [BLOCK_M, BLOCK_N], al.ascend_address_space.UB)
    al.fixpipe(acc, ub_buf, dma_mode=al.FixpipeDMAMode.NZ2ND)
```

### 示例 3：Cube-Vector 协同中的 fixpipe

```python
@triton.jit
def cube_vector_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
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
        acc_tensor = bl.to_tensor(ub_buf)
        result = tl.sigmoid(acc_tensor)
        tl.store(c_ptr + ..., result)
```

## NPU 适配要点

1. **仅限 Ascend 910_95**：fixpipe 当前仅在 Ascend 910_95 系列上支持，其他平台调用会抛出 `RuntimeError`。

2. **src 必须是 tensor，dst 必须是 buffer**：fixpipe 的源数据必须是 `tl.tensor`（L0C 上的数据），目标必须是 `bl.buffer`（UB 上的缓冲区）。

3. **dst 必须在 UB 地址空间**：目标缓冲区的地址空间必须是 `ascend_address_space.UB`，否则会抛出 `TypeError`。

4. **对齐要求严格**：不同位宽和 DMA 模式有不同的对齐要求，违反对齐约束会导致运行时错误。

5. **NZ 格式理解**：Cube 输出的数据以 NZ（Narrow Z）格式存储在 L0C 中，NZ 是昇腾特有的分块数据排布格式。`NZ2ND` 模式将 NZ 转换为标准的 ND（密集）格式，是最常用的模式。

## 常见问题

**Q: fixpipe 和直接 load/store 有什么区别？**
A: fixpipe 是硬件级的数据搬运流水线，直接从 L0C 读取数据并写入 UB，同时支持格式转换和融合后处理。直接 load/store 需要经过 GM 中转，效率较低。

**Q: 什么时候使用 NZ2ND vs NZ2NZ？**
A: 如果后续 Vector 操作需要 ND 格式（大多数情况），使用 `NZ2ND`。如果后续操作可以接受 NZ 格式（如某些特殊的 Cube-Vector 融合场景），使用 `NZ2NZ` 可以避免格式转换开销。

**Q: dual_dst_mode 是什么用途？**
A: 双目的模式允许 fixpipe 将一个 Cube 结果拆分为两个输出，分别写入不同的 UB 缓冲区。`COLUMN_SPLIT` 按列拆分，`ROW_SPLIT` 按行拆分。这在某些融合算子中可以减少数据搬运次数。

**Q: 为什么 fixpipe 的预量化和预 ReLU 在 Python API 中没有暴露？**
A: 当前 Python 层的 `fixpipe` 函数固定使用 `NO_QUANT` 和 `NO_RELU`，这些高级功能通过内部语义层调用。未来版本可能会开放这些参数。

## 相关文档

- [02-pipe-and-core.md](./02-pipe-and-core.md) - PIPE 枚举（PIPE_FIX 详解）
- [04-sync-operations.md](./04-sync-operations.md) - 同步操作（fixpipe 后的 Cube-Vector 同步）
- [07-buffer-model.md](./07-buffer-model.md) - Buffer 编程模型（UB 缓冲区分配）

## 源码参考

- [core.py: fixpipe 函数](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py#L273-L333) - fixpipe 函数定义及对齐校验
- [core.py: FixpipeDMAMode 等枚举](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py#L247-L270) - DMA 模式、双目的模式、预量化模式、预 ReLU 模式枚举
- [semantic.py: fixpipe 语义](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/semantic.py#L132-L148) - fixpipe 的 IR 生成
