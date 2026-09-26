# 分块策略

## 概述

Tiling（分块）是 NPU 性能优化中最核心的技术手段。由于 NPU 的片上内存（UB）容量有限（A2 系列为 192KB，910_95/950 为 256KB），算子计算时必须将大数据切分为小块，每次只加载处理其中一部分数据。合理的 Tiling 策略不仅能避免 UB 溢出，还能实现存算并行，最大化流水线效率。本文档详细描述 Tiling 的概念、约束、计算方法和代码示例。

## 关键概念

| 概念 | 说明 | 约束/典型值 |
|------|------|-------------|
| Tiling | 将大数据切分为适配片上内存的小块进行处理 | 必须满足 UB 容量和对齐约束 |
| BLOCK_SIZE | 一次处理的数据元素数量（一级分块） | 受 UB 容量限制 |
| BLOCK_SIZE_SUB | 二级分块大小，用于核内 for 循环进一步切分 | 用于避免 UB 溢出 |
| UB 容量 | Vector Core 的片上存储空间 | A2: 192KB (1572864 bits), 910_95/950: 256KB |
| 32B 对齐 | Vector 算子场景下，Tensor 尾轴大小须被 32Bytes 整除 | VV 类算子必须满足 |
| 512B 对齐 | CV 融合算子场景下，Tensor 尾轴大小须被 512Bytes 整除 | CV 类算子必须满足 |
| multiBuffer | 双缓冲机制，为同一张量创建 2 个副本实现存算并行 | 开启后 UB 可用空间减半 |
| 存算并行 | MTE2 搬运与 Vector 计算重叠执行 | 依赖合理的 Tiling 和 for 循环 |

## UB 大小约束

### 各型号 UB 容量

| 芯片型号 | UB 大小 | 开启 multiBuffer 后可用空间 | 对应 bits |
|----------|---------|---------------------------|-----------|
| Ascend910B (A2) | 192 KB | 96 KB | 1572864 bits |
| Ascend910_95 | 256 KB | 128 KB | 2097152 bits |
| Ascend950 | 256 KB | 128 KB | 2097152 bits |

源码参考：[utils.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/runtime/utils.py#L28-L61)

### UB 溢出错误

当单次分块的数据量超过 UB 容量时，编译器会报错：

```
E [ConvertLinalgRToBinary] encounters error:
E loc("xxx.ttadapter.mlir":2:1): error: ub overflow, requires 3072256 bits while 1572864 bits available!
(possible reason: large or block number is more than what user expect due to multi-buffer
 feature is enabled and some ops need extra local buffer.)
```

### UB 空间计算

UB 空间需要容纳所有在片上同时存在的张量，包括输入、输出、中间结果和 multiBuffer 副本：

```
所需 UB 空间 = (输入张量数 + 输出张量数 + 中间张量数) * BLOCK_SIZE * dtype_bytes * multiBuffer副本数
```

例如，一个 float16 的 add_kernel，有 2 个输入 + 1 个输出，开启 multiBuffer（2 副本）：
```
所需空间 = 3 * BLOCK_SIZE * 2 * 2 = 12 * BLOCK_SIZE bytes
A2 上限 = 192 * 1024 = 196608 bytes
最大 BLOCK_SIZE = 196608 / 12 = 16384 elements
```

## 对齐要求

### Vector 算子（VV 类）- 32B 对齐

当算子仅使用 Vector Core 计算时，UB 要求 Tensor 的尾轴大小能被 32Bytes 整除：

```python
# float16 (2 bytes): 尾轴元素数须为 16 的倍数
# float32 (4 bytes): 尾轴元素数须为 8 的倍数
# int8    (1 byte):  尾轴元素数须为 32 的倍数

# 正确示例：BLOCK_SIZE = 1024, float16
# 1024 * 2 = 2048 bytes, 2048 % 32 == 0 ✓

# 错误示例：BLOCK_SIZE = 100, float16
# 100 * 2 = 200 bytes, 200 % 32 != 0 ✗
# 硬件会自动补齐到 224 bytes (7*32)，浪费空间且影响性能
```

### CV 融合算子 - 512B 对齐

当算子同时使用 Cube Core 和 Vector Core 时，要求 Tensor 的尾轴大小能被 512Bytes 整除：

```python
# float16 (2 bytes): 尾轴元素数须为 256 的倍数
# float32 (4 bytes): 尾轴元素数须为 128 的倍数

# 矩阵乘法场景：HEAD_DIM 须为 256 的倍数（float16）
# BLOCK_N = 256  # 256 * 2 = 512 bytes ✓
```

### 对齐对性能的影响

| 场景 | Shape | 问题 | 优化方法 |
|------|-------|------|----------|
| 短轴不对齐 | (2048, 3) bf16 | 尾轴 3*2=6B，需补齐到 32B | 借轴转置 |
| 单元素轴 | (2048, 1) bf16 | 尾轴 1*2=2B，需补齐到 32B | 1D load + reshape |
| 离散访存 | stride=(12832, 128) | 非连续访问导致额外搬运 | 调整 stride 为 (32, 1) |

### 借轴转置技巧

适用于 `tensor.numel() % 256Byte == 0` 的场景：

```python
# conv_state = tensor([2048, 3], bfloat16)
# 问题：尾轴 3 不满足 32B 对齐

# 解决：当成 1D tensor load，避免自动补齐
conv_state = tl.load(conv_state_ptr + conv_batch_offs * conv_batch_stride
                     + doffs * 3 + tl.arange(0, 2048 * 3))

# 长轴(2048)裂出一根对齐轴(16)借给短轴(3)，让两个轴都对齐
conv_state_T = conv_state.reshape(128, 16 * 3).trans().reshape(16, 3 * 128).trans().reshape(3 * 2048,)
```

## 存算并行

### 原理

存算并行是 NPU 性能优化的核心手段，通过让数据搬运（MTE2）和计算（Vector/Cube）重叠执行，消除空闲等待时间。

```
存算串行（低效）：
MTE2:    |==搬入==|              |==搬入==|
Compute:          |==计算==|              |==计算==|
MTE3:                    |==写出==|              |==写出==|

存算并行（高效）：
MTE2:    |==搬入1==|==搬入2==|==搬入3==|
Compute:           |==计算1==|==计算2==|==计算3==|
MTE3:                      |==写出1==|==写出2==|
```

### 实现条件

编译器默认配置 `multiBuffer=True`，默认支持存算并行。但以下条件需同时满足：

1. **for 循环 Tiling**：算子内必须有多次迭代，才能形成"搬入+计算"的流水线
2. **无数据依赖**：当前迭代的计算不依赖下一迭代搬入的数据
3. **UB 空间充足**：multiBuffer 需要额外 UB 空间存储双缓冲副本

### multiBuffer 失效的常见原因

| 原因 | 说明 | 解决方法 |
|------|------|----------|
| 数据搬运和计算存在依赖 | 必须依赖 Vector 运算后才能触发 MTE 搬运 | 使用 care_padding=False 减少同步 |
| 无多个数据加载 | 算子内无 Tiling 切分，单次执行完成 | 添加 for 循环实现 Tiling |
| UB 空间不足 | multiBuffer 需额外 UB 空间 | 减小 BLOCK_SIZE_SUB |

## 分块配置计算方法

### 一级分块（BLOCK_SIZE）

一级分块决定了每个核处理的总数据量，影响 grid 分核数：

```python
# 计算公式
NUM_BLOCKS = ceil(N / BLOCK_SIZE)
coreDim = NUM_BLOCKS

# 约束
# 1. coreDim <= 65535 (UINT16_MAX)
# 2. 推荐对齐到物理核数
# 3. BLOCK_SIZE 应为 2 的幂次
```

### 二级分块（BLOCK_SIZE_SUB）

二级分块决定了 for 循环内每次迭代处理的数据量，影响 UB 使用量：

```python
# 计算公式
num_sub_blocks = ceil(BLOCK_SIZE / BLOCK_SIZE_SUB)

# 约束
# 1. 单次迭代 UB 使用量 <= UB 容量 / multiBuffer副本数
# 2. BLOCK_SIZE_SUB 应满足对齐要求
# 3. BLOCK_SIZE_SUB 应为 2 的幂次
```

### UB 空间估算

```python
def estimate_ub_usage(num_tensors, block_size, dtype_bytes, multibuffer=True):
    """估算单次迭代的 UB 使用量"""
    num_copies = 2 if multibuffer else 1
    ub_per_tensor = block_size * dtype_bytes
    total_ub = num_tensors * ub_per_tensor * num_copies
    return total_ub

# 示例：masked_fill 算子
# 输入: inp(float16) + mask(int8) + value(float16) + 输出: out(float16)
# num_tensors ≈ 4 (含中间结果)
# BLOCK_SIZE_SUB = 1024, dtype_bytes = 2
# ub_usage = 4 * 1024 * 2 * 2 = 16384 bytes ≈ 16KB (远小于 192KB)
```

## 代码示例

### 示例1：简单 Vector 算子的分块

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n, BLOCK_SIZE)
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)
```

### 示例2：二级分块避免 UB 溢出

```python
@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N,
                       BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE

    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)

    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N

        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)

        tl.store(out + offsets, input_vals, mask=mask)

        value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        overwrite_vals = tl.where(fill_mask_vals, value_to_write, input_vals)
        tl.store(out + offsets, overwrite_vals, mask=mask)
```

### 示例3：矩阵乘法的分块策略

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
            other=0.0
        )
        b = tl.load(
            B + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)

    c = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### 示例4：多核任务分配的分块

```python
import torch_npu
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]

NUM_CORE = vectorcore_num
grid = (NUM_CORE,)

@triton.jit
def _attn_fwd(Q, K, V, M, Out, acc, scale,
              stride_qz, stride_qh,
              Z: tl.constexpr, H: tl.constexpr,
              N_CTX: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr):
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)

    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
```

## NPU 适配要点

1. **UB 容量是硬约束**：分块大小必须确保不超出 UB 容量，开启 multiBuffer 后可用空间减半
2. **对齐是性能关键**：32B/512B 对齐要求必须满足，否则硬件自动补齐会浪费空间和带宽
3. **for 循环是存算并行的前提**：没有 for 循环就无法形成流水线，multiBuffer 无法使能
4. **分核数应对齐物理核数**：避免多轮调度开销，在核内通过 for 循环处理多块数据
5. **BLOCK_SIZE 参数应为 2 的幂次**：autotune 的候选值必须是 2 的幂次

## 常见问题 (Q&A)

**Q1: 出现 "ub overflow" 错误怎么办？**

A: 减小 BLOCK_SIZE 或引入 BLOCK_SIZE_SUB 进行二级分块。计算公式：`所需空间 = 张量数 * BLOCK_SIZE_SUB * dtype_bytes * multiBuffer副本数 <= UB容量`。

**Q2: BLOCK_SIZE 和 BLOCK_SIZE_SUB 应该如何选择？**

A: BLOCK_SIZE 决定每个核处理的总数据量，应尽量大以减少核启动开销，但需保证 `coreDim = ceil(N/BLOCK_SIZE) <= 65535`。BLOCK_SIZE_SUB 决定 for 循环内每次迭代的数据量，应在 UB 容量允许范围内尽量大，以最大化存算并行效率。

**Q3: 为什么开启 multiBuffer 后性能反而下降？**

A: multiBuffer 会将 UB 可用空间减半，如果原本 UB 空间就紧张，开启后可能导致分块过小，反而增加搬运次数。此时可以尝试减小 BLOCK_SIZE_SUB 或关闭 multiBuffer。

**Q4: 如何判断当前的 Tiling 是否合理？**

A: 使用 msprof 采集性能数据，查看 `aiv_mte2_ratio` 和 `aiv_vec_ratio`。如果 MTE2 占比过高（>50%）且 Vector 占比低，说明 Tiling 可能过小，搬运开销大。

## 相关文档

- [01-optimization-overview.md](./01-optimization-overview.md) - 优化策略总览
- [03-autotune-guide.md](./03-autotune-guide.md) - Autotune 自动搜索最优分块
- [06-data-movement-optimization.md](./06-data-movement-optimization.md) - 数据搬运优化

### 源码参考

- [programming_guide.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/programming_guide.md) - Tiling 优化章节
- [tile_generator.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/runtime/tile_generator.py) - 分块配置生成器
- [utils.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/runtime/utils.py) - UB 容量和硬件参数
