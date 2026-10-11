# 数据搬运优化

## 概述

NPU 的数据搬运是影响算子性能的关键因素。与 GPU 不同，NPU 的数据通路具有明确的层级结构（GM -> L1 -> L0C / GM -> UB），每条通路有独立的延迟特征和带宽约束。数据搬运优化的核心目标是最大化搬运与计算的重叠度（存算并行），减少搬运等待时间，并合理利用各级缓存。本文档详细描述 NPU 数据通路的特征、存算并行的实现方法、MultiBuffer 双缓冲机制、for 循环中的 Tiling 策略以及 L1 缓存的利用方法。

## 关键概念

| 概念 | 说明 | 关键参数 |
|------|------|----------|
| GM (Global Memory) | 全局内存，HBM，容量大但延迟高 | 带宽约 1.8TB/s |
| UB (Unified Buffer) | Vector Core 片上存储，数据搬入/计算/搬出的工作区 | A2: 192KB, 910_95: 256KB |
| L1 Buffer | Cube Core 片上存储，用于矩阵乘法数据缓存 | 与芯片型号相关 |
| L0C | Cube Core 累加器输出缓冲区 | 固定大小 |
| MTE2 | GM -> UB 的搬运流水线 | 与 MTE3 共享 GM 带宽 |
| MTE3 | UB -> GM 的搬运流水线 | 与 MTE2 共享 GM 带宽 |
| MTE1 | GM -> L1 的搬运流水线 | 独立带宽 |
| 存算并行 | MTE2 搬运与 Vector/Cube 计算重叠执行 | 依赖 multiBuffer 和 for 循环 |
| MultiBuffer | 双缓冲机制，为同一张量创建 2 个副本 | A2/A3 默认开启，910_95 默认关闭；开启后 UB 可用空间减半 |
| ping-pong | 双缓冲的工作模式，交替使用两个缓冲区 | buffer0 搬入 + buffer1 计算 |

## NPU 数据通路的延迟特征

### 内存层级与延迟

```
延迟从低到高：
L0C < UB < L1 < GM
|    |    |    |
|    |    |    +-- HBM，~100ns 量级延迟，~1.8TB/s 带宽
|    |    +------- 片上 SRAM，~10ns 量级延迟
|    +------------ 片上 SRAM，~5ns 量级延迟
+----------------- 寄存器级，~1ns 量级延迟
```

### 搬运流水线

| 流水线 | 方向 | 典型带宽 | 说明 |
|--------|------|----------|------|
| MTE2 | GM -> UB | ~1.8 TB/s (共享) | Vector 数据搬入 |
| MTE3 | UB -> GM | ~1.8 TB/s (共享) | Vector 数据写出 |
| MTE1 | GM -> L1 | 独立带宽 | Cube 数据搬入 |
| FIX (fixpipe) | L0C -> UB（910_95）/ L0C -> GM/L1（A2/A3） | 片内带宽 | Cube 结果传递给 Vector（910_95）或输出到 GM/L1 |

> 注意：MTE2 和 MTE3 同时搬运时会共享 GM 带宽，实际带宽约为峰值的一半。

### 理论搬运耗时计算

```python
# 搬运理论耗时 = 搬运数据量 / 理论带宽
# 示例：float16, 4096*4096 矩阵搬运
data_size = 2 * 4096 * 4096  # 32MB (float16 = 2 bytes)
bandwidth = 1.8e12  # 1.8 TB/s
latency = data_size / bandwidth  # ≈ 17.8 us

# MTE2 + MTE3 同时搬运时
# 总耗时 = (MTE2搬运量 + MTE3搬运量) / GM带宽
```

## 存算并行（MTE2 搬运与 M 计算重叠）

### 原理

存算并行是 NPU 数据搬运优化的核心手段。通过合理设计 Tiling 策略，使得在当前批次数据计算过程中，能够提前准备下一阶段所需的数据，实现数据搬运与计算过程的并行化。

### 实现方式

Triton-Ascend 支持两种数据处理模式：

| 模式 | 执行流程 | 效率 |
|------|----------|------|
| 存算串行 | 搬入 -> 计算 -> 搬出 -> 搬入 -> 计算 -> 搬出 | 低，存在空闲等待 |
| 存算并行 | 搬入1/计算0/搬出-1 -> 搬入2/计算1/搬出0 -> ... | 高，流水线重叠 |

编译器默认配置 `multiBuffer=True`（A2/A3 系列）或 `multiBuffer=False`（910_95 系列），默认支持存算并行（A2/A3）。

### 存算并行的前提条件

1. **for 循环 Tiling**：算子内必须有多次迭代，才能形成流水线
2. **无数据依赖**：当前迭代的计算不依赖下一迭代搬入的数据
3. **UB 空间充足**：multiBuffer 需要额外 UB 空间

### 存算并行失效的常见原因

| 原因 | 详细说明 | 解决方法 |
|------|----------|----------|
| 数据搬运和计算存在依赖 | Vector 运算后才能触发 MTE 搬运 | 使用 care_padding=False 减少同步 |
| 无 Tiling 切分 | 算子内单次执行完成，无 for 循环 | 添加 for 循环实现 Tiling |
| UB 空间不足 | multiBuffer 需额外 UB 空间 | 减小 BLOCK_SIZE_SUB |

## MultiBuffer（ping-pong 双缓冲）

### 概述

MultiBuffer 是存算并行的底层实现机制，为同一张量创建 2 个副本（ping 和 pong），交替使用两个缓冲区实现搬运与计算的重叠。

源码参考：[multibuffer.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/triton_api/Extension_Ops/multibuffer.md)

### 工作原理

```
无 MultiBuffer（串行）：
UB:     |==搬入数据==|==计算==|==写出==|==搬入数据==|==计算==|==写出==|

有 MultiBuffer（并行）：
Buffer0:|==搬入数据0==|           |==计算0==|==写出0==|
Buffer1:              |==搬入数据1==|           |==计算1==|==写出1==|
                      ↑ 搬入与计算重叠
```

### API 使用

```python
from triton.language.extra.cann import extension as al

@triton.jit
def triton_compile_hint(in_ptr0, out_ptr0, xnumel,
                        XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        # 为 tmp0 设置双缓冲
        al.multibuffer(tmp0, 2)
        tmp2 = tmp0
        tl.store(out_ptr0 + (xindex), tmp2, xmask)
```

### multibuffer 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| src | tensor | 需要进行多缓冲设置的源张量 |
| size | int / constexpr | 要创建的缓冲区副本数量（当前仅支持 2） |

### MultiBuffer 对 UB 空间的影响

| 配置 | UB 可用空间 | 说明 |
|------|------------|------|
| multiBuffer=False | 192KB (A2) / 256KB (910_95) | 单缓冲，全部 UB 可用 |
| multiBuffer=True | 96KB (A2) / 128KB (910_95) | 双缓冲，UB 可用空间减半 |

> 注意：开启 multiBuffer 后 UB 可用空间减半，可能导致原本不溢出的分块配置变得溢出，需要相应减小 BLOCK_SIZE_SUB。

### AutoTilingTuner 中的 MultiBuffer 处理

AutoTilingTuner 会自动为每个候选配置生成 multiBuffer 开/关的变体：

```python
# 基础配置
Config({'BLOCK_SIZE': 1024}, num_warps=1, num_stages=1)

# 自动生成的 multiBuffer 变体
Config({'BLOCK_SIZE': 1024, 'multibuffer': True}, num_warps=1, num_stages=1)   # 默认
Config({'BLOCK_SIZE': 1024, 'multibuffer': False}, num_warps=1, num_stages=1)  # 对比
```

## for 循环中的 Tiling 策略

### 为什么 for 循环是存算并行的前提

没有 for 循环时，算子执行流程是"搬入 -> 计算 -> 搬出"的串行模式，multiBuffer 无法使能。添加 for 循环后，可以将数据切分为多个子块，形成"搬入子块N+1 / 计算子块N / 写出子块N-1"的流水线。

### 优化示例

#### 优化前：单次处理，无存算并行

```python
@triton.jit
def alloc_extend_kernel(
        pre_lens_ptr, seq_lens_ptr, free_page_ptr, out_indices,
        bs_upper: tl.constexpr, page_size: tl.constexpr,
        max_num_extend_tokens: tl.constexpr,
):
    pid = tl.program_id(0)
    # ... 省略部分逻辑 ...

    # 单次加载所有数据，无 Tiling，无法存算并行
    offset_many_page = tl.arange(0, max_num_extend_tokens)
    page_start = tl.load(
        free_page_ptr + new_page_start_loc + offset_many_page // page_size,
        mask=offset_many_page < num_part2,
    )
    tl.store(
        out_indices + output_start_loc + offset_many_page,
        page_start * page_size + offset_many_page % page_size,
        mask=offset_many_page < num_part2,
    )
```

#### 优化后：for 循环 Tiling，实现存算并行

```python
@triton.jit
def alloc_extend_kernel(
        pre_lens_ptr, seq_lens_ptr, free_page_ptr, out_indices,
        bs_upper: tl.constexpr, page_size: tl.constexpr,
        max_num_extend_tokens: tl.constexpr,
        BLOCK_SIZE: tl.constexpr = 1024,  # 新增分块参数
):
    pid = tl.program_id(0)
    # ... 省略部分逻辑 ...

    # 使用 for 循环分块处理，实现存算并行
    num_loop = tl.cdiv(max_num_extend_tokens, BLOCK_SIZE)
    blk_offset = tl.arange(0, BLOCK_SIZE)
    for i in range(num_loop):
        offset_many_page = blk_offset + i * BLOCK_SIZE
        page_start = tl.load(
            free_page_ptr + new_page_start_loc + offset_many_page // page_size,
            mask=offset_many_page < num_part2,
        )
        tl.store(
            out_indices + output_start_loc + offset_many_page,
            page_start * page_size + offset_many_page % page_size,
            mask=offset_many_page < num_part2,
        )
```

### for 循环 Tiling 的注意事项

1. **数学等价性**：增加 Tiling 后需确保计算结果与原始逻辑等价
2. **BLOCK_SIZE_SUB 选择**：应在 UB 容量允许范围内尽量大，以减少循环次数
3. **mask 处理**：每个子块都需要独立的 mask 检查
4. **循环变量类型**：确保循环变量和偏移量计算使用正确的整数类型

## L1 缓存利用

### 概述

L1 Buffer 是 Cube Core 的片上存储，用于缓存矩阵乘法的输入分块。合理利用 L1 可以减少 GM 访问次数，提升矩阵乘法性能。

### 数据通路

```
矩阵乘法数据流：
GM --MTE1--> L1 --Cube--> L0C --fixpipe--> UB --MTE3--> GM
             (缓存A/B分块)  (矩阵乘)  (结果传递)  (写出)
```

### L1 利用策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| 缓存 A 矩阵分块 | 将 A 的分块缓存在 L1，减少重复搬运 | K 维度循环，A 分块复用 |
| 缓存 B 矩阵分块 | 将 B 的分块缓存在 L1，减少重复搬运 | K 维度循环，B 分块复用 |
| 双缓冲 L1 | 为 L1 中的数据创建双缓冲 | 大矩阵乘法 |

### copy_from_ub_to_l1 / copy

```python
from triton.language.extra.cann import extension as al

# 将数据从 UB 拷贝到 L1
al.copy(src_ub_tensor, dst_l1_buffer)

# 已弃用：al.copy_from_ub_to_l1(src, dst)
```

## 代码示例

### 示例1：先搬运到 UB 再 select

在 NPU 的离散场景下，先搬运大量数据到 UB，再从 UB 中 select 目标值，比多次小批量通过 L2 搬运更高效：

```python
@triton.jit
def pick_kernel(x_ptr, idx_ptr, y_ptr, stride_x, stride_idx, stride_y,
               M: tl.constexpr, N: tl.constexpr):
    pid = tl.program_id(0)
    rn = tl.arange(0, N)
    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

    # 优化前：离散地址加载，多次小批量 L2 搬运
    # val = tl.load(x_ptr + idx * stride_x, mask=mask)

    # 优化后：先整体搬运到 UB，再 select
    rm = tl.arange(0, M)
    x_shared = tl.load(x_ptr + rm * stride_x)  # [M]
    val = tl.gather(x_shared, idx, 0)

    tl.store(y_ptr + rn * stride_y, val, mask=mask)
```

### 示例2：存算并行 + care_padding

```python
@triton.jit
def optimized_kernel(input_ptr, output_ptr, n,
                     BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE
    num_sub_blocks = BLOCK_SIZE // BLOCK_SIZE_SUB

    for sub_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < n

        # care_padding=False 减少同步 + for 循环实现存算并行
        data = tl.load(input_ptr + offsets, mask=mask, care_padding=False)
        result = data * 2.0 + 1.0
        tl.store(output_ptr + offsets, result, mask=mask)
```

### 示例3：矩阵乘法中的 L1 利用

```python
@triton.jit
def matmul_with_l1(A, B, C, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn,
                   stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K 维度循环，A/B 分块可利用 L1 缓存
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

## NPU 适配要点

1. **存算并行是首要优化**：通过 for 循环 Tiling + multiBuffer 实现搬运与计算重叠
2. **MTE2/MTE3 共享带宽**：同时读写 GM 时，带宽约为峰值的一半
3. **multiBuffer 会减半 UB 可用空间**：需相应调整 BLOCK_SIZE_SUB
4. **先搬后选优于离散加载**：大量数据整体搬运到 UB 再 select，比多次小批量离散加载更高效
5. **对齐影响搬运效率**：32B/512B 对齐是高效搬运的前提
6. **L1 缓存适合矩阵乘法**：矩阵乘法的 K 维度循环中，A/B 分块可缓存在 L1

## 常见问题 (Q&A)

**Q1: 如何判断存算并行是否生效？**

A: 使用 msprof 采集性能数据，查看 `aiv_mte2_time` 和 `aiv_vec_time`。如果 MTE2 和 Vector 时间有显著重叠（总耗时远小于两者之和），说明存算并行已生效。也可以使用仿真流水图直观查看流水线重叠情况。

**Q2: multiBuffer 开启后 UB 溢出怎么办？**

A: 减小 BLOCK_SIZE_SUB，或者关闭 multiBuffer（牺牲存算并行）。也可以通过 autotune 自动搜索最优配置。

**Q3: 为什么 for 循环 Tiling 后性能反而下降？**

A: 可能的原因：(1) BLOCK_SIZE_SUB 过小，循环开销大；(2) 循环内逻辑复杂，编译器无法有效优化；(3) 数据依赖导致无法并行。建议通过 msprof 分析具体瓶颈。

**Q4: MTE2 搬运时间过长怎么办？**

A: (1) 检查 Tiling 是否过小，导致发射冗余搬运指令；(2) 检查是否存在非对齐访存；(3) 使用 care_padding=False 减少同步开销；(4) 考虑先整体搬运再 select 的策略。

## 相关文档

- [01-optimization-overview.md](./01-optimization-overview.md) - 优化策略总览
- [02-tiling-strategy.md](./02-tiling-strategy.md) - 分块策略详解
- [04-care-padding.md](./04-care-padding.md) - care_padding 优化
- [07-profiling-guide.md](./07-profiling-guide.md) - 性能分析与瓶颈定位

### 源码参考

- [programming_guide.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/programming_guide.md) - 存算并行章节
- [performance_guidelines.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/migration_guide/performance_guidelines.md) - 指令并行优化
- [multibuffer.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/triton_api/Extension_Ops/multibuffer.md) - MultiBuffer API 文档
