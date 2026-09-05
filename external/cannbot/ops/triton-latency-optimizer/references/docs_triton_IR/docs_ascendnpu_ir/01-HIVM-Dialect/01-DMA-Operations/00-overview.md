# DMA 操作总览

> 关键词：DMA、数据搬运、Pipeline、Address Space、MTE2、MTE3、FIX

## 概述

DMA（Direct Memory Access）操作是 HIVM 方言中最核心的操作类别之一，负责在 Ascend NPU 的不同内存层级之间搬运数据。DMA 操作直接映射到硬件的 DMA 引擎，通过不同的 Pipeline 执行，是实现高效数据流的关键。

Ascend NPU 的内存层级和 DMA 操作构成了一个层次化的数据通路，理解这些通路对于编写高性能内核至关重要。

## 内存层级与数据通路

### 内存层级

```
+----------------------------------------------------------+
|                         GM (Global Memory)                |
|                    片外全局内存，大容量高延迟               |
+----------------------------------------------------------+
          |                                    ^
          | MTE2 (Load/ND2NZ)                  | MTE3 (Store/NZ2ND)
          v                                    |
+----------------------------------------------------------+
|                         L1 (Cube Buffer)                  |
|                    片上缓存，Cube 专用                      |
+----------------------------------------------------------+
          |                                    ^
          | MTE1 (L12UB)                       |
          v                                    |
+----------------------------------------------------------+
|                         UB (Unified Buffer)               |
|                    统一缓冲区，Vector 专用                   |
+----------------------------------------------------------+
          |                                    ^
          |                                    | V (Copy UB->UB)
          v                                    v
+----------------------------------------------------------+
|                         UB (Unified Buffer)               |
+----------------------------------------------------------+

+----------------------------------------------------------+
|                         L0A / L0B                         |
|              Cube 矩阵输入缓存（由 MMAD 隐式使用）          |
+----------------------------------------------------------+
          ^
          | M (MMAD 写入 L0C)
          |
+----------------------------------------------------------+
|                         L0C                                |
|              Cube 矩阵累加器缓存                           |
+----------------------------------------------------------+
          |
          | FIX (Fixpipe)
          v
+----------------------------------------------------------+
|                    GM / UB / L1                            |
|              Fixpipe 输出目标                              |
+----------------------------------------------------------+
```

### 数据通路图

```
                    GM
                   / | \
                 /   |   \
        MTE2   /     |     \   MTE2
        Load  /      |      \  ND2NZ
             v       |       v
            UB      L1       L1
            |        ^       ^
            | Copy   |       |
            v        | MTE3  |
            UB ------+       |
            |               /
     MTE3   |             /
     Store  |           / Copy (Ascend950)
            v         v
            GM       UB
            

L0C ---- FIX (Fixpipe) ----> GM / UB / L1
         支持随路量化/激活
```

## Pipeline 归属表

每个 DMA 操作都归属于特定的硬件 Pipeline，这决定了操作的执行单元和同步方式。

| 操作 | Pipeline | 枚举值 | 说明 |
|------|----------|--------|------|
| `hir.load` | MTE2 | `PIPE_MTE2` | GM 到本地缓冲区的数据加载 |
| `hir.nd2nz` | MTE2 | `PIPE_MTE2` | GM 到 L1 的 ND 到 NZ 布局转换加载 |
| `hir.store` | MTE3 | `PIPE_MTE3` | 本地缓冲区到 GM 的数据存储 |
| `hir.nz2nd` | MTE3 | `PIPE_MTE3` | L1 到 GM 的 NZ 到 ND 布局转换存储 |
| `hir.l12ub` | MTE1 | `PIPE_MTE1` | L1 到 UB 的数据搬运 |
| `hir.fixpipe` | FIX | `PIPE_FIX` | L0C 到其他层级的搬运 |
| `hir.copy` | 动态 | 取决于 src/dst | 根据源和目标地址空间动态确定 |
| `hir.atomic_cas` | 无固定 | - | 原子比较并交换 |
| `hir.atomic_xchg` | 无固定 | - | 原子交换 |

### Copy 操作的动态 Pipeline

`hir.copy` 操作的 Pipeline 根据源和目标地址空间动态确定（定义于 [HIVMDMAOps.cpp:L607-L632](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L607-L632)）：

| 源地址空间 | 目标地址空间 | Pipeline |
|-----------|-------------|----------|
| UB | UB | `PIPE_V` |
| L0C | GM | `PIPE_FIX` |
| GM | L1 | `PIPE_MTE2` |
| UB | L1 | `PIPE_MTE3` |

## 同步要求概述

DMA 操作涉及多个 Pipeline 之间的数据依赖，需要通过同步机制保证数据一致性。HIVM 提供了以下同步原语：

### Pipeline 内同步

- **`hir.set_flag` / `hir.wait_flag`**：Pipeline 间的标志同步，用于生产者-消费者模式
- **`hir.pipe_barrier`**：Pipeline 内屏障，确保同一 Pipeline 内操作的顺序性

### 块间同步

- **`hir.sync_block`**：不同 Kernel 间的同步
- **`hir.sync_block_set` / `hir.sync_block_wait`**：细粒度块间标志同步

### 典型同步模式

```
hir.load   (MTE2) ---- set_flag[MTE2, V] -->
hir.vadd   (V)    ---- wait_flag[MTE2, V] -->
                      set_flag[V, MTE3] -->
hir.store  (MTE3) ---- wait_flag[V, MTE3] -->
```

## DMA 操作一览

| 操作 | IR 语法 | 数据通路 | Pipeline | 布局转换 | 随路功能 |
|------|---------|---------|----------|---------|---------|
| `hir.load` | `hir.load ins(...) outs(...)` | GM -> UB | MTE2 | 无 | Padding、Eviction |
| `hir.store` | `hir.store ins(...) outs(...)` | UB -> GM | MTE3 | 无 | 原子操作 |
| `hir.copy` | `hir.copy ins(...) outs(...)` | UB->UB, GM->L1, UB->L1 | 动态 | 无 | Padding |
| `hir.fixpipe` | `hir.fixpipe ins(...) outs(...)` | L0C->GM/UB/L1 | FIX | NZ2ND/NZ2DN/NZ2NZ | 量化、ReLU、双目标 |
| `hir.nd2nz` | `hir.nd2nz ins(...) outs(...)` | GM -> L1 | MTE2 | ND -> NZ | 初始化缓冲区 |
| `hir.nz2nd` | `hir.nz2nd ins(...) outs(...)` | L1 -> GM | MTE3 | NZ -> ND | 无 |
| `hir.l12ub` | `hir.l12ub ins(...) outs(...)` | L1 -> UB | MTE1 | NZ -> ND | 无 |
| `hir.atomic_cas` | `hir.atomic_cas ins(...) outs(...)` | GM <-> UB | 无 | 无 | 原子 CAS |
| `hir.atomic_xchg` | `hir.atomic_xchg ins(...) outs(...)` | GM <-> UB | 无 | 无 | 原子交换 |

## 支持的数据通路汇总

| 通路 | 操作 | 硬件约束 |
|------|------|---------|
| GM -> UB | `hir.load` | 所有 Ascend 型号 |
| GM -> L1 | `hir.nd2nz` | 所有 Ascend 型号 |
| UB -> GM | `hir.store` | 所有 Ascend 型号 |
| L1 -> GM | `hir.nz2nd` | 所有 Ascend 型号 |
| L1 -> UB | `hir.l12ub` | 所有 Ascend 型号 |
| UB -> UB | `hir.copy` | 所有 Ascend 型号 |
| GM -> L1 | `hir.copy` | 所有 Ascend 型号 |
| UB -> L1 | `hir.copy` | 仅 Ascend950 系列 |
| L0C -> GM | `hir.fixpipe` | 所有 Ascend 型号 |
| L0C -> L1 | `hir.fixpipe` | 所有 Ascend 型号 |
| L0C -> UB | `hir.fixpipe` | 仅 Ascend950 系列 |

## 相关文档

- [hir.load 详解](01-load.md)
- [hir.store 详解](02-store.md)
- [hir.nd2nz 详解](03-nd2nz.md)
- [hir.nz2nd 详解](04-nz2nd.md)
- [hir.copy 详解](05-copy.md)
- [hir.fixpipe 详解](06-fixpipe.md)
- [原子操作详解](07-atomic.md)
- [Gather/Scatter 详解](08-gather-scatter.md)
- [间接访问详解](09-indirect-access.md)
- [随路功能详解](10-padding-quantization.md)
- 源码参考：
  - [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td) - DMA 操作 TableGen 定义
  - [HIVMDMAOps.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp) - DMA 操作实现
  - [dma-ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/dma-ops.mlir) - DMA 操作测试用例
