# 通算融合 Compute 优化

本文针对 SHMEM 通算融合算子中的计算部分（主要是 Matmul/GEMM）提供优化指引。适用于 `dispatch_gmm_combine`、`matmul_reduce_scatter`、`matmul_allreduce` 等包含 Cube 计算的算子。

## 0. CATLASS 源码参考入口

优化 L2/L3 计算段时，先参考真实 CATLASS 源码，避免凭记忆改 TileShape、DispatchPolicy 或 TLA 接口：

- 优先路径：`${SHMEM_REPO}/3rdparty/catlass`
- 缺失时：`git clone https://gitcode.com/cann/catlass.git ${SHMEM_REPO}/3rdparty/catlass`
- 重点目录：
  - `examples/13_basic_matmul_tla`、`14_optimized_matmul_tla`、`43_ascend950_basic_matmul`：基础/优化 matmul TLA
  - `examples/08_grouped_matmul`、`02_grouped_matmul_slice_m`、`05_grouped_matmul_slice_k`：GroupedMatmul / MoE/GMM 参考
  - `examples/27_matmul_gelu`、`28_matmul_silu`、`64_ascend950_matmul_evg`：matmul + epilogue/activation 参考
  - `examples/31_small_matmul`：小矩阵低并行度场景参考
  - `include/catlass/gemm/`、`include/catlass/epilogue/`：真实类型、dispatch policy、swizzle、epilogue 接口

CATLASS 负责 compute 段参考

## 1. Matmul TileShape 调优

### 1.1 核心原理

C 矩阵按 `L1TileShape::M` 和 `L1TileShape::N` 切分为基本块，分配给 AIC 核计算：

```
total_blocks = CeilDiv(M, L1TileShape::M) × CeilDiv(N, L1TileShape::N)
```

**目标**：调整 TileShape 使 `total_blocks` 等于（或整除于）可用 AIC 核数，实现负载均衡。

### 1.2 约束

| 约束 | 说明 |
| --- | --- |
| 倍数要求 | TileShape 各维度 **MUST** 为 16 的倍数 |
| 硬件上限 | 不超过 L1/L0A/L0B/L0C 容量 |
| L0 关系 | `L0TileShape::M == L1TileShape::M`，`L0TileShape::N == L1TileShape::N`，`L0TileShape::K ≈ L1TileShape::K / 4` |
| AIC 核数 | Atlas A2 有 20 个 AIC 核 |

### 1.3 调优案例

| Case | Shape | 默认 TileShape | Blocks | 优化 TileShape | Blocks | 加速比 |
| --- | --- | --- | --- | --- | --- | --- |
| fp16 M=1024,N=576,K=6144 | `<128,256,256>` | 24 (不均衡) | `<256,128,256>` | 20 (均衡) | 1.49× |
| fp16 M=20,N=6144,K=16384 (zN) | `<128,256,256>` | 24 | `<32,320,128>` | 20 | 1.30× |
| fp32 M=1,N=768,K=5120 | `<128,128,256>` | 6 (大量核空闲) | `<16,32,1024>` | 24 | 2.34× |

### 1.4 调优步骤

1. 计算当前 TileShape 下的 `total_blocks`
2. 对比可用 AIC 核数（Atlas A2 = 20）
3. 若 `total_blocks % core_num != 0`，调整 M/N 维度使之整除
4. 若 `total_blocks < core_num`，缩小 TileM 或 TileN 增加并行度
5. 保持约束：16 倍数、不超硬件容量

## 2. DispatchPolicy 选择

| Policy | 特点 | 适用场景 |
| --- | --- | --- |
| `MmadAtlasA2Pingpong` | Pingpong 策略，基础安全 | 通用场景、首次实现 |
| `MmadAtlasA2Preload` | Preload + ShuffleK，减少 bank conflict | 性能敏感场景 |
| `Iterate<false>()` (async) | 避免 AIC/AIV 每次迭代同步消息 | MIX mode 通算融合 |

### ShuffleK 说明

ShuffleK 通过偏移 K 轴访问地址模式减少 bank conflict：

- 不同的 K 起始位置让相邻核的 L1 访问不冲突
- 适合 K 维度较大的场景
- 在 `MmadAtlasA2Preload` 中自动启用

### Async Iterate（MIX mode）

| 模式 | 行为 |
| --- | --- |
| 同步 `Iterate()` | 每次迭代 AIC 向 AIV 发同步消息 |
| 异步 `Iterate<false>()` | 只在首次发消息，后续无同步开销 |
| 适用条件 | AIV 不需要逐迭代消费 Cube 结果时 |

## 3. Swizzle 策略

### 3.1 基本概念

`GemmIdentityBlockSwizzle<a, b>` 控制基本块到核的映射关系：

- `b` = SwizzleDirection：0 优先沿 M 轴分配，1 优先沿 N 轴分配
- `a` = SwizzleOffset：控制映射的交错程度

### 3.2 选择规则

| 条件 | 推荐 Swizzle |
| --- | --- |
| M > N（高瘦矩阵） | `<a, 0>`（沿 M 轴优先） |
| M < N（矮胖矩阵） | `<a, 1>`（沿 N 轴优先） |
| M ≈ N | 两者皆可，实测选优 |

### 3.3 调优步骤

1. 先确定 direction：根据 M/N 关系选 0 或 1
2. 再调 offset：从 3 开始，尝试 2、4、5，选最优
3. **MUST** 实测：Swizzle 对性能的影响取决于具体 shape 和内存布局

### 3.4 案例

fp16 M=160,N=6144,K=2048：
- `swizzle<3,1>`：40.6us
- `swizzle<4,1>`：35.3us（↓13%）

## 4. SplitK

### 4.1 适用场景

| 条件 | 说明 |
| --- | --- |
| M 或 N 很小 | 例如 M=1 或 M=20，常规切分只能用少数核 |
| K 很大 | 有足够数据量在 K 轴上并行 |
| 核利用率低 | `total_blocks << core_num` |

### 4.2 原理

将 K 轴切分到多个核，每个核计算 partial sum，最后 reduce 得到完整结果：

```
standard:  core_i computes C[m_i][n_i] = sum_k(A[m_i][k] × B[k][n_i])
splitK:    core_i computes partial_C[m][n] = sum_{k_start..k_end}(A[m][k] × B[k][n])
           final:  C[m][n] = reduce(partial_C across cores)
```

### 4.3 注意

- 需要额外 workspace 存放 partial results
- reduce 步骤有额外开销
- 只在 M×N blocks 远少于 core_num 时有收益

## 5. 存储层次优化

### 5.1 L0C 累加

| 项目 | 说明 |
| --- | --- |
| 瓶颈 | 多次 matmul 的结果先写 GM workspace，再读入 UB 累加 |
| 优化 | 使用 Mmad 内建累加（`enAtomic=1`），结果留在 L0C 直接累加 |
| 收益 | 避免 L0C→GM→UB→Add→GM 路径，约 12% cycle 减少 |

适用场景：`C = A1×B1 + A2×B2 + ...`，多个 matmul 结果累加到同一输出。

### 5.2 BT Buffer 存 Bias

| 项目 | 说明 |
| --- | --- |
| 瓶颈 | Bias 从 GM → UB → Add to matmul result |
| 优化 | 将 Bias 存入 BT Buffer (C2)，在 Mmad CopyOut 时自动 fuse |
| 效果 | 省去 UB 加法和额外搬运 |

### 5.3 FP Buffer 做量化

| 项目 | 说明 |
| --- | --- |
| 瓶颈 | matmul 输出 fp32 → GM → UB → quantize → GM |
| 优化 | 将量化参数存入 FP Buffer，利用 Fixpipe 在 CopyOut 过程中完成量化 |
| 效果 | 省去 GM 中间写读和 UB quantize 步骤 |

### 5.4 小矩阵驻留 L1

| 项目 | 说明 |
| --- | --- |
| 场景 | A 和 B 大小差异大（如 A 很小，B 很大） |
| 优化 | 将小矩阵一次性加载到 L1 保持驻留，只循环搬运大矩阵 |
| 前提 | 小矩阵 size ≤ L1 容量的一半 |

## 6. 通算融合专用优化

### 6.1 Compute-Comm Workspace 交替

通算融合中 compute 和 comm 共享 workspace 时，使用 STAGES 交替避免冲突：

```cpp
constexpr uint32_t WORKSPACE_STAGES = 2;
CrossCoreFlag flagFinishCompute[WORKSPACE_STAGES];
CrossCoreFlag flagFinishComm[WORKSPACE_STAGES];

for (uint32_t idx = 0; idx < loops; ++idx) {
    uint32_t stage = idx % WORKSPACE_STAGES;
    if (idx >= WORKSPACE_STAGES) {
        CrossCoreWaitFlag(flagFinishCompute[stage]);  // 等上一轮计算完成
    }
    // comm: 填入当前 stage workspace
    comm_fill(workspace[stage]);
    CrossCoreSetFlag(flagFinishComm[stage]);
}
```

### 6.2 通信 Epilogue 设计

| 模式 | 说明 |
| --- | --- |
| Matmul → CopyOut → Comm | compute 完成后 comm 搬运结果到远端 |
| Matmul → CopyOut to symmetric → Signal | 结果直接写入 symmetric buffer 并通知远端 |
| Comm → Accumulate → Matmul | 从远端拉 partial result，本地累加后再做 matmul |

选择依据：减少 GM 中间写读，尽可能在 UB/L0C 层面完成数据交接。
