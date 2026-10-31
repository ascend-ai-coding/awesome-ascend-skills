# MatMul 族策略选型指南

> 算子族：`matmul`
> 适用于 `matmul` 族所有变体：`matmul`, `matmul_mxfp4`, `matmul_a16w16`, `batch_matmul`, `matmul_all_reduce`, `grouped_matmul` 等
> 关联设计：[knowledge-strategy-architecture-v3.2](../../../../../docs/design/knowledge-strategy-architecture-v3.2.md) §3.2

## 选型决策树

```
是否 MN 够并行？(totalBlockCnt > aicNum × 2)
├── 是 → 默认走 pingpong (P1) + 流水同步 (P5)
│        ├── 存在尾碎片负载不均 → SWAT 类策略 (P4, P47)
│        ├── 有小侧矩阵 (≤ L1/2) + 对侧循环 ≥ 2 → Full-load (P71 IBShare)
│        └── 假 MTE2 bound (busy 高 + 带宽利用 < 70%) → Scale Coalescing (P52)
└── 否 → MN 欠并行 + 长 K (≥ 2048) → Stream-K (P72 Multi-Core Split-K)
```

## 族特定策略表（23 张）

| 策略 | Tier | Playbook | 触发场景 | 核心手段 |
|---|---|---|---|---|
| **P14** CV 流水预发射 | L2 | ✅ | FA 系列 / V 模板 + K/V premerge | Kernel 三段结构（首轮 PING / 预取 PONG / 消费），与 pingpong 正交叠加 |
| **P17** CV Parallel Block Sizing | L1 | — | AIC/AIV 协作 + 流水气泡 | 基本块大小匹配 Cube 粒度 + Vector 容量 |
| **P46** MatmulImpl 高阶 API | L1 | — | 标准 matmul，避免手写 mmad 调度 | 用 CANN 内置 MatmulImpl 模板，自动 tiling + 同步 |
| **P47** 对角线块调度 | L2 | — | 上/下三角 matmul + 不规则负载 | 对角线优先调度，避免空 block |
| **P48** 多量化模式编译期分发 | L1 | — | matmul_quant / matmul_mxfp4 等量化变体 | TilingKey 在 host 选模板，kernel 零分支 |
| **P49** 硬件加速反量化 | L1 | — | 量化 matmul，dequant 在 fixpipe 之后 | 用 FixPipe BT/FP buffer 内置反量化 |
| **P51** 动态 AIC/AIV 核配比 | L1 | — | matmul 后跟 vector 后处理 | Host 根据 shape 计算 1:1 / 1:2 / 2:1 比例 |
| **P52** L2 Cache Hint | L1 | — | B 矩阵跨多核重复读 | `SetL2CacheHint(NORMAL)` 启 L2 cache，热数据复用 |
| **P60** KV NZ 格式优化 | L1 | — | FA decode + KV cache | 把 KV cache 从 ND 转 NZ，搬运对齐 |
| **P63** Iterate 异步 | L1 | — | matmul + vector 强耦合 | IterateAll / EnQueAll 解耦 AIC/AIV 同步 |
| **P70** FP Buffer 随路量化 | L1 | — | quant matmul + scale 小块 | scale 走 FixPipe Buffer，避免 UB 占用 |
| **P71** Matmul IBShare L1 共享 | L1 | — | 多个 matmul 共享 B 矩阵 | L1 buffer 跨 matmul 实例复用 |
| **P72** Matmul 多核切 K | L1 | — | MN 欠并行 + K ≥ 2048 | Stream-K 切 K 到空闲核 + workspace 归约 |
| **P73** CV 并行 AIC:AIV + multi-Workspace | L1 | — | GroupedMatmul 等 CV 强耦合 | 多 workspace 减少 AIC/AIV 互等 |
| **P78** 片上缓存加速 matmul | L1 | — | 长 K + 多次 B 复用 | B 矩阵 L1 driven，AB padding 对齐 32B |
| **P79** Grad Cube2+Cube3 融合 | L1 | — | FA backward, d=64/128 | 把多个 cube 算合并到一次 mmad 流 |
| **P83** L1 Resident Reuse + Multi-Buffer | L1 | — | FlashAttention 反向 | L1 多 buffer 分区，A/B/dQ/dK 各占独立区 |
| **P86** Matmul Internal Data-Path Tuning | L1 | — | matmul busy 高但带宽未饱和 | 调 stepK / nBufferNum 等内部数据通路 |
| **P87** 手动 Mmad 流水时序 | L2 | — | matmul busy >70%, 通用调度不够 | 手动 SetFlag/WaitFlag 控制 mmad 发射 |
| **A3** Rounding Mode Control | L0 | — | 多版本精度对齐 | `SetMaskCount / SetRoundingMode` 显式控制 |
| **D3** TilingKey Type Dispatch | L0 | — | 多 dtype 编译期分发 | TilingKey 编码 dtype，模板特化 |
| **P4** 多核负载均衡 | L0 | — | block 数不能整除 aicNum | 把尾块均匀分散到所有核 |
| **P7** 32B Alignment + DataCopyPad | L0 | — | shape 非对齐 | GM 地址 32B 对齐 + DataCopyPad 处理尾边 |

## 通用策略（omni）

参见 [`families/omni_guide.md`](omni_guide.md)（待 Phase A6 后续补）——所有 L0 策略（P1/P5/P10 等 13 张）对 matmul 族都适用。

## 叠加正交性表

来自 cannbot ascendc-performance-best-practices 的经验：

| 优化 | 与 pingpong (P1) | 与 SWAT (P4/P47) | 与 Stream-K (P72) | 与 Full-load (P71) |
|---|---|---|---|---|
| **P14 CV 预发射** | 上层增强 | 正交 | 正交 | 正交 |
| **P52 L2 Cache** | 正交 | 正交 | 正交 | 正交 |
| **P72 Stream-K** | 不推荐叠加 | 互斥 | — | 互斥 |
| **P71 IBShare** | 正交 | 正交 | 互斥 | — |

## 量化收益（待 Phase D 自动填充）

跑过 evo loop 后，wm_ops refine 会自动把高分变体的 speedup 写到对应策略卡的 `quantified_gain` 字段。

参考 cannbot mte2_preload_design 给的范例数据：
- `matmul` BF16 `M=1024, K=4096, N=2048`：66.0μs → 63.3μs (-4%)
- MTE2 段：37.6μs → 32.7μs (-13%)

---

## Ascend C Matmul Tiling 系统化设计

> 来源：插件本地 [`tiling-design/matmul/ascendc-api-matmul-tiling.md`](../tiling-design/matmul/ascendc-api-matmul-tiling.md)
> 适用路径：Ascend C **Matmul 高阶 API**（`MatmulImpl` + Host 侧 `MatmulApiTiling::GetTiling`）
> 适用平台：Atlas A2 / A3（`DAV_2201`，如 Ascend910B / Ascend910_93）
> 不适用平台：Ascend 950（`DAV_3510`）→ 应走 Blaze / tensor_api 路径

### 1. 三级 Tiling 架构

```
GM (Global Memory)
  │  A[M,K] × B[K,N] → C[M,N]
  ▼
┌─────────────────────────────────────┐
│         L2 级 Tiling                 │  M/N 方向切分，serpentine 遍历
│   mTileBlock × nTileBlock 个 base 块 │
└─────────────────────────────────────┘
  │  ┌──────────┐  ┌──────────┐
  ▼  ▼          ▼  ▼          ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Core 0  │ │ Core 1  │ │ Core N-1│   Block 级（核间切分 + 错位）
│singleM×N│ │singleM×N│ │singleM×N│
└─────────┘ └─────────┘ └─────────┘
  │            │            │
  ▼ MATMUL    ▼ MATMUL    ▼ MATMUL
┌─────────┐ ┌─────────┐ ┌─────────┐
│ BaseBlk │ │ BaseBlk │ │ BaseBlk │   BaseBlock 级（cube 一次发射）
│M0N0K    │ │M1N1K    │ │MnNnK    │
└─────────┘ └─────────┘ └─────────┘
```

与 Vector 类算子的关键差异：计算单元从 AIV 变为 **AIC（Cube）**，片上存储从 UB 变为 **L1(A1/B1) + L0A/L0B/L0C**。

核心数据结构：

- `TCubeTiling`：CANN 标准 cube 参数（~196 字节），由 `MatmulApiTiling::GetTiling` 自动填充。
- `TilingHeader`：自定义轻量头部，内嵌 `TCubeTiling` 并附加 `mTotalCnt / nTotalCnt / totalBlock`。
- `L2cacheTilePara`：L2 级切分参数（`mTileCntL2 / nTileCntL2 / mTileBlock / nTileBlock / calOrder`）。
- `MultiBatchInfo`：BatchMatMul 专用（`batchUsedCoreNum / iterBatch` 等）。

### 2. L2 级 Tiling（M/N 方向切分）

目的：减少 L2 Cache 抖动、提高数据局部性、配合 serpentine 遍历复用 A 行 / B 列。

切分公式：

```cpp
mTileCntL2 = ceil(mTotalCnt, mTileBlock);
nTileCntL2 = ceil(nTotalCnt, nTileBlock);
mCnt = ceil(mTotalCnt, mTileCntL2);
nCnt = ceil(nTotalCnt, nTileCntL2);
totalTileCnt = mCnt * nCnt;
```

- **小算子退化**：若 `M * N * dtypeSize ≤ L2 / 2`，置 `mTileCntL2 = nTileCntL2 = 1`，跳过 L2 切分。
- **Serpentine 反向遍历**：N 方向在行切换时交替反向，复用上一行末端缓存的 B 行。
- **calOrder**：`ROW_FIRST` 优先复用 A 行（`M ≫ N`），`COL_FIRST` 优先复用 B 列（`N ≫ M`）。

### 3. Block 级 Tiling（核间切分）

- **整核/尾核策略**：所有核之间最多差 1 个 base 块。

```cpp
round       = ceil(totalTileCnt, usedCoreNum);
preCoreNum  = totalTileCnt % usedCoreNum;
if (preCoreNum == 0) preCoreNum = usedCoreNum;
// 其余 (usedCoreNum - preCoreNum) 个核处理 round-1 次
```

- **错位分核**：避免相邻核同时访问同一行 A / 同一列 B 造成 bank 冲突。

```cpp
newBlockIdx = (GetCurrentBlockIdx() + usedCoreNum - blockIdxStart) % usedCoreNum
            + roundIdx * usedCoreNum;
mIdx = newBlockIdx % mCntUse;
nIdx = (newBlockIdx + newBlockIdx / Lcm(mCntUse, nCntUse)) % nCntUse;
```

- **尾块处理**：Kernel 内按 `blockIdx → (mIdx, nIdx)` 计算当前 base 块实际尺寸并调用 `SetSingleShape`。

### 4. BaseBlock 级（cube 一次发射）

- 推荐经验值（优先用 `MatmulApiTiling::GetTiling` 自动给出）：

| 数据类型 | 推荐 baseM | 推荐 baseN | 推荐 baseK |
|---|---|---|---|
| float16 / bfloat16 | 128 / 256 | 128 / 256 | 32 / 64 |
| float32 | 64 / 128 | 64 / 128 | 16 / 32 |

- **对齐约束**：`baseM / baseN / singleCoreM / singleCoreN` 必须是 `ALIGNED_H = 16` 的整数倍，`baseK` 通常也为 16 的整数倍。

### 5. 多核切分扩展

`MatmulApiTiling::GetTiling` 可能返回 `singleCoreM = M`，导致 `totalBlock = 1`，多核完全未生效。**当 `totalBlock < aicCoreNum` 时，必须手动缩小 `singleCoreM / singleCoreN`**，同时保持 16 对齐且不低于 `baseM / baseN`。

算法：交替对半缩小 M/N 方向的 singleCore 尺寸，直到 `totalBlock ≥ aicCoreNum` 或已达下限。

```cpp
if (totalBlock < aicCoreNum) {
    while (totalBlock < aicCoreNum && singleCoreM > baseM) {
        singleCoreM = std::max(baseM, (singleCoreM / 2) / ALIGNED_H * ALIGNED_H);
        mTotalCnt = CeilDiv(M, static_cast<int64_t>(singleCoreM));
        totalBlock = mTotalCnt * nTotalCnt;
    }
    while (totalBlock < aicCoreNum && singleCoreN > baseN) {
        singleCoreN = std::max(baseN, (singleCoreN / 2) / ALIGNED_H * ALIGNED_H);
        nTotalCnt = CeilDiv(N, static_cast<int64_t>(singleCoreN));
        totalBlock = mTotalCnt * nTotalCnt;
    }
}
```

完成后同步更新 `cubeTiling.singleCoreM / singleCoreN / usedCoreNum`，并计算 `mBaseTail / nBaseTail` 供 Kernel 端使用。

### 6. Batch 维度处理

通用广播规则：

| A Batch | B Batch | C Batch | 模式 |
|---|---|---|---|
| N | 1 | N | B 广播 |
| 1 | N | N | A 广播 |
| N | N | N | 一对一 |
| N | M | LCM(N,M) | 多维广播 |

- **Batch 合并到 M**：当 `batchB == 1 && !transA` 时，把 A 的 batch 维合并到 M，以提升单核任务粒度并简化调度。

```cpp
if (batchB == 1 && !transA) {
    M = batchA * M;
    batchA = 1;
}
```

- **Batch 负载均衡**：`iterBatch = min(batchC, coreNum)`，`batchUsedCoreNum = min(batchC, coreNum)`。

### 7. Host 端 Tiling 计算流程

```
1. 输入校验（dtype、shape、K 维度匹配）
2. 通过 PlatformAscendC 获取硬件参数（coreNumAic / l1Size / l0A / l0B / l0C / ubSize）
3. （可选）batch 合并、shape 规范化
4. 调用 MatmulApiTiling::GetTiling(TCubeTiling&)
5. 强制覆写 M/N/Ka/Kb
6. 计算派生计数 mTotalCnt / nTotalCnt / totalBlock
7. 多核切分扩展（当 totalBlock < aicCoreNum 时）
8. （大算子）计算 L2 级 mTileCntL2 / nTileCntL2 / mTileBlock / nTileBlock
9. （BatchMatMul）填充 MultiBatchInfo
10. TilingHeader 序列化到 device tensor（blocking copy）并 EXEC_KERNEL_CMD
```

### 8. 平台路由

| NpuArch | 推荐路径 | 说明 |
|---|---|---|
| `DAV_2201`（Atlas A2 / A3） | `MatmulImpl` + `MatmulApiTiling` | 本文档描述的 Tiling 体系 |
| `DAV_3510`（Ascend 950） | Blaze / tensor_api | 使用 `ascendc-blaze-best-practice` 相关路径 |

新增架构出现时，按 `NpuArch` 新增 reference 文件或更新平台适配说明。

### 9. Tiling 设计检查清单

- [ ] 维度匹配：`Ka == Kb`；BatchMatMul 满足广播规则。
- [ ] 对齐：`baseM / baseN / singleCoreM / singleCoreN` 是 `ALIGNED_H = 16` 的整数倍。
- [ ] 核数：`usedCoreNum ≤ GetCoreNumAic()`。
- [ ] `blockDim = min(usedCoreNum, totalBlock)`，防止富余核越界。
- [ ] 多核切分扩展：当 `totalBlock < aicCoreNum` 时，可缩小 `singleCoreM / N`。
- [ ] Kernel 入口有 `if (GetBlockIdx() >= totalBlock) return;` 越界守卫。
- [ ] Kernel 端按 `blockIdx → (mIdx, nIdx)` 计算偏移和尾块尺寸，调用 `SetSingleShape`。
- [ ] `TilingHeader → device tensor` 必须 blocking copy（`non_blocking = false`）。
- [ ] `GetTiling` 后强制覆写 `M / N / Ka / Kb`。
- [ ] `CubeMathType` host/kernel 配对：`NORMAL ↔ SetHF32(false, 0)`；`HF32 ↔ SetHF32(true, 1)`。
- [ ] Batch 维度 ≤ 4。
- [ ] 小算子（`M * N * dtypeSize ≤ L2 / 2`）跳过 L2 切分。
- [ ] 平台路由：根据 `NpuArch` 选择正确 API 路径。

### 10. 与进化策略的映射

上述系统化设计直接对应以下策略卡：

| 设计环节 | 策略卡 | 关注点 |
|---|---|---|
| 多核切分扩展 | **P104** matmul_multi_core_tiling_expansion | totalBlock 不足时拆分 singleCoreM/N |
| L2 Cache 切分 | **P105** matmul_l2_cache_tile_sizing | mTileBlock / nTileBlock 与 cache 容量匹配 |
| Batch 合并到 M | **P106** matmul_batch_merge_to_m | batchB==1 && !transA 时折叠 batch 维 |
| Serpentine 遍历 | **P107** matmul_serpentine_traversal | L2 tile 反向遍历提升 B 行复用 |
| 平台路由 | **P108** matmul_platform_routing | 按 NpuArch 选择 MatmulImpl / Blaze 路径 |
