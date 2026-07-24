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
