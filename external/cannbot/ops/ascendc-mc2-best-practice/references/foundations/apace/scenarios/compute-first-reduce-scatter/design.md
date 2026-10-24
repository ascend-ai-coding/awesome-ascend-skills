# compute-first-reduce-scatter 场景设计指导

> **定位**：apace 定制扩展场景设计合同。需求语义命中本场景判据时默认查阅（规则见 [`../index.md`](../index.md)）；设计细节（flag 硬件规则、分核惯例、Win 区布局）以 `../../fundamentals/fusion.md` §6.2 与 `../../fundamentals/communication.md` 为唯一事实源，本文不重复推导，只冻结场景级决策。

## 1. 场景头

| 项目 | 内容 |
|:---|:---|
| 场景 ID | `compute-first-reduce-scatter` |
| 支持范围 | compute-first 模式，ReduceScatter 语义 + QuantMatmul 融合，输出沿 M 轴按 rank 切分，UDMA 直调 |
| 准入条件（语义判据） | **逻辑语义为 ReduceScatter**（每 rank 输出 M 轴分片、跨 rank 求和）；localMatmul 为 QuantMatmul；compute-first（先算后通信）；通信走 UDMA |
| 状态 | 已实现（有生产实现：先本地计算、再 AllToAll PUT + 本地增量归约实现 ReduceScatter 语义——通信实现方式为本场景设计决策，非准入前提）。⚠️ 官方 master 在 pin 快照之后已新增 **ReduceScatter UBMEM 路径模板**（`8fbd92b1d`，`kernel/fusions/quant_matmul_reduce_scatter/` 等 17 文件）——本场景仍定位 UDMA 直调路径的自研编排；设计前须按 [`communication.md`](../../fundamentals/communication.md) §8 漂移登记核对 master 是否已直接覆盖需求 |

## 2. Consumes

| 输入工件 | 来源 | 消费内容 |
|:---|:---|:---|
| `matmul 链路事实` | 设计前核对（内联于 DESIGN.md §0.3） | QuantMatmul 链路事实：Blaze 模板组件（`BlockMmad` + `BlockScheduler` + `MatmulWithScaleMx`）、dtype 组合、M/N/K 与对齐约束、localMatmul 模式候选 |

缺少 matmul 链路事实 或其中任一事实为 `unknown`/`not_found` 时，停止并返回 Step 2 补充调查，禁止在本场景中推测补齐。

## 3. 设计边界（compute-first 专属合同）

### 3.1 通信方向与编排

| 项目 | 合同 | 事实源 |
|:---|:---|:---|
| 通信方向 | compute-first（compute → communication）：先完成本地 mm，再以 AllToAll PUT + 本地增量归约实现 ReduceScatter 聚合语义 | fusion.md §6.2 |
| 编排形式 | 严格分离（专职化）：**后 R 核通信 / 前 (核数-R) 核归约**，`AllToAll(t) ∥ Reduce(t-1)` 错位流水 | fusion.md §6.2.2 |
| 分核映射 | 通信核：`jobIndex = GetBlockNum() - 1 - GetBlockIdx()`，最大 blockIdx → jobIndex 0；归约核：`blockIdx < rsCoreNum`。**禁止**套用通信在前算子的"前 R 核"惯例 | communication.md §2 |

### 3.2 切分与数据分布

| 项目 | 合同 |
|:---|:---|
| 切分轴 | 输出沿 **M 轴**按 rank 切分 |
| 数据分布 | **输入不切分**：每 rank 完整持有 A `[M, K]`（各 rank 为各自独立数据，**非**全局 A 的 M 分片）与 B `[K, N]`（全 rank 共享）；每 rank 本地计算完整 `C = A @ B` `[M, N]`。**切分仅在输出分布**：C 按 M 轴切 R 段，向 targetRank 交付其 `M/rankSize` 输出行段（`chunkM = M / rankSize`），聚合后每 rank 输出本段归约结果 `[M/R, N]`。**golden 推导式**：`out_r = Σ_j C_j[r·chunkM : (r+1)·chunkM, :]`（`C_j` 为 rank j 的完整 mm 输出，求和跨全部 rank、仅取第 r 个 M 段；FP32 累加后转输出 dtype）——gen_data.py 依此逐 rank 生成，切勿把输入 A/B 误切分 |
| staging 即通信源 | mm 输出写入 staging（workspaceGM）为连续 `[M, N]`；第 targetRank 段恰好等于发往该 rank 的 chunk，**零重排、零额外拷贝**（本场景关键红利） |
| self rank 处理 | 本卡段 `stagingGm + rankId×chunkBytes` 本地合并，不经通信链路；Win self 槽闲置为已知取舍 |

### 3.3 通信轮次 T 推导

| 项目 | 合同 |
|:---|:---|
| T 公式 | 优先 `T = mSeg / headMSize` 整除（无尾块，实现最简）；**单尾块合法**：`tailMSize = mSeg % headMSize`（16 对齐且 ≤ headMSize）时 PUT 直传（偏移精确，生产实证 m4032：3×512+480 与 7×128+112 均通过），tail 轮用 tail tiling（入参 m=R×tailMSize）；多尾块或尾块>头块才必须走 padding 策略 |
| 上限 | 计数式固定 flagId 下 T 上限按 case 实测确定（计数器 0-15 衡量未消费积压，紧邻配对下 ≈1-2 不触顶，不构成 T 上限）；轮次索引型 flagId 时轮次索引 < FLAG_ID_MAX(16) 且避开 SyncAll 保留区（`SyncAll<true>` 占 14）。**不设 host 强制数值拒绝** |
| 调试策略 | 两阶段：精度调试 `T=1` 串行基线；性能调优再扫描合法 T 取值 |

### 3.4 Flag 编排（compute-first）

| 项目 | 合同 |
|:---|:---|
| 配对方向 | AIC `CrossCoreSetFlag<0x2, PIPE_FIX>(flagId)` ⇔ AIV **无模板** `CrossCoreWaitFlag(flagId)`（官方配对惯例"模板 Set + 无模板 Wait"，见 `ascendc-api-best-practices` skill `references/api-crosscore-sync.md` §4.1）；Set 用 `PIPE_FIX`（fixpipe 排空语义），AIV 侧 Wait 后紧随标量流通信下发——模板形态（如 `<0x2, PIPE_MTE2>`）不阻塞标量流，Commit 会在 flag 生效前执行 → PUT 读未写完的 staging（生产实测间歇精度失败），**禁止** |
| T=1 | 循环外单次配对 |
| T>1 | 逐轮计数式配对（T 次 Set ⇔ T 次 Wait，紧邻配对下积压 ≈ 1-2） |
| 硬约束 | **Set/Wait 严格配对**（不配对 = 未定义行为/挂死）+ flagId 数量 16（0-15）；计数器 0-15 衡量未消费积压（官方有据，紧邻配对下不触顶），不作 host 强制校验 |
| flagId 选取 | flagId ∈ [0, FLAG_ID_MAX)；工程验证使用计数式 flag，event ID 0（flagA）和 1（flagB） |
| 双 flag 编排（**强制**） | **compute-first 场景必须遵循** [`fusion.md`](../../fundamentals/fusion.md) §6.2.2/§6.2.3 localLast 双 flag 编排：fragment 重排 `[remote..., local]` + 两个 flagId 分工（flagA=remote 段算完提前启动 PUT、flagB=本轮全部算完），未跨边界核兜底 Set flagA。**禁止以每轮 Set 两个 flag 替代 localLast 双 flag**——此替代使 flagId 用量翻倍且丧失通信提前启动能力。实现要点见 development.md §5.8 |

### 3.5 增量归约

| 项目 | 合同 |
|:---|:---|
| 归约形式 | 手工 UB 分批（batch）求和：每轮处理 R 个来源（R-1 个 Win 槽 + 本卡 staging 自 chunk） |
| 中间精度 | **FP32 中间累加**，最终轮次转回输出 dtype 写 yGm |
| 流水位置 | 归约核处理第 t-1 轮与通信核 PUT 第 t 轮并行（错位流水） |

### 3.6 Win 区布局

| 项目 | 合同 |
|:---|:---|
| 元数据分离 | Win 数据区与元数据/barrier 区分离：官网布局 barrier 在独立 BARRIER_BUF、数据区从 0 可用；共享布局须按约定偏移跳过头部（`localFlag(32B)+teamSyncCounter(32B)+crossCoreFlag(32B)=96B` → 对齐 128），偏移按 host 建链布局确定、host/kernel 同源（[`communication.md`](../../fundamentals/communication.md) 陷阱 #12） |
| **winOffset 强制（共享布局）** | **共享布局（barrier 与数据同一 Win 区）时 winOffset 必须显式设置，禁止 0 偏移**（[`development-guide.md`](../../operator-design/development-guide.md) §3.5 已禁止）。官网布局（barrier 在独立 BARRIER_BUF）时数据区从 0 可用，winOffset=0 合法。AllToAll Init 的 winOffset 按 host 建链布局确定（共享布局一种已验证实现为 128B） |
| 同源约束 | host 侧预留、PUT 写入偏移、归约读取偏移三处必须同源；0 偏移覆盖元数据会造成"假通过"，精度验证无法发现，属设计红线 |
| 布局判定 | Win 区是否含头部元数据由 host 建链布局决定，以 communication.md 陷阱 #12 为唯一事实源 |

## 4. 联合合同（计算 × 通信组合）

三级流水：**mm → staging → PUT → 增量归约**。

| 级 | 执行者 | 动作 | 交接机制 |
|:---|:---|:---|:---|
| 1 | AIC | RunMatmul 逐 tile 计算，结果写 staging（连续 [M, N]） | 每轮 CrossCoreSetFlag 通知 AIV |
| 2 | AIV 通信核（后 R 核） | WaitFlag 门控后 Commit/Wait，将 staging 的 targetRank 段 PUT 至对端 Win 槽 | Win 区数据段（偏移 128 起） |
| 3 | AIV 归约核（前核） | 对上一轮聚合 R 来源，UB 分批 FP32 求和 → yGm | 与第 2 级错位并行 |

架构边界声明：compute-first 下通信依赖完整 mm 输出，**mm 段天然暴露**（无法被通信掩盖）；跨方案性能对比时须显式声明此为架构边界而非回归。

## 5. Step 3 输入/源码前提

| 前提 | 说明 |
|:---|:---|
| 场景命中 | 官方未覆盖项语义命中本场景判据，零命中/多命中判 `unsupported` |
| Investigation 闭合 | matmul 链路事实全部 confirmed，无未闭合项 |
| 修改范围 | 仅允许在 `kernel/<op>/` 下新建文件；禁止修改 `block/`、`tiling/` |
| 直调约束 | 仅 UDMA 直调；HCCL windows 不支持直调 |
| 核配比 | 由 `KERNEL_TYPE_MIX_AIC_1_1` 保证，禁止 `__schedmode__(1)` |

## 6. 验证合同摘要

| 项目 | 合同 |
|:---|:---|
| golden 语义 | 逐 rank：本地完整 [M, N] mm 输出，聚合全部 rank 的本 rank M 段，FP32 求和后转输出 dtype；gen_data.py 依据切分轴=M 生成 golden |
| 精度基线 | 先 `T=1` 串行基线全绿，再扫描合法 T；阈值按输出 dtype 精度标准 |
| 边界矩阵 | rankSize 端点 × shape 对齐/非对齐 × T 各合法取值（优先 `T \| mSeg` 无尾块；T 上限按 R9 口径 case 实测确定——计数式固定 flagId 0/1 生产验证 T=16，计数器 0-15 衡量未消费积压不触顶，不设 host 强制数值拒绝） |
| 红线用例 | Win 区 0 偏移元数据覆盖检测（共享布局下偏移同源校验）；flagId ∈ [0,15] 硬件数量上限（`FLAG_ID_MAX=16`，计数式经 Set/Wait 严格配对规避溢出） |
| 性能 | mm 段暴露为架构边界；性能采集须含 L2 flush 证据，与精度验证分离 |
