# put-all-to-all-quant-matmul 场景设计指导

> **定位**：本文件是 `put-all-to-all-quant-matmul` 定制扩展场景的唯一设计指导。设计前核对记录的官方未覆盖项语义命中本场景判据时，默认查阅本文件（规则见 [`../index.md`](../index.md)），并据此完成 DESIGN.md 的定制设计；跨场景拼接设计元素须在 DESIGN.md 中显式论证。
>
> **场景边界（重要）**：标准 PUT AllToAll+QuantMatmul 需求由官方 kernel `all_to_all_quant_matmul` 直接覆盖，应判 `apace_native`，**不读取本场景**。本场景仅服务官方 kernel 未覆盖的**变体需求**（新 dtype 组合、bias/scale 融合、不同切分轴或编排形态等），准入前提是官方未覆盖项非空且语义命中本场景判据。

## 1. 场景头

| 项目 | 内容 |
|:---|:---|
| 场景 ID | `put-all-to-all-quant-matmul` |
| 支持范围 | PUT 模式 AllToAll+QuantMatmul 的变体扩展，输入 A 按 K 轴按 rank 切分（每卡全 M 行 × 本卡 K 段）、输出按 M 轴分布，UDMA 直调 |
| 官方参考算子 | `all_to_all_quant_matmul`（`apace/kernel/all_to_all_quant_matmul/`，K 轴切分、UDMA 直调范式来源） |
| 授权修改范围 | 仅 `kernel/<op>/` 下新建文件；禁止修改 `block/`、`tiling/` |
| 状态 | 已实现 |

### 准入条件（须全部满足，且与其他场景互斥）

| # | 条件 | 判据来源 |
|:--|:---|:---|
| 1 | 通信原语为 AllToAll | Step 2 `requirement_portrait` |
| 2 | localMatmul 为 QuantMatmul（含 scale） | Step 2 候选组装事实 |
| 3 | 通信方向为 PUT（通信→计算） | Step 2 需求四要素 |
| 4 | 核间数据沿 K 轴切分 | Step 2 切分轴事实 |
| 5 | 通信走 UDMA（非 HCCL windows 直调） | Step 2 ABI 事实（入口含 `CommContext`） |
| 6 | 存在官方 kernel 未覆盖的 native gap（变体需求） | 设计前核对非空 |

## 2. 消费输入

| 输入 | 来源 | 用途 |
|:---|:---|:---|
| `matmul 链路事实` | 设计前核对（内联于 DESIGN.md §0.3） | 提供 M/N/K、dtype、scale 编码、Blaze matmul 链（`BlockMmad` + `BlockScheduler` + `MatmulWithScaleMx`）与 tiling 基线事实，是切分/Win 区/轮次推导的前提 |
| 官方未覆盖项 | 设计前核对 | 授权本场景的唯一依据 |
| 通信接口事实 / 入口 ABI 事实 | 设计前核对 | Win 区容量、入口签名、`CommContext` 传递方式 |

## 3. 设计边界（PUT 模式专属合同）

### 3.1 通信与切分合同

| 项目 | 设计结论 | 事实源 |
|:---|:---|:---|
| 通信方向 | PUT：AIV 先推数据到远端 Win 区，AIC 从 Win 区读取计算（通信→计算） | `../../fundamentals/communication.md` §3.2 |
| 编排形式 | 严格分离或时分复用（**前 R 核通信**为官方惯例：`GetBlockIdx() < rankSize` 守卫包裹 Commit/Wait，`SyncAll<true>()` 与 SetFlag 在守卫外全 AIV 执行） | `../../fundamentals/communication.md` §2 AIV 分核惯例 |
| 切分轴 | 输入 A 按 K 轴按 rank 切分：每 rank 持有**全部 M 行 × 本卡 K/rankSize 列段**（`[M_total, Ka]`，M_total = m×rankSize）；**B 全量复制**（`[K, N]`，R 个 K 段拼接逻辑布局，PUT 模式 B 不切分，见 [`../../fundamentals/architecture.md`](../../fundamentals/architecture.md) §4 B 分布表） | Step 2 `matmul 链路事实` |
| 通信原语实现 | `CollectiveComm<AllToAll, PUT, AType, TeamBarrier>` → `AllToAllCommPutImpl` | `apace/block/aiv_comm/all_to_all/all_to_all_udma_put.h` |

### 3.2 Win 区布局（两段式）

```text
本 rank Win 区（commBufferAddrs[rankId]）
├─ data 段：rankSize × rankDataBytes          ← winOffset = 0
└─ scale 段：rankSize × scaleKaSize × axisM   ← winOffset = rankSize × rankDataBytes
rankDataBytes = axisM × axisKa × sizeof(AType)
```

- data 与 scale 两个通信对象复用同一 Win 区，经 `winOffset` 分段；各对象独立 UB commBuf（512B × 2 + barrier 32B = 1056B）。
- host 侧必须校验：Win 区需求（data 段 + scale 段）≤ HCCL 内置 buffer 实测容量。
- 数据区/元数据区分离红线：官方布局 barrier flag 在独立 BARRIER_BUF，Win 数据区从偏移 0 可用；共享布局须按 host 约定偏移跳过头部，三处偏移同源（`../../fundamentals/communication.md` 陷阱 #12）。

### 3.3 通信轮次 T 推导

| 项目 | 公式/结论 |
|:---|:---|
| T（commTurn） | `T = splitAxisTileCnt + splitAxisTailCnt`（`GetCommTurn()` 返回值） |
| 切分不变量 | `切分轴总元素数 = splitAxisTileSize × splitAxisTileCnt + splitAxisTailSize × splitAxisTailCnt` |
| 两阶段策略 | 精度调试 `tileCnt=1` 串行基线；性能调优扫描 `{1,2,4,8,16}`（官方 cases.csv 实测档位 tileCnt∈[2,8]；上限受下方 commTurn 约束） |
| 硬上限 | flagId 直接取轮次 tid 时轮次索引 < FLAG_ID_MAX(16) 且避开 SyncAll 保留区（`SyncAll<true>` 占 14，实际安全上界 13；本场景每轮 `SyncAll<true>`）；`waitedMask` 为 uint32 → 通信 tile 总数 ≤ 32 |
| UDMA 可靠性 | 单轮 PUT 大小**无官方硬上限**（bring-up 期过大单轮曾见间歇失败，边界未定——512KB 量级系 bring-up 经验值；同平台更大单轮亦有稳定实例）。风险提示级：大单轮按 case 复核（连续多轮精度 + 重复 launch 一致性），不作 host 强制拒绝；间歇 FAIL 时优先增大 T（`../../fundamentals/communication.md` 陷阱 #13；R14 口径） |

### 3.4 Flag 编排（PUT）

| 项目 | AIV（通信侧） | AIC（计算侧） |
|:---|:---|:---|
| 逐轮动作 | `Commit(scale) → Commit(data) → Wait<BARRIER_DEVICE>() → SyncAll<true>() → SetFlag<0x2, PIPE_MTE3>(tid)` | `WaitFlag<0x2, PIPE_MTE2>(tid)`（经 `UdmaCommWaitPolicy::WaitTile`）后从 Win 读 A[tid] → Matmul |
| 依赖计算 | — | `CalcDependTileIdx(mPos + blockM - 1, ...)` 推导依赖 tile，`waitedMask` 按位去重，循环末尾尾部兜底 |
| self 跳过 | DoCommit 跳过 `targetRankId == rankId`；DoWait 仅非 self 执行 `Drain`，barrier 含 self | self 段经本地 GM/Win 直读 |
| 回压 | 无（PUT 不需要环形回压） | — |

## 4. 联合合同（通信 × 计算组装）

| 项目 | 设计结论 |
|:---|:---|
| 组装形态 | AIV 驱动 UDMA PUT 与 AIC Blaze matmul 在同一 kernel 内逐 tile 流水；`Commit()` 非阻塞是重叠关键 |
| localMatmul 模式 | 默认 `0`（REMOTE-only，官网 ST 基线）；通算并行选 `1`（LOCAL+REMOTE+AtomicAdd，须加 `PipeBarrier<PIPE_ALL>` 修复）；精度优先选 `2`（DEFERRED_SYNC，仅 UDMA impl 可达，L0C 须满足 `baseM × baseN × 4B ≤ 128/256KB`） |
| 累加语义 | 各 rank 部分和在同一 L0C FP32 累加器累加，计满 `splitKNum` 触发单次 fixpipe → C；L0C 需求与 rankSize 无关 |
| 核配比 | `KERNEL_TYPE_MIX_AIC_1_1`，禁止 `__schedmode__(1)`；通信核数 = rankSize。TeamBarrier `totalJobs = rankSize`（官方 A2A/AG 惯例：前 R 个通信核各以 `GetBlockIdx()` 作 jobIndex 参与，`teamBarrier_.Init(buf, ctx, rankSize, GetBlockIdx())`，CrossDevice 由 `Wait<BARRIER_DEVICE>` 内建触发，分布式轮询覆盖全部 rank） |
| 同步面 | 单向通道 AIV→AIC（通信就绪 flag）；PUT 无 AIC→AIV 回压 |

## 5. Step 3 输入/来源前提

| # | 前提 | 不满足时 |
|:--|:---|:---|
| 1 | 设计前核对事实已**内联记录于 DESIGN.md §0.3**（含 `matmul 链路事实`、官方未覆盖项非空）；独立 `apace-investigation-report.md` 仅在委托方明确要求时才需（见 step2-investigation.md） | 返回 Step 2 补充调查（最多一次） |
| 2 | 场景语义命中本场景判据 | 零命中/多命中 → `unsupported` |
| 3 | 官方参考算子 `all_to_all_quant_matmul` 源码可读（UDMA impl、tiling_data、ST） | 停止并澄清 |
| 4 | 已读 `../../fundamentals/communication.md`、`../../fundamentals/fusion.md` §2/§3/§4/§5、`../../operator-design/operator-anatomy.md` §3 | 补齐后再设计 |

## 6. 验证合同摘要

| 项目 | 合同要点 |
|:---|:---|
| golden 语义 | 设全局逻辑 M 总量 `M_total = m × rankSize`（m 为单卡输出行数）。**输入分布**：每卡持有 A 的本卡 K 段 `[M_total, K/rankSize]`（**全部 M 行 × 本卡 K 列段**，K 轴切分）+ B 全量复制 `[K, N]`（R 段 K 段拼接布局，各卡一致）。**通信语义**：本卡 A 的第 r 个 M 行块（m 行）PUT 给 rank r；收齐后本卡持有自己 m 行对应的全部 K 段。**输出分布**：每卡输出本卡 M 段 `[m, N]`（全局输出 `[M_total, N]` 按 M 轴分布，**不是每卡完整 C**）。golden 公式：`C_r = concat_K(各源 rank 发来的行块) × B`（K 段拼接后单次 matmul，非逐 rank 输出求和）。依据：官方 ST `gen_data.py` `a_local=(M_total, Ka)` + chunk 按 M 行块分发、`main.cpp` 输出 buffer 为 `m×n` |
| 量化解码 | 设备输入为 MX/FP8 编码字节时，golden 须从最终写入设备的实际字节解码计算；ScaleB 为 DN 布局列主序（host 侧按 `[N, K/32]` C-order 写入，官方 gen_data 已修复此坑） |
| 精度基线 | 先 `tileCnt=1` 串行基线 PASS，再扫描 tileCnt；nonfinite 门与 dtype 容差按 DESIGN §4.2 |
| 多卡矩阵 | rankSize 端点 × shape（对齐/非对齐/tail）× tileCnt × dtype 组合 |
| 性能基线 | 须含 L2 flush 证据；Step 3 不实现、不运行设备、不写设备 PASS |
