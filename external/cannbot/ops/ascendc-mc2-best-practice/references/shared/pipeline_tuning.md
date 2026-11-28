# 通算并行调优（tileCnt / headMSize）

本文档承载 MC2 skill 的"通算并行调优子能力"。核心结论：**`tileCnt` 是通算流水的深度旋钮**，分两阶段调优——Step 2-4 在容量及接口允许时用 `tileCnt=1` 做串行基线（简化精度/审查调试），Step 6 性能验收阶段扫描 `tileCnt` 找最优值。设计时可用 §2.1 的公式缩小候选范围，最终仍以实测选优。

> 参考实现默认 `headMSize=512`（即 `tileCnt = M/512`）只是经验值，**不是最优值**。不同 M/K/dtype 组合下最优 `tileCnt` 不同，必须以 `msprof --ai-core=on --aic-mode=task-based` 实测为准。

---

## 1. tileCnt 是什么

```cpp
// include/tiling/all_to_all_matmul_tiling_data.h
struct AllToAllCommTilingData {
    uint32_t tileCnt;      // M 轴切分块数 = 通算流水的深度
    uint32_t bufferSize;   // SHMEM buffer 轮转数（独立于 tileCnt，默认 4）
};

// src/all_to_all_matmul.cpp host 侧
uint32_t headMSize = 512;                       // 每块的 M 行数
uint32_t tileCnt = (m - tailMSize) / headMSize; // 完整主块数；此样例要求 tailMSize 为 0 且能整分
tilingData.commTilingData.tileCnt = tileCnt;
```

示例按等长主块计算，固定 M 时，块数越多，每块行数越少。若实现支持尾块，主块覆盖的行数加上实际尾块行数须等于 M，块数是否包含尾块按该实现的 Host 和 Kernel 约定。

- AIV 把 M 轴切成 `tileCnt` 块，每块 `headMSize` 行，逐块 Put；
- AIC 在通信 buffer 上逐块做 Blaze Matmul；
- 通过 `CrossCoreSetFlag<0x2, PIPE_MTE3>(i)` / `CrossCoreWaitFlag<0x2, PIPE_MTE2>(i)` 让 AIV 的 Put(i+1) 与 AIC 的 MMAD(i) 并行——这就是通算并行。

> **路线差异**：上述 `PIPE_MTE2` 编排适用于 blaze-shmem 路线和 apace PUT 模式（AIC 侧等通信数据，gate MTE2 装载路径）。
> apace GET 模式（契约级推导，官网无样例）使用不同的配对：
> - GET 模式 AIV WaitFlag：**无模板形态** `CrossCoreWaitFlag(i)`（默认 mode 0 + PIPE_S，阻塞标量流——AIV Wait 后紧随标量流 Commit 下发，模板形态不阻塞标量流会引入竞态，见 `ascendc-api-best-practices` skill `references/api-crosscore-sync.md` §4.1）
> - GET 模式 AIC SetFlag：`PIPE_FIX`（而非 `PIPE_MTE3`）
> 详见 [`../foundations/apace/fusion.md`](../foundations/apace/fundamentals/fusion.md) §3（GET/PUT flag 编排模式）与 `ascendc-api-best-practices` skill `references/api-crosscore-sync.md`（API 签名与平台生效性）。

---

## 2. tileCnt 与通算并行度

```
tileCnt=1:
  AIV:  Put(0) ━━━━━━━━━━━━━━━━━
  AIC:            wait ━━ MMAD(0)
  → 通信延迟完全暴露（AIC 等 AIV Put 完才能开始算）

tileCnt=2:
  AIV:  Put(0) ━━ Put(1) ━━━━━━━
  AIC:     wait ━━ MMAD(0) ━━ MMAD(1)
  → Put(1) 与 MMAD(0) 并行，1 级流水掩盖

tileCnt=N (N≥2):
  AIV:  Put(0) ━ Put(1) ━ Put(2) ━ ... ━ Put(N-1) ━
  AIC:     wait ━ MMAD(0) ━ MMAD(1) ━ ... ━ MMAD(N-1)
  → N-1 处跨块重叠机会；计算足够长、缓冲和依赖允许时，稳态通信可被掩盖
```

**关键洞察（针对上述逐块 Put → MMAD 示例）**：
- `tileCnt=1` 不是"没有通算融合"——AIV/AIC 仍然在两个核上同时启动 kernel，但只有一个数据块，没有掩盖空间；
- `tileCnt>1` 才有跨块通算并行的机会（Put 与 MMAD 重叠）；实际等待和缓冲复用仍可能使其串行；
- `tileCnt` 越大，掩盖级数越多，但 `headMSize = M/tileCnt` 越小，UDMA 单次 Put 数据量越小，带宽利用率下降。

### 2.1 公式化 Tiling：估时与候选

优先使用当前路线已有的计算/通信模型，其次使用配置相同的有效测量。参数须匹配产品、通信引擎、原语、Rank 数、拓扑、dtype 和消息大小；AICPU、CCU、SHMEM 的提交成本与容量约束不能互相套用。通信按消息大小分段拟合时，分块跨过区间边界后重新估时。缺少可信参数时，从资源允许的已有配置起步，不编造耗时。

**外层分块与核组联合选型**：外层块分配给独立流水的核组（协同处理同一块的 AIC/AIV）时，应按实际分配统计每组连续处理的块数。每组只有一块时，组内没有跨块重叠机会；增加每组块数可延长流水，但减少核组数也会降低空间并行度。联合比较块长与核组数，按每组实际块序列计入负载不均、启动和排空成本，以最慢组的完成时间评估整体延迟，并与 §5 的内部矩阵波次利用率分别核对。

假设 `k` 个等长块，每阶段内部串行、两阶段可重叠且缓冲足够，单块计算/通信耗时为 `c(k)`、`d(k)`：

```text
T(k) = c(k) + d(k) + (k−1) × max(c(k), d(k))
c(k) = A/k + u，d(k) = B/k + v
计算较慢时：T(k) = A + k*u + B/k + v，候选 k* = sqrt(B/u)
通信较慢时：T(k) = B + k*v + A/k + u，候选 k* = sqrt(A/v)
```

`A/B` 是计算/通信可随分块均分的成本，`u/v` 是每块固定开销。平方根候选仅在分母为正、相应的较慢阶段判断成立时适用；不能直接当作合法整数最优解。候选来自当前合法配置、可行的单块方案、平方根解邻近整数、计算通信配平点、拟合区间和资源边界。先按 §4 合法化主尾块，再按真实块大小重算，去掉重复候选；缺少可信模型时，仍可使用 §3 的经验候选进行实测。

**量化前后处理的资源占用**：量化准备与反量化、bias、Cast 等后处理共用 AIV 时，应按真实执行顺序累计不能重叠的占用时间。缓冲独立且完成通知允许时，可比较“准备当前块并启动 Cube，再处理上一块输出”的交错顺序，同时估算后处理对下一块供给的影响。这类流水按实际资源占用与块间依赖估算首轮、稳态和排空，不能直接套用两个独立阶段的公式。

### 2.2 主尾块、有限缓冲与本地计算

对依次经过生产、消费两个阶段的块，记耗时为 `p[i]/q[i]`、完成时间为 `P[i]/Q[i]`，从第 1 块开始。若有 `b` 个轮转缓冲，且每块从生产开始到消费结束一直占用该缓冲：

```text
P[0] = Q[0] = 0
release[i] = 0（i ≤ b），否则为 Q[i−b]
P[i] = max(P[i−1], release[i]) + p[i]
Q[i] = max(P[i], Q[i−1]) + q[i]
总耗时 = Q[k]
```

计算后通信时生产者是计算，通信后计算时生产者是通信。若实际更早释放缓冲，应使用真实释放时刻。单缓冲在此生命周期下会串行；三块均为生产 12、消费 12 时，双缓冲完成时间为 48，单缓冲为 72（模型时间单位）。长短块顺序也影响启动与排空，应按实际顺序递推。

例如，GM 中间结果搬入独立 UB 且最后一次读取完成后，原 GM 区即可供生产者复用；后续运算使用的 UB 与独立通信窗口仍分别等待自身的最后一次访问完成。

通信先行且存在不依赖通信结果的本地计算时，可用本地工作掩盖首轮通信。若它占用同一计算消费者、不阻塞通信，也不占用轮转缓冲，令 `Q[0]=T_local`；前 `b` 块的空缓冲仍可立即使用。首个通信块从 `max(P[1],T_local)` 开始计算。首块和后续块可按此依赖分别选长，不必强制等长。

### 2.3 各类通信的分块依赖与估时

计算在通信之前还是之后，由算子数据流决定，不能仅凭通信原语名称判断。上述模型要求每块能按当前接口支持的粒度提交、确认完成并交给消费者；接口只报告整次通信完成时，不能把内部数据包到达当作块就绪。

| 通信原语 | 一块数据何时可以消费 | 分块与估时要点 |
| --- | --- | --- |
| AllGather | 当前计算块所需的各来源 Rank 数据均已就绪；本地输入若可独立计算，可使用 §2.2 的本地工作模型 | 区分每 Rank 发送量与收集后的输出量；分别估算本地计算和远端计算，按拼接布局确定计算块覆盖哪些来源 |
| AllReduce | 该块汇集所有参与 Rank 的贡献并完成规约；本地部分结果不能当作最终结果 | 按实际规约算法估算通信轮次和规约成本，不能只用输出字节数除以带宽；切分保持各 Rank 对应的规约元素一致 |
| ReduceScatter | 当前 Rank 所属输出片段已完成跨 Rank 规约 | 区分参与规约的输入总量与本 Rank 接收量；切分对应输出归属，不能把完整输入量当作每 Rank 的输出量，也不能默认沿矩阵行切分 |
| AllToAll | 当前计算块所需的各来源片段已接收，并按目标布局就绪 | 按每个对端的等长片段估时，核对切分轴、拼接轴及打包/重排成本；不能把整份输入量当作每个对端的传输量 |
| AllToAllV | 当前计算块所需的变长片段按实际接收计数和偏移就绪 | 使用真实的逐对端收发量与块顺序，考虑负载较重 Rank 对完成时间的影响；不以平均消息量代替不均衡分布，不直接套均匀块的平方根候选 |

不等长块按 §2.2 逐块递推；没有匹配的通信模型时，用合法候选实测选优。GMM 使用实际 Group 工作量估算计算时间；量化路径计入 scale、Cast、布局转换及物理传输字节，已有模型包含的成本不重复相加。

例如，计算块跨越多个变长通信片段时，按接收计数和偏移取各段交集，所需数据及关联 scale 就绪后再计算；若跨越 Expert 边界，则分别使用对应权重。通信分段与计算分块分别确定，估时使用交集的实际行数，并计入空段与尾块开销。

两次通信须分别建模，按真实依赖和通信资源确定能否并行。计算与通信争用内存带宽时，单独运行的耗时只作为初估，重叠成本用实际 trace 核对。

---

## 3. 两阶段调优策略

### 阶段 A：`tileCnt=1` 串行基线（Step 2 设计 → Step 4 审查）

设计时结合 §2.1 和 §4 确认合法分块；容量及接口允许时，采用以下串行基线，否则使用最简单的合法分块。

**DESIGN.md 中配置**：
```cpp
// host 侧
uint32_t headMSize = m;   // 整个 M 一块
uint32_t tileCnt = 1;
```

**为什么从 `tileCnt=1` 开始**：
- 数学语义不随 `tileCnt` 改变；精度问题先用 `tileCnt=1` 排查，减少跨块流水干扰；多块同步、尾块和缓冲复用仍需验证；
- AIV/AIC 同步链路最短，便于定位 CrossCore Flag 配对和数据依赖错误；
- Reviewer 在 Step 4 审查时可以专注于算法正确性 + 三大约束，不必同时纠结通算性能；
- Blaze `tilingEngine.GetTilingData(headMSize=m, n, ka, ...)` 仍然会推导内部 `baseM/baseN/baseK`（典型 baseM=128/256），单卡 Matmul 正确性独立于通信切分。

**门禁**：精度 verify_result.py PASS + REVIEW.md 判定 PASS / PASS WITH NOTES。

### 阶段 B：扫描 `tileCnt` 找最优（Step 6 性能验收）

**扫描候选值**：`tileCnt ∈ {1, 2, 4, 8, 16, 32}`（建议为 2 的幂，便于均匀切分和调试）。先按 §4 筛选合法值；有匹配模型时，优先测试 §2.1 的公式候选及附近合法值，不要求扫完固定集合。

> 注意：`bufferSize` 独立于 `tileCnt`，默认 4。`tileCnt` 决定通信块总数，`bufferSize` 决定 SHMEM 中同时存在的 buffer 数。`bufferId = mLoopIdx & (bufferSize - 1)` 的位掩码约束作用于 `bufferSize`（要求 `bufferSize` 为 2 的幂），不约束 `tileCnt`。

**每个 `tileCnt` 的测试流程**：

```bash
PROJ="$(pwd)"

# 1. 改 host 侧 headMSize 计算（src/{op_name}.cpp）
# 例如 tileCnt=8: headMSize = m / 8

# 2. 重新编译
cmake --build build -j

# 3. 跑精度验证（确保 tileCnt 改动未破坏正确性）
bash run.sh

# 4. msprof task-based 采集（不传 --warm-up；程序多轮运行，每轮刷 L2，后处理取后 5 轮）
msprof --ai-core=on --aic-mode=task-based \
    --output="${PROJ}/docs/perf/tileCnt_8" \
    --application="${PROJ}/build/{op_name} 2048 8192 3584 4 perf"

# 5. 多卡后处理：使用 ops-profiling skill 的 msprof_perf_summary.py
python3 ${SKILL_PATH}/scripts/msprof_perf_summary.py "${PROJ}/docs/perf/tileCnt_8" ops/{op_name}
```

**选最优的标准**（优先级从高到低）：
1. **整体 Task Duration（4 卡 max）最小**——直接性能指标；其他卡数覆盖实际参与的全部 Rank，基线与候选采用相同用例和采样条件；
2. **cube ratio**——辅助判断 AIC 是否被充分喂满，不以固定的 40–70% 区间判定全部 MC2 算子；
3. **实际通算重叠**——用时间线核对 Put 与 MMAD 的重叠，AIV/AIC 汇总时间接近不能证明通信被掩盖（详见 `profiling_mc2.md §5.2`）。

---

## 4. headMSize 的约束

对上述等长切分的 Blaze 样例，切换 `tileCnt` 时检查以下约束：

| 约束 | 说明 | 违反后果 |
|------|------|---------|
| `headMSize > 0` | `tileCnt ≤ M` | 编译期 OK，运行期 tile 为空 |
| `M % tileCnt == 0` | 否则要显式处理 tail | host 侧 `tileCnt = (m - tailMSize) / headMSize` 当前不支持非零 tail |
| `headMSize ≥ Blaze baseM`（典型 128/256） | Blaze `BlockMmad` 需要最小块 | `GetTilingData` 报错或推导失败 |
| `headMSize × kPerRank × sizeof(dtype)` 落在 UDMA 带宽高效区间 | 数十 KB ~ 512 KB（带宽高效区间）；单轮 PUT 可靠性上限无官方硬边界（512KB 系 bring-up 期经验，同平台官方实现单次 12.6MB 稳定——bring-up 期可按 ≤512KB 起步，放大后按 case 复核） | UDMA 带宽利用率下降，Put 开销暴露；单轮放大后未复核时存在通信可靠性风险 |
| `bufferSize × headMSize × kPerRank × rankSize × sizeof(dtype) ≤ SHMEM_SPACE_SIZE` | SHMEM 空间预算（默认 1 GB） | `aclshmem_align` 失败 |

**512 不是最优值**——参考实现固定 `headMSize=512` 只是为了在 `M=2048, K=8192, rankSize=4` 的典型 shape 下取得一个合理起点。其他 shape/dtype 下最优值不同：
- M 大、K 小时，`tileCnt` 增大（`headMSize` 减小）让通信粒度更细，掩盖级数更多；
- K 大时，单次 MMAD 已足够长，`tileCnt` 可减小（`headMSize` 增大）减少通信次数。

**按 rank 数分档的标定值**（Ascend 950PR/950DT（dav-3510）平台实测标定，不跨平台复用）：

| rankSize | 推荐 headMSize | 理由 |
|---|---|---|
| ≤ 2 | 512 | 大 tile、少 SyncAll、通信效率高（分片大，同步次数敏感） |
| ≥ 4 | 128 | 小 tile、多 commTurn，`AllToAll(t) ∥ Reduce(t-1)` 流水掩盖效果好 |

> ⚠️ **flag 约束（按 flagId 模式区分）**：**计数式固定 flagId**（推荐，apace compute-first 默认）下 T 无数值上限要求——每 flagId 计数器 0-15 衡量**未消费积压**，紧邻配对（每轮 Set 后立即 Wait）下积压 ≈1-2 不触顶，生产验证 T=16 稳定；**轮次索引式 flagId**（flagId=tid）须满足轮次索引 < 16 且避开 SyncAll 保留区（`SyncAll<true>` 占 14，实际安全上界 13）。按所选模式核对 `headMSize ≥ ceil(mSeg / 安全上界)`，不为满足分档值突破 flag 约束。口径详见 apace 路线 `fusion.md` §3.3 / R9。

同时受 Blaze baseM 约束：headMSize 需为 baseM（减半后典型 128）的整数倍；SWAT baseM=256 且 headMSize 可被 128 整除时可手动减半 baseM 至 128（见 fusion.md §6.2.8）。

---

## 5. 每块 Matmul 的 Tiling：`tilingEngine.GetTilingData`

每块通信数据的 Matmul tiling 由 host 侧 `tilingEngine.GetTilingData(headMSize, n, ka, ...)` 自动推导：

```cpp
// src/all_to_all_matmul.cpp host 侧
QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;

// 只需一份 tiling（按 headMSize 切块）
tilingEngine.GetTilingData(headMSize, n, ka, false, true, tilingData.tileQbmmTilingData);
```

**推导关系**：

| 输入 | 输出（自动推导） |
|------|----------------|
| `headMSize` | Blaze 内部 `baseM`（≤ headMSize，对齐 16/128） |
| `n` | `baseN`、`nTailTile` 等 |
| `ka = k/rankSize` | `baseK`、`scaleKL1`、`stepK` 等 |
| — | `usedCoreNum`（实际启动的 AIC 核数） |

**切换 `tileCnt` 时必须重新调 `GetTilingData`**——`headMSize` 变了，所有 Blaze tiling 字段都会跟着变。

**`tileCnt=1` 时**：`headMSize=m`（全 M），`GetTilingData` 推导出的 `baseM` 仍然是 128/256 这种小值，Blaze 内部仍会做多 block 切分（在 AIC 内部把 `headMSize` 再切成多个 baseM 块），因此 `tileCnt=1` 不等于单核计算；单卡 MMAD 效率仍取决于内部 Tiling 和有效核数，不能仅凭通信块数判断。

外层块长改变也会影响矩阵效率，不能把完整矩阵耗时始终按 `tileCnt` 均分。联合检查三层：外层通算流水、核间/L2 分块、Cube 基本块及 L0/L1 容量。对非空、等成本的稠密输出块，可先估算：

```text
输出块数 Q = ceil(M_tile/baseM) × ceil(N/baseN)
波次数 W = ceil(Q/available_cores)
理想波次利用率 = Q / (W × available_cores)
```

该比值只反映空核与尾波，不是实测 Cube 利用率；BMM/GMM 按真实 batch/group 调度与各块成本扩展。使用配套 Tiling 接口重新取得完整字段，核对 `baseM/baseN/baseK`、step/depth、双缓冲和 scale 的驻留字节；区分完整 K 循环与一次驻留窗口，不在求解后孤立修改某个 base 值。Host 的主尾块大小、Kernel 循环、通信计数、完成等待与缓冲复用须使用同一分块方案。

---

## 6. 调优决策树

```
Step 2-4: tileCnt=1（合法时）, 跑通精度 + 审查通过
    │
    ▼
Step 6: 扫描 tileCnt
    │
    ├── 经验候选：1, 2, 4, 8, 16, 32（按 §4 筛选）
    ├── 有匹配模型时，优先测试公式推荐值及附近合法值（§2.1）
    │
    ▼
每个 tileCnt 按 §3 构建并验证精度，再跑 msprof task-based，记录 Task Duration（实际 Rank 的 max）
    │
    ├── 候选精度失败 → 恢复此前通过的版本
    ├── Task Duration 随 tileCnt 增大单调下降 → 约束和预算允许时继续增大
    ├── Task Duration 先降后升 → 取已测且精度通过的最低点
    └── Task Duration 随 tileCnt 增大单调上升 → 保留已测最快配置，允许 tileCnt=1
    │
    ▼
最优 tileCnt 写入 DESIGN.md "性能调优记录" 小节
    │
    ▼
Step 7: 汇报（含最优 tileCnt + 对应 Task Duration）
```

切换候选前保留当前最佳版本及其测试结果；性能差异接近采样波动时复测。最终使用精度通过的最快版本并复验，不以最后一次尝试替代它。

---

## 7. 常见误区

| 误区 | 真相 |
|------|------|
| `tileCnt` 越大越好 | 过大会让 `headMSize` 太小，Blaze `baseM` 推导失败 / UDMA Put 带宽利用率下降 |
| `tileCnt=1` 等于没有通算融合 | AIV/AIC 仍同时启动；只是通信延迟无掩盖 |
| 固定 `headMSize=512` 最优 | 仅是参考实现经验值；不同 shape/dtype 最优值不同 |
| 改 `tileCnt` 后不用重测精度 | 数学语义不变，但内部 Tiling、尾块、同步和缓冲复用可能改变，候选均需复验 |
| `tileCnt` 必须是 `bufferSize` 的整数倍 | 不必——`bufferId = mLoopIdx & (bufferSize - 1)` 自动轮转 |
| 改 `tileCnt` 后 Blaze tiling 不变 | 错——`GetTilingData` 必须重新调用，`baseM` 等会跟着 `headMSize` 变 |

---

## 8. 与其他参数的关系

| 参数 | 关系 |
|------|------|
| `headMSize` | 每个主块的 M 行数，换算条件见 §1 |
| `bufferSize` | 独立——SHMEM 中同时存在的 buffer 轮转数（默认 4）。增大 `tileCnt` 不需要改 `bufferSize` |
| `baseM / baseN / baseK` | Blaze 内部 tiling，由 `GetTilingData(headMSize, n, ka, ...)` 推导 |
| `SHMEM_SPACE_SIZE` | 总空间 ≈ `bufferSize × headMSize × kPerRank × rankSize × sizeof(dtype)` + scale 段 |

---

## 9. 后续阅读

| 想了解 | 读 |
|--------|---|
| 性能采集流程（msprof task-based + L2 flush + 4 卡后处理） | `profiling_mc2.md` |
| MC2 通算流水架构 | `mc2_architecture.md` §4 "4-Buffer 流水" |
| Blaze tiling 字段含义 | `matmul_blaze.md` §4 "Tiling 数据流" |
| Step 6 验收门禁 | blaze-shmem：`../foundations/blaze-shmem/workflow_integration.md` §Step 6；apace：`../foundations/apace/workflow/step4-implementation.md` §4 精度与性能验收 |
