# 通算并行调优（tileCnt / headMSize）

本文档承载 MC2 skill 的"通算并行调优子能力"。核心结论：**`tileCnt` 是通算流水的深度旋钮**，在 CANNBot 流程中分两阶段调优——Step 2-4 用 `tileCnt=1` 做串行基线（简化精度/审查调试），Step 6 性能验收阶段扫描 `tileCnt` 找最优值。

> 参考工程默认 `headMSize=512`（即 `tileCnt = M/512`）只是经验值，**不是最优值**。不同 M/K/dtype 组合下最优 `tileCnt` 不同，必须以 `msprof --ai-core=on --aic-mode=task-based` 实测为准。

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
uint32_t tileCnt = (m - tailMSize) / headMSize; // M 轴一共切多少块
tilingData.commTilingData.tileCnt = tileCnt;
```

`tileCnt` 与 `headMSize` 是反向关系：`headMSize = M / tileCnt`（假设不处理 tail）。

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
  → N-1 级流水掩盖；稳态下通信延迟被 MMAD 完全吸收
```

**关键洞察**：
- `tileCnt=1` 不是"没有通算融合"——AIV/AIC 仍然在两个核上同时启动 kernel，但只有一个数据块，没有掩盖空间；
- `tileCnt>1` 才有真正的通算并行（Put 与 MMAD 重叠）；
- `tileCnt` 越大，掩盖级数越多，但 `headMSize = M/tileCnt` 越小，UDMA 单次 Put 数据量越小，带宽利用率下降。

---

## 3. CANNBot 两阶段调优策略

### 阶段 A：`tileCnt=1` 串行基线（Step 2 设计 → Step 4 审查）

**DESIGN.md 中配置**：
```cpp
// host 侧
uint32_t headMSize = m;   // 整个 M 一块
uint32_t tileCnt = 1;
```

**为什么从 `tileCnt=1` 开始**：
- 算法正确性独立于 `tileCnt`——精度问题用 `tileCnt=1` 排查最简单（无流水干扰）；
- AIV/AIC 同步链路最短，CrossCore Flag 出错时直接 deadlock 而非"数据错位"；
- Reviewer 在 Step 4 审查时可以专注于算法正确性 + 三大约束，不必同时纠结通算性能；
- Blaze `tilingEngine.GetTilingData(headMSize=m, n, ka, ...)` 仍然会推导内部 `baseM/baseN/baseK`（典型 baseM=128/256），单卡 Matmul 正确性独立于通信切分。

**门禁**：精度 verify_result.py PASS + REVIEW.md 判定 PASS / PASS WITH NOTES。

### 阶段 B：扫描 `tileCnt` 找最优（Step 6 性能验收）

**扫描候选值**：`tileCnt ∈ {1, 2, 4, 8, 16, 32}`（建议为 2 的幂，便于均匀切分和调试）

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

# 4. msprof task-based 采集（无 warm-up；L2 flush 由 perf 主循环内部保证）
msprof --ai-core=on --aic-mode=task-based \
    --output="${PROJ}/docs/perf/tileCnt_8" \
    --application="${PROJ}/build/{op_name} 2048 8192 3584 4 perf"

# 5. 多卡后处理：使用 ops-profiling skill 的 msprof_perf_summary.py
python3 ${SKILL_PATH}/scripts/msprof_perf_summary.py "${PROJ}/docs/perf/tileCnt_8" ops/{op_name}
```

**选最优的标准**（优先级从高到低）：
1. **整体 Task Duration（4 卡 max）最小**——直接性能指标；
2. **cube ratio 接近理论上限**（MC2 算子 40-70%）——AIC 被充分喂满；
3. **AIV/AIC time 接近**——通信延迟被计算掩盖（详见 `profiling_mc2.md §5.2`）。

---

## 4. headMSize 的约束

切换 `tileCnt` 时，`headMSize = M / tileCnt` 必须满足：

| 约束 | 说明 | 违反后果 |
|------|------|---------|
| `headMSize > 0` | `tileCnt ≤ M` | 编译期 OK，运行期 tile 为空 |
| `M % tileCnt == 0` | 否则要显式处理 tail | host 侧 `tileCnt = (m - tailMSize) / headMSize` 当前不支持非零 tail |
| `headMSize ≥ Blaze baseM`（典型 128/256） | Blaze `BlockMmad` 需要最小块 | `GetTilingData` 报错或推导失败 |
| `headMSize × kPerRank × sizeof(dtype)` 落在 UDMA 带宽高效区间 | 数十 KB ~ 512 KB（带宽高效区间）；单轮 PUT 可靠性上限无官方硬边界（512KB 系 bring-up 期经验，同平台官方实现单次 12.6MB 稳定——bring-up 期可按 ≤512KB 起步，放大后按 case 复核） | UDMA 带宽利用率下降，Put 开销暴露；单轮放大后未复核时存在通信可靠性风险 |
| `bufferSize × headMSize × kPerRank × rankSize × sizeof(dtype) ≤ SHMEM_SPACE_SIZE` | SHMEM 空间预算（默认 1 GB） | `aclshmem_align` 失败 |

**512 不是最优值**——参考工程固定 `headMSize=512` 只是为了在 `M=2048, K=8192, rankSize=4` 的典型 shape 下取得一个合理起点。其他 shape/dtype 下最优值不同：
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

**`tileCnt=1` 时**：`headMSize=m`（全 M），`GetTilingData` 推导出的 `baseM` 仍然是 128/256 这种小值，Blaze 内部仍会做多 block 切分（在 AIC 内部把 `headMSize` 再切成多个 baseM 块），所以**单卡 MMAD 性能不会因为 `tileCnt=1` 而下降**——下降的只是通算掩盖效果。

---

## 6. 调优决策树

```
Step 2-4: tileCnt=1, 跑通精度 + 审查通过
    │
    ▼
Step 6: 扫描 tileCnt
    │
    ├── 候选值：1, 2, 4, 8, 16, 32（2 的幂）
    │
    ▼
每个 tileCnt 跑 msprof task-based 采集，记录整体 Task Duration（4 卡 max）
    │
    ├── Task Duration 随 tileCnt 增大单调下降 → 继续增大 tileCnt
    ├── Task Duration 先降后升 → 取最低点
    └── Task Duration 随 tileCnt 增大单调上升 → 保持 tileCnt=2 或 4
    │
    ▼
最优 tileCnt 写入 DESIGN.md "性能调优记录" 小节
    │
    ▼
Step 7: 汇报（含最优 tileCnt + 对应 Task Duration）
```

---

## 7. 常见误区

| 误区 | 真相 |
|------|------|
| `tileCnt` 越大越好 | 过大会让 `headMSize` 太小，Blaze `baseM` 推导失败 / UDMA Put 带宽利用率下降 |
| `tileCnt=1` 等于没有通算融合 | AIV/AIC 仍同时启动；只是通信延迟无掩盖 |
| 固定 `headMSize=512` 最优 | 仅是参考工程经验值；不同 shape/dtype 最优值不同 |
| `tileCnt` 影响精度 | 不影响——只改变通信-计算时序，不改变数据内容 |
| `tileCnt` 必须是 `bufferSize` 的整数倍 | 不必——`bufferId = mLoopIdx & (bufferSize - 1)` 自动轮转 |
| 改 `tileCnt` 后 Blaze tiling 不变 | 错——`GetTilingData` 必须重新调用，`baseM` 等会跟着 `headMSize` 变 |

---

## 8. 与其他参数的关系

| 参数 | 关系 |
|------|------|
| `headMSize` | `headMSize = M / tileCnt`（反向） |
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
