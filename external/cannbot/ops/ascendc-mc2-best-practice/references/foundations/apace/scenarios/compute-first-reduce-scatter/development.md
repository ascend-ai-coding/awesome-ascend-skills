# compute-first-reduce-scatter 开发指导

> **场景 ID**：`compute-first-reduce-scatter`
> **适用路线**：`implementation_route=apace_custom`，`selected_scenario=compute-first-reduce-scatter`（DESIGN 冻结后编译进 PLAN 的记录值）
> **执行阶段**：Step 4（Implementation）只执行 PLAN，不改设计、不改接口/ABI；通用工程规范见 [`development-guide.md`](../../operator-design/development-guide.md)，架构原理见 [`fusion.md`](../../fundamentals/fusion.md) §6.2，通用方法论（D1-D6 决策点）见 [`dev-methodology.md`](../../operator-design/dev-methodology.md)。本文只写本场景差异化要点。

---

## 1. 有序开发动作（PLAN §4 编译模板）

> **bring-up 两路径（按起点选）**：① 场景合同完备（design.md 已冻结）→ 直接按 A1-A7 FragmentTensor 一步到位；② 仅有语义无合同且存在可借用参考代码 → 先写 R 循环最简版跑通精度（临时裁剪项在 PLAN 登记"后续补齐"），再按 A1 优化为 FragmentTensor。

| # | 动作 | 验收锚点 | 证据 |
|:---|:---|:---|:---|
| A1 | 实现 FragmentTensor mm kernel（消 R 循环，R16） | FragTensorA/FragScaleA/FragTensorC 打包 R 个 rank 段地址 + `QmmMxBlockMmadFragment` per-fragment L1 隔离，一次调用覆盖 `R × curTileM` 行 | 编译通过；grep 无 R×T 子调用循环 |
| A2 | 实现 staging 即通信源 | mm 输出连续 `[M, N]` staging（GM workspace），rank 段 = chunk；PUT 钩子 src 偏移与 mm 写入偏移同源 | per-tile 契约不变量 3（[`development-guide.md`](../../operator-design/development-guide.md) §3.5） |
| A3 | 实现 AIV 严格分离 | 后 R 核通信 / 前核归约；分核公式 `jobIndex = GetBlockNum()-1-GetBlockIdx()`；零 tile 核无条件 SetFlag | AIV 分支骨架与 [`fusion.md`](../../fundamentals/fusion.md) §6.2.1 一致 |
| A4 | 实现增量归约 | 独立 `reduce_sum_ref.h`；手动 UB 批量形态：多 slot 布局 + src 双缓冲 + FP32 中间累加 + N 分段；禁止 TPipe/TQue 逐行模型 | R17/R18/R19（见 §5） |
| A5 | 实现 `Wait<BARRIER_NONE>` + 手动 CrossDevice | 本场景（后 R 核映射）TeamBarrier 须 totalJobs=1 + 指定核显式 `CrossDevice()`；沿用 totalJobs=rankSize 会因 jobIndex 映射错位/守卫早退致跨卡同步失效（机制见 [`communication.md`](../../fundamentals/communication.md) §2.2） | grep `BARRIER_DEVICE` 为空 |
| A6 | 完成 flag 编排 | T=1 单次 SetFlag / T>1 逐轮配对（Set/Wait 严格配对，flagId ∈ [0, FLAG_ID_MAX)）；SyncAll 在分核守卫外 | R5/R9 检查通过 |
| A7 | 完成最终 drain | 循环结束消费残留事件（含次数守卫）；最后一轮归约与 drain 全量覆盖 | 挂死不出现 |

> 执行纪律：每完成一个动作立即冒烟（改完即测）；bring-up 顺序 T=1/rank=2 → T>1 → 全量矩阵。

---

## 2. 文件级合同（[MODIFY] 清单）

> 布局与命名规范以 [`development-guide.md`](../../operator-design/development-guide.md) §1.2 为唯一事实源；共享层零复制、CMake 直引 CANN 内置 apace。

| 目录 | 文件 | 职责（本场景） |
|:---|:---|:---|
| kernel/ | `{op_name}_tiling_data.h` | 单份完整 mm tiling + `commTilingData` + 通信派生字段（每卡行数/chunk 字节/staging 大小/归约粒度）+ 就地 `CommContext`；**无 `localMatmul` 字段** |
| kernel/ | `{op_name}_impl.h` | Impl：Init/Run 编排；AIC 统一 `for t` 循环（T=1 自然退化）+ SetFlag；AIV 逐轮 WaitFlag 门控 + Commit/Wait + SyncAll + 归约 |
| kernel/ | `{op_name}_frag_kernel.h` | FragmentTensor mm 内核（A1），独立 namespace |
| kernel/ | `reduce_sum_ref.h` | 归约文件（A4），独立 namespace |
| src/ | `kernel_launcher.h` | 4 个 dtype 变体 `__global__` 入口（E4M3E4M3/E5M2E5M2/E4M3E5M2/E5M2E4M3），各含 `KERNEL_TYPE_MIX_AIC_1_1` |
| src/ | `main.cpp` | 前置校验 → T 派生 → staging 分配（`M×N×sizeof(CType)`）→ dtype dispatch → fork 多 rank + TCP rootInfo + 建链 + launch（perf 模式内嵌 L2 flush） |
| src/ | `root_info_exchanger.h`、`utils.h` | [REUSE] 从参考算子复制 |

---

## 3. Host 侧要求

### 3.1 前置校验清单（fork/建链前执行，main.cpp 与 gen_data.py 双侧）

**正确性类强制校验**：

| # | 校验项 | 违反后果 |
|:---|:---|:---|
| 1 | `M % rankSize == 0` | M 轴切分错位 |
| 2 | 对齐约束（`K%32`、`N%16`、单尾块 tailM 16 对齐且 ≤ headMSize 等，按 dtype/fixpipe 32B 行对齐推导） | 跨卡 Win 槽污染 |
| 3 | `usedCoreNum ≥ rankSize` 且 `usedCoreNum - rankSize ≥ 1`（归约核至少 1 个） | 无核执行归约/rendezvous 挂死 |
| 4 | Win 容量 `M×N×sizeof(CType)` ≤ `HcclGetHcclBuffer` 实测值 | 通信越界 |
| 5 | `R ≤ 32`（MAX_FRAGMENT_COUNT 单次构建 fragment 数上限，与 T 无关） | FragmentTensor 越界 |
| 6 | 新增输入 buffer 预算（bias 等 L1/BT 兜底核算） | L1/BT 溢出 |

**风险提示类（不作 host 强制拒绝，按 case 复核）**：
- flag 计数：计数器 0-15 衡量未消费积压（紧邻配对下 ≈1-2 不触顶，不构成 T 上限）；以 Set/Wait 严格配对为硬约束，T 上限按 case 实测确定（计数式配对已在多 tile 场景工程验证）。
- 单轮 PUT 大小：无官方上限（bring-up 期过大单轮曾见间歇失败），大单轮须 case 复核（连续多轮精度 + 重复 launch 一致性）。

### 3.2 T 派生

- 优先 `T | mSeg` 无尾块（R10，实现最简）；**单尾块直传合法**（`tailMSize = mSeg % headMSize`，16 对齐且 ≤ headMSize）：如实填 `commTilingData` 尾块字段、tail tiling 入参 `m = rankSize × tailMSize`（注意 tail tiling 的 M 语义是 R 段拼接的逻辑总行数）
- 多尾块或尾块 > headMSize 时走策略 A：`paddedCurTileM` 32 对齐 + `realFragmentSize` 限读 + 多套 tiling（全量 + head + tail）；三条路径均合法，见 [`fusion.md`](../../fundamentals/fusion.md) §6.2.7
- tileM 选择按计算最优优先；T 上限按 case 实测确定

> **约束联合推导原则**：单条约束各自满足 ≠ 联合可行。host 校验与 T 派生必须覆盖派生边界（尾块对齐、Win 容量、归约核数），cases.csv 必须包含约束边界用例（strided 场景、尾块边界、R 边界）——硬件隐式上限类缺陷（静默丢零、无报错）只在边界用例下暴露。

### 3.3 dtype dispatch

- host 按 `dtypeA`/`dtypeB` **运行期分派**到 4 变体入口，禁止硬编码单入口（硬编码 → 异 dtype 字节流被错误模板解释，精度系统性失败）
- dispatch 宏模板见 [`development-guide.md`](../../operator-design/development-guide.md) §3.5
- Win 数据区偏移按 host 建链布局确定，kernel/host 偏移同源（R14）

---

## 4. 验证矩阵（compute-first 特有项）

| 类别 | 项 | 达标条件 |
|:---|:---|:---|
| 精度 | T=1 与 T>1 双路径 | 只测 T=1 会掩盖多 tile 布局 bug，双路径必须全 PASS |
| 精度 | tail/非对齐 shape + 多 commTurn | tail 路径特征信号暴露 |
| 精度 | per-tile 契约不变量 | 每轮 mm 只算本轮子区间（禁止"全量 mm × T 次"）；归约每轮只处理本轮行区间 |
| 精度 | golden 语义 | 每卡"完整 K、切 M"，输出 `[M/R, N]`；golden FP32 累加路径 + 固定种子 |
| 精度 | dtype 全覆盖 | 4 变体各自 PASS；E5M2 全错 → 查 dispatch |
| 性能 | tileCnt 扫描 | 精度期 tileCnt=1 串行基线 → 性能期扫描（上限按 R9 口径），切 tileCnt 必须重调 `GetTilingData` |
| 性能 | 投产门槛（R15） | 真实大 shape × R=2/4 双档 × 与同语义参考路径对标归档（融合算子或分步实现）；toy shape = FAIL；性能 shape 向业务方索取真实清单 |
| 性能 | L2 flush 实接线（R20） | perf 每轮调 flush kernel，msprof 记录数 == 轮数 |

### 4.1 对标基线工程已知坑（R15 落地，生产试错记录）

> 环境口径：dav-3510 / CANN 9.2.0 / fork 直调多 rank。R15 要求与参考路径对标归档，但 aclnn 基准算子的**调用契约**（参数校验规则、group 机制、引擎模式）无现成文档——以下为实测踩坑记录，完整契约**未闭环**，设计期即应确定对标路径并预留试错时间。

| 候选基线 | 已知坑（实测） | 结论 |
|:---|:---|:---|
| `aclnnMatmulReduceScatterV2`（融合） | CCU 引擎在 fork 直调场景报 "CCU tiling accelerator not support"（需框架侧机制注入，超出直调合理成本）；FP8 输入**必须带 scale**（per-tensor 量化语义，无 MX scale 输入，对标须声明差异）；scale dtype/形状校验严格（FP32/E8M0、[M/128,K/128] 或 [K/128,N/128] 类分块）；world group 不可手动创建（须用已初始化通信域）；streamMode/commMode 参数组合有硬约束；宿主链接 libhccl.so 可能误链系统 9.1.0 版本 | 融合路径在直调场景常判 N/A（须留证归档），转分步路径 |
| `aclnnQuantMatmulV5` + `HcclReduceScatter`（分步） | 两段调用 + 中间同步，参数校验同样严格；须确认 aclnn 侧 scale 语义与被测算子对齐 | **推荐优先尝试**的分步对标路径 |
| mc2 融合算子直调 | 候选检索方法见 R15（按 op 名模式搜 `$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/kernel/`），勿只查单一算子名即下 N/A 结论 | 首选（同语义同框架） |

> 工程纪律：对标试错属**设计期活动**——在 PLAN 中登记对标路径与预期成本；试错 ≥10 轮仍不可行时归档 N/A 证据（报错原文 + 参数组合）转下一候选，禁止无限试错。

---

## 5. 实现细节模板（本场景专属落地形态）

> 通用方法论以 [`fusion.md`](../../fundamentals/fusion.md) §6.2 为唯一事实源；本节为落地骨架，新算子照搬结构、数值按自身 shape/dtype 重新推导。

### 5.1 AIV 严格分离编排骨架

```
AIV (RunAllToAllAndReduceSum):
  jobIndex = GetBlockNum() - 1 - GetBlockIdx()      ← 后 R 核通信
  isCommBlock    = (jobIndex < rankSize)
  isComputeBlock = (blockIdx < usedCoreNum - rankSize)  ← 归约核 = 前 (核数-R) 核

  for t = 0..T-1（统一循环，T=1 自然退化）:
     全 AIV: CrossCoreWaitFlag(flagId) → SyncAll   ← 无模板形态（Wait 后有标量流 Commit 下发，见 fusion.md §6.2.3）
    通信核: Commit → Wait<BARRIER_NONE>（仅 Drain）
    全 AIV: SyncAll
    jobIndex=0: teamBarrier_.CrossDevice()          ← 显式跨卡 fence
    全 AIV: SyncAll
    归约核: t≥1 时 ReduceSum(t-1)（与下一轮通信重叠）
  尾轮: 归约核 ReduceSum(T-1)；通信核 Finalize
```

> 同步机制选型依据（后 R 核映射下 TeamBarrier totalJobs=1 + 显式 CrossDevice，与官方前 R 核映射的 `Wait<BARRIER_DEVICE>` 模式互斥，机制分析见 [`communication.md`](../../fundamentals/communication.md) §2.2）；SyncAll 无跨设备能力，跨卡可见性只能走 CrossDevice。

### 5.2 批量归约循环骨架

> 归约实现必须同时提供两路径（[`fusion.md`](../../fundamentals/fusion.md) §6.2.6）：逐行版（T>1 多 tile 路径）与批量版（tileCnt=1 路径，一次处理多行摊薄 flag 次数）。

**归约多核分治（必做）**：归约核集合（前 `rsCoreNum = usedCoreNum - rankSize` 核）内按行块均分本轮 `tileM` 行（连续行块、余数前摊），禁止单核归约（如 `GetBlockIdx()==0` 独担全量行）——归约耗时独占且多核误写同一输出区产生写竞争。

```
batchRows = UB预算 / (每行字节数 × 每元素 buffer 系数)   // host 按 UB 推导
for batch in 按 batchRows 分批（覆盖本核行区间）:
    for i = 0..R-1（R 个来源，pingpong 双缓冲）:
        2D DataCopyPad(srcBuf[i%2], 来源 i 的本批行区间, blockCount=本批行数, 64B pitch)
        Cast(srcFP32[i%2], srcBuf[i%2], CAST_NONE)      // 独立 srcFP32，禁止 in-place
        PipeBarrier<PIPE_V>()
        Add(accFP32, accFP32, srcFP32[i%2])
    Cast(dstBF16, accFP32, CAST_RINT)
    2D DataCopyPad(yGm 本批行区间, dstBF16, blockCount=本批行数)
    // 一批数据一次 SetFlag/WaitFlag（FIFO 特性摊薄）
```

### 5.3 归约事件配对模板（R19 落地形态）

4 类 HardEvent 的语义与 Set/Wait 时机以 [`fusion.md`](../../fundamentals/fusion.md) §6.2.6 为准；落地要点（pingpong slot = i%2）：

- Init 时预初始化 `MTE3_V`（首次 Wait 前需有对应 Set）
- slot 复用前 `WaitFlag<V_MTE2>(slot)`；搬入后 `SetFlag<MTE2_V>` → Wait → Cast/Add → `SetFlag<V_MTE2>` 释放
- 循环结束**消费残留事件**（保持计数平衡，漏消费 → 后续挂死）；R=1 边界注意 slot 1 从未 Set 的未定义行为
- 输出路径：等上批 MTE3 完成 → Cast → `V_MTE3` 配对 → 搬出 → 为下批预置 `MTE3_V`

### 5.4 手动 UB 布局与预算

> **UB 管理机制红线**：归约区 UB 管理必须与通信区使用同一机制——`TPipe::InitBuffer` 与 `Te::MakeMemPtr` 禁止混用（偏移空间不共享 → 地址重叠 → MTE2 越界）。推荐 `MakeMemPtr` 手动偏移（与官方算子一致，[`operator-anatomy.md`](../../operator-design/operator-anatomy.md) §4.3）。

slot 结构（示例布局）：srcBuf[2]（pingpong 搬入）+ srcFP32[2]（pingpong Cast 目标，独立）+ accBuf（FP32 累加器，单份）+ dstBuf（输出）。

host 侧 UB 预算推导（参数化公式，系数按 dtype 路径代入）：

```
perElemBytes = 2×sizeof(CType) + 2×sizeof(float) + sizeof(float) + sizeof(CType)
availableUB = UB 总量 - guard 通信区
maxElements = availableUB / perElemBytes
redUbN      = min(N, 单次列宽上限) 且按对齐要求取整
redUbM      = min(tileM, maxElements / redUbN)
            // N > redUbN（strided 场景）时追加 redUbM ≤ 32（fusion.md §6.2.6 纪律 3 硬件限制）
```

> *工程提示：整片申请 UB 后 kernel 内运行期算偏移（按 [sum｜bf16×2｜f32×2｜out] 划分）与 host 传参布局同为合法形态，按 UB 预算与批量宽度选择。*

### 5.5 多套 tiling 字段契约（tiling_data.h）

```cpp
struct {OpName}TilingData {
    CommTilingData commTilingData;          // 通信切分（5 字段）
    QuantMatmulTilingData mmTilingData;     // 全量 tiling（GetTilingData(m, n, k)），T=1 退化用
    QuantMatmulTilingData subMmTilingData;  // head 子问题 tiling（GetTilingData(headMSize, n, k)），T>1 时强制存在
    QuantMatmulTilingData tailMmTilingData; // tail tiling（GetTilingData(rankSize*tailMSize, n, k)），无尾块时字段仍须声明
    // 通信派生字段：coreNum / rankSize / mSeg / chunkBytes / tileMaxBytes / stagingSize / redUbM / redUbN
};
// kernel 侧选择：(commTurn == 1) ? mmT : (isHeadTile ? subMmT : tailMmT)；blockDim 恒用 mmTilingData.usedCoreNum
```

> **T>1 时 subMmTilingData 强制存在**：全量 tiling 的 baseM 可能大于 headMSize 导致 tile 分布不合理，必须为 headMSize 独立调 `GetTilingData`。

### 5.6 headMSize 标定原则

决策原则（通用）："分片小 → 减小 tile 让通信尽早启动；分片大 → 增大 tile 减少同步次数"。分档实质由 `tileMaxBytes = tileM×N×sizeof(CType)` 主导（N 维度），非纯 R 档——同 R 下小 N 与大 N 的最优 tileM 方向相反。**数值为示例标定，新算子按 tile 粒度与流水深度目标自行标定**：异常特征 case（tileCnt 非最优、SyncAll 占比偏高）须按通信覆盖率与收益实测搜索，单点最优值不跨 case 复用。对齐：16 的倍数；单尾块 16 对齐且 ≤ headMSize（host 校验）。

### 5.7 自研 FragmentTensor mm kernel 骨架

参考 AllGather `qmm_mx_kernel_ag_udma.h`（`QmmMxKernelAgUdma`）改造，独立 namespace。改造时区分**必须保留**与**可省略**：

| 必须保留（砍了即错） | 可省略（compute-first 特有） |
|:---|:---|
| per-fragment L1 隔离（`QmmMxBlockMmadFragment` 跨 rank 边界自动切 fragment） | dependId 预触发 / `WaitFlag`（计算不依赖通信） |
| fragment 地址公式：A 段 `aGM + r×mPerRank×fullK`（**fullK = 完整 K 行 stride，A 不切分**），C 段 `stagingGm + r×mPerRank×cBytesPerM`；`cFragAddrs_` 保持原始 rank 顺序 | Win 区离散地址解析（A 为本卡 GM 连续 rank 段） |
| localLast 编排（强制，见 §5.8）：fragment 重排 `[remote..., local]` + 边界提前 SetFlag | winDataBase（无 Win 区读 A） |
| `SetL2Cache`（fullMBlock + 128B 对齐判定）+ tail tile `sch.UpdateTailTile` | — |

> **TransA/TransB 参数化**：Impl 模板必须含转置参数（Layout 条件选择），禁止固定 Layout。
>
> **地址公式红线**：A 行 stride = 完整 K；行宽必须用**字节**（`r × mPerRank × K × sizeof(AType)`）——把元素数当字节偏移仅在 1 字节 dtype 巧合成立，FP16/BF16 扩展时静默错位。官方 kernel 用显式 `dataBytesPerMRow` 字节行宽，照搬时勿"简化"。
>
> **借鉴官方 AG kernel 必改项**：其 `cFragAddrs_[8]` 为硬编码 8 槽——R > 8 静默越界，必须扩为 `MAX_FRAGMENT_COUNT=32`；其三区 HEAD/MAIN/TAIL 懒构建结构在 per-tile 调用形态下可按统一 for-t 轮次重建形态省略。

**Params 结构契约**（核心字段）：`mmTile`（当前轮 tiling 指针）/ `qbmmParams` / `rankSize, rankId` / `mPerRank` / `tileM`（problem M = R × tileM）/ `k, n, scaleKLen`（k = 完整 K）/ `aGM, aScaleGM, bGM, bScaleGM`（B 全 rank 共享）/ `stagingGm`（C 输出 = PUT 通信源）/ `cBytesPerM, tileMOffset`。

核心方法：`BuildFragmentTensors`（cFragAddrs_ 按 staging 布局填、addrList localLast 重排、cFragAddrs_ 保持原始顺序）→ `Run`（构造 scheduler + FragL1Params + mmadFrag.Init）→ `Process`（调度循环：SetL2Cache → Slice → `mmadFrag(...)` 17 参调用散射到 staging）。

### 5.8 localLast 双 flag 通信提前启动（compute-first 默认编排）

机制原理见 [`fusion.md`](../../fundamentals/fusion.md) §6.2.2/§6.2.3。落地要点：

1. **localLast 重排只作用于 FragmentTensor addrList**：构造时跳过本 rank 段、最后追加；**`cFragAddrs_` 必须保持原始 rank 顺序**（mmadFrag 内部 L1 cache 管理依赖原始顺序）——写错 = 阻塞级错误。
2. **边界预计算**：`localFragBoundary = headMainRows - fragM`；调度循环内首次 `mPos >= localFragBoundary` 时 `SetFlag(flagA)`。
3. **兜底补 Set**：未跨边界核（无本卡 tile）必须补 Set flagA——否则 AIV `WaitFlag(flagA)` 挂死。
4. **AIV 两侧分等**：通信核 `WaitFlag(flagA)` 后启动 AllToAll；reduceSum 前 `WaitFlag(flagB)` 确认本卡段完成。

### 5.9 per-tile 子区间契约代码模板（AIC RunMatmul，T>1 实现红线）

```cpp
for (uint32_t t = 0; t < commTurn; ++t) {
    uint32_t curTileM    = GetTileM(t);        // t < headTileCnt ? headTileM : tailTileM
    uint64_t tileMOffset = GetTileMOffset(t);  // t < headTileCnt ? t*headTileM : headTileCnt*headTileM
    const auto& tileT = (commTurn == 1) ? mmTiling : (isHeadTile ? subMmTiling : tailMmTiling);
    params.aGM      = aGm         + tileMOffset * axisK;                 // A 按行偏移
    params.aScaleGM = scaleAGm    + tileMOffset * scaleARowBytes;
    params.cGM      = workspaceGm + tileMOffset * axisN * sizeof(CType);
    mmKernel_(params);                          // problem M = R × curTileM
    CrossCoreSetFlag<0x2, PIPE_FIX>(flagId);
}
```

不变量与违反后果见 [`development-guide.md`](../../operator-design/development-guide.md) §3.5 per-tile 契约表（唯一事实源）。

---

## 6. 合规映射（本场景重点项）

> 全量红线见 [`review-checklist.md`](../../review-checklist.md)；违反任意红线 = FAIL。

| # | 约束 | 本场景落点 |
|:---|:---|:---|
| R9 | flag Set/Wait 严格配对 + flagId ∈ [0, FLAG_ID_MAX) | A6；以"每轮 Set 双 flag"替代 localLast = 性能 FAIL |
| R10 | 尾块策略 | §3.2 T 派生：默认 `T \| mSeg` 无尾块，否则策略 A padding + 多套 tiling |
| R14 | Win 数据/元数据分离（硬红线）+ 单轮 PUT 大小（风险提示） | 偏移三处同源（PUT 写/归约读/host 建链）；单轮大小按 case 复核 |
| R16 | mm 默认 FragmentTensor | A1；"vendor kernel + FragmentTensor C 输出"= 阻塞级错误 |
| R21 | localLast 编排禁止移除 | A1/§5.8 |
| R17 | 归约禁止逐行搬运 | A4：2D DataCopyPad blockCount=本批行数；strided redUbM ≤ 32 或 1D 退化 |
| R18 | 归约独立 srcFP32 双缓冲 | A4：禁止 in-place 加宽 Cast |
| R19 | 归约事件配对完整 | A4/A7：四类 HardEvent 同迭代配对 + 残留事件消费 |
| R3 | 入口变体 + dtype dispatch | kernel_launcher.h 4 变体 + §3.3 |
| R12 | UB 静态通信区隔离 | TPipe 与 MakeMemPtr 二选一 |
| R20 | perf L2 flush 实接线 | §4 性能项 |

---

## 后续阅读

| 文档 | 何时读 |
|:---|:---|
| [`fusion.md`](../../fundamentals/fusion.md) §6.2 | compute-first 架构原理与编排模式（通用方法论唯一事实源） |
| [`dev-methodology.md`](../../operator-design/dev-methodology.md) | 五阶段流程 + D1-D6 决策点（本场景为 D1=计算在前的取值示例） |
| [`operator-anatomy.md`](../../operator-design/operator-anatomy.md) §7 | 文件级契约与 MAX_FRAGMENT_COUNT |
| [`development-guide.md`](../../operator-design/development-guide.md) §3.5 | per-tile 子区间契约、dispatch 宏模板 |
| [`review-checklist.md`](../../review-checklist.md) | 全局红线 + 场景约束与 FAIL 诊断 |
