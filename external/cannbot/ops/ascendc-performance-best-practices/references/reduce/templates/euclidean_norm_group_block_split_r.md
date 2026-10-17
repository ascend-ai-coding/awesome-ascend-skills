# EuclideanNorm R 轴分核基础模板（group / BlockSplitR）

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

> 模式族：**R 轴跨核分核（BlockSplitR）**，见 [guide.md](../guide.md)「R 轴分核归约模式族」。
> 本模板对应源码 kernel 类 `EuclideanNormKernel<DType,isTailR>` 的 `InitGroup` / `ProcessGroup` 入口
> （与 normal 的 `Init`/`Process` **共用一个类**，group 复用 normal 全套 UB/引擎）。
> 源码：`euclidean_norm.h`（`InitGroup`~1018、`ProcessGroup`~1034、`SyncAll`~1099），
> 对应 host `ShouldUseGroup` / `ComputeGroupSplit`。

## 1. 模板用途

EuclideanNorm（`y = sqrt(Σ x²)`，典型 reduce 算子）**用得最多的基础离线模板**，等价 batch_norm 的
**BlockSplitR** 模式：**大 R case——R 超出单核 UB 容量，且 A 维乘积小（单纯按输出/行分核用不满核）**时，
沿 **R 轴跨核分核**。20 个 EuclideanNorm 用例里 **9 个走它（#1–9 等）**，是最主力的路径。

**与本库其它模板的本质区别（R 轴分核 vs A 轴分核）**：

| | 切分对象 | 跨核合并 | 屏障 | 代表模板 |
|--|---------|---------|------|---------|
| **A 轴分核** | 输出行（A/A1） | 无（每核独占输出行） | 无 | softmax 五模板、normal、交错-tail-A |
| **R 轴分核（本模板）** | 归约轴 R（跨核切 R） | **需要**（各核部分和写 workspace 再合并） | **SyncAll + Phase2** | group / BlockSplitR |

softmax 五模板全部"切 A（行）不切 R"：一行的完整 R 落在一个核内，核间输出不重叠、无需合并。
但当 **A 维乘积 < core_num** 时，切 A 最多只能用满 `A` 个核，其余核闲置；而此时 R 又很大——把 R 切到
多核、各核算一段 R 的部分平方和、再跨核合并，才能**用满核**。这正是 group / BlockSplitR 存在的理由。

## 2. 输入输出布局和归约轴

- 输入 `x`：归一后的严格交替 pattern `[A,R,A,R,...]`（A 起头，偶数位轴 = A，奇数位轴 = R），行优先 dense。
  单 R case（如 `[64000,32]` axes=[0]）经预处理补 leading A 后即 `[A,R,A]`（axisNum=3）；也支持多 R（ARAR…）。
- 归约轴：所有奇数位 R 轴，沿它们求平方和。
- 输出 `y`：所有偶数位 A 轴的 dense 张量，元素数 `aTotal = ∏(A 轴 size)`。
- **分核网格**：2D `[aLoopCntTotal, rGroupCnt]`。
  - **A 维**（`aLoopCntTotal` = ∏外层A × aSplitChunkCnt）：每个 A chunk 对应一段输出。
  - **R 维**（`rGroupCnt`）：把外层 R loop（`rLoopCntTotal` 个 R chunk）等分到 `rGroupCnt` 个核，
    每核只算 R 的一段部分和。
- **workspace 布局** `[rGroupCnt, aTotal]`（fp32 dense，无 padding）：第 `rChunkIdx` 行存该 R 分组的部分和，
  第 `wsColOff` 列对应该 A chunk 的输出线性列偏移。

## 3. 适用 shape、dtype 和 UB 条件

| 条件 | 要求 |
|------|------|
| dtype | fp32 / fp16 / bf16 / int32 均可（内部一律升 fp32 累加，sqrt in-register，最后 Cast 回原精度） |
| 何时命中 group | `aLoopCntTotal ≤ coreNum/2`（切 A 用不满核）**且** `rLoopCntTotal > 1`（R 已分块，有跨核 R 可切），见 `ShouldUseGroup` |
| 大 R | R 超出单核 UB → `ComputeRUbFactor` 把 R 切成 `rLoopCntTotal>1` 个 chunk（否则 R 全载，无需 group） |
| 小 A | A 维乘积（`aLoopCntTotal`）小到切 A 用不满核（`≤ coreNum/2`） |
| workspace | host 传 `workspaceBytes = rGroupCnt × aTotal × 4`（fp32） |

> 不满足（A 已够多，切 A 就能满核；或 R 未分块）→ `ShouldUseGroup` 返回 false → 走 normal（切 A）。
> normal 与 group 共用一个类，仅入口 `Init/Process` vs `InitGroup/ProcessGroup` 与 host `templateType` 不同。

## 4. 类名、构造函数和 Init 接口

```cpp
namespace NsEuclideanNorm;

template <typename DType, bool isTailR>
class EuclideanNormKernel {
public:
    __aicore__ inline EuclideanNormKernel();
    // normal 入口（A 轴分核）
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const EuclideanNormTilingData* td);
    __aicore__ inline void Process();
    // group / BlockSplitR 入口（R 轴分核）—— 本模板
    __aicore__ inline void InitGroup(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const EuclideanNormTilingData* td);
    __aicore__ inline void ProcessGroup();
};
```

- **构造函数**：无参（`TPipe pipe_` 为类成员）。
- **InitGroup**：先调 `Init(x,y,td)` 复用 normal 全套 UB/字段初始化，再额外设置 `wsGm_`（workspace 基址）、
  `rGroupCnt_`（= `td->rGroupCnt`）、`aTotal_`（= ∏ A 轴 size = workspace 列数）。
- **ProcessGroup**：两段式（Phase1 各核部分和 → workspace → `SyncAll()` → Phase2 组内 ReduceSum 合并 + sqrt）。

## 5. 对应的 TilingData 和字段说明

对应 `EuclideanNormTilingData`（[euclidean_norm_tiling_data.h](euclidean_norm_tiling_data.h)），group 在 normal
字段之上多读一个 `rGroupCnt`：

| 字段 | 含义 |
|------|------|
| axisNum / axisShape[] / axisStride[] | 归一后的严格交替 pattern（size + dense stride） |
| aSplitAxisIdx / rSplitAxisIdx | UB 内 A/R 的切分轴下标 |
| aUbFactor[Align] / rUbFactor[Align] | A/R 切分因子（含对齐） |
| innerAProd[Align] / innerRProd[Align] | UB 内 A/R 内块元素数（含对齐） |
| **aLoopCntTotal** | 外层 A loop 扁平计数 = 2D 网格 A 维 |
| **rLoopCntTotal** | 外层 R loop 扁平计数（被 `rGroupCnt` 等分） |
| aSplitChunkCnt / aBigCoreLoopCnt / aSmallCoreLoopCnt / aBigCoreCnt | 大小核均衡（Phase2 复用同款逻辑） |
| **usedCoreNum** | 实际使用核数 = `aLoopCntTotal × rGroupCnt` |
| **rGroupCnt** | ★ group 专属：R 分组数 = workspace 行数（`ProcessGroup` 用它把 blockIdx 拆成 2D 网格） |
| preReduceUbSize / postReduceUbSize / tmpBufUbSize / cacheBufUbSize | UB buffer 字节数（Phase2 反推 aUbFactorP2 用） |

## 6. Host tiling 参数计算方法（ShouldUseGroup / ComputeGroupSplit）

使用 [euclidean_norm_tiling.h](euclidean_norm_group_tiling.template)（完整驱动 `ComputeEuclideanNormTiling`）：

```cpp
euclidean_norm_host::Result r;
bool ok = euclidean_norm_host::ComputeEuclideanNormTiling(
    xShape, reduceAxes, /*dtypeCode fp32=*/0, coreNum, ubSize, r);
// ok==true 且 r.templateType==1 → 用本模板（group）；r.td 即 EuclideanNormTilingData
// r.workspaceBytes = rGroupCnt × aTotal × 4
```

主入口先试交错 fast（`TryComputeFastTiling` / `TryComputeFastRARATiling`），不命中再走
`ComputeAUbFactor → ComputeRUbFactor → ExpandAIfRFullyLoaded → ComputeFusedALoopSplit → ComputeRLoopCnt`，
最后：

```cpp
// —— 门控：是否走 R 轴分核 group —— （euclidean_norm_tiling_host.h:566）
inline bool ShouldUseGroup(const Ctx& ctx) {
    if (ctx.aLoopCntTotal > ctx.coreNum / 2) return false; // 切 A 已够满核 → normal
    if (ctx.rLoopCntTotal <= 1)             return false;  // R 未分块，无跨核 R 可切 → normal
    return true;                                            // 大 R + 小 A → group
}

// —— R 分核多核公式 —— （euclidean_norm_tiling_host.h:577）
inline void ComputeGroupSplit(Ctx& ctx) {
    int64_t totalOuter = ctx.aLoopCntTotal * ctx.rLoopCntTotal;     // 2D 网格总格数
    int64_t perCoreNum = CeilDiv(totalOuter, ctx.coreNum);
    int64_t numBlocks  = CeilDiv(totalOuter, perCoreNum);
    // 把 numBlocks 对齐到 aLoopCntTotal 的整数倍（保证每 A 列的 rGroupCnt 相等，aPerCore=1）
    if (CeilAlign(numBlocks, ctx.aLoopCntTotal) <= ctx.coreNum)
        numBlocks = CeilAlign(numBlocks, ctx.aLoopCntTotal);
    else
        numBlocks = FloorAlign(numBlocks, ctx.aLoopCntTotal);
    ctx.usedCoreNum = numBlocks;
    ctx.rGroupCnt   = numBlocks / ctx.aLoopCntTotal;               // 每 A 列切成 rGroupCnt 个 R 分组
    ctx.isGroup     = true;
}

// workspaceBytes = rGroupCnt × aTotal × 4（aTotal = ∏ 所有 A 轴 size）
```

关键量：`rGroupCnt`（R 分组数）、`usedCoreNum = aLoopCntTotal × rGroupCnt`、
`workspaceBytes = rGroupCnt × aTotal × 4`。

## 7. Phase1 / Phase2 流程（ProcessGroup）

```
ProcessGroup（每核 blockIdx）:
  if blockIdx >= usedCoreNum: return              // idle 核早退
  # 2D 网格解码：网格 [aLoopCntTotal, rGroupCnt]，aPerCore=1
  aChunkIdx = blockIdx / rGroupCnt                # 0 .. aLoopCntTotal-1（输出 A 列）
  rChunkIdx = blockIdx % rGroupCnt                # 0 .. rGroupCnt-1（R 分组 = workspace 行）
  # 本核 R 区间 = rGroupCnt 等分 rLoopCntTotal（CeilDiv + 末核兜底）
  rPerCore = CeilDiv(rLoopCntTotal, rGroupCnt);  rStart=rChunkIdx*rPerCore;  rEnd=min(rStart+rPerCore, rLoopCntTotal)
  unravel aChunkIdx → (外层A坐标, aSplitChunk) → chunkGmOff（输入）, chunkOutOff（输出列偏移）

  # ── Phase1：本核算 R 段 [rStart,rEnd) 的部分平方和 → 写 workspace ──
  DoOneAChunkGroup(chunkGmOff, aLen, rStart, rEnd):
    SumSqInitVf()                                 # sumSq[A_elements] = 0
    for i in [rStart, rEnd):                      # 只遍历本核的 R chunk
      rOff = UnravelOuterR(i)                      # 解码 R idx → GM 偏移
      CopyIn → CastSquareVf(x²) → [ClearChunkExtensionVf 清尾] → ReduceRPattern(消解内层R) → SumSqAccumulateVf
    # 结束后 sumSq = 本核 R 段的 fp32 部分和
  Phase1OutputToWorkspace(chunkOutOff, aLen, rChunkIdx):
    V→MTE3 fence
    DataCopyPad(wsGm_[rChunkIdx*aTotal + chunkOutOff], sumSq, ...)   # MTE3 UB→GM，dense

  SyncAll()                                       # 所有核 workspace 写完成（跨核屏障）

  # ── Phase2：workspace 沿 rGroupCnt 维 ReduceSum 合并 + sqrt → 输出 ──
  Phase2Kernel():                                 # RA mini-kernel，见 §8
```

## 8. Phase2 mini-kernel 与 workspace 合并（Phase2Kernel）

Phase2 是一个**独立的 RA 归约 mini-kernel**：把 workspace `[rGroupCnt, aTotal]` 沿 R 分组维（`rGroupCnt`）
ReduceSum 掉，保留 A 维（`aTotal`），再 sqrt 输出。**所有核重新按 A 列分工**（不再是 Phase1 的 2D 网格）：

```
Phase2Kernel（每核）:
  # kernel 侧现算 aUbFactorP2：R 全载（rGroupCnt 行全进 UB），A 方向切 tile
  aUbFactorP2 = floor(preInElems / rGroupCnt) → FloorAlign(8) → clamp(outBuf/tmpBuf/aTotal 三上界)
  按 aTotal 切成 aSplitChunkCntP2 个 A tile，大小核均衡分到 usedCoreNum 核
  for 本核 A tile [a_off, a_off+a_len):
    MTE2: workspace[0:rGroupCnt, a_off:a_off+a_len] → preInBuf[rGroupCnt, a_len_ub]（行 pad 0）
    ReduceSum<fp32, Pattern::Reduce::RA, srcInnerPad=true>：srcShape={rGroupCnt, a_len_ub}
        沿 outer R(rGroupCnt) 求和，保留 inner A(a_len_ub) → tmpBuf[a_len_ub]
    VF: Sqrt(fp32) + Cast→D_T → outBuf（只处理 a_len 有效元素）
    MTE3: outBuf → yGm_[a_off]
```

- **为何用 `Pattern::Reduce::RA`**：workspace 布局 `[rGroupCnt, aTotal]`，要消解的 R 分组维在外层、保留的 A 维在
  内层 → 恰是 RA 排布（外 R、内 A）。`srcInnerPad=true` 让硬件按 sum identity(0) 处理 A 方向的右 padding。
- **sqrt 在 Phase2 合并后一次性做**：Phase1 只累加平方和（可交换），跨核合并后才 `Reg::Sqrt`，数值等价单核。

## 9. workspace 布局与 SyncAll

- **workspace 布局** `[rGroupCnt, aTotal]`，fp32 dense：
  - 行索引 = `rChunkIdx`（R 分组号，`0..rGroupCnt-1`）。
  - 列索引 = 输出线性列偏移 `wsColOff`（= Phase1 的 `chunkOutOff`，与输出 dense 序一致）。
  - `wsOff = rChunkIdx * aTotal + wsColOff`。
  - 大小 = `rGroupCnt × aTotal × sizeof(float)` = host 的 `workspaceBytes`。
- **SyncAll 的作用**：Phase1 各核把部分和写到 workspace 的**不同行**（同一 A 列、不同 R 分组）；Phase2 要读
  同一列的全部 `rGroupCnt` 行做求和，因此必须等**所有核** Phase1 写完 → `SyncAll()` 全局屏障。
  写 workspace 前有 `V→MTE3` fence 保证 VF 写对 MTE3 可见；`SyncAll` 保证跨核可见。
- **无原子累加**：跨核合并靠"各核写不同行 + Phase2 单核读整列 ReduceSum"，避免 GM 原子写竞争。

## 10. 独立 Kernel 入口示例

```cpp
#include "euclidean_norm_tiling_data.h"
#include "dav310/euclidean_norm_group_block_split_r.h"

// LoadTilingData：把 GM 上 TilingData 逐字段搬进核内局部结构
extern "C" __global__ __aicore__ __vector__ void
euclidean_norm_group_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    EuclideanNormTilingData td;
    LoadTilingData(reinterpret_cast<__gm__ const EuclideanNormTilingData*>(tiling), &td);
    // isTailR 由 host tilingkey 决定；此处以 fp32 + tail-A(单R 补 leading A) 为例
    NsEuclideanNorm::EuclideanNormKernel<float, /*isTailR=*/false> op;
    op.InitGroup(x, y, workspace, &td);   // ★ group 入口，需传 workspace
    op.ProcessGroup();
}
```

> workspace 大小 = `rGroupCnt × aTotal × sizeof(float)`，由 host tiling 的 `workspaceBytes` 给出。
> dtype/isTailR 两维模板参数由 host tilingkey 分发（fp16/bf16/int32 同样支持，内部升 fp32 计算）。

## 11. 数值语义、代表 case 与实测

- **数值语义**：平方和是**可交换归约**（`Σ x²` 与累加顺序无关）→ 各核任意切 R、部分和跨核合并、Phase2 再 sqrt，
  结果与单核逐 R 累加**逐位等价**。sqrt 只在最终 Phase2 合并后做一次（in-register `Reg::Sqrt`，无需 TilingData 传系数）。
- **精度**：fp16/bf16/int32 输入一律升 fp32 累加 + sqrt，输出 Cast 回原精度（int32 用 `CAST_TRUNC`，与 TF `reduce_euclidean_norm(int32)` 一致）。

| # | case | 合轴/布局 | baseline | merged | 加速 | 选型 |
|---|------|----------|--------:|------:|-----:|------|
| 1 | ra_group_0001 | RA 大R `[64000,32]` axes[0] | 14.3 | 13.3 | 1.08× | **group** |
| 2 | ra_group_0002 | RA fp16 | 13.3 | 12.2 | 1.09× | **group** |
| 3 | ra_group_0003 | RA bf16 | 13.4 | 12.5 | 1.07× | **group** |
| 4 | ra_group_0004 | RA | 17.3 | 17.4 | 1.00× | **group** |
| 5 | r_group_0005 | R fp16 | 9.9 | 10.9 | 0.91× | **group** |
| 6 | ara_group_0001 | ARA `[4,3200,24]` axes[1] | 11.4 | 10.5 | 1.08× | **group** |
| 7 | ara_group_0002 | ARA fp16 | 10.9 | 10.9 | 0.99× | **group** |
| 8 | arar_group_0001 | ARAR（4轴）`[2,320,14,320]` axes[1,3] | 16.7 | 16.8 | 1.00× | **group** |
| 9 | ararara_0002 | 7轴 tail-A，slabCnt=8（交错 fast 判负回退） | 61.8 | 60.4 | 1.02× | **group** |

> 平台：Ascend 950 / arch35(dav-3510)，CANN 9.0.0，coreNum=72，UB≈192KB。数据源 `PERF_TUNING_REPORT.md §2`。
> host 自检（g++ 纯 host）：#1 `[64000,32]` axes[0] → `templateType=1`（group），`rGroupCnt=61`、
> `usedCoreNum=61`、`workspaceBytes=61×32×4=7808`；#6 → `rGroupCnt=21`、ws=8064；#8 → `rGroupCnt=2`、ws=224。
> group 是最主力路径（9/20 case），加速温和（~1.0–1.1×）——它的价值是**把大 R 摊到多核用满核**，而非某个病态点的大赢。

## 12. 不适用场景和可选替代

| 不适用场景 | 说明 / 替代 |
|-----------|------------|
| A 维乘积够大（`aLoopCntTotal > coreNum/2`） | 切 A 就能用满核，无需跨核切 R + 合并开销 → 走 **normal**（同类，`Init`/`Process`） |
| R 未分块（`rLoopCntTotal ≤ 1`，R 全载 UB） | 无跨核 R 可切 → 走 normal |
| 8 轴 tail-R 交错 + burst 病态小 | 先试 [交错-tail-R（fast）](euclidean_norm_interleaved_contiguous_tailr.md)，不命中回退 group |
| 7 轴 tail-A 交错 + 内块每行 ≥2KB | 先试 [交错-tail-A（fast-RARA）](euclidean_norm_interleaved_contiguous_tail_a.md)，不命中（如 #9 slab 太少）回退 group |
| 简单 AR / ARA（可合成单 R，且 A 够大） | softmax_v2 系列 AR/ARA 模板 |
