# EuclideanNorm 交错连续载入-两段式模板（路径 A / tail-R）

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

> 模式族：**多轴交错归约（interleaved A-R）**，见 [guide.md](../guide.md)「多轴交错归约模式族」。
> 本模板对应源码 kernel 类 `EuclideanNormFastKernel`。

## 1. 模板用途

针对 **偶数轴严格交错 `ARARAR...AR`（tail-R，≥6 轴）** 布局的 EuclideanNorm（`y = sqrt(Σ x²)`）。这类 pattern 下，普通归约模板的 CopyIn 为了把 R 聚拢做"边搬边转置"，最内 burst 只有 `sizeof(最内R轴)×dtype` 字节（代表 case 仅 4~8B），MTE2 完全打满（ratio≈0.995，有效带宽仅 ~145 GB/s）。

本模板利用**平方和可交换**：不再按逻辑 (A,R) 顺序碎读，而是**按内存物理连续顺序做大 burst 载入（16~64KB）**，把 A/R 分离从"昂贵的 HBM 碎读"搬到"廉价的 UB 内规约"。支持 fp32/fp16/bf16（载入即升 fp32，输出即降回）。

## 2. 输入输出布局和归约轴

- 输入 `x`：预处理归一后的偶数轴 pattern `[A,R,A,R,...,A,R]`（tail-R，去 1 / 合轴 / 补 leading A / 补 R 之后，A 起头严格交替），行优先 dense。
- 归约轴：所有奇数位轴（R），沿它们求平方和。
- 输出 `y`：所有偶数位轴（A）的 dense 张量，元素数 = ∏(A 轴 size)。
- **内块** = 末 4 轴 `[A0,R0,A1,R1]`（`fastInnerStart=axisNum-4`，物理连续），UB 内保留 (A0,A1)、规约 (R0,R1)。
- **rows 轴** = 轴 `fastInnerStart-1`（R，`stride==innerElems`），沿它多核切分 + 连续大 burst 载入。
- 外层轴 `0..fastInnerStart-2`：外层 A → 分组（group，与输出 slab 一一对应）；外层 R → 核内 `AccumulateOuterR` for 循环累加。

## 3. 适用 shape、dtype 和 UB 条件

| 条件 | 要求 |
|------|------|
| dtype | **fp32 / fp16 / bf16**（载入即升 fp32 累加/规约，输出降回；int32 不走 fast） |
| 轴数 / 类型 | 预处理后 `axisNum` 偶数且 ≥6，严格 `ARAR...AR`（`isTailR==true`） |
| 内块连续 | 末 4 轴 `[A,R,A,R]` 物理连续：`axisStride[N-1]==1` 且 `axisStride[rowsAxisIdx]==innerElems` |
| 内块对齐 | `innerElems × dtypeSize` 须 32B 对齐（kernel 连续 `DataCopy` 载入要求） |
| burst 病态 | 最内 R 轴 `R1 × dtypeSize ≤ 16B`（连续大 burst 的收益区，否则回退） |
| rows 轴够大 | `rowsSize ≥ 8`（小 rows 时两步规约+屏障开销反超 MTE2 增益） |
| 内块可整块载入 | `chunkA0 == szA0`（**内块整块 fits UB，无需分块**）——需分块时实测负收益（-9%），门控直接回退 |
| 多核 | `groupCnt = ∏(外层A轴) ≤ coreNum`，且 rows 轴够切分 |

## 4. 类名、构造函数和 Init 接口

```cpp
namespace NsEuclideanNorm;

template <typename DType>  // D_T ∈ {float, half, bfloat16_t}
class EuclideanNormFastKernel {
public:
    __aicore__ inline EuclideanNormFastKernel();
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const EuclideanNormTilingData* td);
    __aicore__ inline void Process();
};
```

- **构造函数**：无参（`TPipe pipe_` 为类成员，非外部传入）。
- **Init**：镜像 TilingData（pattern + fast* 字段），推导内块几何（szA0/szR0/szA1/szR1、innerElems、outLanes、chunkA0），设置 GM/workspace，分配 UB（inQ 双缓冲 + acc/tmp/red/part/outQ + Phase2 的 wsInQ）。`D_T` 决定 inQ/outQ 与 GM 的元素类型；workspace 恒为 fp32。
- **Process**：`coresPerGroup==1` 时直接 `WriteOutDirect`（sqrt→yGm，无 workspace）；`coresPerGroup>1` 时两段式（Phase1 各核部分和 → workspace → SyncAll → Phase2 组内求和 + sqrt）。

## 5. 对应的 TilingData 和字段说明

对应 `EuclideanNormTilingData`（定义见 [euclidean_norm_tiling_data.h](euclidean_norm_tiling_data.template)），本模板读取 pattern 字段 + `fast*` 字段：

| 字段 | 含义 |
|------|------|
| axisNum / axisShape[] / axisStride[] | 归一后的偶数轴 pattern（size + dense stride） |
| usedCoreNum | 实际使用核数 = groupCnt × coresPerGroup |
| fastInnerStart | 内块起始轴下标（= axisNum-4） |
| fastGroupCnt | 分组数 = ∏(外层 A 轴 size) = 输出 slab 数 |
| fastCoresPerGroup | 每组核数（切 rows 轴） |
| fastRowsAxisIdx | 被切分 + 连续载入的轴（= fastInnerStart-1，R 轴，stride==innerElems） |
| fastRowsPerLoad | 每次 DataCopy 连续载入的 row 数（增大 burst，1~8） |
| fastInnerChunkA0 | 内块最外 A0 分块大小；命中门控时 = szA0（不分块） |

## 6. Host tiling 参数计算方法

使用 [euclidean_norm_tiling.h](euclidean_norm_tiling.template) 中的 `TryComputeFastTiling`（或驱动 `ComputeInterleavedTiling`）：

```cpp
euclidean_norm_host::Result r;
bool hit = euclidean_norm_host::ComputeInterleavedTiling(
    xShape, reduceAxes, dtypeCode, coreNum, ubSize, r);
// dtypeCode ∈ {0=fp32, 1=fp16, 2=bf16}；hit==true 且 r.templateType==2 → 用本模板；r.td 即 EuclideanNormTilingData
```

关键计算 / 门控顺序：
1. `dtype ∈ {fp32,fp16,bf16}` && `axisNum` 偶数且 ≥6 && `isTailR`。
2. 连续性校验（`axisStride[N-1]==1`、`axisStride[rowsAxisIdx]==innerElems`）。
3. `innerElems×dtypeSize` 32B 对齐（连续 `DataCopy` 安全）。
4. `R1×dtypeSize ≤ 16B`（burst 病态判据，连续化收益区）。
5. `rowsSize ≥ 8`（rows 太小则两步规约+屏障开销反超增益）。
6. `chunkA0 = (ubSize - 固定buf) / perA0`，clamp 到 `[1, A0]`；要求 `chunkA0 == A0`（不分块）。
7. `groupCnt = ∏(外层A轴)`，要求 `groupCnt ≤ coreNum`。

## 7. CopyIn、归约、写出流程

```
Process（每核 blk → group g = blk/coresPerGroup, core-in-group c = blk%coresPerGroup）:
  Phase1:
    for A0-chunk (命中时整块一次):
      ZeroAcc(acc)
      AccumulateOuterR: 遍历外层 R 组合 →
        AccumulateRows: 本核 rows 段 [rowStart,rowEnd) 连续大 burst 载入切片
                        → (Cast) VF 平方 + 累加到 acc[A0,R0,A1,R1]
      ReduceInnerBlock: 两步 AR 规约 → part[A0*A1 段]
  if coresPerGroup == 1:
    WriteOutDirect: Sqrt(part) → (Cast→D_T) → yGm[g*outLanes]    // 无 workspace，省去两段式同步
  else:
    WritePartToWorkspace(part, g, c)   // MTE3 → workspace[g, c, outLanes]
    SyncAll()
    Phase2（仅 c==0 的核）:
      载入本组 coresPerGroup 份 part → VF 组内求和 → Sqrt → (Cast→D_T) → yGm[g*outLanes]
```

### 两步 AR 链式规约（`ReduceInnerBlock`）

arch35 的 `ReduceSum` **只支持 AR/RA 两级 pattern**，多维 `ARAR` 会编译报错。故内块 `[A0,R0,A1,R1]` 规约拆两步：

1. **step1**：`ReduceSum<Pattern::Reduce::AR>` 规约最内 `R1`：`src[A0*R0*A1, R1] → red[A0*R0*A1]`。
2. **step2**：逐 `(a0,a1)` 沿 `R0` **标量累加** → `part[(a0c+a0)*A1+a1]`。不用 adv_api `ReduceSum`：其 AR 规约在 `dimR==vlSize(64)` 边界处结果偏大（实测 `szR0=64` 的 6 轴 case 平方和高估 ~4.5%）。step2 非 MTE2 热点（每核每 A0-chunk 仅一次），标量规约既正确又开销可忽略。

### coresPerGroup==1 快路径

`coresPerGroup==1` 时本核独占整个 group（rows 轴不跨核切分），省去 `WritePart→SyncAll→Phase2` 的两段式：直接 `Sqrt(part) → yGm`。实测该 workspace 往返在 `coresPerGroup==1` 时存在竞态（偶发结果偏大），快路径同时修掉该问题。

## 8. 关键技术

- **平方和交换律 → 连续大 burst 载入**：抛弃逻辑 (A,R) 碎读，按物理连续顺序整段搬入，MTE2 从 4~8B/burst 拉到 16~64KB/burst。
- **A/R 分离下沉到 UB**：HBM 只做连续读，A/R 的分离与规约在 UB 内用 VF + ReduceSum 完成。
- **VF 平方累加 + dtype Cast**：`__VEC_SCOPE__` 内 `Mul(x,x)` + `Add(acc)`，逐 `VL_FP32=64` 块，尾块用 `UpdateMask`；fp16/bf16 载入即 `DIST_UNPACK_B16` + Cast 升 fp32。
- **DataCopy 连续直搬**：门控保证 `chunkA0==szA0` ⟹ 相邻 row 在 GM 完全连续，用 `DataCopy(in, xGm[...], n)` 直接连续大 burst 载入（非 `DataCopyPad`），避免大 blockLen 多 block 下的搬运异常。
- **两步 AR 链式规约（step2 标量）**：step1 `ReduceSum<AR>` 规约 R1；step2 逐 `(a0,a1)` 标量累加 R0（规避 adv_api AR 在 dimR==64 边界的高估 bug）。
- **两段式多核（仿 batch_norm BlockSplitR）**：`coresPerGroup>1` 时 Phase1 每核部分和写 workspace，`SyncAll` 后 Phase2 组内 c==0 核求和 + sqrt；`coresPerGroup==1` 走 `WriteOutDirect` 快路径，省 workspace 往返。
- **ReduceSum VL 余量（kPadElems）**：adv_api `ReduceSum` 以整 VL 粒度 Load/Store，末尾多读多写最多 1 VL，故参与 ReduceSum 的 acc/red buffer 末尾各留 64 元素余量防越界。

## 9. 主块、尾块、对齐和数值精度

- **多核切分**：rows 轴 `CeilDiv(rowsSize, coresPerGroup)` + 尾核兜底（`rowEnd` clamp）。
- **对齐**：UB buffer 按 `CeilAlign32`（8 元素 = 32B）对齐；part/outQ 按 outLanes 对齐。
- **数值精度**：全程 fp32 累加/规约；fp16/bf16 输入载入即 Cast 升 fp32，输出 sqrt 后 Cast 降回 `D_T`（`DIST_PACK_B32`）。workspace 恒为 fp32（部分和精度）。sqrt 在最终 `part` 上一次性 `Reg::Sqrt` in-register 完成，无需 TilingData 传系数（与 mean 需 `invRTotal` 不同）。
- **group→输出映射**：外层 A 轴按 axis 序（外→内）编码 `groupIdx`，与输出 dense 序一致，`groupIdx` 即输出 slab 号，无需额外重排。

## 10. 独立 Kernel 入口示例

```cpp
#include "euclidean_norm_tiling_data.h"
#include "dav310/euclidean_norm_interleaved_contiguous_tailr.h"

// LoadTilingData：把 GM 上 TilingData 逐字段搬进核内局部结构（等价 GET_TILING_DATA_WITH_STRUCT）
template <typename DType>
extern "C" __global__ __aicore__ __vector__ void
euclidean_norm_fast_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    EuclideanNormTilingData td;
    LoadTilingData(reinterpret_cast<__gm__ const EuclideanNormTilingData*>(tiling), &td);
    NsEuclideanNorm::EuclideanNormFastKernel<DType> op;
    op.Init(x, y, workspace, &td);
    op.Process();
}
```

> workspace 大小 = `groupCnt × coresPerGroup × outLanes × sizeof(float)`，由 host tiling 的 `workspaceBytes` 给出（`coresPerGroup==1` 时 kernel 不使用 workspace，但 host 仍按公式预留）。

## 11. 代表 case 与实测

| case | shape | 基线瓶颈 | 优化前 | 优化后 | 加速 |
|------|-------|---------|--------|--------|------|
| #18 arararar_0002 | (4,2,2,640,32,32,2,2) fp32 | 8 轴，burst **8B**，MTE2 ratio 0.995 | 1128.5 µs | **136.1 µs** | **8.29×** |

> 平台：Ascend 950 / arch35(dav-3510)，CANN 9.0.0，coreNum=72，UB≈192KB。这是全套 case 中唯一的"大赢家"——病态碎读被连续大 burst 彻底解决。

## 12. 不适用场景和可选替代

| 不适用场景 | 说明 / 替代 |
|-----------|------------|
| int32（或非 fp32/fp16/bf16） | 门控直接回退普通归约模板 |
| `R1×dtypeSize > 16B`（burst 不病态） | 连续化收益 < 固定开销，回退普通模板（见 [guide.md](../guide.md)「关键洞察」） |
| `rowsSize < 8` | 两步规约+屏障开销反超 MTE2 增益，回退 |
| `innerElems×dtypeSize` 非 32B 对齐 | 连续 `DataCopy` 不安全，回退 |
| 内块需分块（chunkA0 < szA0） | 分块屏障 + 逐平面规约吃掉 MTE2 增益（-9%），回退普通模板 |
| 奇数轴 tail-A（ARAR...A） | 用 [交错连续载入-无 workspace（路径 B）](euclidean_norm_interleaved_contiguous_tail_a.md) |
| 简单 AR / ARA（非交错） | 用 softmax_v2 系列 AR/ARA 模板（本库归约骨架通用） |
