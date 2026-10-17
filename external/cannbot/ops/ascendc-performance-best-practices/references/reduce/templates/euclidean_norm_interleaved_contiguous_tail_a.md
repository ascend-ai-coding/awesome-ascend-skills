# EuclideanNorm 交错连续载入-无 workspace 模板（路径 B / tail-A）

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

> 模式族：**多轴交错归约（interleaved A-R）**，见 [guide.md](../guide.md)「多轴交错归约模式族」。
> 本模板对应源码 kernel 类 `EuclideanNormFastRARAKernel`。

## 1. 模板用途

针对 **奇数轴严格交错 `ARARAR...A`（tail-A，≥7 轴）** 布局的 EuclideanNorm（`y = sqrt(Σ x²)`）。与路径 A（tail-R）同属交错归约，同样用**连续大 burst 载入**破解交错布局的 MTE2 碎读，但因 **tail-A（内块最外是被规约的 R，输出维在内块内）、输出多、单输出规约小**，采用**每核独占若干 slab、无 workspace / 无 SyncAll** 的结构，比路径 A 更轻量。支持 fp32/fp16/bf16（载入即升 fp32，输出即降回）。

## 2. 输入输出布局和归约轴

- 输入 `x`：归一后的奇数轴 pattern `[A,R,A,R,...,A]`（tail-A），行优先 dense。
- 归约轴：奇数位轴（R）。
- 输出 `y`：偶数位轴（A）dense 张量。
- **内块** = 末 4 轴 `[R0,A0,R1,A1]`（`fastInnerStart=axisNum-4`，物理连续）：保留内块 A 轴 (A0=轴S+1, A1=轴S+3)，规约 R 轴 (R0=轴S, R1=轴S+2)。
- **slab** = ∏(内块外 A 轴)：每 slab 输出 `outLanes = A0*A1` 个，每核独占若干 slab。
- **rows 轴** = 内块最外轴 R0（轴 S，`stride==innerRowElems`）：沿它分块连续大 burst 载入。
- 外层 R 轴：核内 for 累加。

## 3. 适用 shape、dtype 和 UB 条件

| 条件 | 要求 |
|------|------|
| dtype | **fp32 / fp16 / bf16**（载入即升 fp32 累加/规约，输出降回；int32 不走 fast） |
| 轴数 / 类型 | 预处理后 `axisNum` 奇数且 ≥7，严格 `ARAR...A`（`isTailR==false`，tail-A） |
| 内块连续 | 末 4 轴 `[R,A,R,A]`：`axisStride[N-1]==1` 且 `axisStride[S]==innerRowElems` |
| 内块对齐 | `innerRowElems × dtypeSize` 须 32B 对齐（kernel 连续 `DataCopy` 载入要求） |
| 并行度 | `slabCnt = ∏(内块外A轴) ≥ coreNum/2`（否则大量核闲置，回退） |
| **内块每行足够大** | **`innerRowElems = A0*R1*A1 ≥ 512`** —— 硬阈值，见下 |
| UB | 内块一行 `[A0,R1,A1]` 至少能放下一块（`chunkR0 ≥ 1`） |

> **`innerRowElems ≥ 512` 是经上板实测校准的硬阈值**：内块每行太小时，连续载入无 MTE2 收益，反被每 slab 的 step2 屏障 + 小规约开销拖垮（实测 innerRowElems=80 → 慢 28%；=8 → 慢 10×）。

## 4. 类名、构造函数和 Init 接口

```cpp
namespace NsEuclideanNorm;

template <typename DType>  // D_T ∈ {float, half, bfloat16_t}
class EuclideanNormFastRARAKernel {
public:
    __aicore__ inline EuclideanNormFastRARAKernel();
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const EuclideanNormTilingData* td);  // 无 workspace 形参
    __aicore__ inline void Process();
};
```

- **构造函数**：无参（`TPipe pipe_` 为类成员）。
- **Init**：镜像 pattern + `rara*` 字段，推导内块几何（R0/A0/R1/A1、innerRowElems、accElems=A0*R1*A1、outLanes），设置 GM（仅 x/y，**无 workspace**），分配 UB（inQ 双缓冲 + acc/tmp/plane/part/outQ）。
- **Process**：strided slab 分配（`slab = blk, blk+coreNum, ...`），逐 slab 独立处理。

## 5. 对应的 TilingData 和字段说明

对应 `EuclideanNormTilingData`（[euclidean_norm_tiling_data.h](euclidean_norm_tiling_data.h)），本模板读取 pattern 字段 + `rara*` 字段：

| 字段 | 含义 |
|------|------|
| axisNum / axisShape[] / axisStride[] | 归一后的奇数轴 pattern |
| usedCoreNum | 实际使用核数 = min(slabCnt, coreNum) |
| fastInnerStart | 内块起始轴下标（= axisNum-4） |
| raraSlabCnt | slab 数 = ∏(内块外 A 轴 size) |
| raraChunkR0 | 内块最外 R 轴(R0)分块大小（沿 R0 分块累加，1~8） |

## 6. Host tiling 参数计算方法

使用 [euclidean_norm_tiling.h](euclidean_norm_tiling.h) 中的 `TryComputeFastRARATiling`（或驱动 `ComputeInterleavedTiling`）：

```cpp
euclidean_norm_host::Result r;
bool hit = euclidean_norm_host::ComputeInterleavedTiling(
    xShape, reduceAxes, dtypeCode, coreNum, ubSize, r);
// dtypeCode ∈ {0=fp32, 1=fp16, 2=bf16}；hit==true 且 r.templateType==3 → 用本模板；workspaceBytes==0
```

关键计算 / 门控顺序：
1. `dtype ∈ {fp32,fp16,bf16}` && `axisNum` 奇数且 ≥7 && `!isTailR`。
2. 连续性校验（`axisStride[N-1]==1`、`axisStride[S]==innerRowElems`）。
3. `innerRowElems×dtypeSize` 32B 对齐（连续 `DataCopy` 安全）。
4. `slabCnt = ∏(内块外A轴)`，要求 `slabCnt*2 ≥ coreNum`。
5. **`innerRowElems ≥ 512`**（否则回退）。
6. `chunkR0 = avail / (2*innerRowElems*dtypeSize)`，clamp 到 `[1, min(R0,8)]`。

## 7. CopyIn、归约、写出流程

```
Process（每核处理 strided slabs）:
  for slab in {blk, blk+coreNum, ...}:
    ProcessOneSlab(slab):
      ZeroAcc(acc[A0,R1,A1])
      for 外层R组合(轴1):
        AccumulateR0: 沿内块 R0 分块 → 连续大 burst 载入 [chunkR0, A0,R1,A1]
                      → VF 平方 + 逐 R0-行累加到 acc[A0,R1,A1]   // step1: 对 R0、外层R 求和
      ReduceStep2: 逐 A0 平面 RA 规约 R1 → part[A0,A1]           // 无 gather
      WriteOut: Sqrt(part) → yGm[slab*outLanes]
```

### step1（VF 平方累加，规约 R0）

内块布局 `[R0,A0,R1,A1]`，固定 R0 的一行 = `[A0,R1,A1]` = `accElems`，**物理连续**。逐 R0-行 `(Cast 升 fp32)` + `Mul(x,x)` + `Add(acc)`，天然把最外 R0 规约掉，无需 ReduceSum。

### step2（单级 RA 规约，无 gather）

`acc` 布局 `[A0,R1,A1]`，每个 A0 平面 `[R1,A1]` 已连续 → 直接 `ReduceSum<Pattern::Reduce::RA>` 规约 R1（外层）保留 A1（内层）→ `[A1]`，逐 A0 写入 `part[a0*A1+a1]`。**RARA 天然连续，省去路径 A step2 的 gather**。

## 8. 关键技术

- **平方和交换律 → 连续大 burst 载入**：与路径 A 相同，破解交错碎读。
- **无 workspace / 无 SyncAll**：输出多、单输出规约小 → 每核独占 slab，输出互不重叠，无需跨核合并。
- **strided slab 分配**：`slab = blk + k*coreNum`，自然负载均衡。
- **两步规约但无 gather**：step1 的 R0 由 VF 逐行 Add 吃掉，step2 单次 RA 规约 R1 保留 A1；RARA 连续布局使 step2 无需重排。
- **DataCopy 直搬**：内块 R0 分块连续，用 `DataCopy(in, xGm[base+r*r0Stride], n)` 直接连续搬入（非 pad）。

## 9. 主块、尾块、对齐和数值精度

- **R0 分块**：`chunkR0` 主块 + 尾块（`rows = min(chunkR0, R0-r)`）。
- **对齐**：part/outQ 按 `CeilAlign32`（8 元素）对齐。
- **数值精度**：全程 fp32 累加/规约；fp16/bf16 输入载入即 `DIST_UNPACK_B16` + Cast 升 fp32，输出 sqrt 后 Cast 降回 `D_T`（`DIST_PACK_B32`）。sqrt 在 `WriteOut` 对 `part[outLanes]` 一次性 `Reg::Sqrt` in-register 完成。
- **slab→输出映射**：内块外 A 轴按 axis 序编码 slab，与输出 dense 序一致，`slab` 即输出 slab 号。

## 10. 独立 Kernel 入口示例

```cpp
#include "euclidean_norm_tiling_data.h"
#include "dav310/euclidean_norm_interleaved_contiguous_tail_a.h"

template <typename DType>
extern "C" __global__ __aicore__ __vector__ void
euclidean_norm_fast_rara_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    EuclideanNormTilingData td;
    LoadTilingData(reinterpret_cast<__gm__ const EuclideanNormTilingData*>(tiling), &td);
    NsEuclideanNorm::EuclideanNormFastRARAKernel<DType> op;
    op.Init(x, y, &td);   // 无 workspace
    op.Process();
    (void)workspace;      // 本模板不使用 workspace
}
```

## 11. 代表 case 与实测

| case | shape 特征 | 优化前 | 优化后 | 加速 | 说明 |
|------|-----------|--------|--------|------|------|
| #11 ararara_0004 | 7 轴 fp32，内块 ~17KB，innerRowElems=4320 | 171.0 µs | **123.2 µs** | **1.39×** | 大内块连续载入有效 |
| #12 | 7 轴 fp32，innerRowElems=3840 | — | 持平微赢 | ~1.0× | 命中，收益小 |

> 平台：Ascend 950 / arch35(dav-3510)，CANN 9.0.0，coreNum=72，UB≈192KB。

## 12. 不适用场景和可选替代

| 不适用场景 | 说明 / 替代 |
|-----------|------------|
| int32（或非 fp32/fp16/bf16） | 门控回退普通归约模板 |
| `innerRowElems×dtypeSize` 非 32B 对齐 | 连续 `DataCopy` 不安全，回退 |
| `innerRowElems < 512`（内块每行太小） | 连续化收益 < 每-slab 固定开销，实测慢 28%~10×，**回退** |
| `slabCnt < coreNum/2` | 并行度不足，大量核闲置，回退 |
| 偶数轴 tail-R（ARAR...AR） | 用 [交错连续载入-两段式（路径 A）](euclidean_norm_interleaved_contiguous_tailr.md) |
| 简单 AR / ARA（非交错） | 用 softmax_v2 系列 AR/ARA 模板 |
