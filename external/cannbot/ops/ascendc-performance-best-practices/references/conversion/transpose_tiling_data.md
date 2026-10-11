# Transpose TilingData 结构与参数说明

本文说明 Transpose 各 tiling 策略所用的 TilingData 结构体(POD blob,host 计算后原样拷入 kernel 的 `tiling` GM 参数)。结构体定义见 [`templates/transpose_tiling_data.template`](templates/transpose_tiling_data.template)。

## 使用方式

1. 根据实际入参(dtype、shape、perm),对照 [`guide.md`](guide.md) 策略速查表选择 tiling 策略。
2. 直接引用 [`templates/dav3510/`](templates/dav3510/) 下对应策略的 kernel 模板文件(见下表映射),按需微调。
3. host 侧计算 TilingData 时,参照本文各字段含义填充。

## 策略 → kernel 模板 → TilingData 结构 映射

| 策略 | kernel 模板文件 | TilingData 结构 | 参考文档 |
|------|-------------|-----------------|----------|
| TENSOR_MOVE | `templates/dav3510/transpose_tensor_move.template` | `TransposeOpTilingData` | [transpose_tensor_move.md](transpose_tensor_move.md) |
| SMALL_SHAPE | `templates/dav3510/transpose_small_shape.template` | `TransposeOpTilingData` | [transpose_small_shape.md](transpose_small_shape.md) |
| CUT_ONCE | `templates/dav3510/transpose_cut_one_axis.template` | `TransposeOpTilingData` | [transpose_cut_one_axis.md](transpose_cut_one_axis.md) |
| CUT_TWICE | `templates/dav3510/transpose_cut_two_axis.template` | `TransposeOpTilingData` | [transpose_cut_two_axis.md](transpose_cut_two_axis.md) |
| N_LAST_TRANSPOSE | `templates/dav3510/transpose_n_last.template` | `TransposeOpTilingData` | [transpose_n_last.md](transpose_n_last.md) |
| BIG_DIM | `templates/dav3510/transpose_big_dim.template` | `TransposeOpTilingData` | [transpose_big_dim.md](transpose_big_dim.md) |
| GATHER_TRANSPOSE | `templates/dav3510/transpose_with_gather.template` | `GatherTransposeTilingData` | [transpose_gather.md](transpose_gather.md) |
| VCONV_TRANSPOSE (5HD) | `templates/dav3510/transpose_transdata_5hd.template` | `TransposeVCONVTilingData` | [transpose_vconv_5hd.md](transpose_vconv_5hd.md) |
| VCONV_021_TRANSPOSE | `templates/dav3510/transpose_transdata_5hd_021axis.template` | `Transpose021VCONVTilingData` | [transpose_vconv_021.md](transpose_vconv_021.md) |

> NDDMA 家族共用同一个 `TransposeOpTilingData`;加速策略(gather/vconv/021)各有独立结构。所有结构均 `#pragma pack(push, 8)` 8 字节对齐。

## 外部依赖(CANN toolkit 头文件)

kernel 代码依赖以下 toolkit 头(不在本目录,由 CANN 环境提供):

- `kernel_operator.h` — AscendC 核心 API(TPipe/TQue/DataCopy 等)
- `op_kernel/platform_util.h`、`op_kernel/math_util.h` — 平台常量与 CeilDiv/FloorDiv 等
- `simt_api/asc_simt.h` — SMALL_SHAPE 的 SIMT 逐元素模型
- `reg_compute/kernel_reg_compute_datacopy_intf.h` — GATHER 的 MicroAPI/Reg 寄存器编程与 `DataCopyGatherImpl`(仅 dav-3510)

---

## 一、TransposeOpTilingData(NDDMA 家族)

NDDMA 家族把逻辑 shape 归约后对齐到固定 5 维(`NDDMA_MAX_DIM_NUM=5`),用带 stride 的多维 DMA 完成搬运 + 换轴。字段分四组。

### 1.1 轴与切分基础

| 字段 | 类型 | 含义 |
|------|------|------|
| `permSize` | int64 | 归约后有效维数(perm 长度)。5 维扩展偏移 = `NDDMA_MAX_DIM_NUM - permSize`。 |
| `inCutIndex` | int64 | 输入方向切分轴下标(归约后坐标)。 |
| `outCutIndex` | int64 | 输出方向切分轴下标(归约后坐标)。CUT_TWICE 判定:`outCutIndex > FindOutIndex(inCutIndex)`。 |
| `inUbFactor` | int64 | 输入切分轴的 UB 分块因子(每块取多少元素)。 |
| `outUbFactor` | int64 | 输出切分轴的 UB 分块因子。 |
| `inTailFactor` | int64 | 输入切分轴尾块大小(`inCutAxisSize % inUbFactor`),0 表示无尾块。 |
| `outTailFactor` | int64 | 输出切分轴尾块大小,0 表示无尾块。 |
| `ubSize` | int64 | 本策略可用的 UB 字节数(已按 double buffer 等预留后的值)。 |

### 1.2 多核切分

| 字段 | 类型 | 含义 |
|------|------|------|
| `realCoreNum` | int64 | 实际参与计算的核数(`blockIdx >= realCoreNum` 的核直接返回)。 |
| `blkFactor` | int64 | 每核处理的 NDDMA 块数(基础值)。 |
| `blkTailFactor` | int64 | 前 `blkTailFactor` 个核各多处理一块,用于均衡余数。见 `ParseMultiCoreRange`。 |
| `totalNddmaNum` | int64 | 总 NDDMA 块数(一维扁平编号总数),供多核均分。 |

### 1.3 CUT_TWICE 的四类块区间(扁平 loopidx 边界)

同时切两轴 → main/inputTail/outputTail/tail 四类块。区间边界由 host `GetIntervalInfoForCutTwice` 预算,kernel 用本核范围与之求交。仅当对应尾 factor 非 0 时区间存在。

| 字段 | 含义 |
|------|------|
| `rangeMainEnd` | main(两轴都取满)区间结束。 |
| `rangeInputTailStart` / `rangeInputTailEnd` | inputTail(输入取尾、输出取满)区间。 |
| `rangeOutputTailStart` / `rangeOutputTailEnd` | outputTail(输出取尾、输入取满)区间。 |
| `rangeTailStart` / `rangeTailEnd` | tail(两轴都取尾)区间。 |

### 1.4 Shape 与 UB 布局数组

| 字段 | 长度 | 含义 |
|------|------|------|
| `inputShape` / `outputShape` | 8 | 归约后输入/输出逻辑 shape(`TRANSPOSE_MAX_AXIS_NUM_TD=8`)。 |
| `perm` | 8 | 归约后 perm。 |
| `baseInShape` | 8 | 输入各维的元素步长(后缀连乘),用于 GM 地址还原。 |
| `baseNddmaShape` / `nddmaIdx` | 5 | BIG_DIM 专用:折叠后 NDDMA 各维的基数与原始轴映射(见 `FlushBaseNumForBigDim`)。 |
| `expandedPerm` | 5 | 5 维扩展后的 perm(换轴 stride 依据)。 |
| `expandedInputShape` / `expandedOutputShape` | 5 | 5 维扩展后的输入/输出 shape。 |
| `inUbMainSrcShape` / `inUbMainDstShape` | 5 | main 块进/出 UB 的 5 维形状(src=读入布局,dst=换轴写回布局)。 |
| `inUbInputTailSrcShape` / `inUbInputTailDstShape` | 5 | inputTail 块的 UB src/dst 形状。 |
| `inUbOutputTailSrcShape` / `inUbOutputTailDstShape` | 5 | outputTail 块的 UB src/dst 形状。 |
| `inUbTailSrcShape` / `inUbTailDstShape` | 5 | tail 块的 UB src/dst 形状。 |

> TENSOR_MOVE / SMALL_SHAPE / N_LAST / CUT_ONCE / BIG_DIM 只用到与自身相关的子集,四类块的 UB 形状数组主要服务 CUT_TWICE;其余策略对应字段可能为 0 或退化值。

外层封装 `TransposeTilingData { TransposeOpTilingData transposeOpTiling; }` 仅为与原始 layout 对齐,kernel 入口传 `&tilingData.transposeOpTiling`。

---

## 二、GatherTransposeTilingData(GATHER_TRANSPOSE)

见 [`transpose_gather.md`](transpose_gather.md)。UB 内向量 gather 重排,MTE2/MTE3 两端连续大 burst。

### 2.1 标量控制

| 字段 | 类型 | 含义 |
|------|------|------|
| `dataTensorSize` | uint32 | 单份 data buffer 字节数(×2 ping-pong ×2 in/out 后须放进 UB)。 |
| `indexTensorSize` | uint32 | gather index buffer 字节数(容纳所有相位的 index)。 |
| `usedCoreCnt` | uint32 | 实际用核数,要求 `>= coreNum/2` 否则该策略放弃。 |

### 2.2 轴切分位置(int8 下标,-1 表示该方向不被切分)

| 字段 | 含义 |
|------|------|
| `blkAxesCnt` | block 轴数量(UB 分块之上、分给核的轴)。 |
| `blkInUbCutPos` / `blkOutUbCutPos` | 输入/输出切轴是否被 block 切分且除不尽(决定尾块相位是否存在)。 |
| `ubAxesCnt` | 进 UB 参与重排的轴数。 |
| `inUbInCutPos` / `inUbOutCutPos` | 输入侧在 UB 内的切分/连续边界位置。 |
| `outUbInCutPos` / `outUbOutCutPos` | 输出侧在 UB 内的切分/连续边界位置(`ubAxesCnt - outUbOutCutPos` = 需 gather 的维数,≤3)。 |

### 2.3 分块因子与 stride

| 字段 | 类型 | 含义 |
|------|------|------|
| `blkFactor` / `blkTailFactor` | int64 | 每核 / 末核处理的 block 数。 |
| `inUbCutAxisSize` / `outUbCutAxisSize` | int64 | 输入/输出切轴的总大小。 |
| `inUbCutAxisFactor` / `outUbCutAxisFactor` | int32 | 输入/输出切轴的 UB 分块因子;尾块 = size % factor。 |
| `axis0/1/2InSrcStride` | int64 | MTE2 搬入的多层 GM 源步长(最多 3 层循环)。 |
| `axis0/1/2OutDstStride` | int64 | MTE3 搬出的多层 GM 目的步长。 |

### 2.4 轴数组

| 字段 | 长度 | 含义 |
|------|------|------|
| `blkAxes` | 8 | 各 block 轴大小(`GATHER_MAX_TRANS_AXIS_NUM_TD=8`)。 |
| `blkAxesInAOffset` / `blkAxesOutAOffset` | 8 | 各 block 轴对应的输入/输出 GM 地址偏移基数(`CalcBlkAddr` 用)。 |
| `inUbAxes` / `outUbAxes` | 6 | UB 内输入/输出各轴大小(`GATHER_UB_MAX_DIM_NUM_TD=6`)。 |
| `ubPerm` | 6 | UB 内轴序(int8),生成 gather index 与循环用。 |

---

## 三、TransposeVCONVTilingData(VCONV_TRANSPOSE 5HD,仅 16bit)

见 [`transpose_vconv_5hd.md`](transpose_vconv_5hd.md)。2D 末轴交换,片上 `TransDataTo5HD` 做 16×16 块转置。

辅助结构 `CoreSplitPara`(核间切分:`AlignBlockFactor`/`BlockFactor`/`BlockCount`/尾块)与 `UbSplitPara`(UB 内切分:主核/尾核的 UB align/factor/count),被下表按 R/C 方向各引用一份。

| 字段 | 类型 | 含义 |
|------|------|------|
| `AvailableUbSize` | int64 | 可用 UB 字节数。 |
| `UsedCoreNum` | int64 | 实际用核数。 |
| `MainCoreLoopCount` / `TailCoreLoopCount` | int64 | 主核 / 尾核的 UB 循环次数。 |
| `RLen` / `CLen` | int64 | 转置矩阵的行(R)/列(C)长度。 |
| `RAlignBlock` / `CAlignBlock` | int64 | R/C 向 16 对齐后的块数。 |
| `RAlignBlockElem` / `CAlignBlockElem` | int64 | R/C 对齐后的元素数。 |
| `IsRSplit` / `IsRCSplit` | bool | 切分模式:仅切 R / R 和 C 都切(都为 false = 仅切 C)。三种模式覆盖不同 UB 容量。 |
| `rSplitPara` / `cSplitPara` | CoreSplitPara | R/C 方向的核间切分参数。 |
| `rUbSplitPara` / `cUbSplitPara` | UbSplitPara | R/C 方向的 UB 内切分参数。 |

---

## 四、Transpose021VCONVTilingData(VCONV_021_TRANSPOSE,支持 8/16/32bit)

见 [`transpose_vconv_021.md`](transpose_vconv_021.md)。`perm=[0,2,1]` 保 batch 转置,按 batch 维循环 + `TransDataTo5HD`。

辅助结构 `Transpose021UbSplitPara`(`UbAlignFactor`/`UbFactor`/`UbCount`/`UbTailAlignFactor`/`UbTailFactor`),被 R/C 方向各引用一份。

| 字段 | 类型 | 含义 |
|------|------|------|
| `AvailableUbSize` | int64 | 可用 UB 字节数。 |
| `UsedCoreNum` | int64 | 实际用核数。 |
| `NLen` / `HLen` / `WLen` | int64 | batch(N)/ 后两维 H / W 的长度。 |
| `HAlignBlockElem` / `WAlignBlockElem` | int64 | H/W 对齐后的元素数。 |
| `NPerCore` / `NTailCore` | int64 | 按 N 维多核切分时每核 / 尾核的 N 数。 |
| `UbLoopCount` | int64 | UB 循环次数。 |
| `UseRConv` | bool | H≥W 时为 true,选 R 为主转置轴(影响 R/C 主轴选择)。 |
| `UseHSplit` | bool | 是否按 H 维(而非 N 维)做多核切分。 |
| `HPerCore` / `HTailCore` | int64 | 按 H 切分时每核 / 尾核的 H 数。 |
| `rUbSplitPara` / `cUbSplitPara` | Transpose021UbSplitPara | R/C 方向的 UB 内切分参数。 |
