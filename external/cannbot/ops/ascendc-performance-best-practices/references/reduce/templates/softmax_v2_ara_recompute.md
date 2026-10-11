# SoftmaxV2 ARA Recompute 模板

## 1. 模板用途

针对 3D `[A1, R, A0]` 布局、R 超出 UB 容量的 Softmax 场景。将 R 切分为多个 bin，通过三阶段重读 GM（求 max → 求 sum → 输出）完成 Softmax。使用 `NlastReduceSum` 跨 bin 二分折叠和 `UpdateCache` 累加树高效合并局部 sum。

## 2. 输入输出布局和 Softmax 轴

- 输入 `x`：`[A1, R, A0]`，行优先，Softmax 沿 R 轴计算。
- 输出 `y`：`[A1, R, A0]`，与输入同 shape、同 dtype。
- UB 内布局为 `(binAddRFactor, tileA0Len)`，沿 R（bin）逐块处理。

## 3. 适用 shape、dtype 和 UB 条件

| 条件 | 要求 |
|------|------|
| 输入维数 | 3D，A0 > 1 |
| R 大小 | R 超 UB 单次载入容量 |
| dtype | FP32、FP16、BF16 |
| UB 约束 | 双缓冲 xQueue/yQueue + xMax/xSum + cache/temp/reduceSumTemp buffer |

## 4. 类名、构造函数和 Init 接口

```cpp
namespace SoftmaxV2Ops;

template <typename T1, typename T2>
class SoftmaxV2ARARecompute : public SoftmaxV2OpsBase {
public:
    __aicore__ inline SoftmaxV2ARARecompute();
    __aicore__ inline SoftmaxV2ARARecompute(const SoftmaxV2ARARecomputeTilingData* tilingDataIn);
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, TPipe* pipeIn);
    __aicore__ inline void Process();
};
```

- **构造函数**：无参默认，或接收 TilingData 指针。
- **Init**：初始化 GM/UB buffer，包括 xQueue/yQueue（双缓冲）、xMaxBuf/xSumBuf、cacheBuffer/tempBuffer/reduceSumTempBuffer。
- **Process**：按 tile 遍历，每 tile 执行三阶段计算。

## 5. 对应的 TilingData 和字段说明

对应 `SoftmaxV2ARARecomputeTilingData`，定义在 [softmax_v2_tiling_data.h](softmax_v2_tiling_data.h)。

| 字段 | 含义 | 单位 |
|------|------|------|
| totalRLen / totalA0Len | R/A0 维总长度 | 元素数 |
| totalTiles | 总 tile 数 = A1 × tileA0Outer | tile |
| tilesPerCore | 每核 tile 数 | tile |
| usedCoreNums | 实际使用核数 | 核 |
| tileA0Outer | A0 方向 tile 总数 | tile |
| tileA0Len / tileA0Tail | 主块/尾块 A0 长度 | 元素数 |
| binAddRFactor | R 方向单 bin 长度（默认 128） | 元素数 |
| binAddRLoop | R 方向完整 bin 数 | bin |
| binAddRTotalLoop | R 方向 bin 总数 | bin |
| binAddRTail | R 方向尾 bin 长度 | 元素数 |
| binAddBasicBlockLoop | 二分树基础块数 | 块 |
| binAddMainFoldCount | 需折叠块数 | 块 |
| binAddCacheBufferCount | cache buffer 数量 | 个 |
| binAddResultCacheID | 最终结果 cache 索引 | 索引 |

## 6. Host tiling 参数计算方法

使用 [softmax_v2_tiling.h](softmax_v2_tiling.h) 中的 `TilingAraRecompute`：

```cpp
softmax_tiling::CaseShape shape{a1, r, a0, dtypeCode};
softmax_tiling::PlatformParam plat{ubSize, numBlocks};
SoftmaxV2ARARecomputeTilingData td;
int64_t blockDim = softmax_tiling::TilingAraRecompute(shape, plat, td);
```

关键计算：
- `binAddRFactor` = 128（默认）
- `binAddRTotalLoop` = CeilDiv(R, binAddRFactor)
- `binAddBasicBlockLoop` = FindNearestPower2(binAddRTotalLoop)
- `binAddCacheBufferCount` = 64 - clz(binAddBasicBlockLoop)（二分树层数）
- `binAddResultCacheID` = GetCacheID(binAddBasicBlockLoop - 1)

## 7. CopyIn、归约、归一化和 CopyOut 流程

三阶段处理，每 tile 独立执行：

```
Process（逐 tile）:
  Step 1 — CalcReduceMax:
    for 每个 R bin:
      CopyInX → 逐 A0 列 VF 求 max，与 xMaxLocal 合并

  Step 2 — CalcReduceSum:
    for basicBlockIdx in [0, binAddBasicBlockLoop):
      a. ProcessMainBlock: CopyIn 主 bin → sub max → exp → 写入 yMain
      b. 若需折叠: ProcessFoldBlock: CopyIn 折叠 bin → sub max → exp → Add 到 yMain
      c. ProcessSummation: NlastReduceSum(yMain) → UpdateCache
    binAddBasicBlockLoop == 0 时: 单 bin 直接处理
    最终: 从 cache[resultCacheID] 拷出 xSum

  Step 3 — 输出:
    for 每个 R bin:
      CopyInX → CalcOutput（sub max, exp, div sum, Cast）→ CopyOutY
```

## 8. NlastReduceSum、bin fold 和 cache 参数

### NlastReduceSum

`NlastReduceSum` 是跨 R（行）方向对 A0（列）做规约求和的核心函数，实现在 [softmax_v2_base.h](dav310/softmax_v2_base.h) 中：

1. **R ≤ 8**：`NlastReduceSumSmallR`，使用 `NlastDichotomyAdd<RSize>` 模板做编译期二分展开。
   - `NlastDichotomyAdd<2>`：基例，2 行 `Add`。
   - 递归展开：`NlastDichotomyAdd<N>` 调用 `NlastDichotomyAdd<(N+1)/2>` 和 `NlastDichotomyAdd<N/2>`。
   - `TailCount` 模板参数：处理尾块不足 8 行时的精确掩码。
2. **R > 8**：`NlastReduceSumLargeR<TailCount>`，按 8 行分组（COMPRESSION=8）做折叠。
   - `FindNearestPower2(rSize)` 确定 foldPoint。
   - mainFold（8 行折叠）、tailFold（尾块折叠）、unFold（展开）三阶段。
   - 递归调用 `NlastReduceSumSmallR` 做最终归约。

### bin fold（MainBlock / FoldBlock 配对）

- **ProcessMainBlock**：载入主 bin，`Sub(maxReg)` → `Exp` → 写入 yMain。
- **ProcessFoldBlock**：载入折叠 bin，`Sub(maxReg)` → `Exp` → `Add` 到 yMain。
- 配对后一次 `NlastReduceSum`，归约次数减半。

### cache 参数

- `binAddCacheBufferCount`：二分树层数 + 1，决定 cacheBuffer 大小。
- `binAddResultCacheID`：`GetCacheID(binAddBasicBlockLoop - 1)`，指向最终合并结果。
- `UpdateCache`：按 `cacheID` 层级将当前 sum 累加到 cache 对应位置。

## 9. 主块、尾块、对齐和数值精度处理

- **主块/尾块**：`tileA0Len` 为主块 A0 长度，`tileA0Tail` 为尾块。`binAddRTail` 为 R 方向尾 bin 长度。
- **折叠尾 bin**：当 `basicBlockIdx == binAddMainFoldCount` 且 `binAddRTail > 0` 且 `binAddRTail != binAddRFactor` 时，折叠块使用 `binAddRTail` 长度。
- **对齐**：A0 按 `a0TileBase`（FP32=8/FP16=16）对齐，`dstStride` 按 `BLOCK_SIZE`（32B）对齐。
- **数值精度**：FP16/BF16 通过 `LoadTensorForDtypeT1` 升至 FP32；max/sum 用 FP32 维护；输出 `Cast` 降回。

## 10. 独立 Kernel 入口示例

```cpp
#include "softmax_v2_tiling_data.h"
#include "dav310/softmax_v2_base.h"
#include "dav310/softmax_v2_ara_recompute.h"

using namespace SoftmaxV2Ops;

extern "C" __global__ __aicore__ __vector__ void
softmax_ara_recompute_fp32(GM_ADDR x, GM_ADDR y, SoftmaxV2ARARecomputeTilingData tiling)
{
    TPipe pipe;
    SoftmaxV2ARARecompute<float, float> op(&tiling);
    op.Init(x, y, &pipe);
    op.Process();
}
```

## 11. 不适用场景和可选替代模板

| 不适用场景 | 推荐替代 |
|-----------|---------|
| R×tileA0 可载入 UB | [ARA FullLoad](softmax_v2_ara_full_load.md)（无重读） |
| A0 = 1（2D） | [AR Recompute](softmax_v2_ar_recompute.md) |
| 需要在线更新（避免 3 次读入） | [ARA Online](softmax_v2_ara_online.md)（2 次读入） |
| 带宽极度受限 | 评估 FullLoad 是否可用（3× 重读代价高） |
