# SoftmaxV2 AR Recompute 模板

## 1. 模板用途

针对 2D `[A1, R]` 布局、R 超出 UB 容量的 Softmax 场景。将 R 切分为多个 chunk，通过三阶段重读 GM（求 max → 求 sum → 输出）完成 Softmax，使用 UB 间二分累加树（UpdateCache）高效合并跨 chunk 的局部 sum。

## 2. 输入输出布局和 Softmax 轴

- 输入 `x`：`[A1, R]`，行优先，Softmax 沿 R 轴计算。
- 输出 `y`：`[A1, R]`，与输入同 shape、同 dtype。

## 3. 适用 shape、dtype 和 UB 条件

| 条件 | 要求 |
|------|------|
| 输入维数 | 2D，A0 = 1 |
| R 大小 | R 超 UB 单次载入容量 |
| dtype | FP32、FP16、BF16 |
| UB 约束 | 预留 max(32B) + sum(32B) + cache(2048B) 后，三缓冲 xQueue + 双缓冲 yQueue + tmpBuffer |

## 4. 类名、构造函数和 Init 接口

```cpp
namespace SoftmaxV2Ops;

template <typename Tx, typename Ty>
class SoftmaxV2ArRecompute : public SoftmaxV2OpsBase {
public:
    __aicore__ inline SoftmaxV2ArRecompute(TPipe* pipe);
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SoftmaxV2ArRecomputeTilingData* tilingData);
    __aicore__ inline void Process();
};
```

- **构造函数**：接收 `TPipe*`。
- **Init**：计算每核行数 `currentRowBlock_`、`resultCacheID_`，初始化 xQueue（三缓冲）、yQueue（双缓冲）及 xTmp/xMax/xSum/cache buffer。
- **Process**：逐行执行三阶段计算。

## 5. 对应的 TilingData 和字段说明

对应 `SoftmaxV2ArRecomputeTilingData`，定义在 [softmax_v2_tiling_data.h](softmax_v2_tiling_data.h)。

| 字段 | 含义 | 单位 |
|------|------|------|
| a | A1 维总长度（行数） | 元素数 |
| r | R 维实际长度 | 元素数 |
| ubFactor | R 方向单 chunk 长度 | 元素数 |
| ubFactorTail | R 方向尾块长度 | 元素数 |
| aBlockFactor | 每核处理行数 | 行 |
| aLoopCountCeil | R 方向 chunk 总数 | chunk |
| basicBlockLoop | 二分树基础块数（≤ aLoopCountCeil 的最大 2^k） | 块 |
| mainFoldCount | 需折叠块数 = FloorDiv(R,ubFactor) - basicBlockLoop | 块 |

## 6. Host tiling 参数计算方法

使用 [softmax_v2_tiling.h](softmax_v2_tiling.h) 中的 `TilingArRecompute`：

```cpp
softmax_tiling::CaseShape shape{a1, r, 1, dtypeCode};
softmax_tiling::PlatformParam plat{ubSize, numBlocks};
SoftmaxV2ArRecomputeTilingData td;
int64_t blockDim = softmax_tiling::TilingArRecompute(shape, plat, td);
```

关键计算：
- `ubFactor` = min(R, (ubFlexible / baseFactor) 按 lcm 对齐)，其中 baseFactor = dtypeSize×3 + dtypeSize×2 + 4
- `aLoopCountCeil` = CeilDiv(R, ubFactor)
- `basicBlockLoop` = FindNearestPower2(aLoopCountCeil)
- `mainFoldCount` = FloorDiv(R, ubFactor) - basicBlockLoop

## 7. CopyIn、归约、归一化和 CopyOut 流程

三阶段处理，每行独立执行：

```
Process（逐行 rowIdx）:
  Step 1 — 求 max:
    for 每个 R chunk:
      CopyIn chunk → CalculateMaxVF（VF 求 max，与 running max 合并）

  Step 2 — 求 sum（二分累加树）:
    for basicBlockIdx in [0, basicBlockLoop):
      a. CopyIn 主块 → MainBlockCastSubExpVF（sub max, exp, 写入 xTmp）
      b. 若需折叠: CopyIn 折叠块 → FoldBlockCastSubExpVF（sub max, exp, Add 到 xTmp）
      c. ReduceSum(xTmp) → UpdateCache（按 cacheID 层级合并）
    basicBlockLoop == 0 时: 单 chunk 直接 ReduceSum

  Step 3 — 输出:
    for 每个 R chunk:
      CopyIn chunk → CalculateOutVF（sub max, exp, div sum, Cast 回原精度）→ CopyOut GM
```

## 8. 重计算与二分累加树优化

- **三阶段重读**：输入读 3 次（max/sum/output），换取 UB 空间——R 可远超 UB 容量。
- **MainBlock / FoldBlock 配对**：主块和折叠块的 exp 结果在 xTmp 上 `Add` 合并后一次 `ReduceSum`，归约次数减半。
- **UpdateCache 二分树**：跨 chunk 的局部 sum 按 `GetCacheID(basicBlockIdx)` 层级二分合并，O(log N) 层而非 O(N) 串行。
  - `GetCacheID(idx)` = `popcount(idx ^ (idx+1)) - 1`，确定当前块在二分树中的位置。
  - `resultCacheID` = `GetCacheID(basicBlockLoop - 1)`，指向最终合并结果。
- **三缓冲 xQueue**：支持 Main/Fold 双 chunk 并行载入。

## 9. 主块、尾块、对齐和数值精度处理

- **主块/尾块**：`ubFactor` 为主 chunk 长度，`ubFactorTail` 为尾 chunk（R 不能整除时）。尾块在 Step 1/3 中通过条件判断使用 `ubFactorTail`。
- **折叠尾块**：当 `basicBlockIdx == mainFoldCount` 且 `ubFactorTail > 0` 时，折叠块使用 `ubFactorTail` 长度。
- **对齐**：`ubFactor` 按 `Lcm(blockSize, Lcm(blockSize/dtypeSize, blockSize/dtypeSize))` 对齐。
- **数值精度**：FP16/BF16 通过 `LoadTensorForDtypeTIn` 升至 FP32 计算；max/sum 用 FP32 维护；输出通过 `Cast` 降回。`xToFp32_` / `yToFp32_` 编译期常量控制转换路径。

## 10. 独立 Kernel 入口示例

```cpp
#include "softmax_v2_tiling_data.h"
#include "dav310/softmax_v2_base.h"
#include "dav310/softmax_v2_ar_recompute.h"

using namespace SoftmaxV2Ops;

extern "C" __global__ __aicore__ __vector__ void
softmax_ar_recompute_fp32(GM_ADDR x, GM_ADDR y, SoftmaxV2ArRecomputeTilingData tiling)
{
    TPipe pipe;
    SoftmaxV2ArRecompute<float, float> op(&pipe);
    op.Init(x, y, &tiling);
    op.Process();
}
```

## 11. 不适用场景和可选替代模板

| 不适用场景 | 推荐替代 |
|-----------|---------|
| R 可载入 UB | [AR FullLoad](softmax_v2_ar_full_load.md)（无重读，带宽更优） |
| R 极小（≤ 16/32） | [AR SmallR](softmax_v2_ar_small_r.md)（转置向量化） |
| 3D [A1,R,A0] 布局（A0>1） | [ARA Recompute](softmax_v2_ara_recompute.md) |
| 带宽极度受限、R 适中 | 评估 FullLoad 是否可用（3× 重读代价高） |
