# SoftmaxV2 AR FullLoad 模板

## 1. 模板用途

针对 2D `[A1, R]` 布局、R 可整段载入 UB 的 Softmax 场景。多行批量载入 UB，单 pass VF 计算 max→sub→exp→sum→div，使用 MicroAPI VF 路径实现大 R 二分归约。

## 2. 输入输出布局和 Softmax 轴

- 输入 `x`：`[A1, R]`，行优先，Softmax 沿 R 轴计算。
- 输出 `y`：`[A1, R]`，与输入同 shape、同 dtype。

## 3. 适用 shape、dtype 和 UB 条件

| 条件 | 要求 |
|------|------|
| 输入维数 | 2D，A0 = 1 |
| R 大小 | R 可整段载入 UB（`ubFactor * rAligned * (dtypeSize*2 + 4) * 2` ≤ UB） |
| dtype | FP32、FP16、BF16 |
| UB 约束 | 预留 1024B + 512B 后，剩余空间分配给 xQueue/yQueue/tmpBuffer |

## 4. 类名、构造函数和 Init 接口

```cpp
namespace SoftmaxV2Ops;

template <typename T_in, typename T_out>
class SoftmaxV2AR : public SoftmaxV2OpsBase {
public:
    __aicore__ inline SoftmaxV2AR(TPipe* pipe);
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SoftmaxV2ARTilingData* tilingData);
    __aicore__ inline void Process();
};
```

- **构造函数**：接收 `TPipe*`。
- **Init**：设置 GM buffer、计算当前核的行数 `blockA_`、初始化 xQueue（双缓冲）、yQueue（双缓冲）和 tmpLocalBuffer。
- **Process**：按 `ubFactor` 分块处理每行。

## 5. 对应的 TilingData 和字段说明

对应 `SoftmaxV2ARTilingData`，定义在 [softmax_v2_tiling_data.h](softmax_v2_tiling_data.h)。

| 字段 | 含义 | 单位 |
|------|------|------|
| a | A1 维总长度（行数） | 元素数 |
| r | R 维实际长度 | 元素数 |
| rAligned | R 对齐后长度（32B/dtypeSize 对齐） | 元素数 |
| ubFactor | 单次 UB 载入行数 | 行 |
| aBlockFactor | 每核处理行数 | 行 |
| rLoopCount | R 方向 VL_FP32 分段数 | 段 |

## 6. Host tiling 参数计算方法

使用 [softmax_v2_tiling.h](softmax_v2_tiling.h) 中的 `TilingArFullLoad`：

```cpp
softmax_tiling::CaseShape shape{a1, r, 1, dtypeCode};
softmax_tiling::PlatformParam plat{ubSize, numBlocks};
SoftmaxV2ARTilingData td;
int64_t blockDim = softmax_tiling::TilingArFullLoad(shape, plat, td);
```

关键计算：
- `rAligned` = CeilAlign(R, blockSize/dtypeSize)
- `ubFactor` = (UB - 1024 - 512) / (rAligned × 4 + 2 × rAligned × dtypeSize × 2)
- `aBlockFactor` = CeilDiv(A1, numBlocks)

## 7. CopyIn、归约、归一化和 CopyOut 流程

```
Process:
  1. 按 ubFactor 分子块（ubLoop）
  2. ProcessUB(ubA, aOffset):
     a. CopyInX: DataCopyPad 批量载入 ubA 行
     b. FirstNormCompute: 逐行 VF 求 max → sub max → exp，写入 xTmpLocal
     c. SecondNormCompute: 二分折叠 ReduceSum 求 sum，再逐行 div sum → Cast 回原精度
     d. CopyOutY: DataCopyPad 写回 GM
```

### FirstNormCompute 流程（VF 路径）

1. 逐行加载 R 方向数据（按 VL_FP32 分段）
2. `MicroAPI::Max` 求行内 max（先处理尾块，再处理整块）
3. `MicroAPI::ReduceMax` 归约到标量
4. 逐段 `Sub(maxReg)` → `Exp` → 写入 xTmpLocal

### SecondNormCompute 流程（大 R VF 二分归约）

1. R ≤ 2×VL_FP32：直接 `ReduceSum` + 逐行 div
2. R > 2×VL_FP32：
   - 按 `FindNearestPower2(ceilVLCount)` 确定 foldPoint
   - 主块（mainFold）：两段 Add 后 ReduceSum
   - 尾块（tailFold）：mask 精确处理非对齐部分
   - 展开块（unFold）：直接 ReduceSum
   - 递归调用 `SecondNormComputePost` 做最终归约和 div

## 8. VF 二分归约优化

- **FindNearestPower2**：确定最优二分折叠点，使 ReduceSum 调用次数最小化。
- **MainFold / TailFold / UnFold 三阶段**：
  - MainFold：成对 VL 段 `Add` 后一次 `ReduceSum`，归约次数减半
  - TailFold：非对齐尾块用 `UpdateMask` 精确掩码
  - UnFold：剩余展开段直接 `ReduceSum`
- **MicroAPI VF**：使用 `RegTensor<float>` + `MaskReg` 直接操作 256B VReg，`ReduceSum` 硬件归约指令。

## 9. 主块、尾块、对齐和数值精度处理

- **主块/尾块**：`ubFactor` 为 UB 内行数分块；R 方向按 `VL_FP32` 分段，尾段用 `UpdateMask` 精确掩码。
- **对齐**：R 按 `blockSize/dtypeSize`（FP32=8, FP16=16）向上对齐到 32B，`dstStride` 填充对齐间隙。
- **数值精度**：FP16/BF16 输入通过 `LoadTensorForDtypeTIn`（`DIST_UNPACK_B16` + `Cast`）升至 FP32 计算；输出通过 `StoreTensorForDtypeTOut`（`Cast` + `DIST_PACK_B32`）降回。`castTraitFp16ToFp32` 使用 `UNKNOWN` 饱和模式，`castTraitFp32ToFp16` 使用 `NO_SAT` + `CAST_RINT`。

## 10. 独立 Kernel 入口示例

```cpp
#include "softmax_v2_tiling_data.h"
#include "dav310/softmax_v2_base.h"
#include "dav310/softmax_v2_ar_full_load.h"

using namespace SoftmaxV2Ops;

extern "C" __global__ __aicore__ __vector__ void
softmax_ar_full_load_fp32(GM_ADDR x, GM_ADDR y, SoftmaxV2ARTilingData tiling)
{
    TPipe pipe;
    SoftmaxV2AR<float, float> op(&pipe);
    op.Init(x, y, &tiling);
    op.Process();
}
```

## 11. 不适用场景和可选替代模板

| 不适用场景 | 推荐替代 |
|-----------|---------|
| R 极小（≤ 16/32） | [AR SmallR](softmax_v2_ar_small_r.md)（转置向量化收益更高） |
| R 超 UB 容量 | [AR Recompute](softmax_v2_ar_recompute.md)（R 切 chunk 三阶段重读） |
| 3D [A1,R,A0] 布局（A0>1） | [ARA FullLoad](softmax_v2_ara_full_load.md) |
