# SoftmaxV2 AR SmallR 模板

## 1. 模板用途

针对 2D `[A1, R]` 布局、归约轴 R 极小（FP32 ≤ 16 / FP16/BF16 ≤ 32）的 Softmax 场景。通过将数据转置为 `(R, A1)` 布局，使短归约轴转到外层、长伴生轴转为向量化方向，将 VReg 利用率从 R/64 提升到接近 100%。

## 2. 输入输出布局和 Softmax 轴

- 输入 `x`：`[A1, R]`，行优先，Softmax 沿 R 轴（最后一轴）计算。
- 输出 `y`：`[A1, R]`，与输入同 shape、同 dtype。
- 内部转置为 `(R, A1)` 布局进行 VF 计算，输出前转置回 `(A1, R)`。

## 3. 适用 shape、dtype 和 UB 条件

| 条件 | 要求 |
|------|------|
| 输入维数 | 2D，A0 = 1 |
| R 大小 | FP32 ≤ 16，FP16/BF16 ≤ 32（R ≤ 2×VL_FP32 = 128 时可转置向量化） |
| dtype | FP32、FP16、BF16 |
| UB 约束 | `tileA0Len * rAligned * (dtypeSize*2 + 4) * 2 + tileA0Len * 4` ≤ UB |

## 4. 类名、构造函数和 Init 接口

```cpp
namespace SoftmaxV2Ops;

template <typename Tx, typename Ty>
class SoftmaxV2ArSmallR {
public:
    __aicore__ inline SoftmaxV2ArSmallR(TPipe* pipe);
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SoftmaxV2ArSmallRTilingData* tilingDataIn);
    __aicore__ inline void Process();
};
```

- **构造函数**：接收 `TPipe*`，保存 pipe 指针。
- **Init**：接收输入 x、输出 y 的 GM 地址和 TilingData 指针，初始化 GlobalTensor 和 UB Buffer。
- **Process**：执行完整 Softmax 计算。

## 5. 对应的 TilingData 和字段说明

对应 `SoftmaxV2ArSmallRTilingData`，定义在 [softmax_v2_tiling_data.h](softmax_v2_tiling_data.h)。

| 字段 | 含义 | 单位 |
|------|------|------|
| totalA0Len | 转置后 A0 维长度（= 原 A1） | 元素数 |
| totalRLen | 归约轴 R 实际长度 | 元素数 |
| totalTiles | A0 方向总 tile 数 | tile |
| tilesPerCore | 每核 tile 数 | tile |
| tileA0Len | 主块 A0 tile 长度 | 元素数 |
| tileA0Tail | 尾块 A0 长度 | 元素数 |
| rTileBase | R 转置基础块（FP32=8/FP16=16） | 元素数 |
| rAligned | R 对齐后长度 | 元素数 |

## 6. Host tiling 参数计算方法

使用 [softmax_v2_tiling.h](softmax_v2_tiling.h) 中的 `TilingArSmallR`：

```cpp
softmax_tiling::CaseShape shape{a1, r, 1, dtypeCode};
softmax_tiling::PlatformParam plat{ubSize, numBlocks};
SoftmaxV2ArSmallRTilingData td;
int64_t blockDim = softmax_tiling::TilingArSmallR(shape, plat, td);
```

关键计算：
- `rTileBase` = FP32 ? 8 : 16
- `rAligned` = CeilDiv(R, rTileBase) × rTileBase
- `tileA0Len` = a0TileNum × VL_FP32，受 UB 容量约束
- `totalTiles` = CeilDiv(A1, tileA0Len)

## 7. CopyIn、归约、归一化和 CopyOut 流程

```
Process:
  1. preload 第一个 tile，CopyInAndTransPose（MultiCopy 转置 (A1,R)→(R,A1)）
  2. for 每个 tile:
     a. CalcMaxSubExp: 逐 A0 列求 max，sub max，exp，写入 tmpLocal 和 tmpLocal2
     b. CopyInAndTransPose: 预载下一个 tile（双缓冲）
     c. CalcReduceSum: 沿 R 轴 ReduceSum（Pattern::Reduce::RA）
     d. CalcOutput: 逐列 div sum，FP16/BF16 做 Cast
     e. CalcTranspose: TransDataTo5HD 转置回 (A1,R)
     f. CopyOutY: DataCopyPad 写回 GM
```

## 8. 转置优化

- **MultiCopy 转置**：使用 `MultiCopyParams<Tx, 2>` + `MultiCopyLoopInfo<2>` 在 CopyIn 阶段直接完成 `(A1,R)→(R,A1)` 转置，避免额外转置 pass。
- **TransDataTo5HD**：输出前使用 `TransDataTo5HD` 将 `(R,A1)` 转回 `(A1,R)`，FP32 和 FP16/BF16 分别使用不同的 block 参数。
- **VF 向量化**：转置后长轴 A1 沿 VReg 方向，每个 VReg 可处理 64 个 FP32 元素，利用率接近 100%。

## 9. 主块、尾块、对齐和数值精度处理

- **主块/尾块**：`tileA0Len` 为主块长度，`tileA0Tail` 为最后一个 tile 的实际长度。尾块通过 `UpdateMask` 精确掩码处理。
- **对齐**：R 按 `rTileBase`（FP32=8/FP16=16）向上对齐，A0 按 `VL_FP32`（64）对齐。
- **数值精度**：FP16/BF16 输入在计算前 `Cast` 升至 FP32，计算完成后 `Cast` 降回原精度。使用 `castTraitFp16ToFp32` / `castTraitFp32ToFp16` 控制饱和模式和舍入。

## 10. 独立 Kernel 入口示例

```cpp
#include "softmax_v2_tiling_data.h"
#include "dav310/softmax_v2_base.h"
#include "dav310/softmax_v2_ar_small_r.h"

using namespace SoftmaxV2Ops;

extern "C" __global__ __aicore__ __vector__ void
softmax_ar_small_r_fp32(GM_ADDR x, GM_ADDR y, SoftmaxV2ArSmallRTilingData tiling)
{
    TPipe pipe;
    SoftmaxV2ArSmallR<float, float> op(&pipe);
    op.Init(x, y, &tiling);
    op.Process();
}
```

## 11. 不适用场景和可选替代模板

| 不适用场景 | 推荐替代 |
|-----------|---------|
| R > 2×VL_FP32（128） | [AR FullLoad](softmax_v2_ar_full_load.md)（R 可载入 UB）或 [AR Recompute](softmax_v2_ar_recompute.md)（R 超 UB） |
| 3D [A1,R,A0] 布局（A0>1） | [ARA FullLoad](softmax_v2_ara_full_load.md) 或 [ARA Recompute](softmax_v2_ara_recompute.md) |
