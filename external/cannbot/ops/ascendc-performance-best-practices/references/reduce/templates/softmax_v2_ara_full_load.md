# SoftmaxV2 ARA FullLoad 模板

## 1. 模板用途

针对 3D `[A1, R, A0]` 布局、R×A0_tile 可载入 UB 的 Softmax 场景。沿 A0 切 tile，跨 R 逐列 VF 计算，根据 R 大小分支选择不同的归约路径（R≤2、R≤4、R≤8 直接累加；R>8 使用 BinaryAddVF 二分折叠）。

## 2. 输入输出布局和 Softmax 轴

- 输入 `x`：`[A1, R, A0]`，行优先，Softmax 沿 R 轴计算。
- 输出 `y`：`[A1, R, A0]`，与输入同 shape、同 dtype。
- 数据在 UB 中布局为 `(R, tileA0Len)`，沿 R 逐行处理，A0 方向向量化。

## 3. 适用 shape、dtype 和 UB 条件

| 条件 | 要求 |
|------|------|
| 输入维数 | 3D，A0 > 1 |
| R 大小 | R×tileA0Len 可载入 UB |
| dtype | FP32、FP16、BF16 |
| UB 约束 | `tileA0Len × (R+1) × (dtypeSize×2 + 4) + tileA0Len × 4` ≤ UB |

## 4. 类名、构造函数和 Init 接口

```cpp
namespace SoftmaxV2Ops;

template <typename T1, typename T2>
class SoftmaxV2ARA {
public:
    __aicore__ inline SoftmaxV2ARA();
    __aicore__ inline SoftmaxV2ARA(const SoftmaxV2ARATilingData* tilingDataIn);
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, TPipe* pipeIn);
    __aicore__ inline void Process();
};
```

- **构造函数**：无参默认构造，或接收 TilingData 指针。
- **Init**：接收 x/y GM 地址和 `TPipe*`，初始化 GlobalTensor 和 UB Buffer（xQueue/yQueue 双缓冲 + xTmpBuf + xReduceBuf）。
- **Process**：按 tile 遍历所有 (A1, A0) 组合，逐 tile 计算 Softmax。

> **注意**：ARA 系模板的构造/Init 方式与 AR 系不同——ARA 在 Init 中接收 `TPipe*`，而 AR 在构造函数中接收。

## 5. 对应的 TilingData 和字段说明

对应 `SoftmaxV2ARATilingData`，定义在 [softmax_v2_tiling_data.h](softmax_v2_tiling_data.h)。

主要字段：

| 字段 | 含义 | 单位 |
|------|------|------|
| totalA1Len / totalRLen / totalA0Len | 各维总长度 | 元素数 |
| totalTiles | 总 tile 数 = A1 × a0Outer | tile |
| tilesPerCore | 每核 tile 数 | tile |
| usedCoreNums | 实际使用核数 | 核 |
| a0Outer | A0 方向 tile 总数 | tile |
| tileA0Len / tileA0Tail | 主块/尾块 A0 tile 长度 | 元素数 |
| binaryAddK | BinaryAdd 外层折叠层数 | 层 |
| binaryAddLast | 最后是否需额外 2→1 折叠 | 标志 |
| binaryAddInnerLoop | BinaryAdd 每层内循环次数 | 次 |
| remainderLoopCount | R 对 8 取余后的折叠循环数 | 次 |
| quotientLoopCount | R 整除 8 后的剩余折叠循环数 | 次 |
| remainderTailOffset0~7 | 余数尾块各行偏移（越界指向 validNumInXUb 做零填充） | 元素数 |

## 6. Host tiling 参数计算方法

使用 [softmax_v2_tiling.h](softmax_v2_tiling.h) 中的 `TilingAraFullLoad`：

```cpp
softmax_tiling::CaseShape shape{a1, r, a0, dtypeCode};
softmax_tiling::PlatformParam plat{ubSize, numBlocks};
SoftmaxV2ARATilingData td;
int64_t blockDim = softmax_tiling::TilingAraFullLoad(shape, plat, td);
```

关键计算：
- `a0TileBase` = FP32 ? 64 : 128
- `tileA0Len` = a0Inner × a0TileBase，受 UB 容量和核数约束
- 当 R ≤ 8 时，binaryAdd* 字段不使用
- 当 R > 8 时，`binaryAddQuotient` = 最大 2^k ≤ R-1，按 8 行分组做 BinaryAddVF

## 7. CopyIn、归约、归一化和 CopyOut 流程

```
Process:
  for 每个 tile (curA1Idx, curA0Idx):
    1. CopyInX: DataCopyPad 批量载入 (curTileRLen × curTileA0Len) 块
    2. Compute:
       a. VFShiftVector: 逐 A0 列求 max → sub max → exp → 写入 xTmpLocal
       b. VFReduceSum: 按 R 大小分支求 sum
       c. VFCalculateOutput: 逐列 div sum → Cast 回原精度
    3. CopyOutY: DataCopyPad 写回 GM
```

## 8. VF、BinaryAddVF 优化

### R 大小分支归约

| R 范围 | 归约路径 | 说明 |
|--------|---------|------|
| R ≤ 2 | `SumRLessThan2` | 逐行 VF `Add` 累加 |
| R ≤ 4 | `SumRLessThan4` | 2 行配对 + 尾行处理（`TwoRowAddWithTail`） |
| R ≤ 8 | `SumRLessThan8` | 4 行配对 + 尾行处理 |
| R > 8 | `SumRMoreThan8` + `BinaryAddVF` | 8 行分组 + 二分折叠 |

### BinaryAddVF 二分折叠（R > 8）

1. **行分组**：将 R 行按 8 行一组分组，每组内用 `TwoRowAdd` 做 4→2→1 折叠。
2. **余数处理**：不满 8 行的余数部分，用 `remainderTailOffset0~7` 精确控制每行偏移，越界行指向 `validNumInXUb`（零填充区域）。
3. **二分树**：`BinaryAddVF` 按 `binaryAddK` 层 × `binaryAddInnerLoop` 次，每次 4→1 折叠，最后若 `binaryAddLast=1` 额外做 2→1。
4. **TwoRowAddWithTail**：通用工具函数，处理 2 行配对 + 2 个尾行偏移。

## 9. 主块、尾块、对齐和数值精度处理

- **主块/尾块**：`tileA0Len` 为主块 A0 长度，`tileA0Tail` 为最后一个 A0 tile 的实际长度。尾块通过 `copyInParams.dstStride` 和 `copyInParams.srcStride` 处理非对齐。
- **对齐**：A0 按 `a0TileBase`（FP32=64/FP16=128）对齐，`dstStride` 按 `BLOCK_SIZE`（32B）对齐。
- **零填充**：R > 8 时，在 `xTmpLocal[validNumInXUb]` 位置写入零值，余数尾块越界行指向该区域，等效于零填充。
- **数值精度**：FP16/BF16 通过 `LoadTensorForDtypeT1`（`DIST_UNPACK_B16` + `Cast`）升至 FP32；输出通过 `Cast` + `DIST_PACK_B32` 降回。

## 10. 独立 Kernel 入口示例

```cpp
#include "softmax_v2_tiling_data.h"
#include "dav310/softmax_v2_base.h"
#include "dav310/softmax_v2_ara_full_load.h"

using namespace SoftmaxV2Ops;

extern "C" __global__ __aicore__ __vector__ void
softmax_ara_full_load_fp32(GM_ADDR x, GM_ADDR y, SoftmaxV2ARATilingData tiling)
{
    TPipe pipe;
    SoftmaxV2ARA<float, float> op(&tiling);
    op.Init(x, y, &pipe);
    op.Process();
}
```

## 11. 不适用场景和可选替代模板

| 不适用场景 | 推荐替代 |
|-----------|---------|
| R×tileA0 超 UB | [ARA Recompute](softmax_v2_ara_recompute.md)（R 切 bin 三阶段重读） |
| A0 = 1（2D） | [AR FullLoad](softmax_v2_ar_full_load.md) 或 [AR SmallR](softmax_v2_ar_small_r.md) |
| 需要在线更新（避免 3 次读入） | [ARA Online](softmax_v2_ara_online.md)（2 次读入） |
