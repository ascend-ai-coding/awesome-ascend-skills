# SoftmaxV2 ARA Online Softmax 模板

## 1. 模板用途

针对 3D `[A1, R, A0]` 布局的独立 Softmax 场景，使用在线（online）算法沿 R 分 chunk 逐块搬入，为每个 A0 列维护 running max 和 running sum。相比 ARA Recompute（输入读 3 次：max/sum/output），Online Softmax 将 max 与 sum 融合为单遍在线更新，输入只需读 2 次（1 遍在线 max+sum，1 遍输出），减少一次 R 全量搬入。

> **重要**：本模板是**独立的 Online Softmax**，不包含 `QK^T`、`P×V`、`O_acc`、causal mask 或 FlashAttention 的 Cube/Vector 融合。它与 [融合 Attention Online Softmax 设计](fused_attention_online_softmax_design.md) 只共享在线 max/sum 数学基础，不是同一个实现。

## 2. 输入输出布局和 Softmax 轴

- 输入 `x`：`[A1, R, A0]`，行优先，Softmax 沿 R 轴计算。
- 输出 `y`：`[A1, R, A0]`，与输入同 shape、同 dtype。
- 输入为已经生成的 Softmax 数据，输出为完整 Softmax 概率。

## 3. 适用 shape、dtype 和 UB 条件

| 条件 | 要求 |
|------|------|
| 输入维数 | 3D，A0 > 1 |
| R 大小 | 任意（R 分 chunk 处理，不受 UB 限制） |
| dtype | FP32、FP16、BF16 |
| UB 约束 | 双缓冲 xQueue/yQueue + xMaxBuf + xSumBuf，按 rChunkFactor 分 chunk |

## 4. 类名、构造函数和 Init 接口

```cpp
namespace SoftmaxV2Ops;

template <typename T1, typename T2>
class SoftmaxV2ARAOnline : public SoftmaxV2OpsBase {
public:
    __aicore__ inline SoftmaxV2ARAOnline();
    __aicore__ inline SoftmaxV2ARAOnline(const SoftmaxV2ARAOnlineTilingData* tilingDataIn);
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, TPipe* pipeIn);
    __aicore__ inline void Process();
};
```

- **构造函数**：无参默认，或接收 TilingData 指针。
- **Init**：初始化 GM/UB buffer，xQueue/yQueue（双缓冲，大小 = `tileA0Len × rChunkFactor`）+ xMaxBuf/xSumBuf。
- **Process**：按 tile 遍历，每 tile 执行在线 max+sum 和输出两遍。

## 5. 对应的 TilingData 和字段说明

对应 `SoftmaxV2ARAOnlineTilingData`，定义在 [softmax_v2_tiling_data.h](softmax_v2_tiling_data.h)。

| 字段 | 含义 | 单位 |
|------|------|------|
| totalRLen / totalA0Len | R/A0 维总长度 | 元素数 |
| totalTiles | 总 tile 数 = A1 × tileA0Outer | tile |
| tilesPerCore | 每核 tile 数 | tile |
| usedCoreNums | 实际使用核数 | 核 |
| tileA0Outer | A0 方向 tile 总数 | tile |
| tileA0Len / tileA0Tail | 主块/尾块 A0 长度 | 元素数 |
| rChunkFactor | R 方向单 chunk 长度（默认 128） | 元素数 |
| rChunkTotalLoop | R 方向 chunk 总数 | chunk |
| rChunkTail | R 方向尾 chunk 长度 | 元素数 |

## 6. Host tiling 参数计算方法

使用 [softmax_v2_tiling.h](softmax_v2_tiling.h) 中的 `TilingAraOnline`：

```cpp
softmax_tiling::CaseShape shape{a1, r, a0, dtypeCode};
softmax_tiling::PlatformParam plat{ubSize, numBlocks};
SoftmaxV2ARAOnlineTilingData td;
int64_t blockDim = softmax_tiling::TilingAraOnline(shape, plat, td);
```

关键计算：
- `rChunkFactor` = min(128, R)
- `rChunkTotalLoop` = CeilDiv(R, rChunkFactor)
- `rChunkTail` = R - (rChunkTotalLoop - 1) × rChunkFactor
- UB 占用（每个 A0 列）：`2 × rChunkFactor × dtypeSize + 2 × rChunkFactor × 4 + 4 + 4` 字节

## 7. 在线更新公式

沿 R 分 chunk，按 A0 列独立维护 running max (`m`) 和 running sum (`l`)：

```text
m_new = max(m_old, chunk_max)
corr  = exp(m_old - m_new)
l_new = l_old * corr + sum(exp(x - m_new))
```

最终 Softmax 输出：`softmax(x) = exp(x - m_final) / l_final`

### 与 ARA Recompute 三次读取输入的区别

| 特性 | ARA Recompute | ARA Online |
|------|--------------|------------|
| 输入读取次数 | 3 次（max/sum/output） | 2 次（在线 max+sum / output） |
| max 与 sum 计算 | 分离（先 max 全量，再 sum 全量） | 融合（单遍在线更新） |
| cache buffer | 需要（二分累加树） | 不需要 |
| 数值稳定性 | Safe Softmax（先减 max） | Safe Softmax（在线减 running max） |
| 带宽 | 3× R 全量读 | 2× R 全量读 |

## 8. CopyIn、归约、归一化和 CopyOut 流程

```
Process（逐 tile）:
  第一遍 — CalcOnlineMaxSum:
    初始化 running max = -inf, running sum = 0
    for 每个 R chunk:
      CopyInX → 逐 A0 列:
        1. 求 chunk 内局部 max (mLocal)
        2. m_new = max(m_old, mLocal)
        3. corr = exp(m_old - m_new)
        4. chunk_sum = Σ exp(x - m_new)
        5. l_new = l_old * corr + chunk_sum
        6. 更新 xMaxLocal = m_new, xSumLocal = l_new

  第二遍 — 输出:
    for 每个 R chunk:
      CopyInX → CalcOutput:
        sub max → exp → div sum → Cast 回原精度
      CopyOutY → GM
```

## 9. 主块、尾块、FP32 running 状态和数值稳定性

- **主块/尾块**：`tileA0Len` 为主块 A0 长度，`tileA0Tail` 为尾块。`rChunkTail` 为 R 方向尾 chunk 长度。
- **对齐**：A0 按 `a0TileBase`（FP32=8/FP16=16）对齐。
- **FP32 running 状态**：running max 和 running sum 始终用 FP32 维护（`xMaxBuf`/`xSumBuf` 为 `float` 类型），即使输入/输出为 FP16/BF16，确保数值稳定性。
- **数值稳定性**：
  - 每步减去 running max `m_new`，避免 exp 溢出。
  - `corr = exp(m_old - m_new)` ≤ 1（因 m_new ≥ m_old），修正历史 sum。
  - 第一遍在线计算最终 max 和 sum，第二遍用最终值做归一化。

## 10. 独立 Kernel 入口示例

```cpp
#include "softmax_v2_tiling_data.h"
#include "dav310/softmax_v2_base.h"
#include "dav310/softmax_v2_ara_online.h"

using namespace SoftmaxV2Ops;

extern "C" __global__ __aicore__ __vector__ void
softmax_ara_online_fp32(GM_ADDR x, GM_ADDR y, SoftmaxV2ARAOnlineTilingData tiling)
{
    TPipe pipe;
    SoftmaxV2ARAOnline<float, float> op(&tiling);
    op.Init(x, y, &pipe);
    op.Process();
}
```

### Host tiling 示例

```cpp
#include "softmax_v2_tiling.h"

softmax_tiling::PlatformParam plat;
plat.ubSize = ubSize;       // 运行时查询
plat.numBlocks = numBlocks; // 运行时查询

softmax_tiling::CaseShape shape;
shape.a1 = 4; shape.r = 2049; shape.a0 = 256; shape.dtypeCode = 0; // FP32

SoftmaxV2ARAOnlineTilingData td;
int64_t blockDim = softmax_tiling::TilingAraOnline(shape, plat, td);
// blockDim 为 <<<>>> 启动核数
```

## 11. 本模板不包含的内容

本模板是**独立的 Online Softmax**，不包含：

- `QK^T`（score 计算）
- `P×V`（概率与 Value 的乘积）
- `O_acc`（输出累加器）
- causal mask（因果掩码）
- Attention workspace
- FlashAttention 的 Cube/Vector 融合

如需了解融合 Attention 中的 Online Softmax 设计，参见 [融合 Attention Online Softmax 设计](fused_attention_online_softmax_design.md)。两者只共享在线 max/sum 数学基础，不是同一个实现。

## 12. 不适用场景和可选替代模板

| 不适用场景 | 推荐替代 |
|-----------|---------|
| A0 = 1（2D） | [AR Recompute](softmax_v2_ar_recompute.md) |
| R×tileA0 可载入 UB 且带宽充裕 | [ARA FullLoad](softmax_v2_ara_full_load.md)（单次载入） |
| R 超 UB 且不需要在线更新 | [ARA Recompute](softmax_v2_ara_recompute.md)（三阶段重读） |
