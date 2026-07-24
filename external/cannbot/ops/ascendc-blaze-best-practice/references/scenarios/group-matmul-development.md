# Grouped MatMul 开发指南

> **适用场景**：M 轴分组矩阵乘，按 `groupList` 将 M 维拆分为 E 个子问题。
>
> **路径**：blaze_custom（tensor_api + 手写 kernel/block）
>
> **融合跳转**：如果需求是 Grouped MatMul + Mul/Add/Bias/GELU 等 Vector Epilogue，转到 [C+V 融合 MatMul 开发指南](fusion-matmul-development.md) 的 Grouped C+V 小节。Grouped C+V 是正式支持场景，但 Epilogue 必须使用 `TileContext` 接口。

---

## §1 场景背景

**数学定义**：将 C[M,N] = A[M,K] × B[K,N] 沿 M 轴按 `groupList` 拆分为 E 个子问题：

```
groupList[e] = m_e,  sum(groupList) == M

C_e[m_e, N] = A_e[m_e, K] × B_e[K, N]    (e = 0..E-1)
```

**输入输出**：

| 张量 | shape | 说明 |
|------|-------|------|
| A | (M, K) | 左输入，按 prefixM 切取 A_e |
| B | (E, K, N) 或共享 | 右输入，按 groupIdx 选取切片 |
| C | (M, N) | 输出，按 prefixM 写回 C_e |
| groupList | (E,) int64_t | GM 缓冲区，每个元素为该 group 的 M 大小 |

**prefixM**：前 e 个 group 的 M 累加和，用于 A/C 的 GM 偏移定位。

---

## §2 组件选择

本节默认描述纯 Grouped MatMul 单算子。Grouped C+V 融合仍使用 `GroupMatmulKernel`，但模板参数需传入非 void Epilogue，并采用 context-based epilogue 接口。

| 组件 | 选择 | 来源 |
|------|------|------|
| 路径 | blaze_custom | `op_kernel/include/blaze_custom/` |
| Kernel | `Kernel::GroupMatmulKernel` | `kernel/group_matmul_kernel.h` |
| BlockMmad | `Block::BlockMmad` (NO_FULL_LOAD) | `block/matmul_block_mmad.h` |
| Scheduler | `Block::GroupMatmulBlockSchedulerSplitM` | `block/group_matmul_block_scheduler.h` |
| Policy | `MatmulMultiBlockPolicy<NO_FULL_LOAD_MODE>` | `policy/dispatch_policy.h` |
| Tiling | 复用对应非 grouped `MatmulTilingSwat` | `assets/op_tiling/matmul/` |

---

## §3 组装代码

### Kernel 入口函数

```cpp
#include "kernel/group_matmul_kernel.h"
#include "block/matmul_block_mmad.h"
#include "block/group_matmul_block_scheduler.h"
#include "policy/dispatch_policy.h"
#include "tiling/blaze_matmul_tiling.h"
#include "tiling/blaze_matmul_tiling_data.h"

template <typename LayoutA, typename LayoutB>
__global__ __aicore__ __cube__ void grouped_matmul_kernel(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dC, GM_ADDR dGroupList,
    const MatmulTilingData& tilingData)
{
    if (ASCEND_IS_AIV) return;

    using AType = bfloat16_t;
    using BType = bfloat16_t;
    using CType = bfloat16_t;
    using layoutA = LayoutA;
    using layoutC = AscendC::Te::NDExtLayoutPtn;

    using DispatchPolicy = MatmulMultiBlockPolicy<NO_FULL_LOAD_MODE>;
    using ProblemShape = MatmulShape;

    using BlockMmad = Block::BlockMmad<DispatchPolicy, AType, layoutA, BType, LayoutB, CType, layoutC>;
    using BlockScheduler = Block::GroupMatmulBlockSchedulerSplitM;
    using MatmulKernelImpl = Kernel::GroupMatmulKernel<ProblemShape, BlockMmad, BlockScheduler>;
    using Params = typename MatmulKernelImpl::Params;
    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename MatmulKernelImpl::L1Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using GroupTiling = typename MatmulKernelImpl::GroupTiling;

    ProblemShape problemShape{
        static_cast<int64_t>(tilingData.m),
        static_cast<int64_t>(tilingData.n),
        static_cast<int64_t>(tilingData.k),
        1L};
    BlockMmadParams mmadParams{dA, dB, dC};
    L1Params l1Params{static_cast<uint64_t>(tilingData.kL1)};
    BlockSchedulerParams schedulerParams{
        static_cast<int32_t>(tilingData.baseM),
        static_cast<int32_t>(tilingData.baseN)};
    GroupTiling groupTiling{
        tilingData.groupNum,
        tilingData.baseM, tilingData.baseN,
        tilingData.baseK, tilingData.l0cDB};

    Params params{problemShape, mmadParams, l1Params, schedulerParams, groupTiling, {}, dGroupList};
    MatmulKernelImpl kernel;
    kernel(params);
}
```

### TilingData → Params 映射

| Params 字段 | TilingData 来源 | 说明 |
|------------|----------------|------|
| `problemShape` | `{m, n, k, 1}` | 总 M（所有 group 之和） |
| `mmadParams` | `{dA, dB, dC}` | 3 个 GM 地址 |
| `l1Params` | `{kL1}` | K 方向 L1 tile 尺寸 |
| `schedulerParams` | `{baseM, baseN}` | 仅 2 个字段 |
| `groupTiling` | `{groupNum, baseM, baseN, baseK, l0cDB}` | 5 个字段 |
| `dGroupList` | GM 指针 | groupList 缓冲区地址 |

---

## §4 Tiling 参数

**Tiling 引擎**：直接复用对应非 grouped `MatmulTilingSwat`，不新增 grouped tiling 算法和 TilingData。

```cpp
MatmulTilingSwat tilingEngine;
MatmulTilingData tilingData;
tilingEngine.GetTilingData(totalM, N, K, inputElemBytes, tilingData, transA, transB, isANz, isBNz, hasBias);
```

**关键输出字段**：

| 字段 | 含义 | Kernel 端使用 |
|------|------|-------------|
| `usedCoreNum` | 启动核数 | `<<<usedCoreNum, ...>>>` |
| `baseM/baseN/baseK` | L0 切分颗粒 | `groupTiling` |
| `kL1` | L1 K 方向窗口 | `l1Params` |
| `l0cDB` | L0C ping-pong 级数 | `groupTiling` |

> Tiling 使用总 M（所有 group 之和）计算切分，不读取 `groupList` 内容。`groupNum/groupList` 是 grouped kernel 参数，不写入 `MatmulTilingData`。

---

## §5 关键约束

1. **Scheduler 生命周期覆盖整个 group loop**：Scheduler 在 group loop 外构造，每个 group 只调用 `UpdateNextProblem` 刷新 shape。禁止在 loop 内重建 scheduler，否则丢失跨 group 轮转状态，导致逐核负载不均衡。

2. **每 group 刷新 problem shape**：`m_e` 可能不同，必须对每个 group 调用 scheduler 刷新 `(m_e, n, k)`，不能复用上一个 group 的派生字段。

3. **per-group GM 偏移**：
   - A：按 prefixM 偏移（前 e 个 group 的 M 累加 × K）
   - B：按 `groupIdx * N * K` 偏移
   - C：按 prefixM 偏移（前 e 个 group 的 M 累加 × N）

4. **尾块处理**：`m_e <= 0` 时 device 侧直接跳过，不参与 tile 调度，但 offset 仍按 `groupList[e]` 更新。

5. **Epilogue 约束**：`Epilogue=void` 走 pure AIC direct writeback；Grouped C+V 是正式支持场景，传非 void Epilogue 时必须使用 context-based view hook，签名为 `operator()(BlockShape, TileContext, flagId)`，不能在 epilogue 内保存 group 状态或读取 `groupList`。

---

## §6 常见陷阱

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| 1 | group 输出错位 | 用 aggregate shape 推导 offset | 使用 prefixM 构造 per-group view |
| 2 | zero group 后结果错乱 | 跳过计算但未更新 offset | offset 始终按 `groupList[e]` 更新 |
| 3 | 小 group 精度/越界异常 | scheduler 未刷新当前 group shape | 每个 group 调用 `UpdateNextProblem` |
| 4 | many-small group 核间不均衡 | group loop 内重建 scheduler | scheduler 在 loop 外构造，每组只刷新 shape |
| 5 | epilogue 与 group 强绑定 | epilogue 保存 group 状态 | kernel 层切好 view，epilogue 保持通用 hook |
