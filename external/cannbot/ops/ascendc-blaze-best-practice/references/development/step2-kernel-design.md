# Step 2: Kernel 设计总纲

> **定位**：告诉 Agent "Kernel 入口函数怎么设计"的通用方法论。具体场景的组装代码见 `references/scenarios/` 目录。

---

## §1 Kernel 签名设计

### 修饰符选择

| 场景 | 修饰符 | 说明 |
|------|--------|------|
| 纯 matmul（无 vector 后处理） | `__cube__` | AIC 独占 Cube Core |
| matmul + epilogue 融合 | `__mix__(aicCount, aivCount)` | AIC + AIV 混合执行 |
| A8W8 量化 matmul（MIX 路径） | `__mix__(1, 2)` | AIC 做 Cube Mmad，AIV 做反量化；必须用 `__mix__`，不能用 `__cube__` |

### GM_ADDR 参数约定

参数顺序：**输入在前、输出在后、tilingData 最后**。额外输入（Scale、Bias）插在输出之前。

```cpp
// 基础 matmul
__global__ __aicore__ __cube__ void my_kernel(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dC,
    const MyTilingData tilingData);

// MX 量化 matmul（额外 Scale 输入）
__global__ __aicore__ __cube__ void mx_kernel(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dScaleA, GM_ADDR dScaleB, GM_ADDR dC,
    const QuantMatmulTilingData tilingData);

// 融合 matmul（额外 Bias 输入）
__global__ __aicore__ __mix__(1, 2) void fused_kernel(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dC, GM_ADDR dBias,
    const MyTilingData tilingData);
```

### 模板参数设计

| 模式 | 适用场景 | 优点 |
|------|---------|------|
| `conditional_t` 推导 | 多种 layout/trans 组合 | 编译期推导，host 端扁平分发 |
| `bool` + `enum` | 量化算子（dtype 运行时可变） | 灵活，支持 runtime dispatch |

**推荐模式**：模板参数 `bool TransA, bool TransB` + Kernel 内部 `conditional_t` 推导 Layout：

```cpp
template <bool TransA, bool TransB>
__global__ __aicore__ __cube__ void my_kernel(GM_ADDR dA, GM_ADDR dB, GM_ADDR dC, ...)
{
    using LayoutA = AscendC::Std::conditional_t<TransA,
        AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutB = AscendC::Std::conditional_t<TransB,
        AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    // ... 后续组装
}
```

NZ 格式需要额外的 `CubeFormat` 模板参数，详见 `references/scenarios/basic-matmul-development.md` §3。

---

## §2 类型链推导

从 `ProblemShape` 出发，逐步推导完整类型链：

```
ProblemShape（问题规模：m, n, k, batch）
   ↓
DispatchPolicy（流水策略：SWAT / MatmulWithScaleMx）
   ↓
BlockScheduler（块调度器）
   ↓
BlockMmad（数据搬运 + MMAD 计算）
   ↓
Kernel（顶层 kernel 类）
   ↓
Params（参数结构体）
```

### 两条开发路径

| 路径 | 适用场景 | 组件来源 |
|------|---------|---------|
| blaze 库路径 | 普通 MatMul 单算子、MX 量化 MatMul、A8W8 等标准库已覆盖场景 | `op_kernel/include/blaze/gemm/` |
| blaze_custom 路径 | 普通 C+V 融合、Group MatMul、Grouped C+V、自定义 Block/Scheduler/Epilogue 扩展 | `op_kernel/include/blaze_custom/` |
| MX C+V 受控组合态 | MXFP8/MXFP4 MatMul + Vector Epilogue | `Kernel::MxMatmulKernelFused` + blaze library MX Block/Scheduler + 自定义 Epilogue |

### 混用禁令

默认情况下，**blaze_custom 模块和 blaze 库模块不能在同一 kernel 入口函数中任意混合使用。**

- blaze_custom 使用 `Kernel::`、`Block::` 命名空间
- blaze 库使用 `Blaze::Gemm::Kernel::`、`Blaze::Gemm::Block::` 命名空间
- 两者的 `BlockMmad` 模板参数数量、Params 结构和 SFINAE 机制不同

普通 MatMul 单算子和纯 MX 量化 MatMul 均使用 blaze 库全套路径。基础 MatMul 场景的 CMake include 应优先采用最小可编译集合；普通 C+V、Grouped C+V 和标准 blaze library 暂不能覆盖的扩展场景使用 blaze_custom 路径。MX C+V 是唯一明确设计的受控例外：`Kernel::MxMatmulKernelFused` 专门桥接 blaze library MX Block/Scheduler 与自定义 Epilogue，不视为任意混用。

### 关键原则

- 选择 blaze 库路径时，所有组件都必须来自 `blaze/gemm/`，不能混用 blaze_custom；MX C+V 只能通过 `MxMatmulKernelFused` 这个受控 bridge 例外接入自定义 Epilogue
- 每个组件都暴露 `::Params` 类型，从中提取字段填充参数结构体
- 不要硬编码 Params 结构，始终从组件类型推导

---

## §3 组件选择决策树

| 场景 | 路径 | Kernel | BlockMmad | Scheduler | Policy | 详见 |
|------|------|--------|-----------|-----------|--------|------|
| 基础 MatMul 单算子 | blaze 库 | `GemmUniversal` | `Blaze::Gemm::Block::BlockMmad` (Basic) | `BlockSchedulerMatmulBasic` | `MatmulMultiBlockBasic` | `references/scenarios/basic-matmul-development.md` |
| A8W8 量化（per-tensor x1Scale） | blaze 库 | `GemmUniversal` | `Blaze::Gemm::Block::BlockMmad` (FixpipeQuant) | `BlockSchedulerQuantBatchMatmulV3` | `MatmulWithScaleFixpipeQuant` | `references/scenarios/a8w8-quant-matmul-development.md` |
| A8W8 量化（全部量化模式） | blaze 库 | `QbmmMixWithoutBatch` | `Blaze::Gemm::Block::BlockMmad` (Mix) | `BlockSchedulerQuantBatchMatmulV3` | `MatmulWithScaleMix` | `references/scenarios/a8w8-quant-matmul-development.md` |
| MX 量化单算子 | blaze 库 | `GemmUniversal` | `Blaze::Gemm::Block::BlockMmad` (ScaleMx) | `BlockSchedulerQuantBatchMatmulV3` | `MatmulWithScaleMx` | `references/scenarios/mx-matmul-development.md` |
| Group MatMul | blaze_custom | `GroupMatmulKernel` | `Block::BlockMmad` | `GroupMatmulBlockSchedulerSplitM` | `MatmulMultiBlockPolicy` | `references/scenarios/group-matmul-development.md` |
| 普通 MatMul + Vector | blaze_custom | `MatmulKernelFused` | `Block::BlockMmad` | `MatmulSwatScheduler` | `MatmulMultiBlockPolicy` | `references/scenarios/fusion-matmul-development.md` |
| MXFP8/MXFP4 MatMul + Vector | 受控组合态 | `MxMatmulKernelFused` | `Blaze::Gemm::Block::BlockMmad` (ScaleMx) | `BlockSchedulerQuantBatchMatmulV3` | `MatmulWithScaleMx` | `references/scenarios/fusion-matmul-development.md` |
| Grouped MatMul + Vector | blaze_custom | `GroupMatmulKernel<..., Epilogue>` | `Block::BlockMmad` | `GroupMatmulBlockSchedulerSplitM` | `MatmulMultiBlockPolicy` | `references/scenarios/fusion-matmul-development.md` |

**LOAD 模式**：本 skill 默认只提供 SWAT 流式路径。Full-load、StreamK、4-buffer 不作为默认开发路径，也不提供对应 tiling engine。

各模块的详细能力（支持的 dtype/trans/format）和参数说明，查阅 `references/modules/` 目录：
- blaze_custom 模块：`references/modules/blaze-custom/`
- blaze 库模块：`references/modules/blaze-library/blaze-modules-index.md`

---

## §4 TilingData 约定

TilingData 是 host 端 Tiling 引擎计算、device 端 Kernel 消费的 POD 结构体。

### 设计规范

- 字段必须与 `BlockScheduler::Params` 对齐
- `#pragma pack(push, 8)` + `alignas(8)` 确保 8 字节对齐
- `#ifndef __CCE_AICORE__` 保护 `<cstdint>` include

### 现有可用 Tiling 引擎

Blaze 路径下的 Tiling 选择统一由 `references/tiling/tiling-selection.md` 维护。本文只说明 TilingData 约定和字段含义，不展开 tiling 算法细节。

| 场景 | Tiling 引擎 | TilingData | 来源 |
|------|------------|------------|------|
| 基础 MatMul 单算子 | `MatmulTilingSwat` | `MatmulTilingData` | `assets/op_tiling/matmul/` |
| MX 量化 matmul | `QuantMatmulTilingSwat<DTypeA, DTypeB>` | `QuantMatmulTilingData` | `assets/op_tiling/mx/` |
| Grouped MatMul / Grouped C+V | 复用对应非 grouped SWAT tiling | `MatmulTilingData` 或 `QuantMatmulTilingData` | `assets/op_tiling/matmul/` 或 `assets/op_tiling/mx/` |

**CV 融合场景**：epilogue 组装不影响 tiling 引擎选择。融合场景的 tiling 引擎与其基础 matmul 场景一致：
- bf16/fp16 matmul + epilogue → 仍按融合场景的 blaze_custom tiling 方案处理
- MXFP8/MXFP4 量化 matmul + epilogue → 同 MX 量化 matmul（`QuantMatmulTilingSwat`）
- Grouped matmul + epilogue → 使用 `{totalM,N,K}` 调用对应非 grouped SWAT tiling；`groupNum/groupList` 独立传给 grouped kernel，不进入 tiling data

**注意**：CV 融合的 V 部分不新增独立 tiling engine。Vector 侧只在 Cube tiling 产出的剩余 UB 中规划 extra input / tmp / output staging 和 `stageRows/stageSize`。详见：

- `references/tiling/tiling-selection.md`
- `references/scenarios/fusion-matmul-development.md`
- `references/modules/blaze-custom/development/epilogue-dev-guide.md`

### 关键字段

| 字段 | 含义 | Launcher 使用位置 |
|------|------|-----------------|
| `usedCoreNum` | 实际使用核数 | `<<<gridDim>>>` |
| `m / n / k` | 问题规模 | kernel 端 `ProblemShape` |
| `baseM / baseN / baseK` | L0 切分颗粒 | kernel 端 `BlockSchedulerParams` / `QBMMTiling` |

各 Scheduler 的 TilingData 字段含义详见 `references/modules/blaze-custom/scheduler-modules.md`。

**注意**：本文档不介绍 tiling 算法原理。Tiling 引擎作为现成模块使用，只需了解输入输出和字段含义；具体选择入口见 `references/tiling/tiling-selection.md`。

---

## §5 Wrapper 函数

Wrapper 函数是 `extern "C"` 包装层，接收运行时 trans/format 参数，通过 if/else 分发到对应模板实例化的 kernel 入口：

```cpp
extern "C" void my_op_launch(
    aclrtStream stream,
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dC,
    const MyTilingData tilingData,
    bool transA, bool transB)
{
    if (transA && transB) {
        my_kernel<true, true><<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dC, tilingData);
    } else if (transA && !transB) {
        my_kernel<true, false><<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dC, tilingData);
    } else if (!transA && transB) {
        my_kernel<false, true><<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dC, tilingData);
    } else {
        my_kernel<false, false><<<tilingData.usedCoreNum, nullptr, stream>>>(dA, dB, dC, tilingData);
    }
}
```

**设计要点**：
- `extern "C"` 确保 C 链接，供 launcher 通过函数声明调用
- 每种 trans 组合一行，编译期实例化正确版本
- `<<<gridDim, nullptr, stream>>>` 中 gridDim 取 `tilingData.usedCoreNum`

NZ 格式需要额外的 `CubeFormat` 模板参数分发（16 种组合），详见 `references/scenarios/basic-matmul-development.md` §3 和 `references/development/step3-launcher.md` §5。

---

**下一步**：→ `references/development/step3-launcher.md`（编写 Launcher）

**按需查阅**：
- `references/scenarios/*.md`（具体场景的组装代码）
- `references/modules/blaze-custom/*.md`（blaze_custom 模块能力查阅）
- `references/modules/blaze-library/blaze-modules-index.md`（blaze 库模块索引）
