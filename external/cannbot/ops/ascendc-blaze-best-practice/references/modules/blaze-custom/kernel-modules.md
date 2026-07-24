# Blaze Custom Kernel 层模块手册

> **定位**：blaze_custom 路径下 Kernel 层各模块的使用手册。Kernel 层负责构建 GM Tensor、驱动 BlockScheduler 循环取块坐标、调用 BlockMmad 完成单 block 计算。

---

## §1 MatmulKernel

| 项目 | 说明 |
|------|------|
| 头文件 | `kernel/matmul_kernel.h` |
| 命名空间 | `Kernel::MatmulKernel` |
| 模板签名 | `template <class ProblemShape, class BlockMmad, class BlockScheduler>` |
| 核心职责 | 纯 AIC matmul，GM→L1→L0→MMAD→L0C→GM |

**模板参数说明**：

| 参数 | 含义 |
|------|------|
| `ProblemShape` | 问题规模结构体，须含 `m`、`n`、`k` 字段 |
| `BlockMmad` | Block 层 MMAD 实现类（须来自 `Block::BlockMmad` 特化） |
| `BlockScheduler` | Scheduler 策略标签（如 `MatmulSwatScheduler<NO_FULL_LOAD_MODE>`） |

**Params 结构体字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `problemShape` | `ProblemShape` | 问题规模 |
| `mmadParams` | `BlockMmad::Params` | GM 地址（aGmAddr, bGmAddr, cGmAddr） |
| `l1Params` | `BlockMmad::L1Params` | L1 流水参数（kL1） |
| `schParams` | `BlockSchedulerOp::Params` | 调度器参数 |
| `qbmmParams` | `MatmulTiling` | Tiling 参数（baseM, baseN, baseK, dbL0C） |

**运行模式**：仅 AIC 核心执行，AIV 直接返回（`if ASCEND_IS_AIV { return; }`）。

**支持的 dtype / trans / format**：

- AType/BType：`float`、`bfloat16_t`、`half`、`int8_t`、`fp8_e4m3_t` 等
- transA/transB：由 `BlockMmad::LayoutA`/`LayoutB` 推导（`TagToTrans`）
- GM Layout：`NDExtLayoutPtn`（行主序）/ `DNExtLayoutPtn`（列主序），C0 = 32 / sizeof(dtype)

---

## §2 MatmulKernelFused

| 项目 | 说明 |
|------|------|
| 头文件 | `kernel/matmul_kernel_fused.h` |
| 命名空间 | `Kernel::MatmulKernelFused` |
| 模板签名 | `template <class ProblemShape, class BlockMmad, class BlockScheduler, class Epilogue>` |
| 核心职责 | AIC/AIV 融合 matmul，CV 同步 + Epilogue 后处理 |

**与 MatmulKernel 的关键差异**：

| 差异点 | MatmulKernel | MatmulKernelFused |
|--------|-------------|-------------------|
| 模板参数 | 3 个 | 4 个（多 `Epilogue`） |
| 核心模式 | 纯 AIC | AIC + AIV 混合 |
| 输出 Tensor | GM Tensor（CopyL0C2GM） | UB Tensor（CopyL0C2UB） |
| CV 同步 | 无 | CrossCoreSetFlag / CrossCoreWaitFlag |
| Epilogue | 无 | AIV 端执行后处理并写回 GM |

**CV 同步协议**：使用 `CvSync::MODE`（=4），AIC 通过 `CrossCoreSetFlag` 通知 AIV 数据就绪，AIV 完成 Epilogue 后通过 `CrossCoreSetFlag` 回复 AIC。Flag ID 按 `count / COUNT_ID_MAX % COUNT_FLAG` 轮转，避免信号冲突。

**Params 额外字段**：在 MatmulKernel 基础上增加 `EpilogueParams epilogueParams`。

---

## §3 GroupMatmulKernel

| 项目 | 说明 |
|------|------|
| 头文件 | `kernel/group_matmul_kernel.h` |
| 命名空间 | `Kernel::GroupMatmulKernel` |
| 模板签名 | `template <class ProblemShape, class BlockMmad, class BlockScheduler, class Epilogue = void>` |
| 核心职责 | M 轴分组 Matmul，每组独立调度 |

**分组机制**：

- `groupListGmAddr`：GM 上的 `int64_t` 数组，每个元素为该组的 M 维度大小
- 外层循环遍历 `groupIdx ∈ [0, groupNum)`，每轮：
  1. 读取 `groupM = groupList[groupIdx]`
  2. 计算 `prefixM`（前缀 M 偏移），用于定位 A/C 的 GM 地址
  3. B 矩阵按 `groupIdx * n * k` 偏移（每组独立 B）
  4. 调用 `CalcBalancedBaseM` 动态调整当前组的 baseM
  5. 调用 `bs.UpdateBaseM(curBaseM)` + `bs.UpdateNextProblem(shape)` 刷新调度器

**Epilogue 默认值**：`Epilogue = void` 时为纯 AIC 直写 GM；传非 void Epilogue 启用 AIC/AIV 融合路径，通过 `GroupMatmulEpilogueTraits` 判断是否启用。

**TileContext 结构体**：融合路径下传递给 Epilogue 的上下文，包含 `groupIdx`、`groupNum`、`prefixM`、`groupM`、`mOffset`、`nOffset`、`curM`、`curN`、`writeM`、`totalM`、`totalN`、`totalK`。

---

## §4 MatmulKernelMxFused（MxMatmulKernelFused）

| 项目 | 说明 |
|------|------|
| 头文件 | `kernel/matmul_kernel_mx_fused.h` |
| 命名空间 | `Kernel::MxMatmulKernelFused` |
| 模板签名 | `template <class ProblemShape, class BlockMmad, class BlockScheduler, class Epilogue>` |
| 核心职责 | MX 量化 matmul + CV 融合 Epilogue |

**与 blaze 库 MX Block 的对接**：

- BlockMmad 使用 blaze 库的 `Blaze::Gemm::BlockMmad`（MX 量化版），而非 blaze_custom Block
- `operator()` 调用签名：`blockMmadOp(gmA, gmB, gmScaleA, gmScaleB, gmBias, ubBlockC, singleShape)`
- `Init` 接受 4D ProblemShape，附加 `isBias` / `dbL0C` 参数
- BlockScheduler 直接使用 blaze 库调度器（无 `BlockSchedulerSelector`）
- C0_SIZE 由 `IsFp4<AType>()` 决定：FP4 → 64，FP8 → 32
- L0CType 固定为 `float`
- Scale Tensor 需要额外的 layout（`ScaleADNLayoutPtn` / `ScaleANDLayoutPtn`）和 Slice
- `MxMatmulKernelFused` 传入 UB Tensor 触发 blaze library MX Block 的 L0C2UB 路径；当前该路径使用 `CopyL0C2UBSplitMTrait / DUAL_DST_SPLIT_M`，Epilogue 必须按 `GetTaskRation()/GetSubBlockIdx()` 消费 M 分片

**QBMMTiling 额外字段**：在 baseM/baseN/baseK/dbL0C 基础上增加 `isBias`（uint32_t）。

---

## §5 混用禁令

默认情况下，**blaze_custom 的 Kernel / Block / Scheduler 只能互相搭配，禁止与 blaze 库模块任意混用。** MX C+V 的 `MxMatmulKernelFused` 是唯一明确支持的受控例外。

| 组合 | 合法性 | 说明 |
|------|--------|------|
| blaze_custom Kernel + blaze_custom Block + blaze_custom Scheduler | 合法 | 标准 blaze_custom 路径 |
| 任意手工混用 blaze_custom Kernel + blaze 库 Block | **禁止** | 模板参数数量、Params、SFINAE 机制不兼容 |
| 任意手工混用 blaze_custom Kernel + blaze 库 Scheduler | **禁止** | Scheduler 接口不匹配 `BlockSchedulerSelector` |
| `MxMatmulKernelFused` + blaze 库 BlockMmad + blaze 库 Scheduler | 合法 | MX 场景的例外，Kernel 本身引用 blaze 库头文件 |

---

## §6 命名空间说明

| 模块 | 命名空间 | 说明 |
|------|---------|------|
| blaze_custom Kernel | `Kernel::` | 如 `Kernel::MatmulKernel`、`Kernel::MatmulKernelFused` |
| blaze 库 Kernel | `Blaze::Gemm::Kernel::` | 如 `Blaze::Gemm::Kernel::KernelMatmulBasic` |
| blaze_custom Block | `Block::` | 如 `Block::BlockMmad`、`Block::MatmulBlockScheduler` |
| blaze 库 Block | `Blaze::Gemm::Block::` | 如 `Blaze::Gemm::Block::BlockMmad` |

两者命名空间完全独立，不存在继承关系。blaze_custom 使用 `Block::BlockSchedulerSelector` 做调度器派发，blaze 库使用 `GemmUniversal` 框架直接绑定。
