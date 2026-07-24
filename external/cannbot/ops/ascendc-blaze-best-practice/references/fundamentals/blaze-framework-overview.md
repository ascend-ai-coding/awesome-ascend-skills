# Blaze 框架总览

> **适用平台**：DAV_3510 / Ascend 950 / CANN 9.1.0
>
> 本文介绍 NPU 执行模型、三层抽象栈、Tensor API 核心概念、Blaze 分层架构和两条开发路径。

---

## 1. NPU 执行模型

### 1.1 存储层级与数据流

昇腾 NPU Cube 单元的存储层级和数据流：

```
                    ┌─────────────────────────────────────────────┐
                    │              GM (Global Memory / DDR)         │
                    └────┬──────────────────────────────┬─────────┘
                         │ MTE1                         │ MTE3
                         ▼                              ▲
                    ┌─────────────────────────────────────────────┐
                    │              L1 (片上 SRAM, ~512 KB)          │
                    └────┬──────────────┬──────────────┬──────────┘
                         │ MTE2         │ MTE2         │
                         ▼              ▼              │
                    ┌──────────┐  ┌──────────┐         │
                    │   L0A    │  │   L0B    │         │
                    │  (~64KB) │  │  (~64KB) │         │
                    └────┬─────┘  └────┬─────┘         │
                         └──────┬──────┘               │
                                ▼                      │
                         ┌──────────────┐              │
                         │    MMAD      │              │
                         │ C += A × B   │              │
                         └──────┬───────┘              │
                                ▼                      │
                         ┌──────────────┐              │
                         │  L0C (~256KB)│              │
                         └──────┬───────┘              │
                                │ Fixpipe              │
                                └──────────────────────┘
```

### 1.2 传输引擎与计算单元

| 组件 | 功能 | 方向 |
|------|------|------|
| **MTE1** | GM ↔ L1 搬运 | 矩阵 A/B 从 DDR 加载到片上 |
| **MTE2** | L1 ↔ L0A/L0B/UB 搬运 | tile 从 L1 送入 Cube 输入或 Vector 工作区 |
| **MTE3 (Fixpipe)** | L0C → GM 搬出 | 累加结果写回 DDR，支持 Cast/量化 |
| **Cube (M)** | 矩阵乘累加 | C_tile += A_tile × B_tile |
| **Vector (V)** | 向量计算 | 非矩阵乘运算（加 bias、激活、Cast 等） |

### 1.3 核心类型

| 核心 | 包含单元 | 用途 |
|------|---------|------|
| **AIC** (AI Cube) | Cube + Vector + MTE1/2/3 + Fixpipe | 矩阵乘主核 |
| **AIV** (AI Vector) | Vector + MTE2/3 | 向量计算核，用于 epilogue 后处理 |

混合核启动：`__mix__(aicCount, aivCount)` 指定 AIC/AIV 核数比例。

### 1.4 缓冲区尺寸（DAV_3510）

| 缓冲区 | 容量 | 用途 |
|--------|------|------|
| L1 | 512 KB | A/B 矩阵 tile 暂存（支持双缓冲 ping-pong） |
| L0A | 64 KB | Cube 输入 A |
| L0B | 64 KB | Cube 输入 B |
| L0C | 256 KB | Cube 累加器（FP32） |
| UB | 256 KB | Vector 工作区 |

### 1.5 事件同步概述

各引擎之间通过 **HardEvent** 同步。DataCopy/DataCopyPad 是**异步 DMA**，必须通过 `SetFlag/WaitFlag` 确保消费者在读数据前生产者已完成。

```
MTE1 ──SetFlag(MTE1_MTE2)──▶ MTE2    // GM→L1 完成后通知 L1→L0
MTE2 ──SetFlag(MTE2_MTE1)──▶ MTE1    // L1→L0 完成后通知下一轮 GM→L1
MTE1 ──SetFlag(MTE1_M)──▶ Cube       // L0 加载完成通知 MMAD
Cube ──SetFlag(M_FIX)──▶ Fixpipe     // MMAD 完成通知 L0C 搬出
```

完整的事件枚举、配对规则和 CrossCore 核间同步，详见 `blaze-sync-patterns.md`。

---

## 2. 三层抽象栈

Blaze 生态构建在三层抽象之上，每层封装不同级别的硬件细节：

```
┌─────────────────────────────────────────────────────────┐
│  Blaze（include/blaze/）                                 │
│  完整矩阵乘 Kernel 实现：Block 调度、策略分发、Epilogue     │
│  组装式开发：选择 Kernel + BlockMmad + Scheduler + Epilogue│
├─────────────────────────────────────────────────────────┤
│  Tensor API（include/tensor_api/）                       │
│  张量抽象：Layout 推导、Copy/Mmad 算法接口、Routing 派发   │
│  编译时最大化：Pattern 决定派发，零运行时开销               │
├─────────────────────────────────────────────────────────┤
│  AscendC 原语（kernel_operator.h）                       │
│  硬件指令：DataCopyPad、Mmad、SetFlag/WaitFlag            │
│  手动管理 buffer、layout、事件同步                        │
└─────────────────────────────────────────────────────────┘
```

| 层级 | 开发者做什么 | 框架封装什么 |
|------|------------|------------|
| AscendC 原语 | 手动管理所有 buffer 偏移、layout 构造、事件同步 | 无 |
| Tensor API | 指定 Pattern + Type，调用 Copy/Mmad | Layout 推导、Routing 派发、硬件指针类型安全 |
| Blaze | 选择组件（Policy + BlockMmad + Scheduler），填充 Params | 完整 tile 循环、双缓冲管理、事件同步、K 迭代 |

---

## 3. Tensor API 核心概念

Tensor API 是 header-only 库（命名空间 `AscendC::Te`），唯一公共入口 `#include "tensor_api/tensor.h"`。

### 3.1 核心四元组

| 概念 | 类型 | 说明 |
|------|------|------|
| **Shape** | `Std::tuple<...>` | 维度大小，支持嵌套（分形 layout） |
| **Stride** | `Std::tuple<...>` | 每维步长 |
| **Coord** | `Std::tuple<...>` | 多维坐标 |
| **Layout** | Shape + Stride + Info | Info = `tuple<Pattern, Trait>`，携带编译时标签 |

### 3.2 Layout Pattern 体系

Layout Pattern 是空标签结构体，用于编译时路由。13 种 Pattern：

| Pattern | 形态 | 主要用途 |
|---------|------|---------|
| `NDExtLayoutPtn` | 二维嵌套 ND | GM 入口（A 行主序、C 输出） |
| `DNExtLayoutPtn` | 二维嵌套 DN | GM 转置入口 |
| `NDLayoutPtn` | 普通 ND | UB / 内层工作 Tensor |
| `DNLayoutPtn` | 普通 DN | UB / 内层 |
| `NZLayoutPtn` | NZ 分形 | L1 A/B、L0A、GM 入口（NZ 预重排） |
| `ZNLayoutPtn` | ZN 分形 | L1 A/B（转置）、L0B、GM 入口（ZN 预重排） |
| `ZZLayoutPtn` | ZZ 分形 | L1 ScaleA（MX 量化） |
| `NNLayoutPtn` | NN 分形（C0=2） | L1 ScaleB（MX 量化） |
| `ScaleANDLayoutPtn` / `ScaleADNLayoutPtn` | A 侧 scale | GM → L1 ZZ |
| `ScaleBNDLayoutPtn` / `ScaleBDNLayoutPtn` | B 侧 scale | GM → L1 NN |
| `DefaultPtn` | 占位 | 内部使用 |

> **Ext 后缀**：Ext = "Extended/外层"，用于 GM 外部入口的双层嵌套 shape；非 Ext 用于内部连续缓冲。Routing 表对 GM ↔ 片上强制区分 Ext / 非 Ext。

### 3.3 C0 速查

C0 是分形格式中的元素粒度，`C0 = 32 / sizeof(dtype)`（fp4 例外）：

| dtype | sizeof | C0_ELEMENT |
|-------|--------|-----------|
| bf16 / fp16 | 2 | 16 |
| fp32 | 4 | 8（L0C 上仍按 16 处理） |
| int8 / fp8_e4m3 / fp8_e5m2 | 1 | 32 |
| fp4×2 | 0.5 | 64（LayoutTraitFP4） |
| fp8_e8m0（scale） | 1 | 2（LayoutTraitScale） |

> **关键常量**：`FRACTAL_FIXED = 16`，`BLOCK_CUBE = 16`。DAV_3510 上 L0C cube 边长恒为 16，不要写成 `32/sizeof(L0CType)`。

### 3.4 FrameLayout 工厂

```cpp
// 根据 Pattern + Trait 自动推导正确的 Shape 和 Stride
auto layout = AscendC::Te::MakeFrameLayout<NDExtLayoutPtn, AscendC::Te::LayoutTraitDefault<half>>(M, K);
// NZ 格式：自动推导分形 shape = ((FRACTAL_FIXED, M1), (C0, N1))
auto nzLayout = AscendC::Te::MakeFrameLayout<NZLayoutPtn, AscendC::Te::LayoutTraitDefault<half>>(M, K);
```

### 3.5 Copy/Mmad Atom 模式

Tensor API 使用 Atom 模式封装搬运和计算操作：

```cpp
// 1. 构造 Atom（编译时）
auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
auto mmadAtom  = AscendC::Te::MakeMmad(AscendC::Te::MmadOperation{}, AscendC::Te::MmadTraitDefault{});

// 2. 注入运行时参数
mmadAtom = mmadAtom.with(mmadParams);  // cmatrixInitVal, unitFlag 等

// 3. 执行（dst 在前）
AscendC::Te::Copy(copyGM2L1, dstL1Tensor, srcGMTensor);
AscendC::Te::Mmad(mmadAtom, dstL0C, srcL0A, srcL0B);
```

### 3.6 两个关键设计点

1. **Pattern 决定派发**：Routing 表按 `(dstLocation, srcLocation, archVersion, dstPattern, srcPattern)` 做编译期派发。Pattern 不在白名单 → `static_assert: Unsupported layout pattern`。
2. **同步是用户责任**：Tensor API 不接管 `SetFlag/WaitFlag`，事件配对由调用方维护（Blaze 在 BlockMmad 层封装了事件管理）。

tensor_api 各 API 的完整签名、参数类型和使用示例，详见 `tensor-api-reference.md` §2。
Copy 操作的合法 Pattern 组合路由表，详见 `tensor-api-reference.md` §3。

---

## 4. Blaze 分层架构

Blaze 是 header-only C++ 模板库（命名空间 `Blaze::Gemm`），构建在 Tensor API 之上。

### 4.1 五层架构

```
┌─────────────────────────────────────────────────────────┐
│  Kernel 层：GemmUniversal / MatmulKernel / ...           │
│  最外层入口，SFINAE 按 ScheduleType 选择偏特化            │
├─────────────────────────────────────────────────────────┤
│  Block 层：BlockMmad + BlockScheduler + BlockEpilogue    │
│  BlockMmad：单 tile 的 K 迭代 + 双缓冲 + 事件同步         │
│  BlockScheduler：tile 到核的映射（蛇形遍历、尾块处理）     │
│  BlockEpilogue：后处理（空操作等）                         │
├─────────────────────────────────────────────────────────┤
│  Policy 层：DispatchPolicy 标签                           │
│  编译时路由：Policy → ScheduleType → SFINAE 选择实现      │
├─────────────────────────────────────────────────────────┤
│  Tile 层：底层搬运/转换原语                               │
│  pad_mx_kl1、shift_w4_to_w8、copy_gm_to_ub 等           │
├─────────────────────────────────────────────────────────┤
│  Epilogue 层：后处理融合                                   │
│  RegBase（__VEC_SCOPE__ + Reg:: API）/ MemBase（AscendC）│
└─────────────────────────────────────────────────────────┘
```

### 4.2 DispatchPolicy 路由

DispatchPolicy 是编译时标签，驱动 SFINAE 选择正确的 GemmUniversal 和 BlockMmad 偏特化：

```cpp
// Policy 定义
struct MatmulMultiBlockBasic { using ScheduleType = KernelMmadMultiBlockBasic; };
struct MatmulWithScaleMx { using ScheduleType = KernelMmadWithScaleMx; };

// SFINAE 选择
template <typename ProblemShape, typename BlockMmad, typename BlockEpilogue, typename BlockScheduler,
          typename Enable = void>
struct GemmUniversal { static_assert(...); };  // 兜底

template <typename PS, typename BM, typename BE, typename BS>
struct GemmUniversal<PS, BM, BE, BS,
    enable_if_t<is_same_v<typename BM::DispatchPolicy::ScheduleType, KernelMmadMultiBlockBasic>>>
{ /* Basic 实现 */ };
```

### 4.3 GemmUniversal 四组件组装

```cpp
using KernelImpl = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape,      // 问题规模类型（如 MatmulShape 或 Shape<int64_t,int64_t,int64_t,int64_t>）
    BlockMmad,         // 计算核心（含 DispatchPolicy + 数据类型 + Layout）
    BlockEpilogue,     // 后处理（void 或 BlockEpilogueEmpty 或自定义）
    BlockScheduler     // tile 调度器
>;
```

### 4.4 BlockMmad 10 参数签名

```cpp
template <
    typename DispatchPolicy_,   // 策略标签（驱动 SFINAE）
    typename AType,             // A 矩阵数据类型
    typename LayoutA,           // A 矩阵 Layout Pattern
    typename BType,             // B 矩阵数据类型
    typename LayoutB,           // B 矩阵 Layout Pattern
    typename CType,             // C 输出数据类型
    typename LayoutC,           // C 输出 Layout Pattern
    typename BiasType = void,   // Bias 数据类型（可选）
    typename LayoutBias = void,  // Bias Layout Pattern（可选）
    typename Enable = void      // SFINAE 占位
>
struct BlockMmad;
```

---

## 5. 两条开发路径

根据算子类型，选择不同的开发路径：

### 5.1 路径选择决策

| 算子类型 | 推荐路径 | 理由 |
|---------|---------|------|
| 普通 MatMul 单算子（fp16/bf16/fp32） | **路径 B**：blaze 库直接组装 | `KernelMatmulBasic + BlockMmadMatmulBasic + BlockSchedulerMatmulBasic + BlockEpilogueEmpty` 已覆盖常规单算子场景 |
| MX 量化 matmul（fp8/fp4） | **路径 B**：blaze 库直接组装 | blaze 已封装 Scale 反量化、fp4 shift、K-padding 等复杂逻辑 |
| A8W8 量化 matmul | **路径 B**：blaze 库直接组装 | blaze 已封装 Fixpipe 反量化路径 |
| matmul + vector epilogue 融合 | **路径 A**：自定义 | AIC/AIV 混合核 + CrossCore 同步 + 自定义 Epilogue |
| Grouped matmul | **路径 A**：自定义 | 需要 GroupScheduler + per-group 迭代逻辑 |

### 5.2 路径 A：自定义（tensor_api + 手写 kernel/block）

```
Launcher (.cpp)
  ├── 选择类型（AType/BType/CType）
  ├── 选择 Layout（NDExt/DNExt/NZ/ZN）
  ├── 实例化：MatmulKernel<ProblemShape, BlockMmad, BlockScheduler>
  ├── 调用 Tiling 引擎 → 填充 Params
  └── kernel<<<usedCoreNum>>>(params)

组件来源：matmul_blaze_template（include/kernel/, include/block/, include/policy/）
依赖：仅 tensor_api（third_party/tensor_api/）
```

### 5.3 路径 B：blaze 库直接组装

```
Launcher (.cpp)
  ├── 选择 blaze DispatchPolicy（普通 MatMul 用 MatmulMultiBlockBasic，MX 量化用 MatmulWithScaleMx）
  ├── 实例化：Blaze::Gemm::Block::BlockMmad<DispatchPolicy, ...>
  ├── 实例化：Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>
  ├── 调用 Tiling 引擎 → 填充 Params
  └── kernel<<<usedCoreNum>>>(params)

组件来源：ops-tensor 仓（include/blaze/）
依赖：tensor_api + blaze（third_party/tensor_api/ + third_party/blaze/）
```

各路径的完整端到端组装代码，详见 `blaze-matmul-assembly.md`。
可用模板的完整清单和场景推荐，详见 `blaze-template-catalog.md`。

---

## 6. NZ 分形格式

NZ 是昇腾 NPU 的分形存储格式，用于 Cube 硬件高效读取。

### 6.1 物理排列

对原始 tensor `(dim0, dim1)` 的 ND 排布，NZ 格式的物理排列为：

```
(dim1/C0, dim0/16, 16, C0)
```

其中 `C0 = 32 / sizeof(dtype)`。非对齐时先补 0 到 16 对齐。

### 6.2 与 ND 的 shape 对应

| 矩阵 | trans | format | 物理排列 | LayoutPtn |
|------|-------|--------|---------|-----------|
| A[M,K] | false | ND | (M, K) | `NDExtLayoutPtn` |
| A[M,K] | false | NZ | (K/C0, M/16, 16, C0) | `NZLayoutPtn` |
| A[M,K] | true | ND | (K, M) | `DNExtLayoutPtn` |
| A[M,K] | true | NZ | (M/C0, K/16, 16, C0) | `ZNLayoutPtn` |
| B[K,N] | false | ND | (K, N) | `NDExtLayoutPtn` |
| B[K,N] | false | NZ | (N/C0, K/16, 16, C0) | `NZLayoutPtn` |
| B[K,N] | true | ND | (N, K) | `DNExtLayoutPtn` |
| B[K,N] | true | NZ | (K/C0, N/16, 16, C0) | `ZNLayoutPtn` |

A/B 矩阵在全部 trans×format 组合下的 LayoutPtn 映射和数据生成方法，详见 `blaze-matmul-layout.md` §1-§2。

---

## 7. 平台信息速查

### 7.1 架构检测

```cpp
// 编译时
#if __NPU_ARCH__ == 3510
    // DAV_3510 特化代码
#endif

// tensor_api 内部
constexpr uint32_t CURRENT_ARCH_VERSION = GetArchVersion{}();  // 取自 __NPU_ARCH__
```

### 7.2 Host 侧平台查询

```cpp
#include "platform_ascendc.h"
auto& mgr = platform_ascendc::PlatformAscendCManager::GetInstance();
int aicNum = mgr.GetCoreNumAiv();       // AIC 核数
int l1Size = mgr.GetCoreMemSize(1);     // L1 容量（字节）
int ubSize = mgr.GetCoreMemSize(0);     // UB 容量（字节）
```

### 7.3 Kernel 侧编译时常量

```cpp
// BlockMmad 中常用
constexpr int HALF_L0_SIZE  = TOTAL_L0A_SIZE / 2 / sizeof(AType);
constexpr int HALF_L1_SIZE  = TOTAL_L1_SIZE / 2;
constexpr int MATMUL_L0C_SIZE = 256 * 1024;  // L0C 容量（字节）
```

### 7.4 Kernel 侧运行时内建

```cpp
int blockIdx  = GetBlockIdx();   // 当前核 ID
int blockNum  = GetBlockNum();   // 总核数
int taskRatio = GetTaskRation(); // AIC/AIV 比例（混合核场景）
```
