---
name: external-cannbot-ops-ascendc-blaze-best-practice
description: Blaze/tensor_api 路径的 Matmul 类算子开发指南（Ascend 950 / DAV_3510）。覆盖框架认知、模板目录、开发指南和扩展开发。触发：在
  A5 平台开发 matmul 类算子（普通 matmul、MX 量化 matmul、Grouped matmul）及 C+V 模式融合算子（上述三类 matmul
  + vector epilogue）时。不适用于纯 Vector 算子和 A2/A3 平台。
original-name: ascendc-blaze-best-practice
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# Blaze Matmul 算子开发指南

## 外部依赖

本 skill 依赖 ops-tensor 仓提供的 **blaze 库**、**tensor_api 库** 和 **官方文档**。

- **仓库**：[gitcode.com/cann/ops-tensor](https://gitcode.com/cann/ops-tensor)
- **适用平台**：DAV_3510 / Ascend 950
- **编码规范**：[ops-tensor/CODING_CONVENTIONS.md](https://gitcode.com/cann/ops-tensor/blob/master/CODING_CONVENTIONS.md)

Blaze 库模块文档查阅：`ops-tensor/docs/API/` → [blaze-modules-index.md](references/modules/blaze-library/blaze-modules-index.md)

## Blaze 及 Tensor API 简介

**Blaze** 是构建在 Tensor API 之上的 header-only C++ 模板库，为 Ascend NPU Cube Core 上的矩阵乘法提供可组合的高层抽象。**Tensor API** 是 Blaze 的底层依赖，封装了 AscendC 原语，提供张量抽象、Layout 推导和 Copy/Mmad 算法接口。

核心设计理念：

- **三层抽象栈**：AscendC 原语（硬件指令）→ Tensor API（张量抽象 + Layout 推导）→ Blaze（完整 Kernel 实现）
- **Pattern 驱动派发**：Layout Pattern 作为编译时标签，驱动 Copy/Mmad 的 Routing 表自动选择正确的硬件指令路径
- **组装式开发**：选择 Kernel + BlockMmad + Scheduler + Epilogue 四个组件，通过模板参数组合实现算子
- **默认库路径**：普通 MatMul 单算子和 MX 量化 MatMul 默认使用 blaze library（`Blaze::Gemm`）；Grouped MatMul、普通 C+V 融合和自定义扩展场景使用 blaze_custom；MX C+V 融合使用 `MxMatmulKernelFused` 受控组合态

完整的 NPU 执行模型、Tensor API 核心概念、Blaze 五层架构和路径选择决策，详见 [Blaze 框架总览](references/fundamentals/blaze-framework-overview.md)。

## 开发路径

基于 Blaze 开发 matmul 类算子，遵循三步主流程：

### Step 1: 工程初始化

拉取 blaze 库、tensor_api，搭建工程目录与 CMake 配置；Grouped MatMul、C+V 融合或自定义扩展场景需要额外拷贝 blaze_custom 模块。MX C+V 融合同时依赖 blaze library 的 MX Block/Scheduler 与 blaze_custom 的 bridge Kernel/Epilogue。
→ [Step 1: 工程初始化](references/development/step1-setup.md)

### Step 2: 定义 Kernel

设计 kernel 入口函数签名，选择并组装 Kernel/BlockMmad/Scheduler/Policy 组件，定义 TilingData 结构体，编写 SWAT tiling 引擎。
→ 总纲：[Step 2: Kernel 设计总纲](references/development/step2-kernel-design.md)
→ Tiling 选择：[Blaze Tiling 选择指南](references/tiling/tiling-selection.md)
→ 按场景查阅组装指南：

| 场景 | 文档 |
|------|------|
| 基础 MatMul | [基础 MatMul 开发](references/scenarios/basic-matmul-development.md) |
| A8W8 量化 MatMul | [A8W8 量化 MatMul 开发](references/scenarios/a8w8-quant-matmul-development.md) |
| MX 量化 MatMul | [MX 量化 MatMul 开发](references/scenarios/mx-matmul-development.md) |
| Grouped MatMul | [Grouped MatMul 开发](references/scenarios/group-matmul-development.md) |
| CV 融合（matmul + epilogue） | [CV 融合 MatMul 开发](references/scenarios/fusion-matmul-development.md) |

→ 组装时查阅模块说明：`references/modules/` 目录

### Step 3: 编写 Launcher

编写 host 端 C++ 入口：ACL 会话、内存管理、文件 I/O、layout dispatch、kernel launch。
→ [Step 3: 编写 Launcher](references/development/step3-launcher.md)

## Blaze Custom 扩展开发

当现有模块无法满足需求时，按层扩展：

| 扩展层 | 文档 |
|--------|------|
| Kernel 层 | [Kernel 层扩展](references/modules/blaze-custom/development/kernel-dev-guide.md) |
| Block 层 | [Block 层扩展](references/modules/blaze-custom/development/block-dev-guide.md) |
| Scheduler 层 | [Scheduler 层扩展](references/modules/blaze-custom/development/scheduler-dev-guide.md) |
| Epilogue 层 | [Epilogue 层扩展](references/modules/blaze-custom/development/epilogue-dev-guide.md) |
| MemBase Epilogue | [MemBase Epilogue 设计](references/modules/blaze-custom/development/epilogue-membase-design.md) |
| RegBase Epilogue | [RegBase Epilogue 设计](references/modules/blaze-custom/development/epilogue-regbase-design.md) |

## 参考手册

| 文档 | 用途 |
|------|------|
| [Blaze 框架总览](references/fundamentals/blaze-framework-overview.md) | 框架认知：NPU 模型、三层抽象栈 |
| [Tensor API 参考](references/fundamentals/tensor-api-reference.md) | tensor_api API 签名、Routing 表 |
| [Blaze 同步模式](references/fundamentals/blaze-sync-patterns.md) | 同步编码：HardEvent、CrossCore |
| [Blaze MatMul Layout](references/fundamentals/blaze-matmul-layout.md) | Layout 格式：ND/NZ/ZN、LayoutPtn |
| [Blaze Tiling 选择指南](references/tiling/tiling-selection.md) | Blaze 路径下选择 SWAT tiling engine 的入口 |
| [Blaze 设计文档模板](references/design/blaze-design-template.md) | Blaze 路线算子 DESIGN.md 骨架（Architect 填充） |

## 约束

- **禁止猜测 API 签名**：必须以 ops-tensor 源码或 [Tensor API 参考](references/fundamentals/tensor-api-reference.md) 为准
- **模板选择必须说明理由**：引用场景文档中的组件选择表
- **默认禁止任意混用 blaze 库和 blaze_custom 库**：普通 MatMul 单算子与纯 MX 量化 MatMul 均走 blaze library；普通 C+V 和 Grouped C+V 走 blaze_custom。唯一受控例外是 MX C+V 融合路径 `Kernel::MxMatmulKernelFused`，它专门桥接 blaze library MX Block/Scheduler 与自定义 Epilogue（详见 [Step 2](references/development/step2-kernel-design.md) §2）
- **默认只提供 SWAT tiling**：本 skill 的 `assets/op_tiling/` 仅维护普通 matmul SWAT 和 MX SWAT 两类 tiling；Grouped 场景复用对应非 grouped tiling
- **伪代码不等于可编译实现**：写代码时参考场景文档中的完整组装代码
