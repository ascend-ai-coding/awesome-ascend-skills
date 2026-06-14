---
name: external-cannbot-ops-ascendc-blaze-best-practice
description: Matmul/Cube/GEMM/BMM 单算子及 matmul+vector 融合算子直调生成（Blaze/tensor_api 路径）。当用户提到
  Blaze 时必须使用此 skill。覆盖纯AIC/StreamK/FixpOpti 三模板选型、RegBase/MemBase 两种 epilogue 路径（推荐
  RegBase）、改造、Tiling 及排错。不适用纯 Vector 逐元素/归约算子或非 Blaze 路径。
original-name: ascendc-blaze-best-practice
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-06-14'
synced-commit: f2b0db65abdced502b1cd1956e2e48e6e0197add
license: UNKNOWN
---

# Ascend C Matmul 单算子生成（Blaze 路径）

该能力解决 Matmul 单算子及 matmul+vector 融合算子生成场景下的问题：tensor_api/Blaze API 约束确认、三模板选型（纯AIC / StreamK / FixpOpti）、RegBase/MemBase 两种 epilogue 路径选择（推荐 RegBase，MemBase 仅用于简单 vector 场景）、统一工程模板复用、SWAT/全载 Tiling 改造、流水与排错。

> **架构限定**：核心验证在 `DAV_3510`。其他架构必须在 DESIGN.md 中显式说明差异并给出适配方案。

> **首次使用**：`matmul_custom/` 依赖 `tensor_api/`，首次需拉取外部依赖。步骤见 [`matmul_pattern.md`](references/matmul_pattern.md) §0.5.0。

## 何时使用

满足以下任一条件时使用该能力：

- 算子类型为 Matmul 族（GEMM/BMM/量化 matmul/matmul+bias/mxfp8 matmul）。
- 算子类型为 matmul+vector 融合（matmul+GELU/matlul+SwiGLU/matmul+scale 等含 Vector 后处理）。
- 代码或讨论中出现 `tensor_api`、`blaze` 等信号。
- 需要选择模板（纯AIC/StreamK/FixpOpti）、选择 epilogue 路径（推荐 RegBase / 简单场景 MemBase）、切换 dispatch mode（NO_FULL/A_FULL/B_FULL_LOAD）、进行 Tiling 改造或流水排错。

不要把该能力当成默认算子开发路径的通用替代品。纯 Vector 逐元素/归约算子（无 matmul 前段）不在本 skill 覆盖范围内。matmul+vector 融合场景使用 FixpOpti 模板做 Cube 段 + RegBase epilogue 做 Vector 段（简单场景可用 MemBase，详见 [`matmul_fixpopti_regbase_epilogue.md`](references/matmul_fixpopti_regbase_epilogue.md)）。

## 三模板速览

| 模板 | 典型场景 | 交付状态 |
|------|---------|---------|
| **纯AIC** | 通用场景（默认） | **已交付** |
| **StreamK** | MN 欠并行 + 长 K（≥4096） | **设计文档** |
| **FixpOpti** | AIC/AIV 流水重叠、可定制 epilogue | **已交付** |

> 三模板共享同一基底 `references/matmul_custom/`（common/、tiling/、scheduler/、utils/）。
> FixpOpti 专用文件（`matmul_fixpopti.cpp`、`matmul_kernel_fused.h`、epilogue/）也在该目录中。

> 详细对比（执行模式、调度器、同步机制、Workspace、ASCII 数据流图、决策树）见 [`matmul_pattern.md`](references/matmul_pattern.md) §10，此处不重复。

## 按角色/阶段查阅

阅读顺序：先速览 [`tensor_api_user_guide.md`](references/tensor_api_user_guide.md) 了解 API → [`matmul_pattern.md`](references/matmul_pattern.md) §10 选模板 → 纯AIC 路径继续 §0 选 mode → §0.5 复制+改造 → 共享基础 §1–§9 + 专属深度文档。

| 阶段/角色 | 主文档 | 辅助 / 深度 |
|---|---|---|
| **方案决策（Architect）** | `matmul_pattern.md` §10 三模板选择 + §0 模式总览 | `tensor_api_user_guide.md`（API 速查） |
| **设计（Architect）** | 选定模板后进入深度文档：<br>- 纯AIC：`matmul_basic.md` / `matmul_full_load.md`<br>- StreamK：`matmul_streamk.md`<br>- FixpOpti：`matmul_fixpopti.md` | `matmul_pattern.md` §1–§9 共享基础 |
| **实现（Developer）** | 统一从 `references/matmul_custom/` 基底出发：<br>纯AIC → `matmul_custom.cpp`（`[MODIFY]` N/C/A）<br>FixpOpti → `matmul_fixpopti.cpp`（`[MODIFY]` N/C/A/E），按 `matmul_fixpopti.md` §3 改造<br>RegBase epilogue（推荐）→ `epilogue_fusion_regbase.h`（`[USER]` T1-T4），按 `matmul_fixpopti_regbase_epilogue.md` §8 开发<br>MemBase epilogue（简单场景）→ `epilogue_fusion_membase.h`（MulEpilogue 参考样例） | `matmul_fixpopti.md`（改造食谱）<br>`matmul_fixpopti_regbase_epilogue.md`（RegBase epilogue 方法论） |
| **Layout/格式开发** | [`matmul_layout_guide.md`](references/matmul_layout_guide.md)（ND/DN/NZ/ZN 格式定义 + 数据生成 + LayoutPtn 选型 + kernel 适配 + 排障） | `tensor_api_user_guide.md` §1.3+§3（API 参考） |
| **审查（Reviewer）** | `matmul_pattern.md` §8 排障速查 | 对应模板深度文档的「常见陷阱」表 |
| **修复调试** | `matmul_pattern.md` §8（编译期 / 精度 / 跑通自检） | `matmul_fixpopti.md` §6 陷阱表 |

## 约束

- **架构验证范围**：DAV_3510。其他架构因 `tensor_api` 依赖和 L1/L0 容量差异，可能不兼容。
- **全载交付状态**：`A_FULL_LOAD_MODE` 已落地（`matmul_block_mmad_a_full_load.h` + `MatmulTilingAFullLoad`，3 行 diff 切换）。`B_FULL_LOAD_MODE` **仅有设计草案**（见 `matmul_full_load.md` §3.2 / §4.2 / §4.3），仓库中无 `B_FULL_LOAD_MODE` 常量、`MatmulMultiBlockPolicy<B_FULL_LOAD_MODE>` 特化、`MatmulSwatScheduler<B_FULL_LOAD_MODE>` selector、`MatmulTilingBFullLoad` 类、`matmul_block_mmad_b_full_load.h` 头文件——使用前必须按对称设计自行镜像补齐，不能套用 `matmul_pattern.md` §0.5.3 的切换 diff（B 全载段已替换为告警 NOTE）。
- **模板复制起手**：纯AIC 从 `references/matmul_custom/matmul_custom.cpp` 入手，FixpOpti 从 `references/matmul_custom/matmul_fixpopti.cpp` 入手；两条路径共享 `references/matmul_custom/` 下的 common/、tiling/、scheduler/、utils/。均按 `[MODIFY]` 标记改造。FixpOpti 额外需要：(1) 替换启动器 `matmul_custom.cpp`→`matmul_fixpopti.cpp`，(2) BlockMmad 的 `CopyL0C2GM` 改为 `CopyL0C2UB`+SPLIT_M Trait（见 `matmul_fixpopti.md` §3.5），(3) 新增 `matmul_kernel_fused.h` 和 epilogue 文件。比纯AIC 多一档 `[MODIFY] E` 用于自定义 Epilogue。
- **Epilogue 路径选择**：推荐 RegBase 路径（`epilogue_fusion_regbase.h`），使用 `__VEC_SCOPE__` + `AscendC::Reg::*` API。MemBase 路径（`epilogue_fusion_membase.h`）仅适用于单个 vector 操作且有明确可用 `AscendC::` API 的场景（如 `AscendC::Mul/Add/Div`）；其他场景应使用 RegBase。RegBase API 约束、签名和陷阱参见 `ascendc-regbase-best-practice` skill。独立 vector 验证的工程脚手架（CMakeLists、run.sh、scripts）参见 `ascendc-direct-invoke-template` skill。
- **API 签名不猜测**：`AscendC::Te::` 系列接口有大量重载和模板特化。查阅 `references/tensor_api_user_guide.md` 或官方文档。
- **伪代码不等于可编译实现**：设计文档中的代码片段为说明概念而简化。写代码时回到对应模板工程——细节已处理好。
- **mode 和模板选择说明依据**：在 DESIGN.md 中写清楚为何选这个 mode/模板。
- **Layout/格式开发**：ND/DN/NZ/ZN 格式定义、数据生成流程、LayoutPtn 选型、kernel 适配点、排障等完整指导详见 [`matmul_layout_guide.md`](references/matmul_layout_guide.md)。

## 与 CANNBot 集成

| CANNBot 阶段 | 调用方 | 加载方式 |
|---|---|---|
| Step 2 设计 | Architect Subagent | 当用户需求路径判定为「tensor_api / CUTLASS 风格 API」时显式加载本 skill |
| Step 3 开发 | Developer Subagent | 按设计选定模板，复用模板工程 |
| Step 4 审查 | Reviewer Subagent | 用 `matmul_pattern.md` §8 排障速查交叉验证 |
