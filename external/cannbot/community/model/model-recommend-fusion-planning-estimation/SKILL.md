---
name: external-cannbot-community-model-model-recommend-fusion-planning-estimation
description: 算子融合规划与理论性能估计（工作流条目 10）。对无现成解法的热点链路与算子适配失败项规划新融合/自定义算子方案，基于本机实测锚定的 Roofline
  做 bound 判定与理论收益排序，决定 kernel 开发优先级。触发：融合规划、理论性能估计、Roofline 分析、bound 判定、算力利用率、融合收益估算、开发优先级排序。
original-name: model-recommend-fusion-planning-estimation
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# 算子融合规划与理论性能估计

## Prerequisites

- [performance-analysis](../model-recommend-performance-analysis/SKILL.md) 的算子级耗时分布（kernel 表 + overlap 机会表 + 融合模式表）
- **[fusion-operator-adaptation](../model-recommend-fusion-operator-adaptation/SKILL.md) 的未适配项清单**（两大输入来源之一，见约束项）
- [structure-analysis](../model-recommend-structure-analysis/SKILL.md) 的 AIC/AIV 归属
- 硬件基准锚定数据见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) Step 2（本机实测算力/带宽）

## 目标

对无现成解法的热点链路与**算子适配失败项**，规划新融合/自定义算子方案，并在动手开发前给出理论收益排序（决定 [kernel-generation](../model-recommend-kernel-generation/SKILL.md) 的开发优先级）。

## 输入 / 输出契约

**输入**：profiling 算子级耗时分布；适配未适配项清单；AIC/AIV 归属。

**输出**：融合候选清单（链路 → 方案 → 理论收益估计 → bound 类型判定 → 开发路径建议 → **来源标记**），按收益/成本比排序。

## 约束项：候选来源与条目 11 唯一入口

- **候选两大来源，缺一不可**：
  1. **热点链路**（profiling 三表提取：小算子串 / GM 往返链路）
  2. **适配未适配项清单**（无算子 / 硬约束 / 语义不匹配 / 精度未达标——需新算子的项必须在本阶段并入规划，与热点链路候选项**同表统一排序**）
- 每个候选必须标注 `source`（hot-chain / adaptation-failure）及其证据（profiling 占比 或 未适配原因）
- 未适配项并入时须补齐 bound 判定与收益估计（适配阶段只给了失败原因，规划阶段补齐"值不值得开发"）——已有 profiling 证据的用实测锚定，无证据的用结构推导并标注
- **[kernel-generation](../model-recommend-kernel-generation/SKILL.md) 的唯一入口是本 skill 的排序结果**：任何新算子开发（含适配失败项的替代开发）必须先出现在本清单中；禁止从未适配项清单或热点分析直接跳到 kernel 开发
- 可用既有手段解决的项（onnx 改写 / GE 选项 / 调用修正）标注"回流 [fusion-operator-adaptation](../model-recommend-fusion-operator-adaptation/SKILL.md) 完成"，不占用开发名额
- 功能补齐型（图跑不起来）不参与性能排序，直接转 [custom-op-development](../model-recommend-custom-op-development/SKILL.md)（标注 `source=functional-gap`）

## 工作流步骤

### Step 1: 热点链路识别与未适配项并入

候选收集（两大来源）：

- **来源 A（热点链路）**：从 profiling 提取相邻 kernel 序列，识别：
  - **小算子串**：kernel launch 开销占比高（AIV/控制类算子密集链，单并发口径显著）
  - **GM 往返链路**：中间张量落回 HBM（memory-bound 链路显著）
  参考三表报告口径，默认 1% 累计占比阈值。
- **来源 B（未适配项清单）**：逐项核对是否已有 profiling 证据（有则并入来源 A 同口径量化，无则结构推导），标注 `source=adaptation-failure`

### Step 2: bound 判定（Roofline）

对每个候选链路计算算术强度（FLOPs/Bytes），对照**本机实测锚定**的：

- 算力峰值（小规模 matmul 实测 TFLOPS；AIC/AIV 分别看）
- 带宽峰值（HBM copy 实测）

判定 compute-bound / memory-bound。混合 bound（部分段计算、部分段访存）分段判定或加权。

### Step 3: 理论收益估计

```
融合收益 = 消除的 kernel launch 开销（小算子串显著，按单算子 launch 开销 × 消除数估）
         + 减少的 GM 读写（memory-bound 链路显著，按中间张量字节数估）
         + 减少的 dtype 转换（Cast 链按转换字节数估）
```

给出保守/乐观区间。

### Step 4: 融合机制选型（Ascend 910B，标准 AscendC）

| 融合机制 | 适用场景 | 说明 |
|---------|---------|------|
| GE 图内融合 pass | 算子级自动融合 | 零开发，先确认是否已被 GE 覆盖（回 [fusion-operator-adaptation](../model-recommend-fusion-operator-adaptation/SKILL.md)） |

**自写 pass 的参数契约（换模型不复用代码的前提）**：pass 里凡是「从模型结构才能确定的量」都不许硬编码，必须按以下顺序取值——① 从图属性/输入 shape 自动推断；② 由 skill 侧给出显式配置（`models/M<id>/config/fusion_pass.json`）；③ 仅在前两者都不可得时才允许环境变量覆盖，并在日志里打印「使用 env 覆盖 + 原因」。典型项：Gather 输出通道数、MSDA 的头数/尺度数、融合子图的输入索引。

```json
{
  "graph_fusion_pass": {
    "gather_out_c": 64,
    "msda_heads": 8,
    "msda_scales": 4,
    "source": "auto|config|env",
    "evidence": "从 graph attr / input shape 推断到的依据"
  }
}
```

pass 侧需对推断失败给出明确错误码与可读提示（不得静默用默认值跑出「看起来成功」的图），并把最终生效值写回 `data/{model_name}/fusion-planning-estimation/effective_params.json`，供复现与 review。
| onnx 等价改写 | 语义等价的形态变换 | 工具层零开发（回流条目 7） |
| AscendC 纯 Vector 融合算子 | elementwise/reduce/查表小算子串 | 标准 AscendC 常规 kernel，单 kernel 消 launch + 减 GM 往返 |
| AscendC Cube 融合算子 | GEMM + 前后处理 | **Matmul 高阶 API**（910B 支持）；950 的 Blaze/tensor_api 路径不适用本机 |
| AIC/AIV 协同 kernel | Cube 段 + Vector 段流水 | 按标准 AscendC 编程模型组织 |

- 不适用本机的机制（勿规划）：Blaze/tensor_api 三模板（950 专属）、SuperKernel 二进制融合、CCU 通算融合、cannbot-dsl 全融合路径
- Tiling 约束随机制变化，开发项移交时附 Buffer 约束提示（运行时 `PlatformAscendC` 获取，禁止硬编码）

### Step 5: 排序输出（双轨分流）

结合开发成本（模板复用度）排序，Top-N 进入 [kernel-generation](../model-recommend-kernel-generation/SKILL.md)。

**先查既有再谈新增**：候选先对照 [fusion-operator-adaptation](../model-recommend-fusion-operator-adaptation/SKILL.md) 的既有手段目录，只有无目录项匹配时才立为新开发项。

**双轨分流**：排序结果分两栏输出——① 可用既有手段解决的项 → **回流条目 7 完成**；② 需新算子开发的项 → **条目 11 开发清单（唯一入口）**。两栏均带来源标记与优先级。功能补齐型单列第三栏 → 条目 12。

**未实施候选保留（收口衔接）**：未进入 Top-N 的候选与暂缓项**不得丢弃**，保留在 `data/{model_name}/fusion-planning-estimation/` 结构化数据中（含 bound 判定与收益估计），供收口环节（[regression-baseline-mgmt](../model-recommend-regression-baseline-mgmt/SKILL.md) 下一步优化动作表）引用。

## 昇腾 NPU 专属要点

- **峰值一律引用本机实测锚定**（`data/env-baseline/`），禁止引用纸面值或他机型数据；锚定数据缺失时先补测（回 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md)）
- AIC/AIV 分离架构：候选链路标注目标核类型；AIV 小算子串融合的 launch 收益按单并发口径估算，高并发下 launch 开销被摊薄，收益区间须分场景给
- 查表类候选（Gather 链）注意索引分布对带宽的影响（结构推导时标注假设）

## 产物目录约束

- **文档格式约束（GFM 删除线）**：Markdown 文档禁止裸单波浪线 `~`——同段落/同表格单元内成对出现会被 GFM 渲染为删除线；数值区间用全宽 `～`（`-15%～-30%`）或 `-`/`至`，约数用 `≈`；交付前 grep 确认文档无裸 `~`（详见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束）
- **文档类交付件** → `docs/{model_name}/fusion-planning-estimation/`：融合候选清单与排序报告（含 bound 判定过程、理论收益估计区间、开发路径建议）
- **数据类交付件** → `data/{model_name}/fusion-planning-estimation/`：候选与收益估计结构化数据（供 [kernel-generation](../model-recommend-kernel-generation/SKILL.md) 程序化引用）
- **中间产物** → `tmp/{model_name}/fusion-planning-estimation/`：roofline 计算草稿、profiling 提取的中间数据
- 全局约定与判定标准见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md)；违反目录约定的产物视为未交付

## 验收标准

- 每个候选有 bound 判定过程（FLOPs/Bytes 计算可复核，峰值引用实测锚定）
- 收益估计与 [performance-analysis](../model-recommend-performance-analysis/SKILL.md) 实测对比误差在声明区间内
- **未适配项清单已逐项并入或显式排除（排除须给理由），候选带来源标记**
- 排序结论明确 Top-N 开发项，且双轨分流清晰（回流条目 7 / 进条目 11 / 功能补齐转条目 12）

## 依据

【实测】本机算力/带宽锚定数据。
【技能库经验】Roofline 分析、三表报告与 fuse-overlap-catalog 思路（先查既有路径，再谈新机会）、唯一入口与双轨分流纪律。
【框架】CANN 自带 AscendC Matmul 高阶 API 头文件与文档（910B 路径）。
