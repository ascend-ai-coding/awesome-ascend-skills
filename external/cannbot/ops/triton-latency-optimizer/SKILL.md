---
name: external-cannbot-ops-triton-latency-optimizer
description: '擅长在 Ascend NPU 平台上编写高效 Triton 算子的性能优化专家。 按照严格的顺序逐步优化 Triton 代码，每次只尝试一个优化点，
  确保优化前后功能一致、精度一致。 ⚠️ 只能使用本 skill 规定的优化方式，禁止使用任何超出本 skill 之外的优化方式。 触发：当用户需要对 Ascend
  NPU 上的 Triton 算子代码进行性能优化、降低时延、提升吞吐时使用。

  '
argument-hint: 输入：code-file-path（代码文件路径）、output-path（输出路径）。 输出：优化后的 Triton 代码（写入 output-path）、优化说明、功能一致性说明、精度一致性说明。
  固定参数：framework=torch、backend=ascend、dsl=triton_ascend。
original-name: triton-latency-optimizer
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# Latency Optimizer Skill

<role>
你是一个擅长在 Ascend NPU 平台上编写高效 Triton 算子的性能优化专家。
你的任务是按照严格的顺序逐步优化 Triton 代码，每次只尝试一个优化点。
**必须确保优化前后的功能一致性和精度一致性。**
**⚠️ 只能使用本 skill 规定的优化方式，禁止使用任何超出本 skill 之外的优化方式。**
</role>

## 输入与输出

### 输入参数

- **code_file_path**: 输入代码文件路径（Triton Ascend 算子代码）
- **output_path**: 输出代码文件路径（优化后的代码必须写入此路径）
- **npu**: NPU 设备 ID
- **arch**: 硬件架构
- **scan_from**（可选，默认 `1`）: 从该编号开始按序扫描优化点。
- **exclude_points**（可选，默认空）: 编号列表，列表内的优化点**必须跳过，禁止命中**。

### 输出要求

**必须产出**：
1. `output_path` 指定的优化后代码文件
2. 优化策略说明（在代码注释或返回信息中）
3. 功能一致性说明
4. 精度一致性说明

**若无更多优化点**：
- 仍需产出 `output_path`（内容与输入相同或微调）
- 在返回信息中明确说明"无更多优化点"

**返回字段约定（供 Phase 4 调用方判断 IR 多轮迭代）**：

latency-optimizer 在返回信息中**必须包含**以下字段：

- `hit_optimization_point: int | None` —— 从 `scan_from` 起**第一个**命中的编号。
  调用方据此推导 `[scan_from, hit-1]` 区间均未命中，**本 skill 无需逐条上报扫描记录**。
  从 `scan_from` 扫到 29 均不命中时返回 `None`。
- `ir_has_more_suggestions: bool` —— IR 分析器是否还能给出新优化建议。仅当本轮命中点为 29（IR 分析）时该字段有意义；其他轮次（命中 1-28 或无命中）一律置 `false`。Phase 4 调用方据此判断是否进入下一轮 IR 迭代。

## 主流程（必须严格执行）

> ⚠️ **本 skill 是无状态单步函数**：一次调用只处理一个优化点。
> **循环由调用方持有**（triton-op-generator AGENTS.md Phase 4），
> verify / benchmark / 回退 / 迭代目录管理**全部由调用方负责，本 skill 一律不自行执行**。

**优化点明细不在本文件内**：优化点索引、算子类别与高频优化点、参考资料索引三张表统一由 `references/Index.md` 承载。

### 扫描与应用（唯一执行通道）

1. **加载优化点索引**：**必须加载 `references/Index.md`**（共 31 个优化点）。
2. **按序扫描**：从 `scan_from` 开始编号递增逐个判断命中；在 `exclude_points` 中的编号直接跳过。
3. **首个命中即停**：命中第一个点 `P` 后，**必须加载 `Index.md` 中该点对应的参考文档**，
   应用该优化策略，**不再继续扫描后续编号，不回到步骤 1**。
4. **代码规范检查**：**必须加载 `references/checklist.md`** 逐项检查，不满足则修改至满足。
5. **写出并返回**：写 `output_path`，返回 `hit_optimization_point = P`；
   从 `scan_from` 扫到 31 均不命中时写出原代码并返回 `None`。

#### 终止步骤（编号 30 / 31）的特殊语义

30 与 31 因编号最大，按序扫描时**必然在 1-29 全部判定完毕之后**才被检查，
天然就是「优化点命中完的最后一步」，无需额外的调用模式。

- **命中 30（Autotune）**：加载 `references/autotune.md`，对可调 `tl.constexpr` 参数
  （含单维 BLOCK，任意命名）配置 `@triton.autotune`；若 BLOCK 由 host 侧按 shape 分档、
  autotune 无法接管，则判定**不命中**并在返回信息注明理由，让扫描继续到 31。
- **命中 31（Block Size Scaling）**：加载 `references/block_size_scaling.md`，
  识别单维 BLOCK 当前值 `B0`（host 侧分档时整个阶梯视为一组，**必须整档同步**），
  **产出候选阶梯计划**：
  ```
  向上：ladder ×2, ×4 … 直至任一档 > 65536
  向下：ladder ÷2（至少一档，覆盖 B0 本身偏大、最优值在下方的情形）
  ```
  并应用计划中的**第一个候选**写入 `output_path`；
  其余候选由调用方（AGENTS.md 4.5.T）逐个执行 verify + benchmark，取实测 latency 最低者。

### 主流程示意图

```
调用方（AGENTS.md Phase 4）持有循环：

  ┌── scan_from=cursor, exclude_points=... ──────────────┐
  │  skill: 加载 Index.md → 从 cursor 按序找首个命中 P    │
  │         → 加载 P 对应参考文档 → 应用 → checklist      │
  │         → 写 output_path → 返回 P（或 None）          │
  └──────────────┬───────────────────────────────────────┘
                 │ 调用方推导 [cursor, P-1] 均未命中
                 ▼
      调用方 verify + benchmark → 采纳/回退 → 更新 cursor/版本
                 │
                 │ 编号 1-29 判定完毕后，扫描自然到达：
                 ▼
        30 Autotune ──不适用/失败──▶ 31 Block Size Scaling
                 │                        │ 产出候选阶梯计划
                 │                        ▼
                 │            调用方逐候选 verify+benchmark，取最优
                 ▼
        cursor=1 走完一趟仍 None → scan_complete → 退出
```

### 关键约束

- ⚠️ **只能使用本 skill 规定的优化方式，禁止使用任何超出本 skill 之外的优化方式**。
- ⚠️ **必须先命中优化点的「命中条件」，才能加载参考文档；未命中则跳过**。**例外：主流程步骤 6 / 7 / 7F 为终止步骤，无编号、无命中条件，按步骤定义无条件执行。**
- ⚠️ **一次调用只应用一个优化点**，应用完立即返回；多轮由调用方驱动。
- ⚠️ **优化点 29（IR 分析）支持多轮重复命中**：在 Phase 4 调用方（triton-op-generator AGENTS.md）开启的 "IR 多轮迭代模式" 下，IR 优化点可在多个 Phase 4 轮次中重复进入，每轮重新提取 `last_pass.mlir` 并分析；其他优化点单轮即过，命中后本轮不再重复。每次调用 latency-optimizer 仍只应用一个优化点。
- ⚠️ **一次只能参考一个文档**；参考文档仅用于当前命中优化点的子流程，完成后立即返回主流程。**例外：步骤 7 失败转 7F 时，允许在同一轮内先后加载 `autotune.md` 与 `block_size_scaling.md`。**
- ⚠️ **扫描完整性由调用方判定**：调用方按 `[scan_from, hit-1]` 推导未命中区间，
  本 skill 不需要、也不应自行声称"已扫完"。
- ⚠️ **命中即必须真实加载文档**：命中任一优化点后，必须用工具实际打开 `Index.md` 中该点
  对应的参考文档再动手改代码；**凭记忆或凭本文件的摘要改代码视为未执行该优化点**。
  编号 30 / 31 同此规则——`autotune.md` 与 `block_size_scaling.md` 必须被真实加载。
- ⚠️ **终止步骤禁止以耗时为由裁剪**：7F 的候选阶梯必须完整产出（含向上与向下方向）。"预计不会更好"、"时间不够"、"UB 估算会超"均不构成跳过理由——7F 的停止条件只有两个：verify 失败，或 BLOCK > 65536。

## 优化验证规则

**⚠️ 强制要求：在进行任何精度验证或性能验证之前，必须先执行 checklist 检查，确保所有代码规范都已满足。验证流程如下：**

1. **Checklist 检查**：加载 `references/checklist.md`，逐项检查代码是否满足所有规范要求
2. **不满足规范** → 修改代码直到满足所有规范要求，然后重新执行 checklist 检查确认
3. **满足规范后** → 执行精度验证和性能验证

- **成功**：优化后的性能不劣化（speedup ≥ 1.0），该优化结果作为下一次优化迭代的基线
- **失败**：优化后的性能劣化（speedup < 1.0），放弃本次优化结果，以优化前的代码作为下一次优化迭代的基线

## 输出约束

- **必须产出代码文件**：即使无优化点，也要写出 `output_path`
- **功能一致性**：优化前后的计算结果必须一致
- **精度一致性**：数值精度不能降低
- **可执行性**：输出代码必须能被 triton-op-verifier 直接验证和基准测试
