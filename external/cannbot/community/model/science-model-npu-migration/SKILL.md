---
name: external-cannbot-community-model-science-model-npu-migration
description: 面向华为 Ascend 的 NPU 代码级迁移（环境门禁、脚本适配、精度/性能对比）。Use when the user asks for
  NPU/Ascend migration, torch_npu, MindSpore Ascend, or NPU adaptation checks.
argument-hint: '[framework] [soc-or-chip] [FP16]'
original-name: science-model-npu-migration
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-11-07'
synced-commit: 2374941495c1549eb4a1a5a82987b567d61855f5
license: UNKNOWN
---

# NPU 迁移到昇腾（Ascend）

技能标识：**`science-model-npu-migration`**。本技能正文在 **`references/`** 分册中。执行时按 [技能分册](#技能分册) 顺序阅读；流程总览见 [workflow.md](references/workflow.md)；说明见 [overview.md](references/overview.md)。

## 何时使用（触发词）

在以下表述时应选用本 skill：`NPU迁移`、`昇腾` / `Ascend`、`torch_npu`、`MindSpore Ascend`；skill 名称 **`science-model-npu-migration`** 及别名 `sci-model-npu-migration`、`sci_model-npu-migration`；「检测 NPU 适配情况」、代码适配、精度回归、延迟/吞吐评估等。

## 适用边界

| 适用 | 不适用（转专项 skill） |
|------|------------------------|
| 训练/推理**代码**从 CUDA、其他 GPU 栈或异构环境**迁到昇腾** | 模型**已在 NPU 推理框架跑通**且有基线，仅需 **Decode 吞吐 / 延迟** 优化 |
| 环境门禁、脚本与依赖适配、smoke、精度/性能**对比归档** | 仅做**推理框架适配**（无完整代码级迁移链路）→ `model-infer-migrator` |
| 「检查 NPU 适配情况」快速路径（part-03 门禁） | **算子融合 / 图优化** → `model-infer-fusion`；**KV Cache** → `model-infer-kvcache`；**图模式** → `model-infer-graph-mode` |

**场景示例**：用户说「CUDA 上 PyTorch 训练仓库要迁到 910 并做精度性能对比」→ **本 skill**；用户说「NPU 上已跑通，想优化 Decode 吞吐」→ **说明本 skill 不适用**，引导上述推理优化专项 skill。

## 场景示例

```text
# 完整迁移
/science-model-npu-migration PyTorch 910 FP16

# 快速适配检查（仅 part-03 门禁）
检查当前工程对昇腾 NPU 的适配情况，不做完整迁移
```

## 资源

- 端到端流程与闭环：[workflow.md](references/workflow.md)（含交付物映射、术语表、Mermaid、闭环检查清单）
- 交付模板：[deliverables-index.md](references/deliverables-index.md)
- 环境与交付索引：[docs-index.md](references/docs-index.md)（含环境目标与快照模板入口）
- **实战参考**（按需）：[reference-code-patterns.md](references/reference-code-patterns.md)、[part-07](references/part-07-commands.md)、[part-09](references/part-09-examples-troubleshooting.md)

## 执行约定

0. **首条回复（缺参必反问）**：用户只说「迁到 NPU」「帮我迁移」等且**未给出框架/芯片/精度**时，**第一条可见回复必须是面向用户的文字**——列出待确认的三项（见 [part-01](references/part-01-scope-and-baseline.md)「快速开始」）或逐项反问；**禁止**在未输出任何文字回复前仅调用 Read/工具读 `references/` 或静默探查工程。
1. **完整迁移顺序**：part-01 → 02 → 03 → 04 → 05 → 08；**不得跳过** [part-03](references/part-03-environment.md) **§4.0** 再建议 NPU 训练/推理。
2. **参数调用**（如 `/science-model-npu-migration ...`）：`$ARGUMENTS[0]`=框架/工程栈、`$ARGUMENTS[1]`=芯片/型号、`$ARGUMENTS[2]`=精度（默认 FP16）；缺参先问（见上文第 0 条与 part-01）。
3. **评测与归档**：先 part-05 落数，再 part-08 同步文档并定稿 **`mig_docs/Summary.md`（最终交付）**；过程记录在 `working/`。
4. **失败与回滚**：评测未通过或运行失败时走 [part-06](references/part-06-risk-rollback.md) + [part-09](references/part-09-examples-troubleshooting.md)，写入 `Mig_report` **§7**，按 [workflow](references/workflow.md) 回流 part-03 / 04 / 05。成功归档后 part-06 仅作可选风险复核。
5. **命令与排查**：[part-07](references/part-07-commands.md)、part-09 按需引用，不阻塞主线。
6. **快速检查**（仅「检查 NPU 适配情况」）：直接从 part-03 开始，不进 part-04～05 与完整归档。
7. **闭环与一致性**：每完成一个执行步，按 [workflow.md](references/workflow.md)「闭环检查」更新对应落盘；步 4 完成后核对 [part-04](references/part-04-code-migration.md) **§5.0 收口**（含 **§5.6 smoke**）；步 5 末与步 6 归档前执行 workflow「文档一致性校验矩阵」，确保 `environment.md`、`Mig_report`、`Mig_Readme`、`Compare`、`Summary` 关键字段无冲突。

## 技能分册

| 顺序 | 文件 | 内容 |
|:----:|------|------|
| 1 | [part-01-scope-and-baseline.md](references/part-01-scope-and-baseline.md) | §1 信息收集、§2 基线 |
| 2 | [part-02-feasibility.md](references/part-02-feasibility.md) | §3 可行性预判 → `Mig_report` §2.2 |
| 3 | [part-03-environment.md](references/part-03-environment.md) | §4 门禁与环境 → `environment.md` |
| 4 | [part-04-code-migration.md](references/part-04-code-migration.md) | §5 代码级迁移 |
| 5 | [part-05-performance-accuracy.md](references/part-05-performance-accuracy.md) | §7 性能、§8 精度 |
| 6 | [part-08-checklist-deliverables-output.md](references/part-08-checklist-deliverables-output.md) | 执行步 6：Checklist、交付、归档（无主流程 §） |
| — | [part-06-risk-rollback.md](references/part-06-risk-rollback.md) | 主流程 §9 回滚（失败路径必用） |
| — | [part-07-commands.md](references/part-07-commands.md) | 命令模板（按需，无主流程 §） |
| — | [part-09-examples-troubleshooting.md](references/part-09-examples-troubleshooting.md) | 示例与排查（按需，无主流程 §） |
| — | [reference-code-patterns.md](references/reference-code-patterns.md) | 代码模式与 CUDA→NPU 对照（按需） |

> **part 编号 ≠ 执行顺序**：执行步 6 为 **part-08**；part-06 为失败路径回滚。三套 § 对照见 [workflow.md](references/workflow.md)「三套编号对照」表。
