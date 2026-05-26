---
name: external-gitcode-ascend-mindspeed-fsdp2-migration-main
description: 用于统筹 MindSpeed-MM FSDP2 端到端迁移。适用于需要协同模型、数据、配置与验证子流程迁移任意新模型时。
original-name: mindspeed-fsdp2-migration-main
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# MindSpeed FSDP2 迁移总控

## 目标

通过调度专用子技能并执行验收门禁，统筹从源仓到 MindSpeed-MM FSDP2 的完整迁移流程。

## 适用场景

- 用户要求将新的开源模型迁移到 MindSpeed-MM FSDP2。
- 用户需要覆盖模型、数据、配置、验证的端到端迁移编排。
- 迁移任务需要跨模块顺序推进并统一汇总报告。

## 不适用场景

- 任务仅涉及模型代码适配。
- 任务仅涉及数据集适配。
- 任务仅涉及 YAML/配置调整。
- 任务仅涉及实施后的验证。

## 输入

- `source_repo_path`
- `target_repo_path`
- `model_identity`（模型名称与模态）
- `source_entrypoints`（训练、数据、模型入口）
- `runtime_assets`（真实运行资产，如 `model_path`、`dataset_path`、可选 `image_root`）
- `constraints`（可编辑路径、禁止改核心、复用优先）
- `acceptance`（功能+可靠性门禁，且一次分布式 E2E 跑通）

## 输出

- `changeset_manifest`
- `compatibility_matrix`（模型/数据/配置映射）
- `verification_evidence`
- `risk_register`
- `next_actions`

## K0 知识储备门禁（执行前必须完成）

1. 必读文档：
   - `../docs/MIGRATION_GUIDE_MINDSPEED_MM_FSDP2.md`
   - `../docs/MIGRATION_GUIDE_MINDSPEED_MM_FSDP2_DEVELOPER.md`
2. 必扫目录：
   - `MindSpeed-MM/mindspeed_mm/fsdp/models/`
   - `MindSpeed-MM/mindspeed_mm/fsdp/data/datasets/`
   - `MindSpeed-MM/examples/fsdp2/`
3. 必交付理解产物：
   - `architecture_chain.md`（最小链路与关键接口映射）
   - `contract_matrix.yaml`（model/data/config 契约矩阵）
   - `similar_case_selection.md`（最相似案例、可复用点、风险点）
   - `preflight_risk_register.yaml`（高/中/低风险与缓解动作）
4. 不通过即停止：
   - 任一产物缺失或内容无法追溯到代码/文档证据，禁止进入实施阶段

## 强制输入校验（执行前必须完成）

1. 路径与资产存在性：
   - `source_repo_path`、`target_repo_path` 必须存在且可读
   - `runtime_assets.model_path`、`runtime_assets.dataset_path` 必须存在且可读
2. 入口有效性：
   - `source_entrypoints.train`、`source_entrypoints.model`、`source_entrypoints.dataset` 必须存在
3. 可编辑范围：
   - 改动路径必须落在 `constraints.editable_paths` 内
4. 缺失即失败：
   - 任一校验失败时，直接输出失败原因与修复建议，不得进入实现阶段

## 工作流

0. 完成 K0 知识储备门禁并交付理解产物。
1. 发现源仓与目标仓入口。
2. 构建迁移映射，划分“可复用”与“必须适配”项。
3. 调度：
   - `mindspeed-fsdp2-model-migration`
   - `mindspeed-fsdp2-data-migration`
   - `mindspeed-fsdp2-config-migration`
4. 合并产物并运行 `mindspeed-fsdp2-verification`。
5. 若验证失败，按归属路由回对应子技能修复。
6. 产出最终迁移汇总报告。

## 分阶段交付契约（防执行偏差）

每个阶段结束时，必须交付“可检验产物”，否则禁止进入下一阶段：

1. 发现阶段交付：
   - `source_target_inventory.yaml`
   - `entrypoint_checklist.md`
2. 映射阶段交付：
   - `compatibility_matrix.yaml`
   - `reuse_vs_adapt.md`
3. model/data/config 阶段交付：
   - 每个子技能各自 `acceptance_checklist.md`
   - `changeset_manifest.yaml`
4. 验证阶段交付：
   - `verification_report.md`
   - `evidence.json`
   - `failed_cases.md`（仅失败时）
5. 汇总阶段交付：
   - `migration_report.md`
   - `risk_register.yaml`
   - `next_actions.yaml`

## 失败归因与回流规则（必须执行）

- `from_pretrained` / `_from_config` / `.loss` 契约问题 -> model
- `dataset_type` / 字段缺失 / collate 问题 -> data
- `model_id` / `dataset_type` / `training.plugin` / strict-extra 分层 -> config
- 无法归因时：
  - 先输出最小复现与证据
  - 再按“最先失败门禁”归属处理，不得模糊归因

## 禁止行为

- 禁止跳过 K0 知识储备门禁直接实施
- 禁止在未完成前置校验时开始代码改动
- 禁止跳过任一阶段交付物
- 禁止无命令证据宣称“已通过”
- 禁止把失败原因写成“可能/大概”而不给出复现证据
- 禁止绕过 `constraints.editable_paths` 修改核心框架

## 强制规则

- 每个子技能产物必须遵循单一职责归属。
- 没有验证证据不得宣称完成。
- 强制 plugin/注册一致性：
  - `model.model_id`
  - `data.dataset_param.dataset_type`
  - `training.plugin`
- 迁移改动必须限制在允许的可编辑路径内。

## 退出标准

- 功能门禁与可靠性门禁均通过。
- 至少一次分布式端到端训练成功。
- 最终报告中的结论均有证据支撑。

## 渐进式资源

- `checks/orchestration-checklist.md`
- `checks/knowledge-gates.md`
- `checks/preflight-gates.md`
- `checks/phase-delivery-gates.md`
- `templates/migration-report-template.md`
- `templates/fsdp2-migration-runbook.md`
- `templates/fsdp2-migration-quickstart-zh.md`
- `templates/knowledge-artifacts-template.md`
- `templates/compatibility-matrix-template.md`
- `templates/failure-routing-template.md`

仅在当前步骤需要详细清单、Runbook 流程、Quickstart 指引或报告格式时再读取这些文件。

## 外部参考（迁移技术要点）

- `../docs/MIGRATION_GUIDE_MINDSPEED_MM_FSDP2.md`
- `../docs/MIGRATION_GUIDE_MINDSPEED_MM_FSDP2_DEVELOPER.md`

