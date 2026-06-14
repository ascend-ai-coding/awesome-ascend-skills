---
name: external-gitcode-ascend-atb-atk-testcase-generator
description: 'ATB ATK 测试用例生成主控技能。负责 6-Gate 流程编排和 HIL 门禁控制。 详细实现拆分到 checks/references/templates/scripts
  资源目录，避免超长单文件。

  '
keywords:
- atb
- atk
- testcase
- gate
- workflow
- yaml
- golden
- 昇腾
metadata:
  author: ascend-transformer-boost-team + Cursor + DeepSeek-v4-pro
  version: 2.1.5
  created: '2026-05-08'
  updated: '2026-05-12'
  skill-type: testcase
  gates:
  - id: gate-1
    description: 路径与算子确认
    trigger: 确认 ATB_REPO_PATH、ATK_PATH、ATB_KNOWLEDGE_PATH 和算子注册状态
  - id: gate-2
    description: YAML 与 Generator 设计确认
    trigger: YAML、Generator、节点配置生成后需用户确认
  - id: gate-3
    description: Golden 实现确认
    trigger: execute_<op>.py 完成后需用户确认
  - id: gate-4
    description: 代表用例验证确认
    trigger: 代表用例精度通过后需用户确认
  - id: gate-5
    description: 全量测试完成
    trigger: 全量精度+性能+CSV 联动结果收集完成
  - id: gate-6
    description: 结果汇总交付
    trigger: 汇总表与产物清单输出完成
hooks:
  PreToolUse:
  - matcher: Write|Edit|Bash
    hooks:
    - type: command
      command: ([ -z "$ATB_REPO_PATH" ] || [ ! -d "$ATB_REPO_PATH" ]) && { echo '[PATH
        CHECK] ATB_REPO_PATH 未设置或无效，请先向用户获取 ATB 路径' >&2; exit 1; }; ([ -z "$ATB_KNOWLEDGE_PATH"
        ] || [ ! -d "$ATB_KNOWLEDGE_PATH" ]) && { echo '[PATH CHECK] ATB_KNOWLEDGE_PATH
        未设置或无效，请先向用户获取知识库路径' >&2; exit 1; }; python3 -c 'import atk' 2>/dev/null ||
        { echo '[PATH CHECK] ATK 不可用，请检查 ATK_PATH / PYTHONPATH 与虚拟环境' >&2; exit 1;
        }
  PostToolUse:
  - matcher: Write
    hooks:
    - type: command
      command: echo '[HIL GATE] 若当前 Gate 完成，请展示确认检查表并等待用户确认通过。'
  Stop:
  - hooks:
    - type: command
      command: echo '[CHECK] 仅当 Gate 1-6 按顺序完成并获得必要确认后方可结束。'
original-name: atb-atk-testcase-generator
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-06-14'
synced-commit: 62f55373f87e10e1d64f3f82f369be625c56b9fc
license: UNKNOWN
---

# ATB ATK Testcase Generator

## Quick Reference

**用途**：作为大型技能主控入口，统一编排 ATK 测试构建的 6 个 Gate 流程。

**渐进式披露阅读顺序**：
1. 先完成「执行前 Checklist」确认范围和输入
2. 再看「流程调度顺序」判断当前阶段
3. 仅在需要时读取对应资源文件（checks/references/templates/scripts）

## 执行前 Checklist

- [ ] 已确认 `ATB_REPO_PATH`、`ATB_KNOWLEDGE_PATH`、`ATK_PATH`
- [ ] 已确认目标算子与输入来源（设计文档 / 既有 YAML / 用户参数）
- [ ] 已确认需要执行的 Gate 范围（全流程或断点续跑）
- [ ] 已确认 Gate 2/3/4 为必须用户确认点

## 功能概述

该技能采用“主控 + 资源文件”结构：
- 主控 `SKILL.md` 负责流程编排、门禁和异常回流
- 详细步骤与模板放在资源目录，按需读取，避免单文件膨胀

## 流程调度顺序

```text
Gate 1 -> Gate 2 -> Gate 3 -> Gate 4 -> Gate 5 -> Gate 6
```

- 任一 Gate 失败：停止并修复，再回到当前 Gate 重试
- Gate 2 / 3 / 4：必须等待用户确认，不允许自动跳转

## 资源目录（按需读取）

### checks/

- `checks/gate-1-path-operator-check.md`
- `checks/gate-2-yaml-generator-design.md`
- `checks/gate-3-golden-implementation.md`
- `checks/gate-4-representative-validation.md`
- `checks/gate-5-full-regression.md`
- `checks/gate-6-report-delivery.md`

### references/

- `references/golden-complexity-guide.md`
- `references/pagedattention-best-practice.md`
- `references/pagedattention-case-study.md`（PagedAttention 精度与输入规范化案例复盘）
- `references/common-faq.md`（含 InputCaseConfig/ranges、contextLen、Golden 解耦、**Q9 perf / PerformanceConfig** 等）

### templates/

- `templates/atk-result-summary-template.md`
- `templates/run-op-sh-template.sh`

### scripts/

- `scripts/atk_gate_commands.sh`
- `scripts/validate_checks_layout.sh`

## 断点续跑规则

- 若已有 `*_gen.yaml` 与 `generator_*.py`，可从 Gate 3 开始
- 若已有 `execute_*.py` 且代表用例已通过，可从 Gate 5 开始
- 续跑前必须重新执行对应 Gate 的 Checklist

## 相关文档

- [ATB Skills 索引](../../SKILL.md)
- [ATB Skills 开发者指南](../../assets/atb-skills-guide.md)
- [Gate 2 设计前置](../atb-aclnn-operator-replacement-designer/SKILL.md)
- [调试问题定位](../atb-debug-guide/SKILL.md)
