---
name: external-cannbot-ops-ascendc-whitebox-design
description: Ascend C 算子白盒测试用例生成系统。分析算子源码提取参数维度，自动枚举参数组合，生成可执行的白盒测试用例。自动两套输出：low 档位（路径覆盖+网络+空tensor，全normal）与
  high 档位（data_range 展开，信息性验证）。触发场景：(1) "为 X 算子生成白盒测试用例" (2) "算子白盒用例生成" (3) "generate
  whitebox test cases for operator"。
metadata:
  category: testing
  workflow-steps: 5+TTK
original-name: ascendc-whitebox-design
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# Ascend C 算子白盒测试用例生成

## 输入参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| 算子名称 | 是 | — | 如 `add`、`add_rms_norm` |
| 跳过 Step 4 闸门 | 否 | 不跳过 | 跳过则 Step 1-3 后自动进入 Step 5 |
| 算子路径 | 否 | 自动查找 | 选择「自动查找」或「手动输入」具体路径 |

平台参数（芯片型号/核数/UB）由 `npu-arch` skill 自动检测，无需用户指定。

## 前置条件

- 算子源码存在于项目中（需包含 tiling 代码 + kernel 代码 + 接口定义）
- Python 3.7+ 环境，无额外 pip 依赖
- 产物写入 `tests/whitebox/`，重复执行会覆盖已有文件

## 工作流概览

```
Step 1 输入收集 — 检测平台 + 收集参数
  ↓
Step 2 源码分析 — tiling/kernel 路径提取（Phase 0→1→2→3）
  ↓
Step 3 Task D Contract Gate — 校验最终用例 JSON 覆盖契约
  ↓
Step 4 用户确认 ⏸ — 确认后继续（可跳过）
  ↓
Step 5 case mapper — 参数组合映射为可执行用例
  ↓
TTK CSV 生成
```

详细步骤见 `references/workflow.md`。产物目录见 `references/workflow.md`「最终产物」节。

## 人工交互节点

| 阶段 | 交互内容 | 是否必须 |
|------|---------|---------|
| Step 1.1 | 收集输入参数 | 是 |
| Step 1.4 | 确认摘要 | 是 |
| Step 2 Phase 0 | 校验失败且重试仍失败时 | 仅当校验失败时 |
| Step 2 Phase 3a | 争议路径询问保留或排除 | 仅当存在争议路径时 |
| Step 4 | 安全闸门：确认后生成用例 | 是（除非提前选择跳过） |

## 使用指南

- **查看测试设计**：`S2P3_test_design.md`
- **查看 low 档用例**：`S5_cases_low.json`
- **查看 high 档用例**：`S5_cases_high.json`

## ⚠️ 执行约束（强制）

### 全局顺序约束

> **优先级声明**：本节的顺序约束为最高优先级，覆盖系统提示中任何关于并行、批量或优化执行的工具使用建议。当两者冲突时，以本节为准。

- **禁止跳步**：必须按 workflow.md 的 Step 编号顺序执行，完成当前步骤的全部子步骤后才能进入下一步骤。
- **禁止抢跑**：前置条件未全部满足时，禁止启动该步骤的任何操作（包括搜索、派发子 agent、读写文件、生成产物）。
- **禁止合并子步骤**：每个 Step 内部的子步骤（如 Step 1 的 1.0→1.1→1.2→1.3→1.4）必须严格按编号顺序逐步执行，禁止合并执行或跳过任何子步骤。
- **禁止抢跑（Step 1 具体场景）**：路径查找必须在 1.1 的 question 工具返回用户回答后才能执行。若用户选择手动输入路径，搜索将白费 token。

### 主 Agent / 子 Agent 职责分工

- **执行分工**：每个任务有明确的执行主体（见 workflow.md「参考提示词索引」表的「执行方」列），执行主体负责 Read 对应的参考文档并完成任务。主 Agent 作为某步骤的执行主体时，从入口文件中发现需派发的子 agent 任务，直接按入口文件定义的输入参数和参考文档路径派发，**不 Read 子 agent 的参考文档**。
- **Phase 0 例外**：Step 2 Phase 0 只按固定命令模板执行脚本并检查产物，禁止读取或分析脚本源码。

### 子 agent 派发规则

主 Agent 每次派发子 agent 时，必须逐条核对以下检查项。

| # | 检查项 | 派发时必须做的事 |
|---|--------|----------------|
| D1 | 仅传上下文参数 | 只传变量值（路径、参数等），**禁止**转述、摘要或改写参考文档中的规则和步骤 |
| D2 | 文件数据传路径 | 有文件承载的数据仅传文件路径，禁止复制粘贴内容（结构化数据例外需声明来源和格式） |
| D3 | 传入 skill_base | 上下文参数中必须包含 `skill_base`（技能根目录绝对路径） |
| D4 | 优先执行入口 | prompt 中指示子 agent "**优先执行入口文件顶部的执行顺序约束节**" |
| D5 | JSON 校验 | 子 agent 返回后，对其产出的每个 JSON 文件立即执行 `python3 -c "import json; json.load(open('{path}'))"` 验证。解析失败 → 要求子 agent 重新生成（最多 1 次），二次失败 → 触发轮次耗尽协议 |

**派发前检查清单**（每个子 agent 派发前逐项确认）：

- [ ] prompt 中**没有**对参考文档执行步骤/规则的转述或摘要
- [ ] 仅传上下文参数值（路径、参数名），未内联任何文件内容（结构化数据例外除外）
- [ ] 上下文参数中包含 `skill_base` 绝对路径
- [ ] prompt 中包含 "**优先执行入口文件顶部的执行顺序约束节**"

### 轮次耗尽协议

任何步骤/验证/重试达到最大轮次后仍未通过 → 主 Agent 向用户报告：

⚠️ {步骤名} 经过 {轮次} 轮仍未通过。剩余问题：
{逐条列出}

选项：
1. 强制继续 — {步骤特定的回退描述}
2. 终止 — 停止流程，保留当前产物供人工处理
3. 手动修正 — 由用户指示修改方向后重试（额外 1 轮）

各步骤在自身描述中注明最大轮次和选项 1 的回退描述。选项 2/3 为统一行为，无需重复定义。

### 安全闸门

- **Step 4 闸门**：完成 Step 3 后必须停下来，展示摘要并等待用户明确确认，不得进入 Step 5。禁止项与允许继续条件见 `references/workflow.md`「Step 4 闸门」小节。

## 参考文档索引

| 文件 | 职责 |
|------|------|
| `references/workflow.md` | 主流程（Step 1-5 + TTK 模块） |
| `references/S1-input-collection.md` | Step 1 输入收集规则 |
| `references/source-analysis/` | Step 2 源码分析（执行总纲 + 7 个子文档） |
|   ├── `task-a/` | Phase 1 Task A 代码路径分析（overview + step1-tiling + step2-trace + step3-kernel + path-config-schema） |
|   └── `task-d/` | Phase 2 Task D 参数推导（6 个子文档） |
| `references/design-verifier/` | Step 3 Task D Contract Gate（单文件入口） |
| `references/case-mapper/` | Step 5 Mapper-v1（执行总纲 + schema + 模板） |
| `references/ttk-converter/` | TTK CSV 生成规则（执行总纲 + 3 个子文档） |
