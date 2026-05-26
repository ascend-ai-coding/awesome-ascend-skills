---
name: external-gitcode-ascend-atb-ops-to-aclnn-migration-workflow
description: 'ATB OPS→ACLNN 迁移标准化工作流主模板。整合前置学习、设计文档生成、CSV用例设计、 实际迁移、编译验证、测试验证全流程，提供明确的阶段
  Gates 和用户确认机制。

  '
keywords:
- atb
- aclnn
- workflow
- migration
- ops
- template
- 标准化流程
- 昇腾
metadata:
  author: ascend-transformer-boost-team + Claude Code + deepseek-v4-pro
  version: 1.6.0
  created: '2026-04-20'
  updated: '2026-04-29'
  skill-type: workflow
  allowed-tools: Bash(docker|bash|cd|ls|grep|find|echo|cat|git) Read(*) Write(*)
  gates:
  - id: gate-1
    description: 设计文档确认
    trigger: Phase 1 完成后，等待用户确认设计文档
  - id: gate-2
    description: CSV 用例确认
    trigger: Phase 2 完成后，等待用户确认 CSV 用例
  - id: gate-5b
    description: 全量回归门控
    trigger: Phase 5b 完成后，所有相关 CSV 必须全部通过，方可进入 Phase 6
  - id: gate-6
    description: 最终交付确认
    trigger: Phase 6 完成后，等待用户确认交付报告
hooks:
  PreToolUse:
  - matcher: Write|Edit|Bash
    hooks:
    - type: command
      command: grep -q 'gate-' ${CLAUDE_PROJECT_DIR}/.claude/plans/*.md 2>/dev/null
        && echo '[planning-with-files] ACTIVE PLAN with Gates detected - check plan
        before proceeding' || true
  PostToolUse:
  - matcher: Write|Edit
    hooks:
    - type: command
      command: echo '[HIL GATE REMINDER] 若刚完成设计文档/CSV用例/全量回归/交付报告，请确认是否需等待用户确认（Gate
        1/2/5b/6）。严禁跳过 Gate 自动进入下一阶段。'
  Stop:
  - hooks:
    - type: command
      command: echo '[CHECK] 确认所有 Phase (0-6) 已完成、Gate 1/2/5b/6 已通过后，方可结束任务。'
original-name: atb-ops-to-aclnn-migration-workflow
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# ATB OPS→ACLNN 迁移标准化工作流

## 功能概述

定义 ATB 算子从 OPS 接口迁移至 ACLNN 接口的**标准化 7 阶段工作流**，含 Human in the Loop 用户确认机制。

## 调用时机

- `{xxx}Quant`、`{xxx}Norm`、`Softmax` 等算子的 OPS→ACLNN 迁移
- 910B/950 设备启用 ACLNN 加速
- 保持对其他设备的兼容性

---

## 详细内容

<details>
<summary>🔍 查看详细内容</summary>

### 阶段 Gates 概览

```
[Phase 0] 前置知识学习
    │
    ▼
[Phase 1] 设计文档生成 ────→ [用户确认 Gate 1] ───┐
    │                                         │
    ▼                                         │
[Phase 2] CSV用例设计 ─────→ [用户确认 Gate 2] ───┤
    │                                         │
    ▼                                         ▼
[Phase 3] 实际迁移（TDD） ←───────────────────────┘
    │
    ▼
[Phase 4] 编译验证（≤3次尝试）
    │
    ▼
[Phase 5a] 小范围验证
    ├─ Step 1: 前台跑 _func.csv（正例+反例, ~30s）
    └─ Step 2: 后台 subagent 跑 _perf.csv（性能, ~10min, 不阻塞）
    │
    ▼
[Phase 5b] 全量回归（所有相关 CSV）──→ [Gate 5b: 全量通过] ──┐
    │ （汇合 subagent 性能结果）                               │
    ▼                                                        ▼
[Phase 6] 交付报告 ←──────────────────────────────────────────┘
```

### Phase 0-6 详细说明

详见下方各阶段详细文档。

</details>

---

## 禁止跳步规则 ⛔

**⚠️ 以下规则在每次 Phase 转换时必须强制执行，不可绕过：**

### Phase 依赖矩阵

| Phase | 前置 Gate | 未通过时的行为 |
|-------|----------|---------------|
| Phase 1 | Gate 0 (文档齐全) | **阻断** — 缺少接口文档时禁止进入设计阶段 |
| Phase 2 | Gate 1 (设计确认) | **阻断** — 用户未确认设计文档前禁止写 CSV |
| Phase 3 | Gate 2 (CSV 确认) | **阻断** — 用户未确认 CSV 用例前禁止写代码 |
| Phase 4 | Phase 3 完成 | 可执行（编译验证） |
| Phase 5a | Phase 4 通过 | **先跑 Phase 2 再跑 Phase 5a** — 用例未设计完禁止测试 |
| Phase 5b | Phase 5a 通过 | **全量回归** — 所有相关 CSV 必须全部运行并通过 |
| Phase 6 | Gate 5b 通过 | 可执行（交付报告） |

### 每阶段开始时必须执行的检查

```
1. 确认上一阶段 Gate 状态:
   - Gate X 是否已获用户确认?
   - 确认记录在哪里 (commit message / working_files / 对话记录)?

2. 如果 Gate 未通过:
   - 立即停止
   - 向用户报告: "Phase Y 需要 Gate X 确认，当前 Gate X 状态: 未确认"
   - 等待用户指示

3. 如果 Gate 已通过:
   - 明确告知用户: "Gate X 已通过，进入 Phase Y"
   - 继续执行
```

### 常见跳步错误与拦截话术

| 错误行为 | Agent 自查问题 | 拦截话术 |
|---------|--------------|---------|
| 编译后直接跑测试 | "CSV 用例是否已设计并确认?" | ⛔ Phase 5a 需要 Gate 2 确认。当前 Gate 2 状态: 未确认。请先完成 Phase 2 CSV 用例设计。 |
| 5a 通过后直接写交付报告 | "全量回归是否已执行?" | ⛔ Gate 5b 未通过: 全量回归测试未完成。请先执行 Phase 5b 并确认所有相关 CSV 通过。 |
| 实现后跳过设计文档 | "设计文档是否已生成?" | ⛔ Phase 3 需要 Gate 1 确认。当前 Gate 1 状态: 未确认。请先完成 Phase 1 设计文档。 |
| WebFetch 失败后从代码反推 | "接口文档是否完整?" | ⛔ Gate 0 未通过: ATB/ACLNN 接口文档获取失败，禁止从代码反推。请用户提供文档。 |
| CSV 设计完直接写代码 | "用户是否已确认 CSV?" | ⛔ Phase 3 需要 Gate 2 确认。请先等待用户确认 CSV 用例。 |

### Agent 启动时的 Phase 状态自检

每次 Agent 被调用执行迁移任务时，必须先执行：

```
1. git log --oneline -5    → 检查当前分支做了什么
2. ls working_files/        → 检查哪些阶段产物已存在
3. 判断当前处于哪个 Phase  → 报告用户当前状态
4. 确认下一 Phase 的前置 Gate 是否通过 → 按依赖矩阵校验
```

---

## Skills 调用时序

| 阶段 | 调用的 Skill |
|------|-------------|
| Phase 1 | `atb-aclnn-operator-replacement-designer` |
| Phase 2 | `atb-csv-testcase-generator` |
| Phase 3 | `atb-aclnn-operator-migration` |
| Phase 4 | `atb-testframework-build` |
| Phase 5a | `atb-csv-tester`（正例/反例默认1次，性能默认400次） |
| Phase 5b | `atb-csv-tester` (Gate 5 全量回归) |
| 调试 | `atb-debug-guide` |

---

## 快速参考

```bash
# Phase 0-1: 设计和用例生成
# → atb-aclnn-operator-replacement-designer
# → atb-csv-testcase-generator
# → [等待用户确认 Gate 1 & 2]

# Phase 3: 实际迁移
# → atb-aclnn-operator-migration

# Phase 4: 编译
# → atb-testframework-build
# → [失败] atb-debug-guide

# Phase 5a: 小范围验证
# → atb-csv-tester（不指定 -t，正例/反例默认1次，性能默认400次）
# → [失败] atb-debug-guide
#
# Phase 5b: 全量回归
# → atb-csv-tester (Gate 5)
# → [失败] atb-debug-guide
```

---

## 详细阶段文档

<details>
<summary>📋 Phase 0: 前置知识学习</summary>

### 目标
理解算子相关技术背景，为后续设计打下基础。

### 前置条件
无

### 输入
- 算子名称（如 `{xxx}Quant`）
- 任务背景文档（如有）

### 输出
- 前置知识总结文档（保存至 working_files）

### 学习内容
| 主题 | 内容 | 输出位置 |
|------|------|----------|
| Quantization 基础 | per-token/per-channel/per-tensor 区别 | `{WORKING_DIR}/quant_basics.md` |
| ACLNN 接口文档 | 官方 API 规格理解 | 记录关键参数 |
| ATB 现有实现 | 当前 OPS 实现分析 | 记录迁移要点 |

### 检查点
| 检查项 | 验证方式 |
|--------|----------|
| Quantization 知识已总结 | 文件存在于 working_files |
| ACLNN 接口关键参数已提取 | 文档中有参数列表 |

### Gate 0
- **类型**: 自动通过（无用户确认）
- **条件**: 学习文档已生成

</details>

<details>
<summary>📋 Phase 1: 设计文档生成</summary>

### 目标
生成完整的算子替换详细设计文档（7章结构）。

### 前置条件
- Phase 0 完成
- ATB/ACLNN 接口文档链接已提供

### Skills 调用
```
atb-aclnn-operator-replacement-designer
```

### 检查点
| 检查项 | 验证方式 |
|--------|----------|
| ATB 接口 URL 已提供 | 文档链接有效 |
| ACLNN 接口 URL 已提供 | 文档链接有效 |
| ATB 算子参数已提取 | 设计文档第 2 章完整 |
| ACLNN 接口参数已提取 | 设计文档第 2 章完整 |
| 参数映射表已创建 | 设计文档第 2.3 节存在 |
| 设计文档 7 个章节完整 | 1-7 章均存在 |

### Gate 1: 用户确认 ⭐

**⚠️ 强制检查点 - 不可跳过**

**未收到确认前，禁止进入 Phase 2**

</details>

<details>
<summary>📋 Phase 2: CSV 用例设计</summary>

### 目标
设计覆盖正例、反例、性能测试的 CSV 测试用例集，**产出两个独立文件**。

### 输出产物
| 文件 | 内容 | 用途 |
|------|------|------|
| `{op}_func.csv` | 正例 + 反例 | Phase 5a 前台快速验证 |
| `{op}_perf.csv` | 性能用例 | Phase 5a 后台 subagent 逐个运行 |
| `{op}_test.csv`（可选） | 全部合并 | Phase 5b 全量回归 |

### 前置条件
- Phase 1 已通过 Gate 1 用户确认

### Skills 调用
```
atb-csv-testcase-generator
```

### 检查点
| 检查项 | 数量 | 验证方式 |
|--------|------|----------|
| 正例覆盖典型场景 | 5-15 个 | 覆盖 float16/bf16 |
| 反例覆盖所有校验分支 | 10-20 个 | 覆盖所有 ERROR_INVALID_* |
| 性能测试覆盖大小 shape | 5-10 个 | 包含大/中/小 shape |
| **已拆分为 _func.csv 和 _perf.csv** | 2 个独立文件 | 各有完整 21 列 header |
| CSV 格式符合规范 | - | 列数=21，分隔符=\| |

### Gate 2: 用户确认 ⭐

**⚠️ 强制检查点 - 不可跳过**

**未收到确认前，禁止进入 Phase 3**

</details>

<details>
<summary>📋 Phase 3: 实际迁移（TDD）</summary>

### 目标
实现 ACLNN Runner 并修改 Operation 启用 910B/950 加速。

### 前置条件
- Phase 2 已通过 Gate 2 用户确认

### Skills 调用
```
atb-aclnn-operator-migration
```

### 实现要点
1. **继承 AclnnRunner**: 新 runner 必须继承基类
2. **实现四大方法**:
   - `BuildAclnnVariantPack`: 处理输入输出张量
   - `SetAclNNWorkspaceExecutor`: 配置工作空间
   - `LaunchAclnnKernel`: 执行计算
   - `LoadMethod`: 动态加载 ACLNN 函数
3. **设备检测**: 使用 `GetSingleton<Config>().Is910B()` 和 `Is950()`
4. **向后兼容**: 非 910B/950 设备保持 OPS 实现

</details>

<details>
<summary>📋 Phase 4: 编译验证</summary>

### 目标
验证代码编译通过，最多尝试 3 次。

### Skills 调用
```
atb-testframework-build
atb-debug-guide（编译失败时）
```

### 编译规则
| 次数 | 操作 |
|------|------|
| 第 1 次失败 | 分析错误，定位问题，修复后重新编译 |
| 第 2 次失败 | 深入分析，查看参考实现，修复后重新编译 |
| 第 3 次失败 | **停止**，记录错误和修改计划，**询问用户** |

</details>

<details>
<summary>📋 Phase 5a: 小范围验证（双文件，两步流程）</summary>

### 目标
分两步运行 Phase 2 产出的 CSV：前台快速验证功能，后台 subagent 运行性能测试。

### 前置条件
- Phase 4 编译通过
- Phase 2 已产出 `{op}_func.csv` 和 `{op}_perf.csv`

### Step 1: 前台运行功能 CSV

```bash
cd <ATB_REPO_PATH>/tests/framework/python/CsvOpsTestTool
python3 atb_csv_ops_test.py -i <CSV_DIR>/{op}_func.csv
```

- 不指定 `-t`，默认 1 次，秒级完成
- **必须等待全部通过**，有失败则停止排查

### Step 2: 后台 Subagent 运行性能 CSV

```python
Agent(
    description="运行 {op} 性能测试",
    prompt="""
逐个运行 {op}_perf.csv 中的性能测试用例，避免 OOM：

cd {ATB_REPO_PATH}/tests/framework/python/CsvOpsTestTool
source {CANN_PATH}/set_env.sh
source {ATB_REPO_PATH}/output/atb/set_env.sh

# 获取性能用例总数
lines=$(tail -n +2 {ATB_REPO_PATH}/tests/apitest/opstest/csv/{op}_perf.csv | wc -l)

# 逐个运行每个性能 case
for i in $(seq 1 $lines); do
    echo "=== 运行性能 case $i/$lines ==="
    python3 atb_csv_ops_test.py -i {ATB_REPO_PATH}/tests/apitest/opstest/csv/{op}_perf.csv -n $i:$i
done

完成后汇总：通过/失败/耗时。
""",
    subagent_type="general-purpose",
    run_in_background=True
)
```

**关键点**：
- 使用 `-n N:N` 逐个运行，每次只加载 1 个性能 case
- `run_in_background=True` 后台执行，不阻塞主流程
- subagent 完成后自动通知，结果汇入 Phase 6 交付报告

### Skills 调用
```
atb-csv-tester（双文件 + 前台/后台两种模式）
atb-debug-guide（测试失败时）
```

### 完成后操作
Step 1 全部通过后，启动 Step 2 后台 subagent，**前台立即继续 Phase 5b 全量回归**。

</details>

<details>
<summary>📋 Phase 5b: 全量回归（所有相关 CSV）</summary>

### 目标
运行所有与目标算子相关的 CSV 文件，确保不引入回归问题。

### 前置条件
- Phase 5a Step 1（前台功能 CSV）全部通过
- Phase 5a Step 2（后台性能 subagent）可仍在运行中

### Skills 调用
```
atb-csv-tester（Gate 5 全量回归门控模式）
```

### 测试范围
枚举并运行所有相关 CSV：

```bash
find <ATB_REPO_PATH>/tests/apitest/opstest/csv/ -name '{op_name}*.csv' -type f | sort
```

### Subagent 结果汇合

若后台性能 subagent 尚未完成，等待完成后汇总：

```markdown
### Phase 5a 后台性能测试结果
| Case | ExecuteTime(us) | precision | 状态 |
|------|----------------|-----------|------|
| case 1 | xxx | 100% | ✅ |
| ...
```

### 测试要求
| 要求 | 说明 |
|------|------|
| 所有 CSV 全部运行 | 不允许部分运行 |
| 所有用例全部通过 | 不允许失败或跳过 |
| 使用默认迭代数 | 不指定 `-t`（正例/反例 1 次，性能 400 次） |
| 汇合后台 subagent 结果 | 等待 subagent 完成并汇总 |

### Gate 5b: 全量回归确认

**强制检查点 — 不可跳过**

```markdown
## Gate 5b — 全量回归结果

| CSV 文件 | 总用例 | 通过 | 失败 | 状态 |
|---------|--------|------|------|------|
| {op}_test.csv | N | N | 0 | ✅ |
| {op}.csv | M | M | 0 | ✅ |

**全部通过，Gate 5b 满足，可进入 Phase 6 交付。**
```

阻断时参见 `atb-csv-tester` Gate 5 阻断话术。

</details>

<details>
<summary>📋 Phase 6: 交付报告</summary>

### 目标
自动聚合 Phase 1/2/5 的输出，生成标准化交付报告。

### 前置条件
- Phase 5a 测试通过
- Phase 5b 全量回归通过（Gate 5b 确认）

### 自动聚合流程

交付报告的数据应从前序阶段自动提取，勿手工填写：

| 报告章节 | 数据来源 | 提取方式 |
|---------|---------|---------|
| 设计文档摘要 | `{WORKING_DIR}/{op}_replacement_design.md` | 读取需求简述/参数映射/风险评估 |
| CSV 用例统计 | `{op}_test.csv` | `wc -l` + `awk` 按 ExpectedError 分组 |
| 代码变更 | `git diff` / `git log` | `git diff --stat` + `git log --oneline` |
| 编译验证 | Phase 4 编译日志 | 提取编译状态 |
| Phase 5a 结果 | 测试输出 | 提取通过/失败/跳过 |
| Phase 5b 结果 | 测试输出 | 汇总所有 CSV |
| 遗留事项 | Phase 5b 未通过项 | 如有失败记录 |

### 自动聚合命令片段

```bash
# 统计 CSV 用例数
tail -n +2 {CSV_FILE} | wc -l

# 统计正例/反例/性能
tail -n +2 {CSV_FILE} | grep -c 'NO_ERROR$'
tail -n +2 {CSV_FILE} | grep -c 'Performance'
tail -n +2 {CSV_FILE} | awk -F'|' '{print $21}' | sort | uniq -c | sort -rn

# 代码变更
git diff --stat HEAD~1..HEAD
git log --oneline -5
```

### 报告模板

```markdown
# {Op} OPS→ACLNN 迁移交付报告

> **自动生成时间**: {timestamp}
> **生成方式**: 自动聚合 Phase 1/2/5 输出

## 任务概述
- **算子名称**: {Op}
- **迁移目标**: OPS → ACLNN
- **目标设备**: 910B/950
- **执行日期**: {date}

## 阶段完成情况
| 阶段 | 状态 | 关键输出 |
|------|------|----------|
| Phase 0 前置学习 | ✅ | working_files |
| Phase 1 设计文档 | ✅ | {op}_replacement_design.md |
| Phase 2 CSV用例 | ✅ | {op}_test.csv（{total} 个用例） |
| Phase 3 实际迁移 | ✅ | 代码变更（{files}个文件） |
| Phase 4 编译验证 | ✅ | 第 {N} 次编译通过 |
| Phase 5a 小范围验证 | ✅ | {designed_pass}/{designed_total} 通过 |
| Phase 5b 全量回归 | ✅ | {regression_total} 通过（{csv_count} 个 CSV） |
| Gate 1/2 确认 | ✅ | 用户已确认 |
| Gate 5b 全量回归 | ✅ | 所有 CSV 通过 |

## 测试结果汇总
### Phase 5a: 小范围验证
- 执行 CSV: {op}_test.csv
- 通过: N/N

### Phase 5b: 全量回归
| CSV | 总用例 | 通过 | 失败 |
|-----|--------|------|------|
| {op}_test.csv | N | N | 0 |
| {op}.csv | M | M | 0 |

## 代码变更
```
{git diff --stat ...}
```

## 遗留事项
[如有未通过用例或已知问题]

## 附件清单
- 设计文档: {path}
- CSV用例: {path}
- 代码变更: git commit {hash}
- 交付报告: {path}
```

### Gate 6: 最终交付 ⭐

**⚠️ 强制检查点**

</details>

---

## 常见问题

<details>
<summary>❓ Human in the Loop 跳过怎么办？</summary>

**禁止跳过 Gate 1 和 Gate 2**。如用户未及时响应：
1. 提醒用户确认 pending
2. 24小时后仍未响应，记录风险并暂停任务
3. 不得擅自进入下一阶段

</details>

<details>
<summary>❓ 编译 3 次失败怎么办？</summary>

1. 记录完整错误日志到 working_files
2. 使用 atb-debug-guide 分析
3. 向用户提问（苏格拉底式）：
   - 现象描述
   - 已尝试的修复
   - 可能的根本原因假设
   - 需要的支持

</details>

---

## 相关文档

- [任务提交模板](TASK_TEMPLATE.md)
- [ATB Skills 索引](../SKILL.md)
