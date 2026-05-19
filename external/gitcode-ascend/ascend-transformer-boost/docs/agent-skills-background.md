# Agent Skills 背景知识

## Skills 概念

**Skills（技能）**是为扩展 AI Agent 能力而设计的模块化功能单元。每个 Skill 封装了指令、元数据及可选资源（脚本、模板），当 Agent 通过意图识别匹配到相关上下文时自动加载。

在 ATB（Ascend Transformer Boost）开发中，Skills 将算子迁移专家经验沉淀为可复用的标准化流程：

- ATB 算子 OPS→ACLNN 迁移全流程（设计→用例→实现→编译→测试）
- CANN/NNAL 环境安装部署
- CSV 泛化测试用例生成与执行
- 编译/测试问题调试

## 新增/修改 Skill 规范

### 目录结构

```
skills/ascend-transformer-boost/
├── SKILL.md                    ← 索引文件（必须更新）
├── docs/
│   ├── atb-skills-guide.md
│   └── agent-skills-background.md
├── rules/
│   └── project-description.md
└── skills/
    └── atb-{name}/             ← 新 Skill（atb- 前缀，kebab-case）
        ├── SKILL.md            ← 技能定义（必须）
        ├── references/         ← 参考文档
        └── templates/          ← 模板文件
```

### 命名规范

- **Skill 目录**：`atb-` 前缀 + kebab-case（如 `atb-csv-tester`）
- **SKILL.md**：必须全大写 `SKILL.md`，不允许变体
- **禁止文件**：目录内不允许 `README.md`，文档内化于 SKILL.md 或存放于 `references/`

### SKILL.md 内容规范

#### Markdown 标题

- **严禁编号**：`##`/`###`/`####` 后不得带数字编号
  - ❌ `## 1. 需求详情` / `### 1.1 项目背景`
  - ✅ `## 需求详情` / `### 项目背景`
- 层次结构通过标题级别表达

#### Frontmatter 模板

```yaml
---
name: atb-{name}
description: >
  Skill 简短描述。TRIGGER when: 触发条件。
keywords:
    - keyword1
    - keyword2
metadata:
  author: ascend-transformer-boost-team + {agent} + {model}
  version: "1.0.0"
  created: "2026-xx-xx"
  updated: "2026-xx-xx"
  skill-type: design | migration | test | env-setup | build | debug | workflow
---
```

#### metadata 规范

| 字段 | 说明 | 示例 |
|------|------|------|
| `author` | `ascend-transformer-boost-team + {agent} + {model}` | `ascend-transformer-boost-team + Claude Code + opus4.7` |
| `version` | 语义化版本 | `"1.0.0"` |
| `skill-type` | `design` `migration` `test` `env-setup` `build` `debug` `workflow` | `test` |

### ⭐ Human in the Loop（HIL）Gate 规范

含有用户确认检查点的 Skill，必须在 metadata 中声明 `gates`：

```yaml
metadata:
  gates:
    - id: gate-design
      description: "设计文档确认"
      trigger: "完成设计文档生成后，等待用户确认再进入下一阶段"
    - id: gate-csv
      description: "CSV 用例确认"
      trigger: "完成 CSV 用例设计后，等待用户确认再进入实际迁移"
```

**Agent 行为规范**：
- 遇到 `[HIL]` / `[用户确认]` / `[Gate]` 标记的步骤时，**必须暂停等用户确认**
- 不得擅自跳过 Gate 继续执行
- 用户未响应时，记录 pending 状态，不得自行决策

### Hook 实现规范

每个 SKILL.md frontmatter 中声明 `hooks:` 块，参考 planning-with-files 格式：

#### Hook 类型

| Hook | 触发时机 | ATB 用途 |
|------|---------|---------|
| `PreToolUse` | 每次工具调用前 | 校验路径参数（CANN_PATH/ATB_REPO_PATH/INSTALL_PATH），提示缺失 |
| `PostToolUse` | Write/Edit 操作后 | Gate 确认提醒（设计文档/CSV 用例）、编译失败次数提醒、950 测试通知 |
| `Stop` | 任务结束时 | 检查 Gate 是否全部通过，未通过则阻止停止 |

#### 实际示例：atb-aclnn-operator-replacement-designer

```yaml
# SKILL.md frontmatter
metadata:
  gates:
    - id: gate-design
      description: "设计文档确认"
      trigger: "设计文档生成完成后，必须等待用户确认"
hooks:
  PreToolUse:
    - matcher: "Write|Edit|Bash"
      hooks:
        - type: command
          command: "([ -z \"$CANN_PATH\" ] || [ ! -f \"$CANN_PATH/set_env.sh\" ]) && echo '[PATH CHECK] CANN_PATH 未设置或无效' >&2; ([ -z \"$ATB_REPO_PATH\" ] || [ ! -d \"$ATB_REPO_PATH\" ]) && echo '[PATH CHECK] ATB_REPO_PATH 未设置或无效' >&2 || true"
  PostToolUse:
    - matcher: "Write"
      hooks:
        - type: command
          command: "echo '[HIL GATE 1] 设计文档已生成。请向用户展示确认检查表并等待\"确认通过\"。'"
  Stop:
    - hooks:
        - type: command
          command: "echo '[CHECK] 确认设计文档已获用户确认（Gate 1）。'"
```

#### 实际示例：atb-ops-to-aclnn-migration-workflow

```yaml
metadata:
  gates:
    - id: gate-1
      description: "设计文档确认"
      trigger: "Phase 1 完成后"
    - id: gate-2
      description: "CSV 用例确认"
      trigger: "Phase 2 完成后"
    - id: gate-6
      description: "最终交付确认"
      trigger: "Phase 6 完成后"
hooks:
  PreToolUse:
    - matcher: "Write|Edit|Bash"
      hooks:
        - type: command
          command: "grep -q 'gate-' ${CLAUDE_PROJECT_DIR}/.claude/plans/*.md 2>/dev/null && echo '[planning-with-files] ACTIVE PLAN with Gates' || true"
  PostToolUse:
    - matcher: "Write|Edit"
      hooks:
        - type: command
          command: "echo '[HIL GATE REMINDER] 若刚完成设计文档/CSV用例/交付报告，请确认是否需等待用户确认（Gate 1/2/6）。'"
  Stop:
    - hooks:
        - type: command
          command: "echo '[CHECK] 确认所有 Phase 已完成、Gate 已通过。'"
```

#### 各 Skill Hook 分配速查

| Skill | PreToolUse | PostToolUse | Stop | gates |
|-------|:---:|:---:|:---:|:---:|
| migration-workflow | ✅ | ✅ | ✅ | gate-1/2/6 |
| replacement-designer | ✅ | ✅ | ✅ | gate-design |
| csv-testcase-generator | ✅ | ✅ | ✅ | gate-csv |
| operator-migration | ✅ | - | - | - |
| csv-tester | ✅ | ✅ | - | - |
| testframework-build | ✅ | ✅ | - | - |
| nnal-installer | ✅ | - | - | - |
| debug-guide | - | - | - | - |

#### 设计原则

1. **PreToolUse 永不阻塞**：所有 hook 命令末尾加 `|| true`（shell）或仅输出 warning
2. **PostToolUse 仅提醒**：不强制中断，通过 echo 输出提示 agent 注意
3. **Stop 仅检查**：输出检查清单，不阻止 CLI 退出
4. **gates 声明在 metadata**：gate 元数据供 PreToolUse hook 扫描识别

### 更新 Index

新增/修改 Skill 后，必须更新 `skills/ascend-transformer-boost/SKILL.md`：

1. **技能索引表**：新增/修改一行
2. **依赖关系图**：更新 ASCII 依赖树
3. **依赖矩阵**：更新前置依赖
4. **场景表**：如有新场景，追加一行
5. **metadata**：更新 version 和 updated

### 更新 atb-skills-guide.md

Index 变更后，自行判断是否需要同步更新 `docs/atb-skills-guide.md`。

## 术语表

| 术语 | 全称 | 说明 |
|------|------|------|
| ATB | Ascend Transformer Boost | 昇腾 Transformer 加速库 |
| ACLNN | Ascend Computing Language Neural Network | 昇腾算子 API 库 |
| CANN | Compute Architecture for Neural Networks | 昇腾计算架构 |
| HIL | Human in the Loop | 人在回路（关键决策点需用户确认） |
| Gate | - | 工作流检查点，需用户确认 |
| Skill | - | 技能模块（ATB 开发专用） |

---

*最后更新: 2026-04-28*
