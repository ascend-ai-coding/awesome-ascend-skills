---
name: external-gitcode-ascend-ascend-transformer-boost
description: '昇腾 Transformer 加速库（ATB）核心技能集索引（Index Skill）。 整合 8 大核心技能：CANN 安装部署、ATB
  测试框架编译、 ATB→ACLNN 算子替换设计文档生成、ATB→ACLNN 算子迁移，覆盖昇腾 NPU 开发全链路。

  '
keywords:
- atb
- ascend
- transformer
- aclnn
- cann
- 昇腾
- npu
- 950
- index
metadata:
  author: ascend-transformer-boost-team + Claude + Claude Code + deepseek-v4-pro
  version: 1.5.0
  created: '2026-04-17'
  updated: '2026-04-29'
  skill-type: index
  allowed-tools: Bash(*) Read(*) Edit(*)
original-name: ascend-transformer-boost
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-19'
synced-commit: 61a50580836017810c0ff005bc53c940ca059f06
license: UNKNOWN
---

# ascend-transformer-boost 技能集

## Quick Reference

**用途**: ATB（Ascend Transformer Boost）开发全流程技能索引，从环境搭建到算子迁移。

**核心命令**:
```bash
# 迁移标准化工作流
atb-ops-to-aclnn-migration-workflow
```

## 调用时机

当用户提到以下关键词时触发此索引：

- `ATB` / `ascend-transformer-boost` / `ascend transformer boost`
- `昇腾` + `开发环境` / `NPU`
- `ATB` + `ACLNN` / `算子迁移`
- `CANN` + `安装` / `910B` / `950`
- `ATB` + `CSV` / `测试` / `调试`

---

## 详细内容

<details>
<summary>🔍 查看详细内容</summary>

### 技能索引表

| 技能目录 | 名称 | 类型 | 核心功能 |
|---------|------|------|---------|
| [atb-ops-to-aclnn-migration-workflow/](skills/atb-ops-to-aclnn-migration-workflow/) | OPS→ACLNN 迁移标准化工作流 | workflow-template | **7 阶段标准化工作流模板，含 HIL 用户确认机制** |
| [atb-nnal-installer/](skills/atb-nnal-installer/) | CANN NNAL 安装 | env-setup | NNAL 安装（Toolkit+Kernels 由 cann-operator-env-config 提供） |
| [atb-testframework-build/](skills/atb-testframework-build/) | ATB 测试框架编译 | build | 全量编译 / 增量编译，GitHub→gitcode 镜像源替换 |
| [atb-aclnn-operator-replacement-designer/](skills/atb-aclnn-operator-replacement-designer/) | ATB→ACLNN 算子替换设计文档生成 | design | 输入 ATB/ACLNN 接口链接，生成 7 章结构化设计文档，**含 HIL 用户确认** |
| [atb-aclnn-operator-migration/](skills/atb-aclnn-operator-migration/) | ATB→ACLNN 算子迁移工具 | migration | 执行算子迁移，910B/950 设备启用 ACLNN 加速 |
| [atb-csv-tester/](skills/atb-csv-tester/) | ATB CSV 测试执行 | test | 运行 CSV 格式 ATB 测试用例 |
| [atb-csv-testcase-generator/](skills/atb-csv-testcase-generator/) | ATB CSV 测试用例生成 | testcase | 正例/反例/性能测试设计，覆盖率分析，**含 HIL 用户确认** |
| [atb-debug-guide/](skills/atb-debug-guide/) | ATB 调试指南 | debug | 环境问题排查、ABI版本、内存错误、CSV测试失败 |

### 技能依赖关系

```
ascend-transformer-boost (Index Skill)
│
├── atb-ops-to-aclnn-migration-workflow         ← 标准化工作流模板
│       │
├── atb-nnal-installer                 ← 依赖 cann-operator-env-config（提供 Toolkit+Kernels）
│       │
├── atb-testframework-build             ← 依赖 atb-nnal-installer
│       │
├── atb-aclnn-operator-replacement-designer  ← 独立，迁移前必需
│       │ (含 HIL 用户确认 Gate 1)
│       │
├── atb-csv-testcase-generator          ← 依赖设计文档
│       │ (含 HIL 用户确认 Gate 2)
│       │
├── atb-aclnn-operator-migration        ← 依赖 atb-nnal-installer + 设计文档
│       │ (依赖 Gate 1 & Gate 2 通过)
│       │
├── atb-csv-tester                      ← 依赖 atb-nnal-installer + 测试框架
│       │
└── atb-debug-guide                    ← 辅助技能，独立使用
```

### 标准化工作流调用时序

```
[Phase 0] 前置学习
    │
    ▼
[Phase 1] 设计文档生成
    └─ atb-aclnn-operator-replacement-designer
        │
        └── [Gate 1: 用户确认] ⭐
                │
                ▼
[Phase 2] CSV用例设计
    └─ atb-csv-testcase-generator
        │
        └── [Gate 2: 用户确认] ⭐
                │
                ▼
[Phase 3] 实际迁移
    └─ atb-aclnn-operator-migration
        │
        ▼
[Phase 4] 编译验证
    └─ atb-testframework-build
        │ (失败) → atb-debug-guide
        ▼
[Phase 5a] 小范围验证（双文件）
    ├─ Step 1: atb-csv-tester（前台 _func.csv，正例+反例）
    └─ Step 2: subagent（后台 _perf.csv，逐个 -n N:N 避免 OOM）
        │ (失败) → atb-debug-guide
        ▼
[Phase 5b] 全量回归（汇合 subagent 结果）
    └─ atb-csv-tester (Gate 5 全量通过)
        │ (失败) → atb-debug-guide
        ▼
[Phase 6] 交付报告
```

### 依赖矩阵

| 调用技能 | 前置技能 | 前置条件 |
|---------|---------|---------|
| `atb-ops-to-aclnn-migration-workflow` | 无 | 作为整体流程参考 |
| `atb-nnal-installer` | cann-operator-env-config | NPU 驱动已安装，Toolkit+Kernels 已由 cann-operator-env-config 安装 |
| `atb-testframework-build` | atb-nnal-installer | CANN 环境、ATB 源码、Docker 容器 |
| `atb-aclnn-operator-replacement-designer` | 无 | ATB/ACLNN 接口文档链接 |
| `atb-csv-testcase-generator` | 无 | ATB 源码、接口规格文档、**设计文档已确认** |
| `atb-aclnn-operator-migration` | atb-nnal-installer + replacement-designer | CANN 环境、ATB 源码、设计文档、**CSV 用例已确认** |
| `atb-csv-tester` | atb-nnal-installer + atb-testframework-build | CANN 环境、ATB 测试框架 |
| `atb-debug-guide` | 无 | CANN 环境、ATB 测试框架 |

### 编译验证约束

当修改 ATB 算子源码（`*_aclnn_runner.cpp`、`*_operation.cpp` 等）后需要重新编译验证。

**编译命令**:
```bash
# 在 Docker 容器内执行
source $ASCEND_TOOLKIT_HOME/set_env.sh
cd $ATB_REPO_PATH
bash scripts/build.sh testframework 2>&1 | tail -50
```

> **注意**：确保环境变量已设置：
> - `ASCEND_TOOLKIT_HOME`（CANN 安装路径）
> - `ATB_REPO_PATH`（ATB 仓库路径）

**约束规则**:
- **最多验证 3 次**：编译失败时记录原因，规划修改方案
- **Step by Step**：每次只修复一个问题
- **无法解决时**：记录问题并询问用户

**编译失败处理**:

| 失败次数 | 操作 |
|---------|------|
| 第1次失败 | 分析错误，定位问题，修复后重新编译 |
| 第2次失败 | 深入分析，查看参考实现 |
| 第3次失败 | 停止，记录错误和修改计划，询问用户 |

</details>

---

## 调用场景索引

| 场景 | 推荐调用技能 | 备注 |
|------|------------|------|
| 安装 NNAL（ATB 加速库） | `atb-nnal-installer` | 依赖 cann-operator-env-config 提供 Toolkit+Kernels |
| 编译 ATB 测试框架 | `atb-testframework-build` | 需要 Docker + CANN 环境 |
| 为算子替换撰写设计文档 | `atb-aclnn-operator-replacement-designer` | 需要 ATB/ACLNN 接口链接 |
| 执行算子迁移代码 | `atb-aclnn-operator-migration` | 支持 910B/950 设备，建议先完成设计文档 |
| 运行 CSV 测试验证算子 | `atb-csv-tester` | 需要 CANN + 测试框架 |
| 编写算子 CSV 测试用例 | `atb-csv-testcase-generator` | 需要接口规格，覆盖 910B/950 |
| ATB 调试/问题排查 | `atb-debug-guide` | 报错分析、环境配置 |

---

## 参考文档

- [ATB Skills 开发者指南](docs/atb-skills-guide.md)
- [Agent Skills 背景知识](docs/agent-skills-background.md)
- [项目描述规则](rules/project-description.md)
