# AGENTS.md - Awesome Ascend Skills Agent Guide

本文件指导 AI coding agent 如何阅读、使用、修改和提交 Awesome Ascend Skills 仓库。它不是用户安装文档的替代品；面向的是在本仓内维护 skill、bundle、marketplace、README 和同步脚本的开发者/Agent。

## 1. 先读什么

进入仓库后按这个顺序建立上下文：

1. 先读 [`README.md`](README.md)：理解用户入口、推荐安装方式、skill 导航和贡献流程。
2. 再读 [`skills/README.md`](skills/README.md)：确认本地 skill 的功能域入口。
3. 进入对应 `skills/<domain>/README.md`：确认该域下实际有哪些 leaf skill / skill set。
4. 只打开任务相关的 `SKILL.md`、`references/`、`scripts/`，不要一次性扫完整仓。
5. 如果涉及分类、bundle、命名或治理规则，再读 [`docs/governance/skill-governance.md`](docs/governance/skill-governance.md)。
6. 如果涉及安装包、公开注册表或外部同步入口，再读 [`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json)。

常用快速定位命令：

```bash
rg --files skills
rg --files external
rg -n '"name": "ascend-|categoryLibrary|domain-skill-set|official-bundle' .claude-plugin/marketplace.json
rg -n '^name:|^description:' skills external .agents/skills
```

## 2. 仓库结构

```text
awesome-ascend-skills/
├── skills/
│   ├── README.md
│   ├── base/                 # 基础环境、服务器连接、容器、设备管理、硬件诊断、torch_npu
│   ├── inference/            # 模型转换、推理服务、量化、评测、在线压测
│   ├── training/             # 通信测试、MindSpeed、VERL、训练流程
│   ├── profiling/            # Profiling 采集、瓶颈分析、MFU 分析
│   ├── ops/                  # AscendC、op-plugin、Triton 迁移、算子 benchmark
│   ├── agent-tools/          # issue/RCA、社区反馈、GitCode 流程、工程知识工具
│   └── ai-for-science/       # AI for Science 总入口和专项模型/框架 skill
├── external/                 # 自动同步的外部 skills；通常不要手工改内容
├── docs/governance/          # taxonomy、命名、分类、bundle 规则
├── scripts/
│   ├── validate_skills.py    # 本地校验入口
│   └── sync_external_skills.py
└── .claude-plugin/
    └── marketplace.json      # 插件注册表、bundle 定义、categoryLibrary
```

本地 skill 的正式路径只允许是 `skills/<domain>/...`。不要在仓库根目录重新放本地 `SKILL.md`，不要恢复旧的 `skills/knowledge/`，当前对应目录是 `skills/agent-tools/`。

## 3. Skill 类型

本仓同时维护五类对象，修改前先判断目标属于哪一类：

| 类型 | 典型位置 | 命名 | 说明 |
|------|------|------|------|
| leaf skill | `skills/<domain>/<skill>/SKILL.md` | 叶子目录名 | 单一能力，应该能独立安装和触发 |
| nested skill | `skills/<domain>/<group>/<leaf>/SKILL.md` | 以 `<group>-` 为前缀 | 属于某个技能包目录下的子能力 |
| domain skill set / router | `skills/<domain>/<group>/...` | 通常 `*-skills` 或领域入口名 | 聚合或分流同一目录树内的多个 skill |
| official bundle | `.claude-plugin/marketplace.json` | `ascend-<domain>` | 官方推荐安装包，必须只引用同一 `skills/<domain>/...` 目录树 |
| external synced skill | `external/<source>/<skill>/SKILL.md` | `external-<source>-...` | 外部同步内容，默认通过同步流程更新 |

`skills/ai-for-science/` 是特殊域：其本地 skill 名称保持 `ai-for-science-*` 语义前缀。

## 4. 如何使用本仓

用户安装推荐使用官方 `npx skills add` 方式。安装技能包/目录时使用 URL + `-s '*'`；安装单个 skill 时使用仓库 shorthand + `-s <skill-name>`。

```bash
npx skills add https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/skills/base -s '*'
npx skills add https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/skills/inference -s '*'
npx skills add https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/skills/training -s '*'
npx skills add https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/skills/profiling -s '*'
npx skills add https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/skills/ops -s '*'
npx skills add https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/skills/ai-for-science -s '*'
npx skills add ascend-ai-coding/awesome-ascend-skills -s npu-smi
```

仓库维护时不要把 README 的推荐安装方式改回旧的 `--skill ascend-*` 形式，也不要把单 skill 示例写成目录 URL 安装。

## 5. 修改 SKILL.md

最小格式：

````markdown
---
name: skill-name
description: 至少 20 个字符，说明何时使用这个 skill，包含关键触发词
keywords:
  - 可选关键词
---

# Skill Title

## Quick Start

```bash
echo "example"
```
````

规则：

- leaf skill 的 `name` 必须匹配叶子目录名。
- nested skill 的 `name` 必须以前一层 group 目录为前缀，例如 `skills/training/mindspeed-llm/... -> name: mindspeed-llm-*`。
- external skill 的 `name` 必须以 `external-<source>-` 开头。
- `description` 用于 Agent 触发匹配，应描述“什么时候使用”，不要只写功能名。
- 核心说明放在 `SKILL.md`，长文档放到 `references/`，脚本放到 `scripts/`。
- Markdown 代码块必须带语言，例如 `bash`、`python`、`json`、`yaml`。
- 内部链接使用相对路径，并确保能被校验通过。
- 不要留下 `[TODO]` 或 `[TODO: ...]`。

## 6. 修改 marketplace.json

每个 `.claude-plugin/marketplace.json` 的 plugin 条目都必须维护：

- `category`：单值主分类，只能来自 `categoryLibrary.primaryCategories`。
- `categories`：多值标签，必须来自 `categoryLibrary`，并包含主 `category`。
- `source`：单个 skill 的实际路径，或者 grouped entry 使用 `"./"`。
- `skills`：仅 grouped entry 使用，必须引用真实存在的 skill 目录。

单个 leaf skill 示例：

```json
{
  "name": "new-skill",
  "description": "Description for agent matching",
  "source": "./skills/inference/new-skill",
  "category": "inference",
  "categories": ["inference", "leaf-skill", "model-serving"]
}
```

同目录 skill set 示例：

```json
{
  "name": "hiascend-forum",
  "description": "昇腾社区论坛抓取与反馈问题分析。",
  "source": "./",
  "strict": false,
  "skills": [
    "./skills/agent-tools/hiascend-forum/hiascend-forum-fetcher",
    "./skills/agent-tools/hiascend-forum/hiascend-forum-analyzer"
  ],
  "category": "agent-tools",
  "categories": ["agent-tools", "domain-skill-set", "community-feedback"]
}
```

关键约束：

- 带 `skills` 数组的 grouped entry 必须收敛在同一个具体目录树下。
- `ascend-base` 只能引用 `./skills/base/...`，`ascend-training` 只能引用 `./skills/training/...`，以此类推。
- domain skill set 也必须是 `../<bundle-name>/<skill-0>`、`../<bundle-name>/<skill-1>` 这种同目录结构。
- external grouped entry 只能引用同一个 `external/<source>/...`。
- 如果新增 category 标签，先更新 `categoryLibrary`，再更新治理文档。

## 7. 新增或调整本地 skill

标准流程：

1. 判断是否真的需要新 skill；如果只是补充已有能力，优先放入现有 `references/`。
2. 选择功能域：`base`、`inference`、`training`、`profiling`、`ops`、`agent-tools`、`ai-for-science`。
3. 创建目录：`mkdir -p skills/<domain>/<skill-name>`。
4. 编写 `SKILL.md`，必要时添加 `references/`、`scripts/`、`assets/`。
5. 更新对应 `skills/<domain>/README.md` 和根 `README.md` 导航。
6. 更新 `.claude-plugin/marketplace.json`，设置 `category` / `categories`。
7. 如果改变 taxonomy、bundle、分类或目录规则，更新 `docs/governance/skill-governance.md`。
8. 运行校验。

## 8. 外部 skills 同步

`external/` 下内容来自外部仓库同步：

- 默认不要手工修改 external skill 内容。
- 同步来源配置在 `.github/external-sources.yml`。
- 同步逻辑在 `scripts/sync_external_skills.py`。
- 同名冲突时本仓本地 skill 优先，外部 skill 被跳过。
- 外部 marketplace grouped entry 使用 `category: "external"`，并包含 `external-skill-set`、`external-sync`。

如果用户明确要求同步 main 或外部源，先确认来源分支，再按同步脚本和当前目录规范合入。

## 9. 校验命令

提交前至少运行：

```bash
python3 scripts/validate_skills.py
```

这个脚本会检查：

- `SKILL.md` frontmatter
- skill 命名规则
- description 长度
- `[TODO]` 占位符
- marketplace JSON、category、categories
- marketplace source / skills 路径是否存在
- grouped bundle 是否落在同一个目录树内

常用补充检查：

```bash
python3 -m json.tool .claude-plugin/marketplace.json >/tmp/awesome-ascend-marketplace.json
python3 -m py_compile scripts/validate_skills.py scripts/sync_external_skills.py
git diff --check -- .claude-plugin/marketplace.json README.md docs/governance/skill-governance.md AGENTS.md scripts
```

如果 `validate_skills.py` 只报告外部同步 skill 的既有 TODO warning，可以在提交说明里明确这是 external 历史内容，不要为通过校验而随意改外部同步文件。

## 10. 提交前自查

提交 PR 或交付变更前确认：

- [ ] 修改路径符合 `skills/<domain>/...` 或 `external/<source>/...` 规则
- [ ] 没有恢复旧的 `skills/knowledge/`
- [ ] `SKILL.md` 的 `name` / `description` / 链接 / 代码块符合规范
- [ ] README 和对应 domain README 已同步
- [ ] marketplace 条目已同步，且 `category` / `categories` 来自 `categoryLibrary`
- [ ] grouped entry 的 `skills` 都在同一个 bundle 目录树内
- [ ] 官方 `ascend-*` bundle 没有混入其他 domain 的路径
- [ ] 外部同步内容没有被不必要地手工改写
- [ ] `python3 scripts/validate_skills.py` 通过
- [ ] 如涉及治理规则，已同步更新 `docs/governance/skill-governance.md`

## 11. 代码风格

Python：

```python
#!/usr/bin/env python3
from typing import Dict, List, Tuple


def validate(path: str) -> Tuple[List[str], List[str]]:
    """Return validation errors and warnings."""
    return [], []
```

Shell：

```bash
#!/bin/bash
set -e
readonly DIR="$(cd "$(dirname "$0")" && pwd)"
```

文档：

- 中英文都可以，但本仓面向中文 Ascend 用户，中文说明应优先清晰。
- 避免营销式描述，直接说明适用场景、输入、输出、命令和限制。
- 更新安装方式时，技能包/目录使用 `npx skills add https://github.com/.../tree/main/... -s '*'`，单 skill 使用 `npx skills add ascend-ai-coding/awesome-ascend-skills -s <skill-name>`。

## 12. 常见错误

- 把本地 skill 放回根目录或旧 `skills/knowledge/`。
- `SKILL.md name` 与目录名不一致。
- grouped marketplace entry 混入多个 domain 的 skill。
- marketplace 只改了 `category`，忘记同步 `categories` 或 `categoryLibrary`。
- 新增 skill 只改了 marketplace，忘记 README / domain README。
- 直接手改 external 内容，而不是通过同步流程或上游源处理。
- 用旧的 `--skill ascend-*` 安装命令替换 README 中的官方 URL 安装方式。
- 把单个 skill 写成目录 URL 安装，而不是 `npx skills add ascend-ai-coding/awesome-ascend-skills -s <skill-name>`。
