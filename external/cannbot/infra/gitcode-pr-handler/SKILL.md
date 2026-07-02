---
name: external-cannbot-infra-gitcode-pr-handler
description: 根据 GitCode PR 的代码变更，重新生成符合约定式提交规范的 PR 标题与符合仓库 PR 模板的 PR 描述（body），然后通过
  GitCode API 写回 PR。当用户提供 PR 链接、要求"更新 PR / 生成标题 / 生成描述 / 改 PR 文案 / 重写 PR 标题描述"时触发此
  skill。
license: CANN-2.0
original-name: gitcode-pr-handler
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-02'
synced-commit: 342679f37ec1b052a8b64d6c9fd59f67e0c89073
---

# GitCode PR 标题 / 描述生成

根据 GitCode PR 的代码变更，重新生成 PR 标题（约定式提交）与 PR 描述（沿用仓库 PR 模板），并通过 API 写回 PR。

---

## 核心原则

### 内容必须来源于代码

- PR 标题与描述必须基于 PR 实际代码变更**重新生成**——原标题/描述仅作为参考，不直接复用
- 通过分析变更文件、diff 内容、commit 信息来总结改动；不要凭经验或假设编造
- 没有代码依据的特性 / 平台 / 数据类型，**不要**写进 PR 描述

### 沿用仓库 PR 模板

PR 描述**必须优先使用仓库 PR 模板**（不同仓库格式不同），只填模板里的占位段。模板优先级：

1. `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md`
2. `.gitcode/PULL_REQUEST_TEMPLATE.md`
3. `.github/PULL_REQUEST_TEMPLATE.md`

仓库无任何模板时降级到本文档「内容生成规范 · 默认 PR 描述格式」。

### 不创建 Issue

本 skill 的写入面**仅限 PR 标题与 PR body**，不会调用 Issue API、不创建 Issue。如果 PR 描述里已经有"关联的Issue"段并填好了 `#issue_number`，沿用即可；如果该段为空，留空（不要 fabricate `#0` 或 `#TBD`）。

### 禁止创建测试内容

严禁向目标仓库发送测试 / 调试用 PR 评论。API 调用问题应通过本地构造 JSON 排查。

### 交互节奏：环境预检 + 终局确认，最多两次卡点

整个流程仅在两类时刻打断用户：

1. **Step 0 环境预检** — token / git / curl / /tmp 等缺失时 AskUserQuestion 询问
2. **Step 7 提交确认** — 完整的新标题、新描述生成后，**一次** AskUserQuestion 询问是否提交

中间过程的所有"判断"——模板选择、标题草稿、描述草稿——**不要**用 AskUserQuestion 打断；按下文「自动决策」执行并在文本里说明"我做了什么、为什么"。

AskUserQuestion 一次只问一个。

### 自动决策

| 决策点 | 做法 |
|--------|------|
| PR 模板选择 | 按上文优先级选定唯一模板；多个候选时按优先级取首个，文本说明所选 |
| 标题 / 描述草稿 | 直接生成；用户在 Step 7 看到完整内容统一确认 |

---

## 工作流程

### Step 0：环境预检（必经）

> 详见 [gitcode-toolkit/references/env-check.md](../gitcode-toolkit/references/env-check.md)

token / git / curl / python3 / /tmp 任一缺失 → AskUserQuestion 询问（一次只问一个）；全部通过则输出预检报告进入 Step 1。

### Step 1-6：解析与上下文获取

1. **解析 PR 链接** → 提取 owner / repo / pr_number（详见 [gitcode-toolkit/references/url-parsing.md](../gitcode-toolkit/references/url-parsing.md)）
2. **获取 PR 详情** → 调用 API 取 title / body / head.ref / base.ref（详见 [gitcode-toolkit/references/gitcode-api.md](../gitcode-toolkit/references/gitcode-api.md)）；原标题/描述只作参考
3. **克隆代码** → `/tmp/gitcode-pr-handler_${owner}_${repo}_${ts}`（详见 [gitcode-toolkit/references/clone-and-checkout.md](../gitcode-toolkit/references/clone-and-checkout.md)）
4. **展示变更列表** → `git diff --numstat` + `--name-status` 表格展示，**直接进入下一步**（不询问）
5. **查找 PR 模板** → 按上文「沿用仓库 PR 模板」优先级独立检查每个路径（不要用 `&&` 链式连接，否则前一个不存在时后续不会被检查）：

```bash
# 逐个检查，任一存在即使用
ls .gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md 2>/dev/null
ls .gitcode/PULL_REQUEST_TEMPLATE.md 2>/dev/null
ls .github/PULL_REQUEST_TEMPLATE.md 2>/dev/null
```

本地克隆失败时通过 API 获取：

```bash
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md?access_token=${GITCODE_TOKEN}"
```

文本说明"已识别 PR 模板：…，使用模板填充描述"或"未发现 PR 模板，使用默认格式"。

6. **生成新标题与新描述** → 见下「内容生成规范」

### Step 7：最终提交确认（流程中**唯一**的提交确认）

完成生成后统一展示：

```
即将执行：
  PR #${pr_number}
    旧标题 → 新标题
    旧描述 → 新描述（diff 摘要或全文）
```

调用**唯一一次** AskUserQuestion：

```
问题: 以上内容是否提交到 GitCode？
选项:
  - 全部按此提交 (Recommended)
  - 取消，保留本地草稿
```

确认后执行 **PATCH** PR title / body（详见 [gitcode-toolkit/references/gitcode-api.md](../gitcode-toolkit/references/gitcode-api.md)），随后 **GET** 回查 PR 验证写入成功（**PATCH 200 不代表写入成功**）。

---

## 内容生成规范

### PR 标题格式（约定式提交）

| 类型 | 前缀 | 示例 |
|------|------|------|
| 新功能 | `feat: ` | `feat: 添加用户登录功能` |
| Bug 修复 | `fix: ` | `fix: 修复登录验证逻辑` |
| 文档 | `docs: ` | `docs: 更新安装文档` |
| 重构 | `refactor: ` | `refactor: 重构用户模块` |
| 测试 | `test: ` | `test: 添加登录单元测试` |
| 性能 | `perf: ` | `perf: 优化批量插入路径` |

类型从主要变更性质判定；同时含多类时取占比最大的一类，混合改动可加 scope（`feat(login): ...`）。

### PR 描述（沿用仓库模板）

如果仓库有 PR 模板：

- 沿用模板的**原章节标题**，仅替换 `<!-- ... -->` 占位
- 不删减、不简化模板字段；模板要求的内容必须从代码里找到依据再填
- 不增加模板没有的章节

### 默认 PR 描述格式（仅在仓库无任何模板时使用）

```markdown
## 描述
[详细描述改动的原因和所采取的方法]

### 改动原因
[描述为什么要做这个改动]

### 改动方法
[描述具体的实现方式]

## 测试
[描述进行了哪些测试]

## 类型标签
- [ ] Bug 修复
- [ ] 新特性
- [ ] 文档更新

## 关联的Issue
- #${issue_number}（已有则填，无则留空——不要 fabricate）
```

---

## API / Token / 错误处理

GitCode API、Token、HTTP 状态码错误处理统一详见：

- [gitcode-toolkit/references/gitcode-api.md](../gitcode-toolkit/references/gitcode-api.md)
- [gitcode-toolkit/references/env-check.md](../gitcode-toolkit/references/env-check.md) / [token-config.md](../gitcode-toolkit/references/token-config.md)

本 skill 的本地约束：

- 调试 API 参数时，先在本地构造 JSON 验证格式，**禁止向目标仓库发送测试请求**
- PATCH 返回 200 不代表写入成功——必须 GET 回查 PR 验证 title / body 实际生效

---

## 参考文档

- [gitcode-toolkit/SKILL.md](../gitcode-toolkit/SKILL.md) — Git 克隆/分支检出、merge-base、diff/log 等共享操作
- [gitcode-toolkit/references/gitcode-api.md](../gitcode-toolkit/references/gitcode-api.md) — GitCode PR API 详细文档
