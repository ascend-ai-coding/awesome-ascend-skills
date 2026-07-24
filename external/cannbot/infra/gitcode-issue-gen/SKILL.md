---
name: external-cannbot-infra-gitcode-issue-gen
description: 根据用户输入自动判断走两条路径之一：(PR路径) 用户提供 GitCode PR 链接时，按变更类型自动选用 Issue 模板，通过 GitCode
  API 创建 Issue 并完成 PR ↔ Issue 双向关联；(手动路径) 用户直接描述问题或要求"提 Issue / 生成草稿"时，交互式收集信息、生成草稿、查重，经确认后提交。当用户提供
  PR 链接、要求"创建 Issue / 关联 Issue / 给 PR 建 Issue"，或用户直接描述问题、要求"提单 / 生成草稿"时触发此 skill。
license: CANN-2.0
original-name: gitcode-issue-gen
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
---

# GitCode Issue 生成与提交

## 核心原则

### 内容真实性

- **PR 路径**：所有技术信息必须从 PR 代码变更中直接获取，禁止基于经验推断或假设
- **手动路径**：不要替用户推断未确认事实；可以根据日志做"可能原因"说明，但必须标注为分析或推测
- 代码中没有明确说明的特性，**不要**写进 Issue body

### 双向关联（仅 PR 路径）

创建 Issue 后必须更新 PR 描述添加 Issue 链接，Issue body 中也必须包含 PR 链接，二者相互引用。

### 禁止创建测试内容

严禁创建测试 Issue 或调试评论到目标仓库。API 调用问题应通过本地构造 JSON 排查，**绝不**通过向目标仓库发请求试错。

### 写操作必须确认

创建 Issue、修改 PR body、Assign 等写操作**必须**经用户确认后才能执行。

---

## 分支判断

收到用户请求后，按以下条件选择路径：

| 条件 | 选择路径 |
|------|---------|
| 用户提供了 PR 链接（`https://gitcode.com/.../pull/...`）或明确要求"给 PR 建 Issue / 关联 Issue" | **Path A：PR → Issue** |
| 用户直接描述问题、要求"提 Issue / 创建 Issue / 生成草稿 / 提单" | **Path B：手动提单** |
| 同时满足两者 | 优先 **Path A**，文本说明"PR 路径已满足，如需要单独创建不关联 PR 的 Issue 请说明" |

---

## Path A：PR → Issue（自动化）

用户提供 PR 链接时，从代码变更自动生成关联 Issue。

### 交互节奏：环境预检 + 终局确认 + 可选 Assign，最多三次卡点

整个流程仅在以下三类时刻打断用户：

1. **Step 0 环境预检** — token / git / curl / /tmp 等缺失时 AskUserQuestion 询问
2. **Step 7 提交确认** — Issue 模板、Issue body、PR 关联写法生成完毕后，**一次** AskUserQuestion 询问是否提交
3. **Step 8 自助 Assign（可选）** — 仅在 Issue 创建成功时弹**一次**，询问是否将新 Issue assign 给当前 token 用户

中间过程的所有"判断"——模板选择、Issue body 草稿——**不要**用 AskUserQuestion 打断，按下文「自动决策」执行并在文本里说明"我做了什么、为什么"。

AskUserQuestion 一次只问一个。

### 自动决策

| 决策点 | 做法 |
|--------|------|
| Issue 模板选择 | 按下表"Issue 模板按变更类型推荐"自动选定，文本说明所选项与原因 |
| Issue body 草稿 | 直接生成；用户在 Step 7 看到完整 body 再统一确认 |

**Issue 模板按变更类型推荐：**

| 变更特征 | 推荐模板 |
|----------|----------|
| RFC 提案 / 架构设计 / 重大变更 | request-for-comments |
| 新增功能 / 新增算子 / 新增接口 | feature-request / requirement |
| Bug 修复 / 异常处理 / 回退 | bug-report |
| 文档变更 | documentation |
| 重构 / 性能优化 | feature-request（无 refactor 模板时退而求其次） |

仓库实际可用的模板与推荐不同时，按"实际存在 ∩ 推荐项"取交集；推荐项不存在则按 `requirement` → `feature-request` → `bug-report` 顺序降级。

### Step 0：环境预检（必经）

> 详见 [gitcode-toolkit/references/env-check.md](../gitcode-toolkit/references/env-check.md)

token / git / curl / python3 / /tmp 任一缺失 → AskUserQuestion 询问（一次只问一个）；全部通过则输出预检报告进入 Step 1。

### Step 1-6：解析与上下文获取

1. **解析 PR 链接** → 提取 owner / repo / pr_number（详见 [gitcode-toolkit/references/url-parsing.md](../gitcode-toolkit/references/url-parsing.md)）
2. **获取 PR 详情** → 调用 API 取 title / body / head.ref / base.ref（详见 [gitcode-toolkit/references/gitcode-api.md](../gitcode-toolkit/references/gitcode-api.md)）；如果 PR body 中已能识别到 `#issue_number`，向用户说明"PR 已关联 Issue #N，是否仍创建新 Issue"（AskUserQuestion 二选一）
3. **克隆代码** → `/tmp/gitcode-issue-gen_${owner}_${repo}_${ts}`（详见 [gitcode-toolkit/references/clone-and-checkout.md](../gitcode-toolkit/references/clone-and-checkout.md)）
4. **展示变更列表** → `git diff --numstat` + `--name-status` 表格展示，**直接进入下一步**（不询问）
5. **查找 Issue 模板** → 分别检查 `.gitcode/ISSUE_TEMPLATE/` 与 `.github/ISSUE_TEMPLATE/`（必须独立检查，禁止 `&&` 链式）；收集所有可用模板后按上文「Issue 模板按变更类型推荐」自动选定，文本说明"已识别可用模板：[…]，自动选用 X，原因：…"
6. **生成 Issue body** → 使用模板填充，结尾固定追加 `关联 PR：https://gitcode.com/${owner}/${repo}/pull/${pr_number}`（模板格式见 [references/issue-templates.md](references/issue-templates.md)）

### Step 7：最终提交确认（流程中**唯一**的提交确认）

完成 Issue 生成后统一展示：

```
即将执行：
  PR #${pr_number}
    将在 PR body 末尾追加："关联的Issue：#${新 Issue 号待生成}"
  Issue（新建）
    模板：${xxx.yml}（自动选择，原因：…）
    标题：…
    body：…
```

调用**唯一一次** AskUserQuestion：

```
问题: 以上内容是否提交到 GitCode？
选项:
  - 全部按此提交 (Recommended)
  - 取消，保留本地草稿
```

确认后顺序执行：
1. **POST** 创建 Issue（API 详见 [gitcode-toolkit/references/gitcode-api.md](../gitcode-toolkit/references/gitcode-api.md)）→ 取回 `new_issue_number`
2. **PATCH** PR body 追加关联链接（不重写其他段落，仅在 PR body 末尾或"关联的Issue"段插入 `#${new_issue_number}`）
3. **GET** 回查 PR / Issue 验证写入成功（**PATCH 200 不代表写入成功**）

### Step 8：Issue 自助 Assign（可选）

**触发前提**：本次 Issue 创建成功（API 返回 201 且 GET 回查到 Issue）。失败则跳过整个 Step 8。

**执行**：

1. 用 token 调 `GET /api/v5/user` 取当前认证用户 login：

```bash
curl -sS "https://api.gitcode.com/api/v5/user?access_token=${GITCODE_TOKEN}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin).get('login',''))"
```

> 用 `/user` 而非要求用户配置环境变量——token 已唯一标识认证主体，再问 username 冗余且容易拼错（GitCode login ≠ 昵称、≠ git config user.name）。

2. 若 `/user` 返回非空 login，发起**唯一一次** AskUserQuestion：

```
问题: Issue #${new_issue_number} 已创建。是否将其 assign 给 @${login}？
        将通过在 Issue 下评论 "/assign @${login}" 实现。
选项:
  - 是，assign 给我 (Recommended)
  - 否，保持未指派
```

3. 若 `/user` 调用失败（401/403/网络）：跳过询问，输出一行说明「无法通过 token 解析当前用户（HTTP X），跳过 Issue 自助 assign」。**不要**追问用户手填 username——根因是 token 权限/网络，问用户也救不回来。

4. 用户选"是"则 POST Issue 评论：

```bash
curl -X POST -H "Content-Type: application/json" \
  -d "{\"body\": \"/assign @${login}\"}" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/issues/${new_issue_number}/comments?access_token=${GITCODE_TOKEN}"
```

> 中文/特殊字符场景推荐 Python `urllib` 构造请求，避免 shell 转义出错。
> 评论 body **必须严格**为 `/assign @${login}`，**不**附加文字/emoji/换行（附加内容会破坏平台指令解析）。

**反模式**：

- ❌ Issue 创建失败 / GET 回查失败也询问 assign
- ❌ 凭 git config user.name 或 PR 作者推断当前用户
- ❌ 把 Step 8 询问和 Step 7 提交确认合并成一条 AskUserQuestion
- ❌ `/user` 失败时降级追问 username

---

## Path B：手动提单（交互式）

用户直接描述问题时，交互式收集信息并生成 Issue 草稿。

### 触发边界

**仅在以下情况触发**：
- 用户明确说"提 Issue / 创建 Issue / 提单 / 提交到 GitCode"
- 用户明确要求"生成 Issue 草稿 / 整理 Issue 内容"
- 上游 agent 明确调用处理 Issue 草稿或提交

**不要在以下情况触发**：
- 用户只是描述 bug、报错、需求、文档问题或咨询问题
- 用户要求调试、定位、修代码、解释原因
- 还没有确认是否要对外提交 Issue

### 工作流程

详见 [references/manual-issue-workflow.md](references/manual-issue-workflow.md)。

**简要流程**：

| 阶段 | 名称 | 完成标准 |
|------|------|---------|
| 1 | 目标确认 | 已确定 owner/repo 与处理模式（草稿 / 提交） |
| 2 | 模板发现 | 已读取本地或远程 Issue 模板，失败时说明回退来源 |
| 3 | 信息收集 | 已补齐模板必填字段，缺失项明确标记待补充 |
| 4 | 草稿生成 | 已生成标题和正文，并展示给用户确认 |
| 5 | 查重检查 | 提交前已查询可能重复的 Issue |
| 6 | 提交与反馈 | 用户确认后提交，返回 URL 或明确错误原因 |

### Issue 编写质量清单

详见 [references/manual-issue-draft.md](references/manual-issue-draft.md)。

### 约束层

详见 [references/manual-issue-workflow.md](references/manual-issue-workflow.md)「约束层」章节。

---

## 共享能力

### 模板读取策略

#### 本地模板

优先读取目标仓库本地模板：

```bash
ls <project_root>/.gitcode/ISSUE_TEMPLATE/*.yml
```

解析字段：
- `name`
- `title`
- `labels`
- `body[].attributes.label`
- `body[].attributes.description`
- `body[].validations.required`

#### 远程模板

当目标仓库不是当前本地仓库，或本地模板不存在时，通过 GitCode Contents API 获取：

```bash
curl -s "https://api.gitcode.com/api/v5/repos/{owner}/{repo}/contents/.gitcode/ISSUE_TEMPLATE?access_token=$GITCODE_TOKEN"
```

模板文件响应中的 `content` 字段通常为 base64 编码，需要解码后解析 YAML。

#### 内置备选模板

当本地和远程模板都不可用时，允许使用内置备选模板，但必须在草稿中说明模板来源：

> 模板来自内置备选，可能与目标仓库最新模板不一致。

---

### GitCode OpenAPI 参考

#### 查询 Issue

| 操作 | Method | URL |
|------|--------|-----|
| 获取仓库所有 Issue | GET | `https://api.gitcode.com/api/v5/repos/{owner}/{repo}/issues` |
| 获取单个 Issue | GET | `https://api.gitcode.com/api/v5/repos/{owner}/{repo}/issues/{number}` |

常用查询参数：

| 参数 | 说明 |
|------|------|
| `access_token` | GitCode API Token |
| `state` | `open` / `closed` / `all` |
| `labels` | 逗号分隔的标签 |
| `page` | 页码 |
| `per_page` | 每页数量 |

#### 创建 Issue

GitCode 创建 Issue 使用：

```text
POST https://api.gitcode.com/api/v5/repos/{owner}/issues
```

参数：

| 参数 | 位置 | 必填 | 说明 |
|------|------|------|------|
| `access_token` | query | 是 | GitCode API Token |
| `repo` | formData | 是 | 仓库路径 |
| `title` | formData | 是 | Issue 标题 |
| `body` | formData | 否 | Issue 正文 |
| `labels` | formData | 否 | 逗号分隔的标签 |

示例：

```bash
curl --location --request POST "https://api.gitcode.com/api/v5/repos/{owner}/issues?access_token=$GITCODE_TOKEN" \
  --form "repo={repo}" \
  --form "title={title}" \
  --form "body={body}" \
  --form "labels={labels}"
```

#### 认证方式

通过环境变量提供 token：

```bash
export GITCODE_TOKEN="your_token_here"
```

安全要求：
- 不输出 token 或 token 片段
- 不把 token 写入草稿、日志或临时文件
- 创建 Issue 前确认 token 具备对应仓库权限

---

## API / Token / 错误处理

GitCode API、Token、HTTP 状态码错误处理统一详见：

- [gitcode-toolkit/references/gitcode-api.md](../gitcode-toolkit/references/gitcode-api.md)（注意 `labels`/`assignees` 必须字符串而非数组）
- [gitcode-toolkit/references/env-check.md](../gitcode-toolkit/references/env-check.md) / [token-config.md](../gitcode-toolkit/references/token-config.md)

本 skill 的本地约束：

- 调试 API 参数时，先在本地构造 JSON 验证格式，**禁止向目标仓库发送测试请求**

---

## 参考文档

- [references/issue-templates.md](references/issue-templates.md) — Feature Request / Bug Report / Request for Comments Issue 模板
- [references/manual-issue-draft.md](references/manual-issue-draft.md) — 手动提单：Issue 编写与信息收集
- [references/manual-issue-workflow.md](references/manual-issue-workflow.md) — 手动提单：工作流程与约束
- [gitcode-toolkit/references/gitcode-api.md](../gitcode-toolkit/references/gitcode-api.md) — GitCode PR / Issue API 详细文档
