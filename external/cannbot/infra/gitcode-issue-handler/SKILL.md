---
name: external-cannbot-infra-gitcode-issue-handler
description: GitCode Issue 端到端处置工具，根据 Issue 内容自动判断走两条路径之一：(PR 路径) 克隆 fork → 代码定位 →
  最小改动 → 跑测试 → 提交并推送 → 创建 PR，覆盖 bug 修复 / 功能增强 / 文档补全等任何需要代码变更的诉求；(Comment 路径) 仅克隆上游主仓只读分析
  → 起草答复 → 提交评论，覆盖答疑 / 设计澄清 / 用法说明等不需改代码的诉求。当用户提到"处理 Issue / 跟进 Issue / 从 Issue 提
  PR / 端到端处理 Issue / 修复 Issue"或仅给出 issue_url 让 Claude 判断要不要改代码时使用此 skill；用户明确说"只回复
  / 答疑 / 评论回复 / 不改代码"则直接走 Comment 路径。
license: CANN-2.0
original-name: gitcode-issue-handler
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
---

# GitCode Issue 端到端处理

把一条 Issue 链接跑到合适的归宿：要么"上游 PR 已创建"，要么"答复评论已发布"。**Step 1.5 模式判定** 先看 Issue 内容是否需要改代码（以及用户消息里的显式倾向），再决定走 PR 路径还是 Comment 路径；不要在没看 Issue 内容前就假设要改代码。

## 通用约束

- 全程中文与用户沟通和思考。
- **PR 路径**：代码改动、commit message、PR 标题与描述统一用英文；坚持最小修改，能两行解决不要重构十行。
- **Comment 路径**：评论语言跟随 Issue 正文主语言（中文 Issue → 中文评论；英文 Issue → 英文评论），读者是 Issue 提问者本人，可读性优先。
- 写操作（commit / push / PR / 评论）前必须 AskUserQuestion 确认，**一次只问一个**；被审查的正文先在对话主流以代码块形式打印，再发问题。不要把多行内容塞进 AskUserQuestion 的 `preview` 字段。

## 输入参数

| 参数 | 必填 | 说明 |
|------|:----:|------|
| `issue_url` | Y | GitCode Issue 链接，如 `https://gitcode.com/cann/ops-math/issues/123` |
| `fork_url` |  | 仅 **PR 路径** 需要：用户自己 fork 出来的仓库链接。PR 路径下若缺失，进入 Step 1.6 询问 / 自动 fork。Comment 路径不需要 fork。 |
| `base_branch` |  | 仅 PR 路径用：上游目标分支，默认从 Issue 上下文或上游默认分支推断（通常 `master`） |

## 总体流程

```
Step 0    环境预检（token / git / /tmp；git author 仅 PR 路径必检）
Step 1    解析 issue_url + fork_url，拉取 Issue 详情，抽取关键信号
Step 1.5  模式判定：Issue 内容 + 用户显式倾向 → 推荐 PR / Comment → 用户确认

PR 路径（详见 references/pr-path.md）：
Step 1.6  fork_url 缺失时询问 / 自动 fork
Step 2    克隆 fork、设置 upstream、同步 base、切工作分支
Step 3    代码定位 + 改动方案 → 用户确认
Step 4    最小改动 + 跑测试
Step 5    Commit（英文）→ 确认 → Push
Step 6    参考 gitcode-toolkit 的 PR 创建工作流（沿用仓库模板）
Step 7    日志 + 报告 PR 链接

Comment 路径（详见 references/comment-path.md）：
Step C-2  克隆上游主仓（只读）
Step C-3  代码与 Issue 联合分析
Step C-4  起草答复评论（语言跟随 Issue）
Step C-5  用户确认 → POST 评论 → GET 回查
Step C-6  日志 + 报告评论链接
```

> 每个写操作前都先用 AskUserQuestion 让用户确认：PR 路径的 Step 3 方案、Step 5 commit、Step 5 push、Step 6 PR；Comment 路径的 Step C-5 评论提交。一次只问一个问题。

---

## Step 0: 环境预检

> 详见 [gitcode-toolkit/references/env-check.md](../gitcode-toolkit/references/env-check.md)

执行时机：拿到 `issue_url` 后、解析任何业务字段前，立即执行。

必检项（任一失败立即通过 AskUserQuestion 询问，一次只问一个）：

1. **GitCode Token**：用户消息 → 环境变量 `GITCODE_TOKEN` → 都没有则询问
2. **git / curl / python3**：缺失则询问是否继续
3. **/tmp 可写**
4. **Git 提交用户信息**：**仅 PR 路径必检**。先做 1~3 项基础预检，git author 留到 Step 1.5 判定为 PR 后再补——Comment 答疑场景下不会无端问用户的 git 身份。按 **local → global → 询问用户** 三级 fallback 读取 `user.name` / `user.email`：先读 `git config --local`，缺失再读 `git config --global`，都缺失则 AskUserQuestion 让用户提供。拿到后**只**在 work_dir 上 `git config`，**禁止改 `~/.gitconfig` 全局**。详见 env-check.md「5. Git 提交用户信息」。

> 为什么早查 git author：缺 author 时 `git commit` 会以 `Author identity unknown` 失败，而那时已走完 clone+改代码+跑测试一整轮，回头补配置浪费上下文。早查早暴露——但别早于"已确定要 commit"。

预检通过后输出一段简短报告（格式参考 env-check.md「预检报告格式」），无需用户确认，直接进入 Step 1。

---

## Step 1: 解析链接并拉取 Issue

### 1.1 URL 解析

按 [gitcode-toolkit/references/url-parsing.md](../gitcode-toolkit/references/url-parsing.md) 解析：

- `issue_url` → `upstream_owner`, `upstream_repo`, `issue_number`
- `fork_url` → `fork_owner`, `fork_repo`

> **校验**：`upstream_repo == fork_repo` 应当成立；不成立说明 fork 链接不是该 Issue 仓库的 fork，立即 AskUserQuestion 让用户更正，不要瞎猜。

### 1.2 拉取 Issue 详情

```bash
curl -s "https://api.gitcode.com/api/v5/repos/${upstream_owner}/${upstream_repo}/issues/${issue_number}?access_token=${token}" \
  --connect-timeout 60 --max-time 180
```

抽出后面分析和 PR 关联用的字段：

| 字段 | 用途 |
|------|------|
| `title` | Issue 标题，用作代码定位起点 / PR 标题灵感 |
| `body` | Issue 正文：复现步骤、报错栈、期望行为，或诉求/疑问描述 |
| `labels` | 标签：bug / feature / question / docs 等，影响处置策略 |
| `state` | `closed` 则提醒用户该 Issue 已关闭，是否继续？|

### 1.3 关键信号提取

从 `title + body` 里手动抽出（不要让模型脑补）：

- 报错信息或日志片段
- 复现命令 / 输入样例
- 涉及的文件路径、函数名、类名（如果 Issue 中提到）
- 期望行为 vs 实际行为

这些信号是后续代码定位 / 联合分析的"种子"，缺失则在 AskUserQuestion 里问用户补一条最小复现或补一条诉求/疑问的具体描述。

---

## Step 1.5: 模式判定（PR / Comment）

读完 Issue 后立即判定。**主要靠 Issue 内容是否需要改代码来判断**，fork_url 是否提供不是决定因素——用户给了 fork_url 也可能后来发现 Issue 只是个问题；没给 fork_url 也可能是奔着修代码来的。

判定细则（关键词显式优先 / 内容信号 / 推荐确认模板）见 **[references/mode-detection.md](references/mode-detection.md)**。

判定结论必须以普通文本打印到对话主流（含 Issue 摘要 + 关键信号 + 推荐路径 + 推荐理由），再 AskUserQuestion 三选一：「走推荐路径 / 改走另一条 / 取消」。用户改路径时吸收选择继续，不要嘴硬辩护。

### 路径确定后

- **选 PR**：若 Step 0 时跳过了 git author 检查（因为模式未明），现在补做。然后进入 **PR 路径**（见下方概览与 [references/pr-path.md](references/pr-path.md)）。
- **选 Comment**：直接进入 **Comment 路径**（见下方概览与 [references/comment-path.md](references/comment-path.md)）的 Step C-2。

---

## PR 路径概览

从 Step 1.5 判定为 PR 时进入。**详细步骤、命令、确认模板见 [references/pr-path.md](references/pr-path.md)**。

骨架：

- **Step 1.6**：fork_url 缺失时 AskUserQuestion 三选一（提供链接 / 自动 fork / 取消）；自动 fork 调 GitCode Fork API，注意 409/422 已存在、403 无权限、异步未就绪三种异常处理
- **Step 2**：先检测当前目录是否已是目标 fork（比对 remote URL），是则复用当前目录、仅新建 branch；否则 clone 到 `/tmp/gitcode-issue-handler_${upstream_repo}_${issue_number}_${ts}` → 按 `GIT_AUTHOR_SOURCE`（local / global / user）处理 git author → 配 upstream → fetch upstream → `checkout -B base upstream/base` → 切 `${type}/issue-${issue_number}-<slug>`（`<type>` 按 Issue 性质选 `fix` / `feat` / `docs` 等，与 commit type 一致）
- **Step 3**：读代码定位相关位置（不脑补、bug 类先稳定复现并追根因、feature/docs 类直接对照诉求圈定改动面、写下假设并验证）→ 输出现象/结论/涉及文件/策略/风险，AskUserQuestion 让用户确认方案后再动手
- **Step 4**：最小改动（注释/变量名/log 文本全英文、遵循文件内既有风格、不顺手重构、不新增依赖）→ 跑相关测试 → 失败回到 Step 4.1 迭代（**不改测试断言来"过测"**）→ 提交前自检 checklist
- **Step 5**：Conventional Commits 风格英文 message（`<type>(<scope>): <subject>` + body；body 只写"这次提交改了什么、为什么这么改"，**不要写 `Fixes #N` / Issue 背景叙述**——Issue 关联放到 PR body 的"关联的Issue"章节；**禁止 Co-Authored-By**）→ 主流打印 commit 全文 + `git diff --stat` → 确认 → push 前再确认一次
- **Step 6**：按 [gitcode-toolkit](../gitcode-toolkit/SKILL.md) 「PR 创建工作流」执行，head 用 `${fork_owner}/${fork_repo}:branch` 格式；**PR body 必须沿用仓库模板的原章节标题**（按优先级 `git show upstream/${base_branch}:.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md` → `.gitcode/PULL_REQUEST_TEMPLATE.md` → `.github/PULL_REQUEST_TEMPLATE.md`），仅替换 `<!-- ... -->` 占位；走 POST 占位 + PATCH 完整 body + GET 回查的套路（**PATCH 200 不代表写入成功**）
- **Step 7**：写日志 + 报告 PR 链接

> 关键不变量：所有写操作前 AskUserQuestion；被审查内容先在主流以代码块打印；commit message / PR body 不塞进 `preview` 字段；commit message 不含 Co-Authored-By；PR 模板章节标题原样保留。

---

## Comment 路径概览

从 Step 1.5 判定为 Comment 时进入。**详细步骤、命令、评论模板见 [references/comment-path.md](references/comment-path.md)**。

骨架：

- **Step C-2**：clone 上游主仓到 `/tmp/gitcode-issue-handler_${repo}_${issue}_${ts}_readonly`（`--depth=200` 而非 1，要追溯历史；后缀 `_readonly` 是自我提醒，全程不 add / commit / push）
- **Step C-3**：用 Read / Grep 实地确认 Issue 提到的代码（不脑补），按问题层次答复——"为什么这么实现"→ 给历史 + 取舍；"是否支持 X"→ 核实边界；"如何用 Y"→ 最小可运行示例。找不到根据明确写"未规约，建议向 xxx 进一步确认"，**不要在评论里塞改动方案 / patch**
- **Step C-4**：起草答复，语言跟随 Issue 正文主语言；建议结构「一句话结论 + 分析（含 `file:line` 引用 + 5~10 行源码片段 + 边界）+ 参考 + 可选建议」
- **Step C-5**：主流以代码块完整打印评论 body → AskUserQuestion 三选一确认 → POST 评论 → GET 回查；长 body（>3KB）走 POST 占位 + PATCH 完整 + GET 验证套路
- **Step C-6**：写日志 + 报告评论链接

> 关键不变量：Comment 路径不 fork、不切分支、不 commit、不在评论里塞代码 patch；分析过程中如果发现真的需要改代码，回到 Step 1.5 让用户切 PR，**抛弃只读目录重起一个非 readonly 工作目录**。

---

## Step 7 / C-6: 日志与最终报告

按 [gitcode-toolkit/references/logging-conventions.md](../gitcode-toolkit/references/logging-conventions.md) 写日志：

- 文件名：`logs/gitcode-issue-handler_{YYYYMMDD}_{HHMMSS}.log`
- PR 路径记录：Issue 摘要 / 模式判定结论与理由 / 工作目录 / 根因结论 / 修改文件清单 / 测试命令与结果 / commit sha / push 结果 / PR 链接
- Comment 路径记录：Issue 摘要 / 模式判定结论与理由 / 只读工作目录 / 分析要点（含 file:line）/ 评论 body 全文 / 评论链接
- **不写 token 明文**

最终向用户输出：

**PR 路径**：
```
Issue #${issue_number} 已处理完成
- Issue : ${issue_url}
- 分支  : ${type}/issue-${issue_number}-${slug}
- 提交  : <sha-short>  <subject>
- PR    : <pr_web_url>
- 日志  : logs/gitcode-issue-handler_xxx.log
```

**Comment 路径**：
```
Issue #${issue_number} 已答复（Comment 路径）
- Issue    : ${issue_url}
- 工作目录 : ${WORK_DIR}（只读）
- 评论     : <comment html_url 或 ${issue_url}#note_id>
- 日志     : logs/gitcode-issue-handler_xxx.log
```

---

## 异常处理与反模式

异常速查表、反模式列表、典型使用示例见 **[references/troubleshooting.md](references/troubleshooting.md)**。遇到 push 被拒 / clone 卡住 / 评论 GET 回查为空 / 用户在确认弹窗里要求"加点改动建议" / 选错路径中途切换等场景先查该文件。

---

## 完整流程检查清单

**共通**：

- [ ] 环境预检通过（token / git / curl / /tmp）
- [ ] Issue 标题/正文已抽取关键信号
- [ ] Step 1.5 模式判定结论已**在对话主流打印**并经用户确认（含「关键信号 + 推荐理由」）
- [ ] 写操作前每一步都用 AskUserQuestion 拿到了用户确认，且被审查内容在主流以代码块形式可见
- [ ] 日志已落盘（含模式判定结论 + 工作目录 + 最终产物链接）；不含 token 明文

**PR 路径专属**：

- [ ] git author 检查通过（user.name + user.email；仅 work_dir local，未污染全局 `~/.gitconfig`）
- [ ] fork ↔ upstream 关系已校验（`parent_full_name == upstream_full_name`，允许 fork 改名）
- [ ] base 分支已同步，工作分支已切出
- [ ] 根因方案经用户确认
- [ ] 改动仅限确认的文件，全部英文（代码、注释、log 文本）
- [ ] 测试通过或手工复现已记录
- [ ] commit message 英文，无 Co-Authored-By；body 只写代码改动本身，不含 `Fixes #N` 或 Issue 背景叙述（Issue 关联放到 PR body）
- [ ] PR 标题英文；PR body **沿用 `git show upstream/${base_branch}:.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md` 的章节标题原样**，仅替换 `<!-- ... -->` 占位
- [ ] 关联 Issue 一栏写 `Fixes #<issue_number>`；类型标签按 commit type 勾一项
- [ ] PR 创建后用 GET 回查 `body` / `description` 长度非 0 与预期一致（PATCH 200 不代表写入成功）

**Comment 路径专属**：

- [ ] 工作目录使用 `_readonly` 后缀，全程未发生 `git add` / `commit` / `push`
- [ ] 评论语言与 Issue 正文主语言一致
- [ ] 评论包含至少一处 `file:line` 引用 + 关键源码片段，不是泛泛而谈
- [ ] 评论 body 已**在对话主流以代码块形式打印**，再发 AskUserQuestion 确认
- [ ] 提交后 GET 回查评论 body 长度与内容均符合预期（长 body 需走 POST 占位 + PATCH 完整 + GET 验证）
- [ ] 没有在评论里写改动 patch / "建议把 X 行改成 Y"——那是 PR 路径的事
