# PR 路径详细步骤

从 Step 1.5 判定为 PR 时进入。所有写操作前必须 AskUserQuestion 确认，一次只问一个；被审查的内容先在对话主流以代码块形式打印（不要塞进 AskUserQuestion 的 `preview` 字段）。

## Step 1.6: fork_url 缺失分支

从 Step 1.5 判定为 PR 后进入。**立即** AskUserQuestion，不要替用户默默 fork（违反"对外可见操作必须确认"原则）：

```
问题: 未检测到 fork 仓库地址，如何处理？
选项:
  - 我来提供 fork 链接（在下一条消息中粘贴）
  - 自动 fork 上游仓库到我账户下（推荐）
  - 取消本次操作
```

### 1.6.1 用户选「提供链接」

等用户在下一条消息中给出，按 Step 1.1 的校验逻辑校验 `fork_owner / fork_repo` 与上游 `parent_full_name` 一致（注意：fork 可能改过名，重点比 `parent_full_name`，不强求 repo 名一致），然后回到 Step 2。

### 1.6.2 用户选「自动 fork」

调用 GitCode Fork API（[gitcode-api.md](../../gitcode-toolkit/references/gitcode-api.md) 仓库部分）：

```bash
curl -X POST "https://api.gitcode.com/api/v5/repos/${upstream_owner}/${upstream_repo}/forks?access_token=${token}" \
  -H 'Content-Type: application/json' \
  --connect-timeout 60 --max-time 180
```

成功响应取以下字段拼出 `fork_url`：

- `namespace.path` 或 `owner.login` → `fork_owner`
- `path` 或 `name` → `fork_repo`
- 拼成 `https://gitcode.com/${fork_owner}/${fork_repo}.git`

**注意点**：

| 情况 | 处理 |
|------|------|
| HTTP 201 / 200 | fork 成功，记下 `fork_url` 进入 Step 2 |
| HTTP 409 / 422 提示已存在 | 该用户名下已有同名仓库，从当前 token 的用户身份推断 `fork_owner`（调 `/user`），拼出 `fork_url` 复用 |
| HTTP 403 | token 没有 fork 权限，AskUserQuestion 让用户换 token 或手动 fork 后回到 1.6.1 |
| fork 异步未就绪 | GitCode fork 可能延迟生效，`git ls-remote ${fork_url}` 失败则 `sleep 2` 重试，最多 5 次；仍失败提示用户稍后重试 |

### 1.6.3 用户选「取消」

记录日志后直接结束，不要带着不完整状态继续。

---

## Step 2: 克隆 fork、设置 upstream、同步

详细 git 命令见 [gitcode-toolkit](../../gitcode-toolkit/SKILL.md) 的速查表与 `references/clone-and-checkout.md`、`references/remote-and-branch.md`。

### 2.1 工作目录

```bash
timestamp=$(date +%Y%m%d_%H%M%S)
WORK_DIR="/tmp/gitcode-issue-handler_${upstream_repo}_${issue_number}_${timestamp}"
```

### 2.2 克隆 fork 并设置 upstream

```bash
git clone --depth=200 "${fork_url}" "$WORK_DIR"
cd "$WORK_DIR"

# Step 0 走 GIT_AUTHOR_SOURCE=user 分支时（即询问用户拿到的 name/email），
# 才在本仓库 local 写一次；走 global 分支则跳过——work_dir 会自动继承全局配置。
if [ "$GIT_AUTHOR_SOURCE" = "user" ]; then
  git config user.name  "$NAME_FROM_USER"
  git config user.email "$EMAIL_FROM_USER"
fi

# 添加上游
git remote add upstream "https://gitcode.com/${upstream_owner}/${upstream_repo}.git"
git fetch upstream --depth=200
```

### 2.3 同步 base 分支

```bash
# 推断 base 分支：优先用户参数 → Issue 中提及 → upstream HEAD（通常 master）
git checkout -B "${base_branch}" "upstream/${base_branch}"
git push origin "${base_branch}"   # 同步到 fork（可选；失败不致命，记日志即可）
```

如果 fork 上 `base_branch` 落后较多，仅同步到本地工作树，不强制推 fork（避免污染用户其他在飞的工作）。

### 2.4 切出工作分支

分支命名（英文）：`<type>/issue-${issue_number}-<slug>`，`<type>` 按 Issue 性质选用 Conventional Commits 类型——bug 类用 `fix`、新特性用 `feat`、文档用 `docs`、性能用 `perf`、重构用 `refactor`、测试用 `test`、其余用 `chore`；slug 来自 Issue 标题的英文化简写（如 Issue 中已有英文关键词则直接用）。例：

```
fix/issue-123-null-deref-in-tiling
feat/issue-456-add-bf16-support
docs/issue-789-clarify-tiling-api
```

```bash
git checkout -b "${type}/issue-${issue_number}-${slug}"
```

---

## Step 3: 根因分析（必须用户确认方案后再改）

这是整条流水线最容易"看似干活、其实跑偏"的步骤。原则：

1. **先读代码再下结论**：用 Read / Grep 去看 Issue 提到的文件、函数；不要凭 Issue 描述脑补实现。
2. **复现优先**：若 Issue 给了复现命令，先在本地复现一次确认失败现象；复现不出来时回到 AskUserQuestion 问用户补充环境信息。
3. **追根因，不打补丁**：找到症状后继续往上追一层"为什么会到这一步"，避免在调用点 try/catch 把异常吞掉。
4. **写下假设并验证**：把"我认为根因是 X，因为 Y"显式记到日志，再去看代码或跑命令验证 Y。

### 3.1 输出修复方案

向用户展示（AskUserQuestion）：

```
Issue #${issue_number} 根因初判
- 现象：<一两句话>
- 根因：<一两句话，定位到文件:行>
- 涉及文件：
    path/to/a.py
    path/to/b.py
- 修复策略：<一句话，体现最小修改>
- 风险/副作用：<已知的回归点；没有则写"无">

是否按此方案修改？
[确认执行] [我来调整方案] [取消]
```

用户选"我来调整"时，吸收用户输入后再问一次，**不要直接动手**。

---

## Step 4: 实施修改并跑测试

### 4.1 改代码（英文）

- 注释、变量名、log 文本统一英文。
- 遵循文件内既有风格（命名、缩进、错误处理范式）。
- 不顺手做无关重构、不"路过修一下别的 lint 警告"。
- 不引入新依赖；确实需要时回到 Step 3 重新跟用户确认方案。

### 4.2 跑测试

```bash
# 优先按仓库自身约定（README / Makefile / pytest.ini / package.json scripts）
# 没有约定时按语言常规：
#   Python:  pytest -x  或  python -m unittest
#   C/C++:   按 README 指引，通常有 build + ctest
#   Node:    npm test / yarn test
```

跑测原则：

- **只跑相关测试**：先跑覆盖到改动文件的测试，全量测试留到通过后再做（避免长跑遮蔽问题）。
- 失败时回到 Step 4.1 迭代，不要直接改测试断言来"过测"。
- 仓库本就没有测试时，至少手工复现一次"修改前失败 / 修改后通过"，把命令和输出记到日志。

### 4.3 自检 checklist

提交前自查：

- [ ] 只动了 Step 3 确认过的文件，没有意外文件改动（`git status` 核对）
- [ ] 改动是英文（含注释）
- [ ] 相关测试通过，或手工复现已记录
- [ ] 没有引入 `print` / debug 残留

---

## Step 5: Commit + Push

### 5.1 Commit message（英文，Conventional Commits）

模板：

```
<type>(<scope>): <subject>

<body: 仅描述这次提交改了什么、为什么这么改>
```

`<type>` 取值参考 [gitcode-toolkit](../../gitcode-toolkit/SKILL.md) 「PR 创建工作流 / Step 3」的类型标签推断规则：`fix` / `feat` / `perf` / `docs` / `refactor` / `test` / `chore`。按 Issue 性质选——bug 类 `fix`、新特性 `feat`、文档 `docs`，与 Step 2.4 切出的工作分支前缀保持一致。

> **硬性要求**：
> - commit message **不能包含** `Co-Authored-By`（CLAUDE.md 要求）。
> - commit body **只写代码改了什么、为什么这么改**：不写 `Fixes #N` / `Closes #N` / `相关 Issue: ...` 等 Issue 关联，也不复述 Issue 背景或现象——commit 是给看 git log 的人的代码变更说明，Issue 关联属 PR 元数据，写进 Step 6 的 PR body「关联的Issue」章节即可，混在一起会让 commit body 被背景叙述淹没。subject 描述改动，body 写"现在的实现做了什么、为什么这么改"，避免出现 Issue 编号 / 标题原文 / "该 Issue 报告…" 这类引导语。

### 5.2 用户确认 commit

**关键：commit message 必须以普通文本打印在对话主流，再用 AskUserQuestion 做"确认/修改/取消"。**

不要把多行 commit message 塞进 `AskUserQuestion` 的 option `preview` 字段——preview 仅在用户聚焦该选项时显示在侧栏，多数情况看不到，而 commit message 是关键审查对象，必须在主区域可见、可滚动、可复制。

操作模板：

1. 先在主对话用代码块明示完整 commit + 改动统计：

       即将创建以下 commit：
       ```
       <type>(<scope>): <subject>

       <body>
       ```

       改动统计：
       ```
       <git diff --stat 输出>
       ```

2. 再发 AskUserQuestion，选项只放"确认提交 / 修改 / 取消"，preview 字段留空或只放一行摘要，不要重复 commit 全文。

用户选"修改"则吸收意见后重复第 1 步重新展示，再问一次。push 前的二次确认（5.3）同理：先展示要 push 的分支名 + 远端目标，再问。

### 5.3 Push 到 fork

确认后：

```bash
git push -u origin "${type}/issue-${issue_number}-${slug}"
```

push 前**再问一次用户确认**（这是对外可见操作）。失败常见原因：

| 现象 | 处理 |
|------|------|
| `403` | token 没有 fork 仓库的写权限，回 Step 0 换 token |
| `rejected (non-fast-forward)` | fork 上同名分支有人推过，AskUserQuestion 让用户选「rebase 后强推 / 换分支名 / 取消」|
| 网络超时 | 重试一次；仍失败检查 gitcode.com 是否可达 |

---

## Step 6: 创建 PR（参考 gitcode-toolkit 的 PR 创建工作流）

按 [gitcode-toolkit](../../gitcode-toolkit/SKILL.md) 「PR 创建工作流」的 Step 5–7 完成推送与 PR 创建。本 skill 负责传入额外上下文：

- `head = ${fork_owner}/${fork_repo}:${type}/issue-${issue_number}-${slug}`
  - 注意 GitCode API 的 head 用 `owner/repo:branch` 格式（不是单纯 `owner:branch`）——当 fork 改过名时尤其重要，否则会 400 `Project not found`。
- `base = ${base_branch}`（默认 `master`）
- PR 标题：英文，沿用 commit subject（或更概括一层）

### 6.1 PR body：必须沿用仓库的 PR 模板

不要自创章节名。从上游仓库取 PR 模板原文，**只把模板中的 `<!-- 占位提示 -->` 注释替换为实际内容，标题与结构保持原样不动**。

为什么重要：维护者按「描述 / 关联的Issue / 测试 / 文档更新 / 类型标签」这些固定章节审查 PR；换成「Description / Related Issue / Tests / ...」的英文重命名版，会让人第一眼觉得不熟悉仓库规范，复核成本与被打回概率都上升。模板的每一章都是契约。

#### 6.1.1 取模板（按优先级）

```bash
# 在已 fetch 的上游里读，不要再调 API：
git show upstream/${base_branch}:.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md   # 最常见（GitCode 中文）
git show upstream/${base_branch}:.gitcode/PULL_REQUEST_TEMPLATE.md         # GitCode 默认
git show upstream/${base_branch}:.github/PULL_REQUEST_TEMPLATE.md          # GitHub 风格备选
```

三个都不存在再用 [gitcode-toolkit](../../gitcode-toolkit/SKILL.md) 「PR 创建工作流 / Step 2」中给出的兜底默认模板，并在日志里记一句"未找到仓库模板，使用默认"。

#### 6.1.2 填充规则

- **保留所有原章节标题**（含中文标题、表情符号、emoji）——不翻译、不增删、不重排顺序。
- **每个章节都要填**，模板中的注释占位 `<!-- ... -->` 全部替换成具体内容；模板没要求的章节不要凭空添加。
- 关联 Issue 一栏写 `Fixes #${issue_number}`（或仓库习惯的写法，比如 `#1511`、Issue 链接）。
- 类型标签按 commit `<type>` 勾选对应项：`fix:`→Bug修复、`feat:`→新特性、`perf:`→性能优化、`docs:`→文档更新、其他→其他。**只勾一项**，用 `[x]` 表示选中，其余保留 `[ ]`。
- 测试章节如实写出运行的命令与结果（包括 SKIP 的用例及原因）——不要空着、不要写"已测试"这种没信息量的话。
- 文档更新章节如无文档变化，明确写"无"或"None"而不是删掉章节。
- 章节标题保持模板原文（多半是中文），章节内容用英文，**和 commit message 风格一致**。这种"模板用原文 + 内容用英文"的组合在 GitCode 国际化仓库中是常见做法。

#### 6.1.3 GitCode API 设置 PR 描述的注意点

部分 GitCode API 路径不接受 POST 时直接带超长 body（曾观察到 400/截断）。稳妥流程：

1. POST `/repos/{owner}/{repo}/pulls` 创建时 body 用一句占位（如 `Fixes #${issue_number}`）。
2. 立刻 PATCH `/repos/{owner}/{repo}/pulls/{iid}` 写入完整模板 body：
   ```bash
   curl -X PATCH "https://api.gitcode.com/api/v5/repos/${upstream_owner}/${upstream_repo}/pulls/${iid}?access_token=${token}" \
     --data-urlencode "body=${full_body}" \
     --connect-timeout 60 --max-time 180
   ```
3. PATCH 响应有时不回显 body，需要 GET 一次确认 `body`/`description` 字段长度非 0 且和预期一致——不要看到 HTTP 200 就以为成功了。

### 6.2 用户确认

和 Step 5.2 一样：**先在对话主流用代码块明示**完整 PR 标题 + 完整 body + head/base 信息，再发 AskUserQuestion 做"创建/修改/取消"。preview 字段不放 body 全文。

成功后保留返回的 PR 链接（`web_url`）用于 Step 7 报告。
