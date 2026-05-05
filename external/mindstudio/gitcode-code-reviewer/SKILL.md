---
name: external-mindstudio-gitcode-code-reviewer
description: 用于审查 GitCode PR，并结合 PR metadata、diff 与整个代码仓上下文生成深度审查结论或发布逐行评论。当用户希望 review
  GitCode PR、检查某个 GitCode PR 链接、分析变更风险、或将审查意见发布到 GitCode PR 时使用。典型触发方式包括“review this
  PR”“检视这个 PR”“检查 PR”，或直接提供 GitCode PR 链接，例如 https://gitcode.com/owner/repo/pull/123
  。
original-name: gitcode-code-reviewer
synced-from: https://github.com/kali20gakki/mindstudio-skills
synced-date: '2026-05-05'
synced-commit: f6695da3cfa864a100db5a94f594b638aeff6389
license: UNKNOWN
---

# GitCode 代码审查助手

针对 GitCode PR 做“有上下文”的代码审查。`diff` 只是入口，不是全部证据；必须结合 PR 标题/描述、变更文件、调用链、相邻模块、历史实现、测试与配置来判断改动是否真的正确。

## 前置要求

### GitCode 访问令牌

本技能需要 GitCode 访问令牌，用于获取 PR 信息和发布评论。令牌可通过以下方式之一配置：

1. **环境变量**（推荐）：`GITCODE_TOKEN`
2. **Git 配置**：`git config --global gitcode.token <your-token>`

如果没有找到令牌，需要提示用户提供。

### 获取 GitCode 令牌

1. 登录 https://gitcode.com
2. 点击头像，进入“设置”
3. 打开“私人令牌”
4. 创建新令牌，并授予以下权限：
   - `pull_requests`（读写）
   - `issues`（读写）
   - `projects`（只读）

## 工作流程

### 1. 解析 PR 信息

当用户提供类似 `https://gitcode.com/Ascend/msprof/pull/109` 的 GitCode PR 链接时：

- 提取：`owner=Ascend`、`repo=msprof`、`pull_number=109`

### 2. 获取 PR 信息

使用 `scripts/fetch_pr_info.py` 获取 PR 详情：

```bash
python scripts/fetch_pr_info.py \
  --owner <OWNER> --repo <REPO> --pull-number <NUMBER> \
  [--token <TOKEN>] [--output-dir <DIR>]
```

该脚本会获取：

- PR 元数据，例如标题、描述、作者、状态
- 发生变更的文件及其 diff 内容
- PR 评论（可选）

### 3. 准备分析工作区

优先使用用户当前 workspace；如果当前目录已经是目标仓库，并且可以拿到 PR 对应分支或 commit，就直接在当前仓库分析，不要重复克隆。

只有在当前 workspace 不是目标仓库，或无法获取 PR 对应代码时，才克隆到临时目录：

```bash
export GITCODE_TOKEN="your-token"
TEMP_DIR="/tmp/gitcode-review-<REPO>-<TIMESTAMP>"
git clone --depth=50 "https://oauth2:${GITCODE_TOKEN}@gitcode.com/<OWNER>/<REPO>.git" "${TEMP_DIR}"
cd "${TEMP_DIR}"

# 获取 PR 分支
git fetch origin "refs/merge-requests/<NUMBER>/head:pr-<NUMBER>-source"
git checkout "pr-<NUMBER>-source"
```

### 4. 构建审查上下文（必做）

先读 `summary.json`、`pr_metadata.json`、`pr_files.json`、`pr_diff.patch`，再开始看代码。必须先回答下面几个问题：

1. **这个 PR 想解决什么问题**
   - 从标题、描述、标签、关联 issue、文件分布中推断目标，是修 bug、加功能、重构、补测试、回滚还是兼容性修复。
   - 如果 PR 描述不足，要根据修改内容自行归纳目的，并在结论里说明你的判断依据。
2. **改动落在系统的哪一层**
   - 识别变更文件分别属于入口层、核心逻辑层、数据模型层、接口/协议层、配置/构建层、测试层、生成产物。
   - 如果是生成文件、lockfile、SDK 产物，继续追到“源头文件”审查，不要只评论生成结果。
3. **这些改动与仓库中的哪些实现有关**
   - 查调用方、被调用方、同类实现、适配层、配置项、feature flag、schema、迁移脚本、测试夹具、文档。
   - 审查时必须看修改行的上下文文件，不要只看 patch hunk。
4. **基线行为是什么**
   - 通过 base 分支代码、历史提交或已有测试确认原本行为，再判断这次改动是否破坏兼容性或遗漏分支。

推荐优先使用这些工具建立上下文：

- `rg -n "<symbol>|<config>|<endpoint>" .`：查调用链、配置项、同类实现、测试覆盖
- `git diff --stat <base_sha>...<head_sha>`：快速识别高风险改动面
- `git diff --name-only <base_sha>...<head_sha>`：按模块分组变更文件
- `git show <base_sha>:<path>`：对比基线实现
- `git log --follow -- <path>`：看文件演进和历史意图
- `git blame -L <start>,<end> -- <path>`：定位历史上下文和责任归属
- 项目自带测试/构建命令：验证推断是否成立

如果某个结论依赖推断而不是直接证据，必须明确写出条件，例如“如果这里允许空集合输入，那么当前实现会……”。不要把猜测写成确定事实。

### 5. 深入审查代码

在完成上下文构建后，从以下维度审查，并尽量结合项目实际实现给出判断：

- **目标一致性**：实现是否真的满足 PR 目标，是否只改了表层而漏掉调用链上的必要改动
- **正确性与回归风险**：老行为是否被意外改变，边界条件、异常路径、状态切换是否完整
- **契约兼容性**：函数签名、接口字段、返回值、序列化格式、错误码、配置项是否与上下游兼容
- **项目一致性**：是否偏离仓库中已有模式，导致后续维护成本升高或行为不一致
- **性能与资源**：是否引入不必要的遍历、重复 I/O、锁竞争、缓存失效、N+1 调用等问题
- **安全性**：输入校验、权限边界、敏感数据处理、命令/SQL/路径注入等风险
- **可测试性**：关键分支是否有测试支撑；没有测试时，至少指出应补哪类用例

### 6. 整理审查结论

按照严重程度组织问题：

- **严重**：会导致 bug、安全问题或破坏性变更
- **建议**：建议优化代码质量、性能或实现方式
- **提示**：格式、风格或其他可选的小问题

默认聚焦真正影响行为、可靠性、兼容性、可维护性的发现；不要把纯风格意见堆成主要结论。

每条发现必须包含：

- 文件路径
- 行号
- 严重程度
- **四段式结构**：
  1. **严重程度**：明确标注问题级别，并与实际影响一致
  2. **问题**：具体指出当前代码存在什么问题
  3. **原因**：解释为什么这是个问题，会带来什么风险或影响，并尽量说明它和仓库中哪段实现、哪条调用链、哪类已有约束相关
  4. **怎么改**：给出可执行的修改建议，必要时附代码示例

额外要求：

- 不要只复述 diff 表面现象；每条重要评论都应体现你查过周边上下文
- 如果问题来自“改动不完整”，说明还缺哪些文件、分支、调用点或测试
- 如果没有发现问题，要明确说出你检查过哪些关键上下文，而不是只说“LGTM”

### 7. 先向用户展示审查结果

在正式发布评论前，先将审查结果展示给用户：

```markdown
## 代码审查摘要

**PR**： [标题](URL)  
**作者**：@author  
**结论**： [通过 / 请求修改]

### 发现的问题

#### 🔴 严重问题（N 个）
1. **文件**：`path/to/file.py`（第 42 行）
   - **严重程度**：严重
   - **问题**：描述问题
   - **原因**：解释原因
   - **怎么改**：提供代码示例

#### 🟡 改进建议（N 个）
...

#### 🟢 细节建议（N 个）
...

### 总结
[给出清晰结论及其原因]
```

在摘要中，除了列问题，还应补一小段“审查覆盖面”，简述你看过的关键上下文，例如“检查了调用方 `A/B`、对照了基线实现 `C`、确认了测试 `D` 的覆盖情况”。

### 8. 将评论发布到 PR（可选）

先询问用户：“是否要把这些发现作为逐行评论发布到 PR？”

如果用户确认，则创建评论 JSON 文件，并使用 `scripts/post_pr_comment.py` 发布：

```bash
python scripts/post_pr_comment.py \
  --owner <OWNER> --repo <REPO> --pull-number <NUMBER> \
  --comments-file /tmp/pr_comments.json \
  --inline
```

**评论正文示例（四段式）**：

````markdown
**严重程度：** 建议

**问题：** 代码中存在不规范的空格

**原因：** `node = int (node_str)` 不符合 PEP 8 规范，影响可读性

**怎么改：**
```python
# 修改前
node = int (node_str)

# 修改后
node = int(node_str)
```
````

优先在 `comments.json` 中使用结构化字段：`severity`、`problem`、`reason`、`fix`。  
`scripts/post_pr_comment.py` 会在发布前将这些字段统一整理为四段式 Markdown 正文，默认输出中文严重程度（`严重`、`建议`、`提示`），也会兼容旧版英文值和旧版三段式 `body` 内容。

### 9. 清理临时目录

审查完成后，清理临时仓库：

```bash
rm -rf /tmp/gitcode-review-<REPO>-<TIMESTAMP>
```

## 审查语气要求

- 保持建设性、专业、友好
- 解释清楚为什么需要修改
- 如果给出通过结论，要明确指出本次提交的价值
- 使用尊重式表达，避免命令式语气
- 所有发现都必须使用**四段式结构**
- 一条评论只聚焦一个问题，严重程度要和实际风险一致
- 行级评论尽量具体、可执行，避免只有结论没有修改方向
- 优先指出真正影响业务或系统行为的问题，少给脱离项目上下文的泛泛建议

## 审查底线

- 不要只基于 diff 下结论；diff 只是定位入口
- 不要为了凑数量输出低价值评论
- 不要忽略测试、配置、调用链和兼容性影响
- 不确定时先补代码搜索和上下文验证，再决定是否提出问题

## 脚本说明

### fetch_pr_info.py

用于从 GitCode API 获取 PR 信息。

```bash
python scripts/fetch_pr_info.py \
  --owner OWNER \
  --repo REPO \
  --pull-number NUMBER \
  [--token TOKEN] \
  [--output-dir DIR] \
  [--include-comments]
```

### post_pr_comment.py

用于向 GitCode PR 发布评论。

**发布逐行行级评论：**

```bash
python scripts/post_pr_comment.py \
  --owner OWNER \
  --repo REPO \
  --pull-number NUMBER \
  --comments-file COMMENTS_JSON \
  --inline
```

**发布整体审查结论：**

```bash
python scripts/post_pr_comment.py \
  --owner OWNER \
  --repo REPO \
  --pull-number NUMBER \
  --body "审查总结" \
  --review-event COMMENT|APPROVE|REQUEST_CHANGES
```

### setup_token.py

用于配置并验证 GitCode 访问令牌。

```bash
# 交互式配置
python scripts/setup_token.py

# 仅验证现有令牌
python scripts/setup_token.py --verify-only
```

## 参考资料

- GitCode API 文档：见 [references/gitcode_api.md](references/gitcode_api.md)
- 如需查看 API 细节，按需阅读对应参考文件

## 技术说明

### API 版本

GitCode 同时支持 v4（兼容 GitLab）和 v5（兼容 GitHub）两套 API：

- **v5 API**：用于获取 PR 信息、文件列表和普通评论
  - 基础地址：`https://api.gitcode.com/api/v5`
  - 认证方式：`PRIVATE-TOKEN` 请求头

- **v4 API**：用于创建逐行行级评论
  - 基础地址：`https://api.gitcode.com/api/v4`
  - 认证方式：`PRIVATE-TOKEN` 请求头
  - 接口：`POST /projects/{encoded_project}/merge_requests/{mr_iid}/discussions`

### 评论 JSON 格式

```json
[
  {
    "path": "relative/path/to/file.py",
    "line": 42,
    "severity": "严重",
    "problem": "描述问题",
    "reason": "解释为什么这是个问题",
    "fix": "给出修改建议",
    "side": "RIGHT"
  }
]
```

- `path`：相对仓库根目录的文件路径
- `line`：新文件中的行号
- `severity`：问题级别，建议使用 `严重`、`建议`、`提示`；脚本兼容旧值 `Critical`、`Improvement`、`Nitpick`
- `problem`：当前代码具体有什么问题
- `reason`：为什么这是个问题，以及会产生什么影响
- `fix`：建议如何修改，可附带 Markdown 代码示例
- `body`：可选的旧版输入；如果提供，脚本会将其规范化为四段式 `严重程度`、`问题`、`原因`、`怎么改`
- `side`：`"RIGHT"` 表示新代码，`"LEFT"` 表示被删除的旧代码
