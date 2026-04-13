---
name: gitcode-merge-flow
description: "Automates GitCode open-source repo merge workflow: commit → push → issue → PR → pipeline → review → /lgtm & /approve → merge. Invoke when user wants to submit code to a GitCode open-source repository."
---

# GitCode 开源仓合入流程自动化

自动化完成从本地提交到代码合入开源仓的完整流程。当需要向 GitCode 开源仓库贡献代码时调用此 Skill。

## 配置要求

执行前必须确认以下配置：

1. **GITCODE_ACCESS_TOKEN**: GitCode API 访问令牌，从项目根目录的 `token` 文件读取，或由用户提供
2. **个人仓**: 用户 fork 的个人仓库（格式：`owner/repo`，如 `skywang2/skill-test`）
3. **开源仓**: 上游开源仓库（格式：`owner/repo`，如 `lyzin/111`）
4. **GITCODE_API_BASE**: API 基础地址，默认 `https://api.gitcode.com/api/v5`

## 完整流程

```
flowchart TD
    A["本地提交"] --> B["远程推送"]
    B --> C["提交Issue"]
    C --> D["提交PR"]
    D --> E["触发流水线"]
    E --> F{"存在流水线失败问题?"}
    F -->|是| G["解决流水线失败问题"]
    G --> E
    F -->|否| H["等待代码评审"]
    H --> I{"存在未解决评审意见?"}
    I -->|是| J["解决评审意见"]
    J --> H
    I -->|否| K{"存在新提交解决的评审意见?"}
    K -->|是| E
    K -->|否| L["等待/lgtm和/approve"]
    L --> M["确认合入"]
```

## 执行步骤

### 步骤1: 前置检查

1. 读取项目根目录 `token` 文件获取 `GITCODE_ACCESS_TOKEN`，若不存在则向用户索取
2. 确认当前工作目录是一个 git 仓库（检查 `.git` 目录是否存在）
3. 向用户确认以下信息（若未提供则询问）：
   - **个人仓** (`owner/repo` 格式): 如 `skywang2/skill-test`
   - **开源仓** (`owner/repo` 格式): 如 `lyzin/111`
   - **目标分支**: 默认 `master`
   - **当前工作分支**: 默认当前 git 分支名
   - **Issue 标题和描述**（参考 `references/issue-templates.md` 中的模板）
   - **PR 标题**
   - **PR 描述**（参考 `references/pr-template.md` 中的模板）

### 步骤2: 本地提交

1. 运行 `git status` 检查是否有未暂存的更改
2. 如果有更改：
   - 运行 `git add -A` 暂存所有更改（或按用户指定文件）
   - 运行 `git commit -m "<commit_message>"` 提交，commit_message 由用户提供或根据更改自动生成
3. 如果没有更改，提示用户确认是否继续

### 步骤3: 远程推送

1. 确认个人仓远程已配置：`git remote -v`
2. 如果没有个人仓远程，添加：`git remote add origin https://gitcode.com/<personal_owner>/<personal_repo>.git`
3. 推送到个人仓：`git push origin <current_branch>`
4. 如果推送失败（如远程拒绝），处理冲突后重试

### 步骤4: 提交 Issue

调用 GitCode API 在**开源仓**创建 Issue。

**API 详情**：请参考 `references/api.md` 中的「创建 Issue」部分。

**执行脚本**：使用 `scripts/create_issue.py` 脚本。

记录返回的 Issue 编号（`number` 字段），后续创建 PR 时关联。

### 步骤5: 提交 PR

调用 GitCode API 从个人仓向开源仓创建 PR。

**API 详情**：请参考 `references/api.md` 中的「创建 PR」部分。

**执行脚本**：使用 `scripts/create_pr.py` 脚本。

**【重要】提交数限制**：
- 如果当前分支的提交数大于 1，**必须使用 squash 合并**
- 在创建 PR 前检查本地提交数量：`git log --oneline origin/main..main`
- 若提交数 > 1，提示用户需要 squash 或 rebase 后再创建 PR

记录返回的 PR 编号（`number` 字段），后续步骤均依赖此编号。

### 步骤5.5: 关联 PR 和 Issue（可选）

如果需要显式关联 PR 和 Issue，可以调用关联 API。

**API 详情**：请参考 `references/api.md` 中的「关联 PR 和 Issue」部分。

**执行脚本**：使用 `scripts/link_pr_issue.py` 脚本。

**注意**：此操作需要仓库维护者权限，普通贡献者可能无法执行。创建 PR 时设置 `close_related_issue: true` 可以自动关联 Issue。

### 步骤6: 检查流水线状态

PR 创建后流水线会自动触发。获取流水线状态并**直接挂起**，让用户自行确认。

**API 详情**：请参考 `references/api.md` 中的「获取 PR 详情」部分。

**执行脚本**：使用 `scripts/get_pr_details.py` 脚本（字段 `mergeable_state.ci_state_passed`）。

**【挂起反馈示例 - 步骤6】**：
```
⏸️ 流程已挂起 - 请确认流水线状态

📋 当前状态：
   • PR 链接: https://gitcode.com/{owner}/{repo}/merge_requests/{number}
   • 流水线状态: ✅ 通过 / ❌ 失败 / ⏳ 运行中
   • 合并状态: ✅ 可合并 / ❌ 不可合并

⏳ 请确认：
   请在 PR 页面确认流水线是否通过
   流水线通过后请告知我继续流程
   若流水线失败，请在修复后告知我

📝 流水线失败处理：
   1. 修复代码问题
   2. 执行 git add → git commit → git push 推送新提交
   3. 新提交会自动重新触发流水线
   4. 流水线通过后告知我继续
```

### 步骤7: 等待代码评审

**【强制要求】无论仓库配置如何，必须等待至少一个维护者进行代码评审通过后方可继续。**

获取 PR 的评审评论并检查评审状态。

**API 详情**：请参考 `references/api.md` 中的「获取 PR 评论」部分。

**执行脚本**：使用 `scripts/get_pr_comments.py` 脚本。

**评审状态判断逻辑**：
1. 如果没有任何评审评论 → 尚未有人评审，**强制挂起流程**
2. 如果存在 `resolved: false` 的评论 → 有未解决的评审意见
3. 如果所有评论 `resolved: true` → 评审意见已全部解决

**【挂起反馈示例 - 步骤7】**：
当流程因等待评审而挂起时，必须向用户清晰展示：
```
⏸️ 流程已挂起 - 等待代码评审

📋 当前状态：
   • PR 链接: https://gitcode.com/{owner}/{repo}/merge_requests/{number}
   • 流水线状态: ✅ 通过
   • 评审状态: ❌ 暂无评审评论 / ⚠️ 有未解决评审意见

⏳ 等待操作：
   需要至少一个维护者进行代码评审

📝 下一步：
   请联系仓库维护者进行评审
   评审完成后请告知我，我将检查评审状态并继续流程
```

**存在未解决评审意见**：
1. 列出所有未解决的评审意见（评论内容和作者）
2. **强制挂起流程**
3. 用户修改代码后，执行 `git add` → `git commit` → `git push` 推送新提交
4. 新提交推送后，告知我重新检查

### 步骤8: 等待 /lgtm 和 /approve

**【强制要求】无论仓库配置如何，必须获得至少一个 `/lgtm` 或 `/approve` 评论表态后方可合入。**

检查 PR 评论，确认是否有人通过评论 `/lgtm` 或 `/approve` 表示同意合入。

**执行脚本**：使用 `scripts/get_pr_comments.py` 脚本。

**判断逻辑**：
- 如果未找到 `/lgtm` 或 `/approve` → **挂起流程**
- 如果找到 `/lgtm` 或 `/approve` → 进入步骤9

**【挂起反馈示例 - 步骤8】**：
```
⏸️ 流程已挂起 - 等待 /lgtm 和 /approve 审批

📋 当前状态：
   • PR 链接: https://gitcode.com/{owner}/{repo}/merge_requests/{number}
   • 流水线状态: ✅ 通过
   • 评审状态: ✅ 已通过
   • 审批状态: ❌ 暂无 /lgtm 或 /approve

⏳ 等待操作：
   需要仓库维护者通过评论 /lgtm 或 /approve 表示同意合入

📝 下一步：
   请仓库维护者在 PR 页面评论 /lgtm 或 /approve
   获得审批后请告知我，我将执行合并
```

### 步骤10: 确认合入

所有条件满足后，调用 API 合入 PR。

**API 详情**：请参考 `references/api.md` 中的「合并 PR」部分。

**执行脚本**：使用 `scripts/merge_pr.py` 脚本。

**【重要】合并方式**：
- **提交数大于 1 时，必须使用 squash 合并**
- `merge_pr.py` 脚本会自动检查 PR 的提交数量，若大于 1 则强制使用 squash
- 提交数 = 1 时，可使用 merge 或 squash（默认 squash）

合入成功后：
1. 告知用户 PR 已成功合入
2. 提供合入后的 PR 链接
3. 建议用户同步本地主分支：`git pull upstream master`

## 挂起与恢复机制

流程中以下步骤需要等待第三方人工操作，会**直接挂起**：

| 挂起点 | 等待原因 | 恢复条件 |
|--------|----------|----------|
| 步骤6: 流水线检查 | 需要用户确认流水线状态 | 用户确认流水线通过后告知我继续 |
| 步骤7: 等待评审 | 需要评审人审查代码 | 用户确认评审完成后告知我继续 |
| 步骤8: 等待 /lgtm 和 /approve | 需要维护者审批 | 用户确认已获得审批后告知我继续 |

**挂起时的行为**：
1. 明确告知用户当前挂起的原因
2. 告知用户需要等待什么操作
3. 提供相关链接（PR 页面、流水线页面等）
4. 用户确认后，从挂起点继续执行

## 错误处理

1. **API 调用失败**：检查 access_token 是否有效，网络是否通畅，参数是否正确
2. **Git 操作失败**：检查远程仓库权限、分支是否存在、是否有冲突
3. **PR 已关闭/拒绝**：告知用户 PR 状态，询问是否重新提交
4. **合并冲突**：提示用户解决冲突后重新推送

## 注意事项

1. 所有 API 调用使用 Python 的 `requests` 库
2. access_token 不得硬编码在代码中，必须从 token 文件读取或由用户输入
3. 推送代码时如需认证，使用 URL 嵌入 token 的方式：`https://oauth2:<token>@gitcode.com/<owner>/<repo>.git`
4. 创建 PR 时 `head` 参数必须包含个人仓 owner 前缀，格式为 `<owner>:<branch>`
5. 轮询流水线状态时注意频率控制，避免触发 API 限流
6. 需要安装依赖：`pip install requests`
