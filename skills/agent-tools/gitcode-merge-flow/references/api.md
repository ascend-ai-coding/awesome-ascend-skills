# GitCode API 参考

## API 速查表

| 操作 | 方法 | URL |
|------|------|-----|
| 创建 Issue | POST | `https://api.gitcode.com/api/v5/repos/:owner/issues` |
| 创建 PR | POST | `https://api.gitcode.com/api/v5/repos/:owner/:repo/pulls` |
| 关联 PR 和 Issue | POST | `https://api.gitcode.com/api/v5/repos/:owner/:repo/pulls/:number/issues` |
| 获取 PR 详情 | GET | `https://api.gitcode.com/api/v5/repos/:owner/:repo/pulls/:number` |
| 获取 PR 评论 | GET | `https://api.gitcode.com/api/v5/repos/:owner/:repo/pulls/:number/comments` |
| 获取评论表态 | GET | `https://api.gitcode.com/api/v5/repos/:owner/:repo/pulls/comment/:comment_id/user_reactions` |
| 合并 PR | PUT | `https://api.gitcode.com/api/v5/repos/:owner/:repo/pulls/:number/merge` |

## 官方文档链接

- **创建 Issue**: https://docs.gitcode.com/docs/apis/post-api-v-5-repos-owner-issues
- **创建 Pull Request**: https://docs.gitcode.com/docs/apis/post-api-v-5-repos-owner-repo-pulls
- **关联 PR 和 Issue**: https://docs.gitcode.com/docs/apis/post-api-v-5-repos-owner-repo-pulls-number-linked-issues
- **获取 PR 评论**: https://docs.gitcode.com/en/docs/apis/get-api-v-5-repos-owner-repo-pulls-number-comments/
- **获取评论表态**: https://docs.gitcode.com/docs/apis/get-api-v-5-repos-owner-repo-pulls-comment-comment-id-user-reactions
- **合并 PR**: https://docs.gitcode.com/docs/apis/put-api-v-5-repos-owner-repo-pulls-number-merge

## 请求参数说明

### 创建 Issue
- `access_token`: 用户授权码（必需）
- `repo`: 仓库名称（必需）
- `title`: Issue 标题（必需）
- `body`: Issue 描述（必需）
- `assignee`: 指派人员（可选）
- `labels`: 标签（可选）

### 创建 PR
- `access_token`: 用户授权码（必需）
- `title`: PR 标题（必需）
- `head`: 源分支，格式为 `owner:branch`（必需）
- `base`: 目标分支（必需）
- `body`: PR 描述（必需）
- `fork_path`: 个人仓完整路径，格式为 `owner/repo`（必需）
- `close_related_issue`: 是否关闭关联 Issue（可选，默认 true）

### 关联 PR 和 Issue
- `access_token`: 用户授权码（必需，通过 `private-token` header 传递）
- `number`: PR 编号（路径参数）
- Request Body: Issue 编号数组，如 `[1, 2]`

**注意**：此 API 需要仓库维护者权限，普通贡献者可能无法执行。

### 合并 PR
- `access_token`: 用户授权码（必需）
- `merge_method`: 合并方式（可选，默认 merge）
  - `merge`: 合并所有提交
  - `squash`: 扁平化合并
  - `rebase`: 变基合并

## 响应字段说明

### Issue 响应
- `number`: Issue 编号
- `title`: 标题
- `body`: 描述
- `state`: 状态
- `html_url`: Issue 页面链接

### PR 响应
- `number`: PR 编号
- `title`: 标题
- `body`: 描述
- `state`: 状态
- `head`: 源分支信息
- `base`: 目标分支信息
- `html_url`: PR 页面链接
- `mergeable`: 是否可合并
- `approval_reviewers`: 评审人信息

### 评论响应
- `id`: 评论 ID
- `body`: 评论内容
- `user`: 用户信息
- `created_at`: 创建时间
- `resolved`: 是否已解决（仅代码行评论）
- `comment_type`: 评论类型（diff_comment/pr_comment）

## 常见错误码

- **401 Unauthorized**: access_token 无效
- **403 Forbidden**: 权限不足
- **404 Not Found**: 仓库或资源不存在
- **422 Unprocessable Entity**: 请求参数错误