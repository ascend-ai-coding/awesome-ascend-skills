# GitCode 合入流程脚本使用说明

本目录包含 Python 脚本，用于执行 GitCode 仓库的合入流程操作。这些脚本使用 requests 库，可在 Linux 和 Windows 上运行。

## 脚本列表

### 1. create_issue.py
**作用**：在开源仓中创建 Issue

**用法**：
```bash
python create_issue.py <access_token> <upstream_repo_info> <issue_title> <issue_body>
```
- `access_token`: GitCode API 访问令牌
- `upstream_repo_info`: 开源仓信息，格式为 `owner/repo`
- `issue_title`: Issue 标题
- `issue_body`: Issue 描述

**返回**：JSON 格式的 Issue 信息

### 2. create_pr.py
**作用**：从个人仓向开源仓创建 PR

**用法**：
```bash
python create_pr.py <access_token> <personal_repo_info> <upstream_repo_info> <branch_name> <target_branch> <pr_title> <pr_body> [close_related_issue]
```
- `access_token`: GitCode API 访问令牌
- `personal_repo_info`: 个人仓信息，格式为 `owner/repo`
- `upstream_repo_info`: 开源仓信息，格式为 `owner/repo`
- `branch_name`: 源分支名称
- `target_branch`: 目标分支名称
- `pr_title`: PR 标题
- `pr_body`: PR 描述
- `close_related_issue`: 是否关闭关联 Issue，可选，默认 `true`

**返回**：JSON 格式的 PR 信息

### 3. get_pr_details.py
**作用**：获取 PR 详情

**用法**：
```bash
python get_pr_details.py <access_token> <upstream_repo_info> <pr_number>
```
- `access_token`: GitCode API 访问令牌
- `upstream_repo_info`: 开源仓信息，格式为 `owner/repo`
- `pr_number`: PR 编号

**返回**：JSON 格式的 PR 详情

### 4. get_pr_comments.py
**作用**：获取 PR 评论

**用法**：
```bash
python get_pr_comments.py <access_token> <upstream_repo_info> <pr_number> [comment_type] [page] [per_page]
```
- `access_token`: GitCode API 访问令牌
- `upstream_repo_info`: 开源仓信息，格式为 `owner/repo`
- `pr_number`: PR 编号
- `comment_type`: 评论类型，可选，`diff_comment` 或 `pr_comment`
- `page`: 页码，可选，默认 `1`
- `per_page`: 每页数量，可选，默认 `100`

**返回**：JSON 格式的评论列表

### 5. merge_pr.py
**作用**：合并 PR

**用法**：
```bash
python merge_pr.py <access_token> <upstream_repo_info> <pr_number> [merge_method]
```
- `access_token`: GitCode API 访问令牌
- `upstream_repo_info`: 开源仓信息，格式为 `owner/repo`
- `pr_number`: PR 编号
- `merge_method`: 合并方式，可选，默认 `squash`

**返回**：JSON 格式的合并结果

### 6. check_pipeline_status.py
**作用**：检查流水线状态（轮询）

**用法**：
```bash
python check_pipeline_status.py <access_token> <upstream_repo_info> <pr_number> [max_attempts] [interval_seconds]
```
- `access_token`: GitCode API 访问令牌
- `upstream_repo_info`: 开源仓信息，格式为 `owner/repo`
- `pr_number`: PR 编号
- `max_attempts`: 最大尝试次数，可选，默认 `20`
- `interval_seconds`: 轮询间隔，可选，默认 `30`

**返回**：成功返回 0，失败返回 1

## 依赖

- Python 3.6+
- requests 库

## 安装依赖

```bash
pip install requests
```

## 使用示例

```bash
# 读取 token
access_token=$(cat token)

# 创建 Issue
issue_response=$(python scripts/create_issue.py "$access_token" "owner/repo" "Bug fix" "Fixes a critical bug")
issue_number=$(echo "$issue_response" | grep -o '"number":[0-9]*' | cut -d':' -f2)

# 创建 PR
pr_response=$(python scripts/create_pr.py "$access_token" "your-owner/your-fork" "owner/repo" "feature-branch" "master" "Fix bug" "Closes #$issue_number")
pr_number=$(echo "$pr_response" | grep -o '"number":[0-9]*' | cut -d':' -f2)

# 检查流水线
python scripts/check_pipeline_status.py "$access_token" "owner/repo" "$pr_number"

# 合并 PR
if [ $? -eq 0 ]; then
  merge_response=$(python scripts/merge_pr.py "$access_token" "owner/repo" "$pr_number")
  echo "PR merged successfully!"
fi
```

## 注意事项

- 所有脚本都需要 `access_token` 参数
- 仓库信息格式必须为 `owner/repo`
- API 调用可能会触发限流，建议适当控制调用频率
- 脚本执行失败时会抛出异常