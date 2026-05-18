# GitCode API 参考

GitCode API 同时支持 GitLab v4 API 和 GitHub v5 API 格式。

## 基础信息

- **v5 API 基础 URL**: `https://api.gitcode.com/api/v5` (GitHub 兼容)
- **v4 API 基础 URL**: `https://api.gitcode.com/api/v4` (GitLab 兼容)
- **认证方式**: `PRIVATE-TOKEN` Header

## 认证

### 获取 Token

1. 登录 GitCode
2. 进入个人设置 -> 私人令牌 (Personal Access Token)
3. 生成新令牌，勾选所需权限：
   - `pull_requests` (读取和写入)
   - `issues` (读取和写入)
   - `projects` (读取)

### 使用 Token

在请求头中添加：
```
PRIVATE-TOKEN: YOUR_TOKEN
```

## v5 API (GitHub 兼容)

### PR 相关

#### 获取 PR 详情
```
GET /repos/{owner}/{repo}/pulls/{number}
```

响应示例：
```json
{
  "id": 12345,
  "number": 109,
  "state": "open",
  "title": "PR标题",
  "head": {
    "sha": "abc123...",
    "ref": "feature-branch"
  },
  "base": {
    "sha": "def456...",
    "ref": "main"
  }
}
```

#### 获取 PR 文件列表
```
GET /repos/{owner}/{repo}/pulls/{number}/files
```

#### 获取 PR Diff
```
GET /repos/{owner}/{repo}/pulls/{number}/diff
```

#### 获取 PR 评论
```
GET /repos/{owner}/{repo}/pulls/{number}/comments
```

#### 创建 PR 普通评论
```
POST /repos/{owner}/{repo}/pulls/{number}/comments
```

请求体：
```json
{
  "body": "Comment text"
}
```

#### 创建 PR Review
```
POST /repos/{owner}/{repo}/pulls/{number}/reviews
```

请求体：
```json
{
  "body": "Review summary",
  "event": "COMMENT"
}
```

`event` 可选值: `COMMENT`, `APPROVE`, `REQUEST_CHANGES`

## v4 API (GitLab 兼容) - 行级评论

### 创建行级评论（内联评论）

GitCode 使用 GitLab v4 API 的 Discussions 功能来创建行级评论。

```
POST /projects/{encoded_project}/merge_requests/{mr_iid}/discussions
```

**URL 编码**: 项目路径 `owner/repo` 需要进行 URL 编码，例如 `Ascend/msprof` → `Ascend%2Fmsprof`

**Headers**:
```
PRIVATE-TOKEN: your_token
Content-Type: application/json
```

**请求体**:
```json
{
  "body": "**严重程度：** 建议\n\n**问题：** 此处存在不必要的空格。\n\n**原因：** `node = int (node_str)` 不符合 PEP 8，降低可读性，也容易让后续风格检查失败。\n\n**怎么改：**\n```python\nnode = int(node_str)\n```",
  "position": {
    "position_type": "text",
    "base_sha": "abc123...",
    "head_sha": "def456...",
    "start_sha": "abc123...",
    "new_path": "src/main.py",
    "new_line": 42
  }
}
```

**Position 对象说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `position_type` | string | 固定为 `"text"` |
| `base_sha` | string | PR base 分支的 commit SHA |
| `head_sha` | string | PR head 分支的 commit SHA |
| `start_sha` | string | 通常与 `base_sha` 相同 |
| `new_path` | string | 文件路径（新文件） |
| `new_line` | int | 行号（新文件的行号） |

**针对删除的代码行**:

如果要评论被删除的代码行，使用 `old_path` 和 `old_line`：

```json
{
  "body": "Comment on deleted line",
  "position": {
    "position_type": "text",
    "base_sha": "abc123...",
    "head_sha": "def456...",
    "start_sha": "abc123...",
    "old_path": "src/main.py",
    "old_line": 10
  }
}
```

### Python 示例

```python
import urllib.request
import json
from urllib.parse import quote

TOKEN = "your_token"
API_V4_BASE = "https://api.gitcode.com/api/v4"

def post_inline_comment(
    project_id: str,  # "owner/repo"
    mr_iid: int,
    path: str,
    line: int,
    body: str,
    sha_info: dict
):
    encoded_project = quote(project_id, safe="")
    url = f"{API_V4_BASE}/projects/{encoded_project}/merge_requests/{mr_iid}/discussions"
    
    headers = {
        "PRIVATE-TOKEN": TOKEN,
        "Content-Type": "application/json"
    }
    
    position = {
        "position_type": "text",
        "base_sha": sha_info["base_sha"],
        "head_sha": sha_info["head_sha"],
        "start_sha": sha_info["start_sha"],
        "new_path": path,
        "new_line": line
    }
    
    payload = {
        "body": body,
        "position": position
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))

# 使用示例
sha_info = {
    "base_sha": "f87b61f23de13c16e1ad201dd5c9615ca7cebf95",
    "head_sha": "e90df8b338f994d00bbf1c79f14c90787603b470",
    "start_sha": "f87b61f23de13c16e1ad201dd5c9615ca7cebf95"
}

result = post_inline_comment(
    "Ascend/msprof",
    109,
    "misc/host_analyzer/cpu_binder/cpu_binder.py",
    226,
    "**严重程度：** 建议\n\n**问题：** 存在不规范的空格。\n\n**原因：** 当前写法不符合 PEP 8，会影响代码风格一致性。\n\n**怎么改：**\n```python\nnode = int(node_str)\n```",
    sha_info
)
```

## 注意事项

1. **API 版本选择**:
   - 获取 PR 信息、文件列表：使用 v5 API
   - 创建行级评论：使用 v4 API (Discussions)

2. **SHA 值获取**:
   - 从 PR 详情接口获取 `head.sha` 和 `base.sha`
   - `start_sha` 通常与 `base_sha` 相同

3. **错误处理**:
   - 400 Bad Request: 参数错误或 SHA 不匹配
   - 401 Unauthorized: Token 无效或权限不足
   - 404 Not Found: PR 或项目不存在

4. **Rate Limiting**:
   - 注意 API 调用频率限制
   - 建议在连续调用之间添加短暂延迟

## 参考链接

- GitCode API 文档: https://docs.gitcode.com/
- GitLab API 文档: https://docs.gitlab.com/ce/api/
- GitHub API 文档: https://docs.github.com/en/rest
