# 环境配置

## GITHUB_TOKEN

```bash
# Windows PowerShell
$env:GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxx"

# Windows CMD
set GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx

# Linux/macOS
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

**Token 权限要求**：只需 `public_repo` 读取权限（或 `repo` 权限用于私有仓库）。

未设置 `GITHUB_TOKEN` 时，GitHub API 每小时仅允许 60 次请求，可能无法完成两个仓库的全量拉取。
