# 手动提单：工作流程与约束

当用户直接描述问题时的交互式提单流程。

---

## 工作流程

### Step 1：目标确认

#### 1.1 确定处理模式

先判断用户要求：
- "只生成草稿 / 先别提交"：只生成草稿，不调用创建 Issue API
- "帮我提交 / 提一个 Issue"：生成草稿并在提交前再次确认

#### 1.2 确定目标仓库

按优先级确定 owner/repo：

| 优先级 | 条件 | 行为 |
|--------|------|------|
| 1 | 调用方传入 `target_repo` | 规范化并使用该仓库 |
| 2 | 用户文本中包含 GitCode 仓库 URL 或 `{owner}/{repo}` | 提取并确认 |
| 3 | 当前目录存在 git remote | 从 `git remote get-url origin` 解析 |
| 4 | 仍无法确定 | 向用户询问目标仓库 |

接受并规范化以下格式：

| 输入格式 | 规范化结果 |
|---------|-----------|
| `owner/repo` | `owner/repo` |
| `https://gitcode.com/owner/repo` | `owner/repo` |
| `https://gitcode.com/owner/repo.git` | `owner/repo` |
| `git@gitcode.com:owner/repo.git` | `owner/repo` |

`target_repo` 规范化后必须恰好包含一个 `/`。

### Step 2：模板发现

按优先级发现模板：

1. **本地模板**：`git -C . rev-parse --show-toplevel` 后读取 `.gitcode/ISSUE_TEMPLATE/*.yml`
2. **远程模板**：通过 GitCode Contents API 获取
3. **内置备选**：本地和远程都不可用时使用，并告知用户来源

### Step 3：信息收集

基于模板必填字段收集信息。缺失信息优先向用户询问，一次不超过 3 个问题。

通用必填项建议：

| 项目 | 说明 |
|------|------|
| 问题描述 | 实际发生了什么 |
| 环境信息 | 操作系统、软件版本、运行环境；无法获取时标记待补充 |
| 复现步骤 | 最小可复现步骤 |
| 预期结果 | 正常情况下应该发生什么 |
| 实际结果 | 当前观察到的行为、日志或截图 |

### Step 4：草稿生成

根据模板生成：
- Issue 标题：模板 `title` 前缀 + 用户问题摘要
- Issue 正文：按模板章节填充
- labels：使用模板 labels 或用户指定 labels

展示完整草稿，等待用户确认。用户未确认前不得提交。

### Step 5：查重检查

提交前使用只读 API 查询已有 Issue：

```bash
curl -s "https://api.gitcode.com/api/v5/repos/{owner}/{repo}/issues?state=all&per_page=20&access_token=$GITCODE_TOKEN"
```

若发现可能重复项，展示标题和 URL，询问用户是否继续提交。

### Step 6：提交 Issue

前置检查：
- `GITCODE_TOKEN` 已设置（不输出 token 内容）
- `curl` 可用
- 目标仓库 owner/repo 已确认
- 用户已确认最终草稿

调用创建 Issue API，提交成功后返回 Issue 编号、状态和 `html_url`。

---

## 约束层

### 强制规则

| # | 规则 | 类型 |
|---|------|------|
| C1 | 仅在用户明确要求 Issue 草稿或提交时触发 | 触发控制 |
| C2 | 创建 Issue 等写操作必须经用户确认 | 流程控制 |
| C3 | 只读 API 可用于模板读取、仓库验证和查重，但应说明用途 | API 边界 |
| C4 | 模板优先级：本地模板 > 远程模板 > 内置备选模板 | 动态设计 |
| C5 | 缺失信息必须标记"待补充"，不得编造 | 数据真实性 |
| C6 | 不输出、不保存 token 或 token 片段 | 凭据安全 |
| C7 | 目标仓库必须规范化为 `{owner}/{repo}` | 输入验证 |

### 幻觉防控

- API 参数和路径必须以 GitCode 官方 OpenAPI 或用户提供文档为准
- 远程模板必须通过 GitCode Contents API 实际获取，失败时说明回退
- 对错误原因、复现条件、环境版本等未验证信息必须标注"不确定"或"待补充"
