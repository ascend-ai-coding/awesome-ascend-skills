# Diffusers API 按版本检索索引

本页用于指导 Agent 在回答 API 问题时，按用户版本定位文档路径。

## 1. 检索流程（必须按顺序）

1. 获取用户实际 Diffusers 版本（如 `0.35.2`、`0.36.0`）。
2. 枚举对应 tag 下 `docs/source/en/api` 文件树。
3. 根据用户问题关键词定位 API 路径。
4. 将 GitHub 路径映射到官网 URL；若官网不可达，使用 `hf-mirror`。

## 2. 路径映射规则

给定：

- 版本：`v{version}`
- 路径：`{path}`（不含 `.md`）

映射为：

- GitHub 源文件：
  - `https://github.com/huggingface/diffusers/blob/v{version}/docs/source/en/api/{path}.md`
- 官方文档：
  - `https://huggingface.co/docs/diffusers/v{version}/en/api/{path}`
- 镜像文档：
  - `https://hf-mirror.com/docs/diffusers/v{version}/en/api/{path}`

## 3. API 类别路径（只列路径，不复制 API 内容）

| 类别 | GitHub API 路径 |
|---|---|
| Main classes | `configuration.md`, `logging.md`, `outputs.md`, `quantization.md`, `parallel.md` |
| Models | `models/*.md` |
| Pipelines | `pipelines/**/*.md` |
| Schedulers | `schedulers/*.md` |
| Loaders | `loaders/*.md` |
| Modular diffusers | `modular_diffusers/*.md` |

## 4. 版本差异示例

### 并行 API（parallel）

- `v0.36.0`：存在 `docs/source/en/api/parallel.md`
  - GitHub: `https://github.com/huggingface/diffusers/blob/v0.36.0/docs/source/en/api/parallel.md`
  - Official: `https://huggingface.co/docs/diffusers/v0.36.0/en/api/parallel`
  - Mirror: `https://hf-mirror.com/docs/diffusers/v0.36.0/en/api/parallel`

- `v0.35.2`：API 树中无 `parallel.md`
  - 说明：该版本不应直接给出 `api/parallel` 方案。
  - 建议：转向该版本已有 API 路径，或建议升级至支持并行 API 的版本。

## 5. 实用命令

列出某个版本 API 文件树：

```bash
gh api "repos/huggingface/diffusers/git/trees/v0.36.0:docs/source/en/api?recursive=1" --jq '.tree[].path'
```

> 将 `v0.36.0` 替换为用户版本即可。
