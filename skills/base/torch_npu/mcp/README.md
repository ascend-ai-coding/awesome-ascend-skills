# torch_npu 文档抓取 MCP

从 [GitCode Ascend/pytorch](https://gitcode.com/Ascend/pytorch) 按分支（版本）抓取 torch_npu 中文文档的 MCP 服务，抓取结果可保存到本 skill 目录下的 `fetched_docs/`。

## 安装

在 `mcp` 目录下执行：

```bash
cd awesome-ascend-skills/torch_npu/mcp
npm install
```

（若 skill 在 `.cursor/skills/torch-npu/` 下，则路径为 `.cursor/skills/torch-npu/mcp`）

### Windows：PowerShell 禁止运行脚本时

若报错「无法加载文件 npm.ps1，因为在此系统上禁止运行脚本」，可任选其一：

1. **不改策略，直接用 cmd 版**（推荐）：
   ```powershell
   cd awesome-ascend-skills\torch_npu\mcp
   npm.cmd install
   ```
   之后若用 npx 也改为 `npx.cmd`。

2. **放宽当前用户执行策略**（仅当前用户，需管理员权限时以管理员打开 PowerShell）：
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
   然后即可正常使用 `npm`、`npx`。

## 配置 Cursor MCP

项目已在 **`.cursor/mcp.json`** 中加入本 MCP 配置，使用相对路径 `.cursor/skills/torch-npu/mcp/index.js`（以工作区根为基准）。若 Cursor 启动 MCP 失败（例如找不到脚本），请改为**本机绝对路径**：

- **Windows**：`"args": ["D:/Desktop/KernelCode/.cursor/skills/torch-npu/mcp/index.js"]`
- **macOS/Linux**：`"args": ["/path/to/KernelCode/.cursor/skills/torch-npu/mcp/index.js"]`

保存后**完全重启 Cursor**，MCP 才会加载。

## 提供的工具

| 工具名 | 说明 |
|--------|------|
| `fetch_torch_npu_doc` | 抓取**单个**文档。参数：`branch`（默认 v2.7.1-7.3.0）、`path`（必填，如 `README.zh.md`）、`save_to_skill`（默认 true）。 |
| `fetch_torch_npu_docs_batch` | **批量**抓取常用中文文档。参数：`branch`、`paths`（不传则使用内置列表）。保存到 `fetched_docs/<branch>/`。 |
| `list_torch_npu_doc_paths` | 列出内置的常用文档路径，便于传给 `fetch_torch_npu_docs_batch`。 |

## 分支与路径说明

- **分支**：与 README.zh.md 中「昇腾辅助软件」表一致，如 `v2.7.1-7.3.0`、`v2.6.0-7.3.0`、`master`。
- **路径**：仓库内相对路径，如 `docs/zh/overview/product_overview.md`、`README.zh.md`。
- **保存位置**：`.cursor/skills/torch-npu/fetched_docs/<branch>/<path>`。

## 使用示例（在 Cursor 中调用 MCP 工具）

- 抓取当前默认分支的 README.zh.md 并保存到 skill：
  - 工具：`fetch_torch_npu_doc`
  - 参数：`{ "path": "README.zh.md" }`
- 抓取指定版本文档并批量保存：
  - 工具：`fetch_torch_npu_docs_batch`
  - 参数：`{ "branch": "v2.7.1-7.3.0" }`（不传 `paths` 则用内置列表）
- 查看内置路径列表：
  - 工具：`list_torch_npu_doc_paths`
  - 参数：`{}`

## 依赖与运行环境

- Node.js >= 18
- 网络可访问 `https://gitcode.com`

若 `@modelcontextprotocol/sdk` 的 API 与当前脚本不兼容，可查看 [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) 文档调整 `index.js` 中的导入与 Server 用法。
