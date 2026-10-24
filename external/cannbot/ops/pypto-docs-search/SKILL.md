---
name: external-cannbot-ops-pypto-docs-search
description: 检索普通 PyPTO（Tensor）算子开发资源——API 文档、按错误码排障、教程/安装/工具文档、算子参考实现与 golden。当要查
  PyPTO 文档、读算子或 API 用法、按错误码排障、找算子参考实现或 golden、或问"这份资源在哪、有哪些"时使用，即使没明说"文档站"也应触发。触发关键词：查
  PyPTO 文档、错误码排障、API 用法、算子参考实现、golden。
original-name: pypto-docs-search
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-24'
synced-commit: ec8a7b0462e3acdd0d1d2f857f1dae652e19f4f2
license: UNKNOWN
---

# pypto-docs-search

按关键词在本地缓存**发现**资源位置，再直接读取全文。缓存是纯文件，可用原生 `Grep`/`Glob`、可离线；文档站不能远程检索，故缓存缺失时只能按确定路径取文档。

## 资源缓存

`$PYPTO_DEVKIT_DIR`（默认 `${XDG_CACHE_HOME:-$HOME/.cache}/pypto-devkit`）下：

| 子目录 | 内容 |
|---|---|
| `docs/` | API / 排障 / 教程 / 安装文档 |
| `ops/` | 算子参考实现（照着写的权威范本） |
| `tests/` | golden / 测试 |

## 准备缓存

首次或需要更新时运行一次：

```bash
python3 scripts/sync_devkit.py
```

它把各类装配到缓存并写 `MANIFEST.json`。当前工作树已含某类资源时符号链接复用、免重复下载——保证磁盘上每类只有一份，`grep` 不会命中两份不一致的副本。成功标准：`$PYPTO_DEVKIT_DIR` 下出现 `docs/ ops/ tests/`。

## 检索

用原生 `Grep`/`Glob` 在缓存三类里按关键词发现（子串即可，`mul` 连带 `matmul`；已知 API 名传精确名如 `pypto-rms_norm` 减噪）。四类模板（`<kw>` 换成关键词）：

- **API 文档名** → `Glob` pattern `**/*<kw>*.md`，path `$PYPTO_DEVKIT_DIR/docs/api/tensor_api`
- **算子参考实现（ops）** → `Grep` pattern `<kw>`，path `$PYPTO_DEVKIT_DIR/ops/pypto_tensor`，`output_mode=content -n`
- **文档全文（docs）** → `Grep` pattern `<kw>`，path `$PYPTO_DEVKIT_DIR/docs`，`output_mode=content -n`
- **golden / 测试（tests）** → `Grep` pattern `<kw>`，path `$PYPTO_DEVKIT_DIR/tests`，`output_mode=content -n`

文档检索排除 `docs/api/pro_api/` 和 `docs/guide/` 下的 `pro/` 子树；测试检索排除 `tests/pypto_pro/`。

命中后直接 `Read` 全文。高级检索（正则、大小写不敏感 `-i`、上下文 `-A/-B`、限定文件类型）直接用 `Grep`/`Glob` 的对应参数。大范围或多角度检索（找某能力的全部参考实现、跨目录定位）派 `Explore` subagent，给明确目标与范围，如"在 `$PYPTO_DEVKIT_DIR/ops/pypto_tensor` 找 attention 的融合实现"。

## 边界

已知确定路径或 URL 的**单点读取**直接 `Read`/`WebFetch`，不经本检索。本检索用于按关键词发现、跨算子/跨文件定位。

## 缓存未就绪时

缓存缺 `docs/ops/tests` 时先运行 `python3 scripts/sync_devkit.py` 装配。无法装配但文档站可访问时，按入口键查下方索引拿**确定文档路径**，直接 `WebFetch https://pypto.gitcode.com/_sources/<sub>.md.txt` 取（`<sub>` 为去掉 `docs/` 前缀的路径，如 `api/tensor_api/operation/pypto-add`）；工具等例外入口见对应索引。算子参考实现与 golden 无文档站形态，仅缓存在场可查。

## 详细索引（按入口键）

- 报错带**错误码** → [`references/error-code-index.md`](references/error-code-index.md)（前缀 → 组件排障文档）
- 知道**算子 / API 名** → [`references/api-index.md`](references/api-index.md)
- 查**教程 / 安装 / 工具文档** → [`references/doc-index.md`](references/doc-index.md)
- 找**算子参考实现 / golden** → [`references/sample-index.md`](references/sample-index.md)
