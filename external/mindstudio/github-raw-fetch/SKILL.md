---
name: external-mindstudio-github-raw-fetch
description: 当用户提供 GitHub 文件页面链接，或希望读取某个仓库中的源码、配置、README、Markdown、docs 内容时，使用此技能。技能不仅支持将
  `github.com/<owner>/<repo>/blob/<ref>/...` 转换为 `raw.githubusercontent.com` 链接，还要求在读取仓库
  docs 前优先读取同仓库同 ref 的 `agent_router.md`，根据其中声明的目录结构或路由规则拼出真实路径，并优先通过 `curl` 获取内容。
original-name: github-raw-fetch
synced-from: https://github.com/kali20gakki/mindstudio-skills
synced-date: '2026-03-26'
synced-commit: 266c7821de7b51b683d4605960d0d86f7d631e03
license: UNKNOWN
---


# GitHub Raw Content 与 Docs Router 读取

## 1. 技能目标

当用户要求读取 GitHub 上的源码、配置、README、Markdown 或 docs 内容时，按下面的顺序执行：

1. 先识别仓库、`ref`、目标文件或目标主题
2. 如果目标属于仓库文档体系，先读取仓库根目录的 `agent_router.md`
3. 根据 `agent_router.md` 中的路由、目录、别名、入口说明拼出真实路径
4. 将最终路径转换成 raw 链接
5. 使用 `curl` 获取文件内容

## 2. 适用范围

- 用户提供的是 GitHub 文件页面链接，例如：
  - `https://github.com/<owner>/<repo>/blob/<ref>/<path-to-file>`
- 用户提供的是 raw 链接，例如：
  - `https://raw.githubusercontent.com/<owner>/<repo>/<ref>/<path-to-file>`
- 用户要读取的是 README、源码、配置、脚本、JSON、YAML、Markdown 等纯文本文件
- 用户要读取的是某个仓库的 docs、文档入口、FAQ、指南、设计文档、API 文档
- 用户只给出了仓库和想看的文档主题，但真实 docs 路径可能受 `agent_router.md` 控制

以下场景不属于本技能的直接处理范围：

- 仓库首页、目录页、Pull Request、Issue、Commit 页面本身
- 需要递归遍历整个仓库或批量抓取大量文件
- 明显的二进制文件，例如图片、压缩包、模型权重

## 3. 核心规则

### 3.1 标准 GitHub 文件页转 raw 链接

如果输入链接满足：

```text
https://github.com/<owner>/<repo>/blob/<ref>/<path-to-file>
```

则转换为：

```text
https://raw.githubusercontent.com/<owner>/<repo>/<ref>/<path-to-file>
```

转换时必须遵守以下规则：

1. 将域名 `github.com` 替换为 `raw.githubusercontent.com`
2. 删除路径中的 `/blob/`
3. 其余路径保持不变
4. 分支、tag、commit SHA 等 `<ref>` 必须原样保留

### 3.2 已是 raw 链接

如果用户提供的本身就是 `raw.githubusercontent.com` 链接，则不要重复转换，直接使用该链接。

### 3.3 读取 docs 前必须先读 `agent_router.md`

只要需求满足下面任一条件，就必须先读取目标仓库根目录下的 `agent_router.md`：

- 用户要看的内容属于 docs、文档、FAQ、指南、Markdown 文档体系
- 用户给出的路径位于 `docs/`、`doc/`、`wiki/`、`manual/` 等文档目录
- 用户没有给出精确文件路径，只说“看某个仓库里关于 X 的文档”

`agent_router.md` 的定位规则：

```text
https://github.com/<owner>/<repo>/blob/<ref>/agent_router.md
```

其 raw 形式为：

```text
https://raw.githubusercontent.com/<owner>/<repo>/<ref>/agent_router.md
```

要求：

1. `agent_router.md` 必须使用与目标文档相同的仓库和相同的 `ref`
2. 不要跳过这一步直接凭经验猜测 `docs/` 目录结构
3. 如果目标文件本身就是 `agent_router.md`，则直接读取它，不需要额外前置步骤
4. 读取完 `agent_router.md` 后，优先依据其中定义的入口文档、目录映射、别名、语言目录、跳转规则来拼最终路径

示例：

```text
https://github.com/kali20gakki/msprof/blob/master/agent_router.md
```

当读取 `kali20gakki/msprof` 仓库中的 docs 时，应优先读取上面的 router 文件，再确定真正的文档路径。

### 3.4 `curl` 是默认抓取方式

获取 raw 内容时，优先使用 `curl`，不要依赖 GitHub HTML 页面渲染结果。

推荐命令：

```bash
curl -L "https://raw.githubusercontent.com/<owner>/<repo>/<ref>/<path-to-file>"
```

如果运行环境是 PowerShell，并且需要避免 `curl` 别名差异，优先使用：

```powershell
curl.exe -L "https://raw.githubusercontent.com/<owner>/<repo>/<ref>/<path-to-file>"
```

### 3.5 docs 路径拼接规则

在读取完 `agent_router.md` 后，agent 应重点识别这些信息：

- docs 实际根目录在哪里
- 是否存在多语言目录
- 是否存在逻辑名称到真实文件路径的映射
- 文档入口是 `README.md`、`index.md` 还是其他文件
- 某些主题文档是否需要从别名跳转到真实路径

最终原则：

1. 先看 router，再决定路径
2. 路径以 router 为准，而不是以仓库默认 `docs/` 习惯为准
3. 如果 router 已明确给出入口或映射，直接按其规则拼路径

## 4. 标准操作流程

1. 识别用户给的是 GitHub 文件页链接、raw 链接，还是“读取某仓 docs”的意图
2. 从链接或上下文中提取 `<owner>`、`<repo>`、`<ref>`、目标路径或目标主题
3. 如果目标属于 docs 体系，先构造并读取 `agent_router.md` 的 raw 链接
4. 根据 `agent_router.md` 推导真实文档路径
5. 将真实路径转换为 raw 链接
6. 使用 `curl` 获取内容
7. 将结果返回给用户：
   - 用户想快速了解内容时，优先给摘要和关键片段
   - 用户明确要求原文时，再返回全文或尽量完整展示
   - 必要时附上最终 raw URL，方便复用

## 5. 推荐命令模板

### 5.1 先读取 router

```powershell
curl.exe -L "https://raw.githubusercontent.com/<owner>/<repo>/<ref>/agent_router.md"
```

### 5.2 再读取真实文档

```powershell
curl.exe -L "https://raw.githubusercontent.com/<owner>/<repo>/<ref>/<actual-doc-path>"
```

### 5.3 直接读取普通文件

```powershell
curl.exe -L "https://raw.githubusercontent.com/<owner>/<repo>/<ref>/<path-to-file>"
```

## 6. 示例

### 示例 1：标准 GitHub 文件页

输入：

```text
https://github.com/actioncloud/github-raw-url/blob/master/index.js
```

转换后：

```text
https://raw.githubusercontent.com/actioncloud/github-raw-url/master/index.js
```

再使用：

```powershell
curl.exe -L "https://raw.githubusercontent.com/actioncloud/github-raw-url/master/index.js"
```

### 示例 2：读取 docs 前先读 router

输入：

```text
请读取 https://github.com/kali20gakki/msprof/blob/master/docs/xxx.md
```

正确流程：

1. 先读取：

```text
https://raw.githubusercontent.com/kali20gakki/msprof/master/agent_router.md
```

2. 根据 router 判断 `xxx.md` 的真实位置
3. 再对真实路径执行：

```powershell
curl.exe -L "https://raw.githubusercontent.com/kali20gakki/msprof/master/<actual-doc-path>"
```

### 示例 3：用户只说“帮我看某仓库的某篇文档”

如果用户只给了仓库和主题，没有给出最终文件路径，先读该仓库的 `agent_router.md`，再根据其中的入口和映射规则推导实际文档位置，而不是直接猜 `docs/<topic>.md`。

## 7. 错误处理与约束

- 如果链接不是 GitHub 文件页或 raw 文件链接，要明确告知该 URL 不符合本技能处理模式
- 如果 `agent_router.md` 返回 404：
  - 先确认仓库、`ref` 是否正确
  - 如果确认无 router，可退化为直接按原始路径转换 raw 链接
  - 退化时要说明“未发现 `agent_router.md`，因此按直接路径尝试读取”
- 如果 router 存在，但无法从中推导出目标文档路径：
  - 明确说明缺少哪类映射信息
  - 不要假装已经确认真实路径
- 如果获取结果返回 404，优先考虑：
  - 路径错误
  - `ref` 不存在
  - router 指向的路径已变化
  - 仓库或文件为私有资源
- 如果返回的是 HTML 而不是文本，说明抓取方式不对，优先检查是否误用了 GitHub 页面链接而非 raw 链接
- 如果目标内容明显为二进制或体积过大，不要强行按纯文本展开；应告知用户文件类型，并优先返回链接或简要说明

## 8. 输出建议

- 如果用户是为了阅读或分析文件，优先提炼关键内容，而不是机械粘贴全文
- 如果用户明确要求 raw content 或原文，再按需返回完整文本
- 如果读取 docs 时经过了 `agent_router.md`，可以顺带说明最终路径是如何由 router 推导出来的
- 分析代码或配置时，可顺带说明关键函数、入口、配置项或用途
