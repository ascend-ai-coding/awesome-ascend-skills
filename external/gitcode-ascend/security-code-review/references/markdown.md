# Markdown 安全审查详细参考

> 本文件包含 Markdown 文件安全审查的完整代码示例和审查要点。由 SKILL.md 按需引用。

---

## 1. XSS 注入（内嵌 HTML）

```markdown
<!-- ❌ 不安全：内嵌恶意 HTML/JavaScript -->
<script>alert('XSS')</script>
<img src=x onerror="alert('XSS')">
<a href="javascript:alert('XSS')">Click me</a>
<div onmouseover="steal(document.cookie)">Hover me</div>

<!-- ✅ 安全：审查时应标记所有内嵌 HTML -->
<!-- 使用安全的 Markdown 渲染器，禁用原始 HTML -->
<!-- 配置: sanitize: true 或使用 DOMPurify -->
```

## 2. 链接安全

```markdown
<!-- ❌ 不安全：javascript: 协议 -->
[Click here](javascript:alert('XSS'))

<!-- ❌ 可疑：外部链接未标注 -->
[下载工具](http://malicious-site.com/tool.exe)

<!-- ✅ 安全：仅允许 https 链接 -->
[文档](https://docs.example.com)

<!-- ✅ 审查要点：检查所有外部链接的合法性 -->
```

## 3. 敏感信息泄露

```markdown
<!-- ❌ 不安全：文档中包含真实密钥 -->
API Key: `sk-1234567890abcdef`
数据库连接: `postgresql://admin:password123@prod-db:5432/mydb`
内部服务地址: `http://10.0.1.50:8080/admin`

<!-- ✅ 安全：使用占位符 -->
API Key: `<YOUR_API_KEY>`
数据库连接: `postgresql://<user>:<password>@<host>:5432/<db>`
内部服务地址: `http://<internal-host>:<port>/admin`
```

## 4. 图片安全

```markdown
<!-- ❌ 可疑：外部图片可能追踪用户 -->
![avatar](http://tracker.evil.com/pixel.gif?user=123)

<!-- ❌ 不安全：超大图片可能导致 DoS -->
![](http://example.com/huge-100mb-image.png)

<!-- ✅ 安全：使用本地或受信任的图片源 -->
![架构图](./docs/images/architecture.png)
![Logo](https://cdn.trusted-domain.com/logo.png)
```

---

## Markdown 审查要点

- 搜索所有 `<script>`, `<iframe>`, `<object>`, `<embed>` 标签
- 搜索所有 `javascript:`, `data:`, `vbscript:` 协议链接
- 搜索所有硬编码的密钥、密码、Token 模式（如 `sk-`, `ghp_`, `AKIA`）
- 检查所有外部 URL 的合法性

## Markdown 安全工具

| 工具 | 用途 | 命令 |
|------|------|------|
| **markdownlint** | Markdown 格式检查 | `markdownlint docs/` |
