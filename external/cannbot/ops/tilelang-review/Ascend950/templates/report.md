# 代码格式与必要检视报告

> 仅供参考的旧报告模板；当前 `tilelang-op-review` 文件检视 workflow 不读取此模板，报告正文由 `scripts/workflow.assemble_report.py` 根据 YAML 生成。
> 当前文件检视的实际保存路径是 `{review_output_dir}/{source_file}_review_summary.md`；`review_output_dir` 由 `instructions.md` 的产物目录规则确定，不要按本模板在 `.agents/` 下额外生成报告。

- **生成时间**：{YYYYMMDD_HHMMSS}
- **检查对象**：{文件路径 / PR / diff / 粘贴内容}

---

## 1. 检查摘要

| 语言 | 文件数 | Lint 问题数 | 格式问题数 | 必要 Review 问题数 |
|------|--------|-------------|------------|--------------------|
| Python / TileLang | {N} | {N} | {N} | {N} |
| C++ | {N} | — | {N} | — |
| Markdown | {N} | — | — | {N} |
| **合计** | {N} | {N} | {N} | {N} |

---

## 2. Python 文件问题详情

> 每个有问题的 Python 文件单独一个 `###` 章节；无问题则删除本节。

### {path/to/file.py}

#### Lint 问题

| 行号 | 代码 | 描述 | 修复建议 |
|------|------|------|----------|
| {N} | {CODE} | {详细问题描述} | {ruff 提供的自动修复建议，无则填 —} |

#### 格式问题

```diff
{ruff format --diff <file> 的输出}
```

- 具体问题：{行宽超限 / 缩进错误 / 空行不规范 等}

---

## 3. C++ 文件问题详情

> 每个需要格式化的 C++ 文件单独一个 `###` 章节；无问题则删除本节。

### {path/to/file.cc}

#### 格式问题

```diff
{clang-format --style=file <file> | diff -u <file> - 的输出}
```

- 具体问题：{对齐 / 换行 / 大括号风格 等}

---

## 4. 必要代码检视详情

> 根据 `instructions.md` 的 reference 路由读取匹配条例，按文件列出确定问题和待确认项；无适用条例则写 N/A。

### {path/to/file.py}

| Canonical 条例身份 | 严重级别 | 状态 | 位置 | 证据与影响 | 修复建议 |
|---|---|---|---|---|---|
| {reference-stem/CLAUSE-ID} | {高/中} | {问题/待确认/通过} | {L<N>} | {可观察证据及影响} | {建议} |

---

## 5. 问题统计摘要

**Lint 问题（按错误代码）**

| 代码 | 出现次数 | 含义 |
|------|----------|------|
| {UP035} | {N} | {说明} |
| {F401} | {N} | {说明} |

**格式问题（按文件数）**

| 语言 | 需格式化文件数 |
|------|----------------|
| Python | {N} |
| C++ | {N} |

**必要 Review 问题（按条例 ID）**

| Canonical 条例身份 | 高 | 中 | 待确认 |
|---|---:|---:|---:|
| {reference-stem/CLAUSE-ID} | {N} | {N} | {N} |

---

## 6. 下一步操作

```bash
# 查看单个文件的 lint 详情
ruff check <file>

# 查看格式差异
ruff format --diff <file>              # Python
clang-format --style=file <file> | diff -u <file> -   # C++

# 全部检查完成并取得确认后，由 common.format-fix.md 执行自动修复
# 注意：自动修复仅处理 lint/格式，不修改必要 Review 中的语义问题
```
