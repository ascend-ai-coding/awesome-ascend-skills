# 报告格式模板

```markdown
# vllm 每日 PR & Issue 追踪报告

> **日期**：YYYY-MM-DD
> **数据来源**：`vllm-project/vllm` | `vllm-project/vllm-ascend`

---

## 📊 执行摘要

| 指标 | 数值 |
|------|------|
| 相关条目总数 | N |
| P0 高优先级 | N 条 |
| P1 重要条目 | N 条 |

---

## 🔴 P0 — 紧急关注

### 🔀 [vllm] #XXXXX PR 标题

| 字段 | 内容 |
|------|------|
| 类型 | PR |
| 状态 | ✅ merged |
| 作者 | @username |
| 链接 | [vllm#XXXXX](url) |
| 模型分类 | DeepSeek |
| 技术方向 | 量化 / 模型Bug Fix |
| 优先级 | **P0** |

**📝 中文摘要**

该 PR 修复了 DeepSeek-R1 在 FP8 量化模式下的数值精度问题，导致输出结果出现 NaN。

**💡 影响分析**

影响所有使用 FP8 量化运行 DeepSeek-R1 的用户，建议尽快升级到最新版本。

---

## 📑 技术方向索引

### PD分离（N 条）
- [vllm #XXXXX](url) — PR 标题

## 🤖 模型关注索引

### DeepSeek（N 条）
- [vllm #XXXXX](url) — PR 标题
```
