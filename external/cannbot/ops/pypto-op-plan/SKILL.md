---
name: external-cannbot-ops-pypto-op-plan
description: PyPTO Kernel requirement planning — structurally-similar example search
  and feasibility setup. Use when planning a new operator's development.
original-name: pypto-op-plan
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# PyPTO Complex Kernel — Planning

> 资料获取统一使用 skill `pypto-docs-search`：按需搜索算子 API 文档、参考实现与 golden 等文件/目录/内容。

## Requirements and environment (optional)

**If requirements are unstructured:** structure them into `SPEC.md` (task summary lives in SPEC.md, not MEMORY.md).

**If environment issues arise:** confirm with the caller whether to ignore them and continue planning.

## Find structurally similar examples

Search for existing kernels with similar structure —— 用 `pypto-docs-search` 搜索 "<kernel type>" 的 算子参考实现与 golden 用法；另可扫描当前工作树 `grep -rn "<kernel type>" custom/`。

Note: golden/用法 is an **API-usage reference only**, not the production implementation standard — its simplified forms (e.g. `pypto.Tensor([])`) may violate the production coding constraints / gates. When a sample conflicts with those constraints, the constraints take priority; do NOT cite such a sample to justify a deviation.


