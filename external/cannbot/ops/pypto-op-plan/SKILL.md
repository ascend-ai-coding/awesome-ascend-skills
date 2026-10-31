---
name: external-cannbot-ops-pypto-op-plan
description: PyPTO 算子开发需求规划（requirement planning）——结构相似样例检索与可行性准备。触发：启动新算子开发的需求规划阶段、检索结构相似参考实现时使用。
original-name: pypto-op-plan
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# PyPTO Complex Kernel — Planning

> 资料获取统一使用 skill `pypto-docs-search`：按需搜索算子 API 文档、参考实现与 golden 等文件/目录/内容。

## Requirements and environment (optional)

**If requirements are unstructured:** 第 1 步，structure them into `SPEC.md` (task summary lives in SPEC.md, not MEMORY.md).

**SPEC.md front matter 的 `supported_dtypes`** 仅收录 P0 输入/输出 tensor 的 dtype 集合，必须与 `REQUIRE.md` 声明一致；权重 buffer 与中间计算 dtype 不写入。模板与生成指引见 skill `pypto-intent-understand`（`templates/spec-template.md`）。

**If environment issues arise:** confirm with the caller whether to ignore them and continue planning.

## Find structurally similar examples

第 2 步，search for existing kernels with similar structure —— 两个并列手段，回答"如何找范本"时须同时给出：

1. 用 `pypto-docs-search` 搜索 "<kernel type>" 的算子参考实现与 golden 用法；
2. 用 `grep -rn "<kernel type>" custom/` 扫描当前工作树，找结构相似的已有 kernel 范本。

Note: golden/用法 is an **API-usage reference only**, not the production implementation standard — its simplified forms (e.g. `pypto.Tensor([])`) may violate lint / gates. When it conflicts with lint, lint takes priority; do NOT cite such a sample to declare lint a false positive.
