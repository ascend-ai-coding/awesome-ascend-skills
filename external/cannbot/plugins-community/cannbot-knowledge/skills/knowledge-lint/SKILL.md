---
name: external-cannbot-plugins-community-cannbot-knowledge-skills-knowledge-lint
description: Use when 用户需要在摄入、生成、勘误或提交 PR 前，检查 Ascend NPU 算子 OKF 知识库的结构、溯源、索引或图谱是否合规。
original-name: knowledge-lint
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# knowledge-lint — 全库知识基本原则体检（确定性 + 可选语义深审）

对 `reference/` + `ops/` + `runbooks/` 的全部 concept 卡，按 SPEC（Reference/Source/Graph/Retrieve）的**基本原则**做一次权威体检。安装后的 skill 入口 `scripts/knowledge_lint.py --knowledge-root <知识库根>` 是**确定性、零-LLM、只读只报**的 linter，一条命令叠加全库原则检查 + 聚合现有 `okf_graph/knowledge_query verify`，单一分级报告。与 `ops-knowledge-reference-ingest` 期逐批的 review（§5，agent 语义审）互补——**knowledge-lint 是全库 post-hoc 体检**。

> 依据：`SPEC-Reference.md` §3（写入规范）/§7（verify）、`SPEC-Source.md`（溯源）、`SPEC-Graph.md`、`SPEC-Retrieve.md`。复用 `ops-knowledge-ingest/scripts/okf_graph.py`（图簇）+ `ops-knowledge-reference-ingest/scripts/okf_source.py`（源簇）、`knowledge-query/scripts/knowledge_query.py` 的解析器。

## §0 引擎接口（用 Bash 调，cwd=knowledge-lint skill 目录）
```
python3 scripts/knowledge_lint.py --knowledge-root <知识库根> [--bundle B] [--kind K] [--json]   # 全部确定性检查 + 聚合 verify
python3 scripts/knowledge_lint.py --knowledge-root <知识库根> --aggregate-only                   # 只跑 okf_graph/knowledge_query verify
python3 scripts/knowledge_lint.py --knowledge-root <知识库根> --sample N [--bundle B] [--kind K] # 抽 N 卡为 JSON worklist，喂 §3 深审
```
- 退出码：**任一 blocker（或聚合 verify 失败）→ 1**；否则 0。`--json` 输出结构化 findings 供 CI/agent 消费。
- 报告按原则分组打印 `✗ blocker / ! warn / ✓ ok`，尾部 `blocker N / warn M` + `OK|FAIL`；同仓连跑逐字节一致。
- **溯源是 per-bundle 的**：asc-devkit/ops=GitCode `@sha blob|tree`；op-dev-guide/profiling=hiascend.com 文档 URL（无 @sha/pin）；glossary/runbooks 允许空 resource。脚本内置该策略表，不会对文档站 URL/空 resource 误报。

## §1 何时用
- 每次摄入/生成**收尾后**（finalize 之后）、**提交前**、或 CI 关口。
- 改动一批卡后只查该范围：`--bundle asc-devkit` / `--kind example`。

## §2 确定性检查清单（脚本，零-LLM）

| 组 | 检查 | 级别 |
|---|---|---|
| **A 命名/结构** | 文件名无前导序号 `^\d+[_-]`；深度 `id` 段数 ≤3；snake_case（CJK 名容忍、runbooks/ops 的 ID 约定豁免大写） | blocker / 空格或大写 warn |
| **B frontmatter** | okf.v1 基础字段在位：`schema_version/kind/type/source_family/title/description/tags/created_at/updated_at`；`kind/type/source_family` 合法；无 legacy `timestamp`；无重复 tag；无 merge 冲突标记 | blocker |
| **C 多源溯源** | 有 `sources`：**唯一** primary、`resource==primary.url`、role∈受控词表；`resource`/`sources[].url` 形态按 **per-bundle 策略**（@sha / 文档 URL / 空 OK） | blocker |
| **D 正文** | 无嵌图 `![..](`/`<img>`；`okf:related` 块至多一对；正文（剥码后）**path-like 相对链接**解析到存在文件；未完成标记 `TODO/FIXME/待补充` | 嵌图/多块 blocker；死链/标记 warn |
| **E index/导航** | 每级 index.md 每条目有非空描述；git-bundle 根 index.md 含 `upstream_repo/ref/commit` pin | blocker |
| **G 日志** | `log/<date>.md`：首行 `# <date>` 与文件名一致、`[HH:MM]` 合法、条目**倒序**（最新在前）。仅全库 lint 跑（不随 `--bundle/--kind/--aggregate-only`）；autofix=`log/sort_logs.py`，不查 operation 词表 | blocker |
| **F 聚合** | shell `okf_graph verify` + `knowledge_query verify`，独立成段（区分"原则违规"与"图谱/索引未重建"） | blocker |

## §3 可选语义深审（agent，超出确定性范围时）
确定性脚本管不了的**语义原则**（SPEC 标记 SEMANTIC）：蒸馏非照搬、description 与正文一致、index 只列本层。流程：
1. `knowledge_lint.py --sample N [--bundle/--kind]` 取 JSON worklist（每项 `path`+`sources`+`excerpt`）。
2. 逐卡（量大时**派 subagent 规模化**）审：
   - **蒸馏非照搬**：抓 `sources` 上游对照，正文是否成段照搬而非蒸馏融合。
   - **description 准确**：是否如实概括正文。
   - **index 渐进式披露**：每级 index.md 是否只列**本层**直接子项、不深入下层。
3. 判据复用 `ops-knowledge-reference-ingest` §5 review（B OKF 原则 / C 正确性）；输出 `findings`（severity+位置+建议）。**只报不改**。

## §4 报告判读 & 收尾
- **blocker 必清零**：逐条修卡后复跑至 `OK`。
- **warn 酌情**：死链/未完成标记是真实存量问题，按优先级清理（knowledge-lint 可反复跑驱动）。
- 聚合段报"cards changed since last judge / 索引 stale"= **图谱/索引未重建**，非原则违规：先跑 finalize（`knowledge_query build` / `okf_graph candidates→judge→inject`）再复跑。

## §5 边界
- 脚本**零-LLM、确定性、只读只报不自动改**；修复交人/上层 agent。
- per-bundle 容忍（文档站 URL、glossary 空 resource、runbooks/ops 的 ID 命名）已内置，**勿**改成一刀切 @sha。
- **聚合**非替换：`okf_graph/knowledge_query verify` 仍可单独跑；`asc_devkit_extract verify`（覆盖账）保留。
- 不替代 `ops-knowledge-reference-ingest` 期 review（§5，逐批语义审）；knowledge-lint 是全库收口体检。
