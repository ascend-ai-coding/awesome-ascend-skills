---
name: external-cannbot-plugins-community-cannbot-knowledge-skills-ops-knowledge-ingest
description: 知识库摄入顶层编排器：大原则（新 source 接入 / commit 级增量 / 大版本升级三路由）+ reference/ops 生产
  skill 与 runbooks 治理边界。持跨树共享图引擎脚本（okf_graph.py / okf_judge_aggregate.py）。不亲自产卡——正文著作委派对应生产
  skill。手动触发。
disable-model-invocation: true
original-name: ops-knowledge-ingest
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# ops-knowledge-ingest — 知识库摄入顶层编排器

把上游官方文档和代码仓编译进本库 `reference/` / `ops/` 内容树，并维护 `runbooks/` 的治理边界。**本 skill 是编排器**：持大原则 + 三路由 + 跨树共享图引擎，**不亲自产卡**——正文著作委派给对应生产 skill。

> 架构定位：本 skill（编排器）+ `reference/ops` 生产 skill（[`ops-knowledge-reference-ingest`](../ops-knowledge-reference-ingest/SKILL.md) / [`ops-knowledge-vv-ingest`](../ops-knowledge-vv-ingest/SKILL.md) / [`ops-knowledge-cv-ingest`](../ops-knowledge-cv-ingest/SKILL.md)）。重组前本 skill 兼管 reference 生产，已拆出 `ops-knowledge-reference-ingest` 承接；本 skill 重写为纯编排器，图引擎脚本（`okf_graph.py` / `okf_judge_aggregate.py`）留在 `ops-knowledge-ingest/scripts/`，由上述 `ops-*` Skill 共享。

## §0 必读背景

- **本库是什么**：cannbot-knowledge 是面向昇腾 NPU / Ascend C 算子开发的 OKF 知识库，治理成可被「大模型检索 + 人渐进式浏览」消费的知识。
- **三个内容树**：
  - **`reference/`** — 上游官方文档结构化抽取的 OKF 知识库（asc-devkit / op-dev-guide / profiling bundle）。生产 skill：[`ops-knowledge-reference-ingest`](../ops-knowledge-reference-ingest/SKILL.md)。
  - **`ops/`** — 按算子的多模板分发与内存复用设计 wiki。生产 skill：[`ops-knowledge-vv-ingest`](../ops-knowledge-vv-ingest/SKILL.md)（纯 Vector 多模板）、[`ops-knowledge-cv-ingest`](../ops-knowledge-cv-ingest/SKILL.md)（cube↔vector 融合）。
  - **`runbooks/`** — 跨算子优化点库 + 实战 field notes + 版本迁移笔记。当前插件未内置基于开发轨迹的生产 skill；社区条目按目标知识库规范与 [`CONTRIBUTING.md`](../../CONTRIBUTING.md) 贡献，版本迁移条目可在大版本升级流程中沉淀。
- **跨树共享层**：`graph/`（知识图谱产物）、`search/`（检索索引）、`log/`（按天审计日志）、根 `index.md`。
- **知识 ≠ 流水账**：卡片记整合后的系统知识；摄入过程（diff 区间、commit、批次、核实动作）只进 `log/`，不进卡片、不进 frontmatter。

## §1 会话定向（每次必做，先读后写）

1. **规范与生产 skill**：目标知识库根目录下的 `SPEC-Reference.md` / `SPEC-Source.md` / `SPEC-Graph.md` / `SPEC-Retrieve.md` / `SPEC-Knowledge-Compile.md` / `SPEC-Version-update.md`；各树生产 skill（见 §6 链接表）。
2. **各树入口**：`reference/index.md`、`ops/index.md`、`runbooks/index.md` — 已有实体，防重复造卡。
3. `log/` 最近 1–3 天 `<YYYY-MM-DD>.md` — 近期操作，防重复劳动。

对大库还要 `grep` 确认目标是否已覆盖，而不只看 index。

## §2 输入路由（三路由）

| 路由 | 触发 | 执行 skill（按树） | 流程子文档 |
|---|---|---|---|
| **新 source 接入** | 首次接入某上游（新 bundle / 新算子仓） | reference→[`ops-knowledge-reference-ingest`](../ops-knowledge-reference-ingest/SKILL.md)；ops-VV→[`ops-knowledge-vv-ingest`](../ops-knowledge-vv-ingest/SKILL.md)；ops-CV→[`ops-knowledge-cv-ingest`](../ops-knowledge-cv-ingest/SKILL.md) | [`reference/new-source-onboarding.md`](reference/new-source-onboarding.md) |
| **commit 级增量** | 上游推进新 commit 区间（同 ref） | 同上（各树 skill 增量更新） | [`reference/incremental-sync.md`](reference/incremental-sync.md) |
| **大版本升级** | ref 切换（如 9.0.0→9.1.0，含架构性变更可能） | reference→[`ops-knowledge-reference-ingest`](../ops-knowledge-reference-ingest/SKILL.md) + 本编排器双 clone diff；详见目标知识库根目录下的 `SPEC-Version-update.md` | [`reference/version-bump.md`](reference/version-bump.md) |

**判路由**：看上游变化性质——新仓=新 source 接入；同 ref 推进 commit=commit 级增量；ref 切换/大版本=大版本升级。`runbooks/` 不直接接上游；当前插件不提供开发轨迹自动进化，社区贡献按目标知识库规范执行，版本升级时可由编排器沉淀架构性变更到 `runbooks/version-migration/`。

## §3 摄入不变量（所有路由通用）

### §3.1 单实体原子动作（建/更新任何一个卡都走这五步）
1. **理解源** — 读全相关一手材料。
2. **影响分析** — 查重（是否已有卡？）、定范围（建新卡 vs 增强既有卡）、找受影响的其它卡。
3. **建/更新卡** — 按对应内容树的规范/生产 skill 写。**更新既有卡前先读它现在写了什么，在其上增量补充，绝不整体覆盖**。
4. **交叉链接** — 相关卡间用相对链接，且**双向**。
5. **维护三件套** — 写完卡必同步：**逐层 `index.md`**（每级只列本层，渐进式披露）；**检索/图谱重建**（见 §4）；当天 **`log/<YYYY-MM-DD>.md`** 顶部插入 `## [HH:MM] <op> | <题>`。

### §3.2 红线（不变量）
- **frontmatter 必填非空 okf.v1 路由字段**：`schema_version`/`kind`/`type`/`source_family`/`title`/`description`/`tags`/`created_at`/`updated_at`；`resource` 按 profile 要求填写。
- **bundle pin 在 bundle 根 `index.md` frontmatter**（`upstream_repo`/`upstream_ref`/`upstream_commit` 或 `upstream_doc`/`upstream_ref`），即 watermark。不建静态 `resource/` 清单层。
- **每卡 ≥1 source、有且仅一 `primary`、`resource==primary.url`、每 url 为上游 @sha（git 源）或文档站 URL**。
- **蒸馏非照搬**、`snake_case` 无数字前缀、目录扁平 ≤3 层（自内容根）、正文不嵌图片。
- **矛盾绝不静默覆盖**：并存 + 日期/来源 + `> ⚠️ 矛盾标记`。
- **每卡自包含**；**正文禁摄入/变更口吻**（「本次 diff / 留待后续」只进 `log/`）。

### §3.3 溯源纪律（本库最硬的一条）
- **核到一手源**（代码/config/算子规格/README 产品表），**prose ≠ 权威**。
- **各层溯源形态**：`reference/` 按 `SPEC-Reference`（蒸馏 + 上游 @sha 链接）；`ops/` 按 `SPEC-Knowledge-Compile` + 各生成器（GitCode @sha blob/tree）；`runbooks/` 回链出处算子卡/轨迹。
- **无一手可核的断言** → 标 `unverified`/`gap`，不冒充事实。

### §3.4 质量护栏
- **不过度提炼**：写得好的机制/公式/关键代码该保留就保留并标出处。
- **一概念一 canonical 卡**：共享算子/参考**丰富既有卡**，不按案例复制第二份同概念卡。
- **复用优先**：已有卡只双向链接、不重复造；**串行单写者**：有概念交集的实体串行摄入。

## §4 跨树共享设施（图引擎）

图引擎脚本留在本 skill 的 `scripts/`，被各树生产 skill 跨树调用（路径不变）：

- `scripts/okf_graph.py` — 知识图谱主程序：`candidates`（确定性召回）→ `judge`（agent fan-out LLM 判定）→ `inject`（写卡片 `# 相关`）→ `viz`（生成 `graph/viz*.html`）→ `verify`（校验闭环）；`related`/`explain` 读时检索。
- `scripts/okf_judge_aggregate.py` — 把判定结果聚合为 `graph/edge_judgments.json`。

两者保持独立：前者管理确定性图谱生命周期，后者只消费外部 LLM fan-out 的批量结果；合并会把外部工作流输入耦合进图引擎主 CLI。

**新增/改动卡片后必须重跑图谱**：`candidates → judge → inject → viz → verify`，更新其 `# 相关`（规则见 `SPEC-Graph.md`）。各树 skill 收尾时按全路径调用 `python3 scripts/okf_graph.py --knowledge-root <知识库根> …`。

> 图谱判定缓存位于目标知识库根目录的 `graph/edge_judgments.json`，按内容指纹增量复判，卡片不变则零 LLM。

## §5 红线速查（任何路由都不许碰）
- [ ] **正文著作委派各树生产 skill**，本编排器不亲自产卡、不重写各树格式。
- [ ] **不建 `resource/` 清单**；bundle pin 在 bundle 根 index.md（= watermark）。
- [ ] **每卡有且仅一 `primary`、`resource==primary.url`、每 url 为上游 @sha**（git 源）。
- [ ] **多源是融合蒸馏非多段拼接**；蒸馏非照搬。
- [ ] **收尾跑完图谱增量**（`candidates → judge → inject → viz → verify`）后才算落地；改动集 + log 一起 commit。
- [ ] **watermark 是唯一「已落地」开关**：大版本升级时由 `finalize-version-bump` 原子推进（见目标知识库根目录的 `SPEC-Version-update.md`）；commit 级增量推进前完成所有卡写入+verify+图谱+日志。
- [ ] **watermark 缺失则停下问基线，不猜**（对齐目标知识库根目录的 `log/README.md §B.1`）。
- [ ] 不擅自改 `SPEC-*.md` / 生产 skill（约定冻结，新约定先经人类批准）。
- [ ] 不创建孤卡（每卡 ≥2 双向链接）；提交/推送/提 MR 只在用户明确要求时做。

## §6 各树生产 skill 链接表

| 内容树 | 生产 skill | 体裁 | 溯源形态 |
|---|---|---|---|
| `reference/` | [`ops-knowledge-reference-ingest`](../ops-knowledge-reference-ingest/SKILL.md) | 上游文档→OKF 蒸馏卡（多源融合） | GitCode @sha blob/tree（git 源）/ hiascend 文档 URL（文档站源） |
| `ops/`（VV） | [`ops-knowledge-vv-ingest`](../ops-knowledge-vv-ingest/SKILL.md) | 纯 Vector 多模板分发设计 wiki（逐模板全链路 + mermaid UB 布局图） | GitCode @sha blob/tree（golden 源码） |
| `ops/`（CV） | [`ops-knowledge-cv-ingest`](../ops-knowledge-cv-ingest/SKILL.md) | cube↔vector 融合设计 wiki | GitCode @sha blob/tree（golden 源码） |
| `runbooks/` | 未内置（按目标知识库规范与 [`CONTRIBUTING.md`](../../CONTRIBUTING.md) 贡献） | 跨算子优化点、field notes 与版本迁移笔记 | 回链出处算子卡 + 可核验证据 |

> **配套工具**：[`knowledge-query`](../knowledge-query/SKILL.md)（检索）、[`knowledge-lint`](../knowledge-lint/SKILL.md)（全库原则体检）。

## §7 路由流程子文档

三路由的详细 pipeline 在 [`reference/`](reference/) 子文档（渐进式披露）：
- [`new-source-onboarding.md`](reference/new-source-onboarding.md) — 新 source 接入 pipeline
- [`incremental-sync.md`](reference/incremental-sync.md) — commit 级增量同步 pipeline
- [`version-bump.md`](reference/version-bump.md) — 大版本升级 pipeline（摘要，权威见目标知识库根目录的 `SPEC-Version-update.md`）
