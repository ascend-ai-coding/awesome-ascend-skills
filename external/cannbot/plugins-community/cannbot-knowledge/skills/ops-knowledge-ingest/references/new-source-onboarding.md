# Pipeline · 新 source 接入（冷启动一个上游）

> **适用**：把一个**首次接入**的上游（新 reference bundle / 新算子仓）一次性编译成第一批稳定知识。一次性、规模大、人类主导编排。
> **不适用**：该上游已接入、只是推进 → 走 [`incremental-sync.md`](incremental-sync.md)；ref 切换大版本 → 走 [`version-bump.md`](version-bump.md)。
> **先读**：[`../SKILL.md`](../SKILL.md) §0–§5 + 对应内容树的生产 skill + `SPEC-Reference.md`（reference 树）/ `SPEC-Knowledge-Compile.md`（ops 树）。

## 范式

编排者**不亲自吃全源**，先规划、再把活拆成叶子子 agent，分批推进、试点先行、收尾收敛：

```
人类指令（手动编排）
  └─ 编排者（先 plan，不亲自吃全源）
       ├─ recon 子 agent（只读）   → 全源事实底图（实体清单 / 类目 / 规模）
       ├─ plan  子 agent（只读）   → 《接入计划》：产哪些卡 / 参考，请人类批准
       ├─ Batch 0  骨架（编排者手建：bundle 根 index.md + 各层骨架）
       ├─ Batch 1  试点 ──★ 复盘循环：审查 → 演进约定 → 补链接 → 再铺开
       ├─ Batch 2..N  逐实体 author 子 agent（调用对应生产 skill）  ⟲ 失败重试 / 截断 resume
       └─ 收尾：人工审查 → 去重 / 重归类 / 链接修复 → 重建 index + 图谱 + 写 log
```

## Runbook

### 0 · 取源 + 建 bundle 根 pin
- 把上游取到生成器约定的 golden 位置；记录 commit sha（git 源）或文档版本（文档站源）。
- **reference bundle**：bundle 根 `index.md` frontmatter 记 pin（git 源 `upstream_repo`/`upstream_ref`/`upstream_commit`；文档站源 `upstream_doc`/`upstream_ref`）。**不建静态 `resource/` 清单层**（SPEC-Source 已废）。
- **ops 算子仓**：算子卡 frontmatter `resource` 指 GitCode @sha blob/tree。
- 各层骨架：`reference/<bundle>/index.md`、`ops/<repo>/index.md` 等，让后续每批从第一天就能进 index。

### 1 · 规划（零写入）
1. 派 **recon 子 agent**（只读）勘探全源：实体清单、类目、规模、目标硬件 → 事实底图。
2. 派 **plan 子 agent**（只读）通读规范 + golden → 产《接入计划》：产哪些卡、是否贡献参考，并标出约定不一致点。
3. **人类批准计划、纠偏**。

### 2 · Batch 1 — 试点 + ★ 复盘循环（质量关键步，不要跳过）
- 选一个**中等复杂度**实体作试点，用对应生产 skill 跑通全范式。
- **试点跑通后不直接铺开**，插一轮：审查产出 → 把暴露问题固化进约定（人类改 SPEC/生成器，叶子子 agent 不碰）→ 补链接 → 再规模化。

### 3 · Batch 2..N — 逐实体 author（委派生产 skill）
- **一个实体（或一组同族）= 一个 Batch = 一个叶子 author 子 agent**：
  - **reference bundle**：调用 [`ops-knowledge-reference-ingest`](../../ops-knowledge-reference-ingest/SKILL.md) §4 curate 准则蒸馏卡。
  - **ops VV 算子**：调用 [`ops-knowledge-vv-ingest`](../../ops-knowledge-vv-ingest/SKILL.md) 按其模板产出。
  - **ops CV 算子**：调用 [`ops-knowledge-cv-ingest`](../../ops-knowledge-cv-ingest/SKILL.md) 按其模板产出。
- 给子 agent 的 prompt 含：① 生产 skill / 规范关键约束；② 当前各层状态（防重复造卡）；③ golden 全部相关路径；④ 200–300 字收尾总结；⑤ 红线（不碰 SPEC/生成器/他人卡）。
- **串行单写者**：共享同一参考/同一类目的实体串行；并行只适用互不相干的全新卡。

### 4 · 收尾 — 一致性收敛（必做）
人工审查后：**去重/合并**（并行子 agent 会造重复变体卡）、**重归类**、**链接补全**。`grep` 验收（无裸路径、无畸形链接、无重复卡）。收尾由编排者**亲自串行**做。

### 5 · 集成（编排者集中处理）
- **逐层 `index.md`**：各层只列本层（渐进式披露）。
- **检索重建**：`python3 ../knowledge-query/scripts/knowledge_query.py --knowledge-root <知识库根> build`。
- **图谱增量**：`python3 scripts/okf_graph.py --knowledge-root <知识库根> candidates → judge → inject → viz → verify`。
- **交叉链接**：相关卡双向链接。
- **当天 `log/<YYYY-MM-DD>.md`** 顶部插入接入记录。

### 6 · 自检 + 分支 + MR
- 自检：每卡 frontmatter 含非空 `schema_version/kind/type/source_family/title/description/tags/created_at/updated_at`；index 逐层完整；本批新引入链接无悬挂；`knowledge-lint` blocker 0。
- 新建分支 → push → 提 MR。**提交/推送/提 MR 只在用户明确要求时做。**

## 本 pipeline 红线
- [ ] 正文著作委派对应内容树的生产 skill，不在本 pipeline 重写其格式。
- [ ] bundle pin 在 bundle 根 index.md frontmatter，不建 `resource/` 清单。
- [ ] 叶子子 agent 不碰 SPEC/生成器/他人卡。
- [ ] 收尾收敛是必做步；图谱增量不可省。
- [ ] 范围是用户裁决点，不擅自扩张本库定位。
