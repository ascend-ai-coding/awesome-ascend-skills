# Pipeline · commit 级增量同步（上游推进新 commit 区间）

> **适用**：某个**已接入**的上游（同 ref）出了新 commit 区间，要把变化编译进库并推进其 watermark。常态、反复、规模远小于接入。
> **不适用**：首次接入 → 走 [`new-source-onboarding.md`](new-source-onboarding.md)；ref 切换大版本 → 走 [`version-bump.md`](version-bump.md)。
> **先读**：[`../SKILL.md`](../SKILL.md) §0–§5 + 对应内容树的生产 skill + 目标知识库根目录的 `log/README.md` §B（watermark 机制）。

## 范式

```
取源@新sha + 读 watermark(旧sha)  →  git diff  →  范围裁决(用户)  →  分批实施(试点先行 / 串行单写者；正文委派生产 skill)
        ↓                                                                      ↓
   新增 / 变更 / 受影响清单                                  每批：实施 → 独立第二视角 review → 编排者裁决 → 修
        ↓
   集中集成（index + 检索 + 图谱增量）→ 推进 watermark → 写 sync log → 收尾 verify
```

## Runbook

### 0 · 取源 + 读 watermark
- 把上游取到 **@目标新 sha**。
- **旧 sha = watermark**，从 bundle 根 `index.md` frontmatter 的 `upstream_commit` 取（reference bundle）；ops 算子卡的 `resource` URL 里的 sha 即该卡 watermark。
- **没有记录 watermark、或它不是可解析的 sha → 停下向用户索取起始 commit，不要用 `HEAD~N`/最近 tag 猜**（对齐目标知识库根目录的 `log/README.md §B.1`）；`upstream_ref` 为 tag 时先 `git rev-parse <tag>` 解析成 sha 再 diff。

### 1 · 算增量 / 定范围
- `git diff --stat <旧sha>..<新sha>` 出文件级清单，逐块看 diff，归三类：**新增实体 / 变更实体 / 受影响的既有卡**。
- **commit/PR 编号只作发现机制 + 溯源锚点，不作分批依据**。
- 采集**意图溯源**（commit subjects、MR 编号）——进相应卡的溯源，不进知识正文。

### 2 · 范围裁决（用户拍板，不擅自扩）
不是所有增量都收。**守住本库定位**与溯源链。**收哪些、建新卡还是只增强既有卡、是否需要扩约定——由用户裁决**。

### 3 · 约定冻结
增量阶段**约定冻结**：新 `type`/新归类规则/新 frontmatter 字段、生成器模板改动，一律**先改对应 SPEC/生成器（人类批准）再用**。

### 4 · 分批实施
- **★ 试点先行**：首批先选一个代表性实体全流程跑通，**停下向用户汇报校准**，再铺开其余。
- **按实体/语义单元分批，不按 commit**；多个增量项映射到**同一卡**时归一个实施单元（避免并发改同一文件）。
- **串行单写者**：共享同一参考/同一类目的单元串行；并行只适用互不相干的全新卡。
- **正文著作委派对应生产 skill**（reference→`ops-knowledge-reference-ingest`；ops VV→`ops-knowledge-vv-ingest`；ops CV→`ops-knowledge-cv-ingest`）。
- **增强既有卡只增不改**：先读它现在写了什么，在其上增量补，保留既有内容与 tags，绝不整体覆盖。

### 5 · 独立第二视角 review + 编排者裁决（不盲从）
- 每个产物经**独立第二视角 review**：对一手源逐条核验，不凭感觉。review 关注面**全局**：与上游/其他卡的冲突、归类错误、重复卡、过度提炼、交叉链接、index 一致、「读起来是知识不是变更流水账」。
- **编排者裁决，不盲从 review**：reviewer 矛盾 → 亲自去一手源核；每处 fix-forward 都对源二次核实。
- 冲突并存标注（保留双方 + 日期/来源），不静默覆盖。

### 6 · 集中集成 + 推进 watermark（编排者统一做）
- **逐层 `index.md`**：受影响层只列本层。
- **检索重建**：`python3 ../knowledge-query/scripts/knowledge_query.py --knowledge-root <知识库根> build`。
- **图谱增量**：`python3 scripts/okf_graph.py --knowledge-root <知识库根> candidates → judge → inject → viz → verify`（按内容指纹只重判改动卡）。
- **反向链接**：新卡被链的卡要回链。
- **推进 watermark（唯一「已落地」开关）**：所有卡写入、review、集成、verify 都完成后，才把 bundle 根 `index.md` 的 `upstream_commit`（及 `updated_at`）推进到**新 sha**。reference bundle 走 `ops-knowledge-reference-ingest/scripts/asc_devkit_extract.py index`（重建时自动写 pin）；ops 算子卡改 `resource` URL 的 sha。
- **写 sync log**：当天 `log/<YYYY-MM-DD>.md` 顶部插入：

  ```markdown
  ## [HH:MM] sync | <bundle>/<scope> 增量同步

  ### Summary
  把 <bundle> 从 <旧sha> 增量同步到 <新sha>：新增 N、更新 M、弃用 K。

  ### Changes
  - created: <op> (<知识库根>/ops/...)
  - updated: <ref> (<知识库根>/reference/...)

  ### References
  - source: <upstream_repo>
  - revision: <旧sha>..<新sha>

  ### Details
  按增量同步流程处理；<范围裁决 / 矛盾标注 / 上游删除按弃用而非物理删除……>
  ```
  变更流水账只进 log，不进卡片。

### 7 · 收尾 verify
- `python3 ../knowledge-lint/scripts/knowledge_lint.py --knowledge-root <知识库根>`（全库 blocker 0，聚合 `okf_graph verify`+`knowledge_query verify`）。
- reference bundle 另跑 `ops-knowledge-reference-ingest/scripts/asc_devkit_extract.py verify`（多源合法+覆盖账）。
- 本批新引入链接无悬挂；watermark 确已推进。

## 幂等与续跑
- **watermark 是唯一「已落地」开关**——全部卡写入、review、集成、verify 完成才推进；推进后同区间 diff 即空。
- **不维护额外账本**——续跑看 watermark + 已落盘的卡 + git 就够。
- **重跑同区间安全**——同一对 sha 的 `git diff` 结果确定，加「只增不改」，重跑不产生重复或错乱。
- 对齐目标知识库根目录的 `log/README.md §B.3`。

## 本 pipeline 红线
- [ ] 范围是用户裁决点，不擅自扩张本库定位。
- [ ] 约定冻结：新 type/归类/字段/生成器模板先经人类批准。
- [ ] 增强既有卡只增不改、保留既有内容与 tags、矛盾并存标注。
- [ ] 正文著作委派对应生产 skill；watermark 推进是收尾最后一步。
- [ ] review 不盲从，每处裁决/修改对一手源二次核实。
