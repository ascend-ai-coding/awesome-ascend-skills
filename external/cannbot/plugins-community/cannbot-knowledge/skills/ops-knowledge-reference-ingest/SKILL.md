---
name: external-cannbot-plugins-community-cannbot-knowledge-skills-ops-knowledge-reference-ingest
description: Use when 用户需要将 asc-devkit、CANN 或 Profiling 上游文档摄入并编译为 Ascend NPU 算子 OKF
  reference 知识卡。
original-name: ops-knowledge-reference-ingest
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# ops-knowledge-reference-ingest — reference/ 多源知识摄入（含 curate / review 准则）

把一个上游 docs bundle（如 asc-devkit）蒸馏进 `reference/<bundle>/`：一张卡可由**多个上游 md 融合**而成，而非 1 源 1 卡。本 skill 是**主 Agent 拉起**的编排入口，自己做确定性胶水（harvest §3 / finalize §6），把**判定+著作**派给 curate subagent（按 **§4** 执行）、把**审查**派给 review subagent（按 **§5** 执行，独立新上下文）。

> 本 skill 是 [`ops-knowledge-ingest`](../ops-knowledge-ingest/SKILL.md) 编排器的 **`reference/` 树生产执行者**：负责 reference bundle 的 curate/review/finalize；大原则（新 source 接入 / 增量 / 版本升级三路由）与跨树共享设施见编排器。图引擎脚本（`okf_graph.py` / `okf_judge_aggregate.py`）留在 `ops-knowledge-ingest/scripts/`，供各 `ops-*` 生产和进化 Skill 共享；本 skill 收尾时按全路径调用。
>
> 设计依据：`SPEC-Reference.md`（多源 frontmatter §3.2 / 正文 §3.3 / index §3.5 / 工具链 §5 / verify §7）、`SPEC-Source.md`、`SPEC-Graph.md`。

## §0 触发与参数
- **触发**：用户/主 Agent 显式拉起（手动）。输入 = `bundle`（默认 `asc-devkit`）+ `scope`（类目，如 `basic_api`；pilot 优先一个类目）。
- **bundle 参数化**：流程不写死 asc-devkit；换 bundle 只换 harvest 的 pin/枚举源（adapter 见 `SPEC-Reference §2` 表）。

## §1 不变量（违反即停）
- **不建静态 `resource/` 清单层**（SPEC-Source 已废）；bundle pin 落 **bundle 根 `index.md` frontmatter**（git 源 `upstream_repo/ref/commit`；文档站源 `upstream_doc/ref`），即 watermark。
- **文档中心**：`ops-knowledge-reference-ingest` 做 docs→概念卡的 **LLM 概念蒸馏**，不做 tiling/kernel 代码分析；后者分别由 `ops-knowledge-vv-ingest` 和 `ops-knowledge-cv-ingest` 处理。
- **知识 ≠ 流水账**：整合后的系统知识进卡片；摄入过程（判定、skip 理由、批次）只进 `log/`。
- **蒸馏非照搬**、`snake_case` 无数字前缀、目录扁平 ≤3 层（自内容根）、正文不嵌图片、frontmatter 全。
- **收尾集中写**：并发 subagent 各产卡，但 index/检索/图谱/日志由编排者**收尾统一写**（单写者）。

**脚本边界**：`asc_devkit_extract.py` 只处理 GitCode asc-devkit bundle，`cann_doc_extract.py` 只处理 hiascend 文档 bundle，`okf_source.py` 提供两者共享的 source/catalog 抽象。三者输入和生命周期不同，不互相合并；均从本 Skill 的 `scripts/` 直接调用，不再提供顶层包装入口。

## §2 生命周期（六步）
```
1 harvest  (本 skill §3)   : pin 上游 → 列 scope 全部 docs → 确定性聚卡种子 → inventory
2 curate·判定 (subagent §4): 逐 doc/种子 → 进库? 新建/更新(检索已有卡)? source role? → ledger
3 curate·著作 (subagent §4): 按目标卡聚合多源融合蒸馏 → 写卡(sources/resource)
4 review   (subagent §5)   : 独立 subagent 审改动卡 → findings
5 修订循环 (本 skill)      : 按 findings 改卡 → 必要时复审，直到 clean
6 finalize (本 skill §6)   : tags → index → 检索重建 → 图谱增量 → 覆盖账 → log(sync)
```
- **批次**：按种子/类目分批；**每批一次独立 review**（新上下文 subagent，不复用著作 agent）。
- **委派契约**（派 curate/review subagent 时在 prompt 里给齐）：① 本 skill 的不变量 + 对应 §4/§5 准则；② 当前 bundle/scope 与已有卡状态；③ inventory/ledger 路径；④ 红线；⑤ 200–300 字收尾摘要要求。

## §3 harvest（确定性，编排者自做）
1. **pin**：`python3 scripts/asc_devkit_extract.py --knowledge-root <知识库根> pin`（clone 上游到 `.build/asc-devkit@<sha>`，记 sha）。
2. **列 docs + 聚卡种子**：`scaffold` 解析 TOC（三级级联）/examples（leaf）得 worklist（每 doc：上游路径/URL/kind/概念名）；按**归一概念基名 / TOC 概念域**把同概念的多份 doc 归为一个**候选卡种子**（如 `Add.md` + `Add接口.md` → `add` 卡种子）。产出 `.build/knowledge/<bundle>/inventory.json`（docs + 正文摘要 + seed clusters）。
3. seed 只是**候选边界**；最终归并/拆分由 curate（§4）语义定夺。

## §4 curate 准则（判定 + 多源融合著作 —— 派 subagent 按本节执行）
处理**一个种子/一批文档**，两相：**A 判定**（决定每份 doc 去向）→ **B 著作**（把分配到同一卡的多源融合成一张概念卡）。产出改动卡 + 决策 `ledger`。
**输入**：`bundle`+`scope`+种子簇（来自 `inventory.json`）；上游正文 `.build/<bundle>@<sha>/…`（只读）；已有卡 `reference/<bundle>/`（用 `knowledge_query` 检索定位）。

**A · 判定（逐 doc/种子 → ledger）**
1. **是否进库**：纯噪声/构建设施/与已有完全重复 → `skip`，记**理由**（进库判据：对"设计/使用该 API 的模型"有知识增量）。
2. **新建 or 更新**：先 `python3 ../knowledge-query/scripts/knowledge_query.py --knowledge-root <知识库根> search --query "<概念/API名 同义/英文>" --bundle <bundle> [--kind ..] -k 8` 检索已有卡；同义词放进同一个检索意图，禁止用多个 query 参数混合不同概念；高置信命中同概念卡 → `update`（本 doc 作该卡新 source），否则 → `new`（建新卡，本 doc 多为 `primary`）。
3. **source role**：`primary`(接口定义/概述,每卡唯一) / `variant`(重载/变体) / `guide`(教程) / `example`(样例) / `faq` / `constraint` / `migration`；代码源 `header`/`impl`。
4. 写 `.build/knowledge/<bundle>/ledger.json`：`{doc, action: new|update|skip, card_id, role, reason}`。种子只是候选，可据语义**归并/拆分**（如 `Add-Broadcast` 不并入 `add`，单列）。

**B · 著作（按目标卡聚合多源 → 写卡）**
- **融合 ≠ 拼接**：理解各源后写**一份连贯的概念知识**（功能/签名/要点/约束/示例）；重叠合一、互补补全、冲突标注，不是每源各抄一段。
- **frontmatter**：`schema_version: okf.v1`、`kind`、canonical `type`、`source_family`、`title`、`description`、精简 `tags`、`created_at`、`updated_at`；`resource` = primary 文档.url；`sources:` 块列全部源(每源 `url`+`role`)。
- **文档+代码结合**（有代码源的 bundle）：`sources` 须同时含**文档页**与**声明该 API 的代码头**（在 `.build/<bundle>@<sha>/include/**/*.h` grep 函数名定位，role `header`；有实现可加 `impl`）。`# 函数原型` 每种形态旁**同时给「文档」与「代码」链接**（代码锚到声明行 `#L<n>`）。
- **正文**（按 kind 控长，SPEC-Reference §3.3）：蒸馏非照搬；保留签名代码块。
- **多源来源标注（多源卡必做，可点击）**：`# 函数原型` 每形态标题旁直接给行内链接，锚文本用源文件名，例如 `**RegBase SIMD（…）** — 来源 Add-20.md (<@sha url>)`；每形态就近标一次。溯源以 frontmatter `resource`/`sources` 为准、正文不再单列书目段。
- **update 既有卡**：在现有内容上**增量融合**新源（不整体覆盖/不重复），追加 `sources`；若新源使主题更准可调 description。
- **OKF 规则**：`snake_case` 无数字前缀；扁平 ≤3 层；正文不嵌图片；每断言可溯源到某 source。

**curate 红线**：每卡有且仅一 `primary`、`resource==primary.url`、每 `sources[].url` 为上游 @sha；多源**融合**非拼接、无照搬整段；frontmatter 齐；**不**自建 index/检索/图谱、**不**写 `log/`（那是 finalize 的事）；skip 必带理由、判定可复跑。产出 200–300 字收尾摘要（建/改/skip 数、关键归并决策、存疑点）给编排者。

## §5 review 准则（独立对抗审查 —— 派独立 subagent 按本节执行）
每批著作后派起，**独立于 curate**（新上下文、默认从严）。只读改动卡 + 其源（`.build/<bundle>@<sha>/…` 只读）+ `ledger.json`，产出 findings；**不改卡**（主 Agent 按 findings 修订）。

**A 多源 schema**：有且仅一 `role: primary`（文档主页）、`resource==primary.url`、每 `sources[].url` 为上游 @sha `blob/|tree/`、`role` 在受控词表；**文档+代码结合**：API 卡 `sources` 含代码头（role `header`），`# 函数原型` 每形态有「文档」+「代码」双链、代码锚抽查 `#L<n>` 确为该签名（缺代码源/锚错 = major）；多源来源标注可点击（缺 = major）；frontmatter 字段齐。
**B OKF 原则**：蒸馏非照搬（无整段复制）；融合非拼接（不退化为每源一段）；`snake_case` 无数字前缀、路径 ≤3 层、正文无嵌图；index 不在卡里手写下层。
**C 正确性**：每条关键断言可溯源到某 source（抽样核对源正文，无杜撰/张冠李戴）；update 卡新源确融入、不冲突/重复，description 与主题相符；交叉链接不死。
**D 判定合理性（抽查 ledger）**：skip 的 doc 确无知识增量；new/update 归属正确（同概念才并卡）。

**findings 输出**：每条 `{severity: blocker|major|minor, 卡片:行/区, 问题, 建议}`；无问题则 `clean`。`blocker`=schema 非法/照搬/杜撰/死链/primary 非唯一；`major`=融合退化为拼接/归并错；`minor`=措辞/长度/tags。**纪律**：独立判断（存疑从严，不复用 curate 结论）；只审不改（不动卡/不重建索引图谱）；给可执行建议供定点修订至 clean。

## §6 finalize（确定性，编排者收尾集中写）
1. `python3 scripts/asc_devkit_extract.py --knowledge-root <知识库根> tags`（path_tags 并入 frontmatter）。
2. `python3 scripts/asc_devkit_extract.py --knowledge-root <知识库根> index`（逐层重建 index.md + bundle 根 pin）。
3. **检索重建**：`python3 ../knowledge-query/scripts/knowledge_query.py --knowledge-root <知识库根> build`。
4. **图谱增量构建**（一等阶段，不可省）：`python3 ../ops-knowledge-ingest/scripts/okf_graph.py --knowledge-root <知识库根> candidates → judge(派 agent，聚合 `okf_judge_aggregate.py`) → inject → viz → verify`。新建/合并卡按内容指纹**只重判改动卡**。（图引擎在 `ops-knowledge-ingest/scripts/`，跨树共享。）
5. **覆盖账**：范围内每 doc 终态 = 进某卡 `sources` 或 skip(理由)；核对 `union(所有卡 sources) ∪ skipped == scope docs`；`python3 scripts/okf_source.py --knowledge-root <知识库根> list <bundle>` 抽查多源表。
6. **verify**：`asc_devkit_extract verify`（多源合法+覆盖账）+ `okf_graph verify` + `knowledge_query verify` 全过（或一条 `knowledge-lint` 聚合体检）。
7. **日志**：先 `now=$(date +"%Y-%m-%d %H:%M")`（时间不臆造），在 `log/<date>.md` **顶部插入** `## [HH:MM] sync | <bundle>/<scope> 多源摄入`（Summary/Details + Changes + References；skip 理由入 Details）。按目标知识库根目录的 `log/README.md` 倒序规则。

> **大版本升级**（如 9.0.0→9.1.0）不在本 skill 常规六步内，走 `ops-knowledge-ingest` 编排器的版本升级路由（见目标知识库根目录的 `SPEC-Version-update.md`）：双 clone diff + 5 类分流 + `finalize-version-bump` 原子推进。

## §7 红线速查
- [ ] 不写 `resource/` 清单；pin 在 bundle 根 index.md。
- [ ] 每卡 ≥1 source、有且仅一 `primary`、`resource==primary.url`、每 url 为上游 @sha（git 源）。
- [ ] 多源是**融合蒸馏**非多段拼接。
- [ ] 收尾跑完 §6 全部步骤（含图谱增量）后才算落地；改动集 + log 一起 commit。
- [ ] curate/review 派 subagent 时把对应 §4/§5 准则喂进 prompt；skip/判定可审计、可复跑。
