---
name: external-cannbot-plugins-community-cannbot-knowledge-skills-knowledge-query
description: Use when 用户正在处理昇腾 NPU 上的 Ascend C/CANN 算子任务，并询问 API 名称或可用性、签名、参数、头文件、调用方式、平台/版本支持、Tiling、数据搬运、AICore
  kernel、实现样例、编译/运行错误、精度、性能或 Profiling；在搜索本地 CANN/Toolkit/SDK 安装包或源码前触发。
allowed-tools: Bash Read Grep Glob
original-name: knowledge-query
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# knowledge-query — 知识检索与证据前置

本 skill 是 Ascend C / CANN / 昇腾 NPU 算子知识库的只读检索入口。它负责把用户问题转成可检索的短 query，查找 reference API、ops 算子样例、runbooks 优化经验和 field notes，补充图谱邻居，再把证据交给后续回答、设计、编码、调试和性能优化使用。

安装后的 skill 入口是本 skill 内的 `scripts/knowledge_query.py`（纯 stdlib，确定性默认）。Agent 调用本 skill 后，优先用 `preflight` 形成短 query、检索路线和第一批证据，再由 Agent 读取卡片、判断充分性，并按任务类型组合 `overview/pipeline/grep/get/neighbors`。

> 依据：`SPEC-Retrieve.md`。知识库 root 解析顺序：`--knowledge-root` / `--knowledge-roots`、`CANNBOT_KNOWLEDGE_ROOT(S)`、`~/.config/cannbot/knowledge.env`、有限结构探测。只有 root 强校验通过时，索引缺失或过期才会自动 build；空知识库默认拒绝写空索引。

参数提示：`preflight --task "<当前任务/报错/疑问>"`、短 `--query`、`--method/--scope/--kind` 等检索参数，或 `<卡片 doc-id>`。任务明确指向 910C / Atlas A3 / A3 时，`preflight` 自动启用 A3 平台过滤；直接检索可显式传 `--platform a3`。

## What this skill solves
- **API/语义查证**：确认 Ascend C/CANN API、宏、头文件、dtype、shape、format、buffer、同步、资源和平台约束。
- **算子实现设计**：查找 AICore kernel、TilingData、Host Tiling、BlockDim、SetBlockDim/GetBlockDim、TilingKey、算子注册、OpDef/aclnn 和模板选择经验。
- **数据搬运与内存问题**：定位 DataCopy/DataCopyPad、GlobalTensor/LocalTensor、UB/GM/workspace、TPipe/TQue/TBuf、流水线、对齐、bank conflict 等约束和样例。
- **并行与计算模式**：检索多核切分、broadcast、reduction、AtomicAdd、CopyIn/Compute/CopyOut、vector/matmul 等实现模式和反模式。
- **调试与诊断**：根据编译报错、运行错误、精度异常、性能异常、msprof/Profiling 线索查找诊断卡、约束说明和已知修复路径。
- **相似样例与经验复用**：查找 ops/ 下相似算子、runbooks/ 下优化经验、reference/ 下 API 依据，并通过图谱邻居扩展上下文。
- **依据与出处补全**：为方案结论、代码修改、性能判断和问题定位补充 doc-id、resource @sha、相关卡片和上游来源。

## Trigger scenarios
- 用户问“怎么写、怎么实现、怎么设计、怎么调、怎么优化、哪里有样例、这个 API 什么含义、这个报错怎么处理”时，只要主题属于 Ascend C、CANN、昇腾 NPU、AICore kernel 或算子开发，先调用本 skill。
- 用户询问昇腾 NPU 算子所用 API 的名称、是否存在、函数签名、参数语义、头文件、调用方式、dtype/shape/format、平台/版本支持、限制、样例或替代接口时，在搜索本地 CANN 安装包或源码前调用本 skill。
- 用户提到或暗含以下知识内容关键词时优先触发：Ascend C、CANN、NPU、AICore、kernel、算子、Tiling、Host Tiling、TilingData、BlockDim、SetBlockDim、GetBlockDim、DataCopy、DataCopyPad、AtomicAdd、UB、GM、workspace、TPipe、TQue、TBuf、GlobalTensor、LocalTensor、OpDef、aclnn、dtype、shape、format、broadcast、reduction、msprof、Profiling、精度、性能、编译失败、运行失败、相似样例。
- 开始方案设计、代码实现、代码修改、代码审查或调试前先做知识库 `preflight`；从设计转实现、从编译转运行、从正确性转性能时再次检索。
- 对某个概念、函数、宏、错误码、配置项、实现路径、最佳实践或反模式不确定时，先查知识库再推断。
- 用户要求补充依据、出处、上游链接、相似样例、最佳实践、性能经验或错误修复路径时，先调用本 skill。
- 不触发：纯机械编辑、格式化、文件移动、用户明确要求不查知识库、与 Ascend C/CANN/NPU 算子知识无关的通用问答、生成新知识卡（应交给 `ops-knowledge-ingest` 路由到对应的 `ops-*` 生产 Skill）。

## API 证据顺序
昇腾 NPU 算子的 Ascend C/CANN API 查证固定执行 `知识库 -> 固定版本上游 -> 本地安装包`：
1. **知识库**：在搜索本地 CANN 安装包、Toolkit、SDK、include/impl 源码或外部资料之前，先运行 `preflight`，逐张读取 `read_first`，并按需用 `get --section` 或 `local_path` 深读定义、约束和样例。
2. **固定版本上游**：如果读卡后仍缺少精确签名、平台/版本约束、实现细节或可运行样例，再沿卡片 `resource` / `sources` 核对固定 revision 的官方文档或源码。
3. **本地安装包**：只有知识库未命中、前两层仍不足，或任务明确要求核验当前已安装 CANN 包的实际行为时，才检查本地 CANN/Toolkit/SDK。检查时同时确认并记录安装版本、目标平台和读取路径，并与知识卡或固定版本上游交叉核对；冲突按版本/平台差异处理并保留两侧依据。

## Default workflow
1. 运行 `knowledge_query.py discover` 确认选中的 knowledge base root；root 不对时显式加 `--knowledge-root <知识库根>`。
2. 在实质性推理、计划或改代码前运行 `knowledge_query.py preflight --task "<当前任务/报错/疑问>"`；上下文紧张时加 `--brief`。API 查证时，这一步先于任何本地 CANN/Toolkit/SDK 源码搜索。
3. 先读取 `route`、`read_first`、`relevance`、`read_purpose`、`verify_after_read`、`next_actions`、`follow_candidates`、`missing_signals`、`sufficiency_rule`。不要把 `results` 当最终答案。
4. 查看 `platform_filter`：A3 任务会排除 frontmatter 严格为 `platforms: [950]` 的卡片；缺少平台元数据和多平台卡片不降权、不排除。A3 与 950 对比/迁移任务不会自动启用该过滤。
5. 如果 `route=browse_first`，先运行 `overview`/`browse` 建立目录视图，再选择类别或目录卡进入；如果 `route=search_first`，先 `get` 或 `Read local_path` 读取 `read_first` 中 1-3 张卡片。
6. `preflight/search/overview/grep/neighbors/pipeline` 输出中的 `path` 是稳定 doc-id；如果同时给出 `local_path`，Agent 可直接 `Read` 该文件做多轮阅读，不必反复调用 `get`。后续检索继续使用 `preflight` 生成的命令；手动执行 `search/pipeline/grep/neighbors` 时保留 `--platform a3`。
7. 读取卡片后由 Agent 判断证据是否满足 `sufficiency_rule`：至少确认 1 张 strong 证据卡，或 2 张 medium 证据卡互相支撑；weak 证据不能单独支撑结论。不满足时，从 `follow_candidates` 里选择 `neighbors`、`pipeline`、`grep` 或二次 `preflight`，不要直接补全猜测。若只缺精确声明、平台/版本约束或源码实现，按“API 证据顺序”继续核对固定版本上游，再判断是否需要本地安装包。
8. 只有证据足够后再回答、设计或修改代码；结论中保留 doc-id、`resource`，使用本地安装包证据时同时给出版本、平台和路径。

## How to query
```
knowledge_query.py preflight --task "<当前任务/报错/疑问>" [--platform a3] [--brief]  # 自动识别 A3，也可显式指定
knowledge_query.py plan      --task "<当前任务/报错/疑问>"                       # 只生成检索计划，不执行检索
knowledge_query.py overview  [--task T|--query Q] [--scope S] [--groups N]       # browse-first：目录/类别视图
knowledge_query.py browse    [--task T|--query Q] [--scope S] [--groups N]       # overview 别名
knowledge_query.py search   --query Q [--scope/--dir S] [--bundle/--kind/--category/--section] [--platform a3] # 确定性 BM25F
knowledge_query.py discover                                                        # 展示当前解析到的 knowledge base root
knowledge_query.py recall   --method bm25|tfidf|tagtype|graph|dense [facets]      # 召回一路 -> hits JSON
knowledge_query.py rerank   --method bm25f|tagidf|quality|reranker|llm-judge --hits -  # 重排 hits
knowledge_query.py pipeline --recall bm25,tagtype --rerank bm25f --query Q [facets] [--platform a3]  # 多路召回+合并+重排
knowledge_query.py grep     <regex> [--scope] [--platform a3] [--only body|frontmatter|all] # 正则(默认 body、剥 related)
knowledge_query.py get      <doc-id>... [--section H] [--max-chars N] [--neighbor-limit N]  # 整卡/片段 + 邻居 + @sha
knowledge_query.py neighbors <doc-id> --hops N [--platform a3]                   # 图谱多跳取周边知识
knowledge_query.py prepare-judge --hits - [--material header|card]               # 备料给 llm-judge(上下文判定)
knowledge_query.py verify [--level index|schema|strict] [--limit N] [--json]     # 默认只验索引可用性
knowledge_query.py eval [--cases FILE] [-k N] [--fail-under FLOAT]               # 小型检索回归评测
```
模型路线参数（opt-in）：`--backend api|hashing` `--embedding-model M`（dense）、`--llm-model M` `--material header|card`（reranker/llm-judge）。
`search` 过滤参数：`--scope/--dir`（接受 `reference/`、`reference/asc-devkit/`、`asc-devkit/`、`ops/`、`runbooks/`；`all` 表示不限制路径）以及 `--bundle --kind --category --section --platform a3`。`recall/pipeline` 还支持完整 facets：`--type --tags --paradigm --severity --confidence --status active|verified|stub|all`。

## Query construction
- **先跑 preflight**：开始需要领域知识支撑的实现、调试、设计、优化或验收前运行 `knowledge_query.py preflight --task "<当前任务/报错/疑问>"`，先看 `route/read_first/next_actions/follow_candidates/missing_signals/sufficiency_rule`，再看 `results/grep_matches/suggested_get`。
- **让 Agent 做决策**：脚本只给检索计划草案和候选证据；Agent 必须读取 `read_first` 卡片，判断证据是否足够，再决定追 `neighbors`、走 `overview`、做二次 `preflight`，或进入回答/编码。
- **看相关性信号**：`relevance.level` 是脚本给出的确定性候选强度，`relevance.reasons/risks` 说明为什么相关或有什么误召回风险；`read_purpose` 是读卡目的，`verify_after_read` 是读卡后的检查清单。
- **直接读本地卡片**：JSON 输出里的 `local_path` 是卡片绝对路径。需要反复查看同一张卡、交叉比对多张卡或引用细节时，优先 `Read local_path`；需要扩展邻居、重新排序或限定范围时，继续用 doc-id 调 `knowledge_query.py`。
- **不要手写长自然语言 query**：不要把完整任务描述塞进单个 `--query`。
- **让 preflight 拆 query**：`preflight` 会优先抽取精确符号、错误码、文件/接口名、knowledge base index 中的 title/alias/tag；同时识别 concept、API usage、similar example、overview、debug、performance 等意图并做轻量重排。它还会给出 `search_first` 或 `browse_first` 路线。领域 hints 只进入 `hint_queries` / `supplemental_results`，不和精确 query 混在同一主排序里。
- **A3 平台约束要贯穿后续检索**：任务出现 910C / Atlas A3 / A3 时，先看 `platform_filter`，并在后续检索中保留 `--platform a3`。本轮规则只排除严格 `platforms: [950]`，不是通用兼容性判定。
- **开放式问题先 browse**：用户问“有哪些、整体怎么查、某类知识范围、优化路径、相关经验总览”时，用 `overview`/`browse` 先看目录和类别，再决定具体 query。
- **一次 search 一个意图**：每次 `search` 只放一个 API/符号、错误码、报错短语或设计/调试/性能短语。不要用多个 `--query` 混合不同 API 猜测；需要多路证据时用 `preflight` / `pipeline`，或分开执行多次 `search`。只有复现旧合并排序时才显式加 `--allow-multi-query`。
- **精确符号再 grep**：对 API、错误码、宏、关键配置项使用 `grep` 复核，再用 `get` 注入整卡。`suggested_get_details` 会标出 `precise_grep`、`exact`、`broad_grep`、`hint` 来源，优先读 exact 和 precise_grep。
- **领域 hints 可替换**：默认 hints 在 `domain_hints.json`；其他知识库可在 knowledge base root 放置 `search/knowledge-query-hints.json` 或 `.cannbot/knowledge-query-hints.json` 覆盖默认触发词和短 query。

Good:
```
knowledge_query.py preflight --task "实现算子 kernel，确认接口语义、dtype 约束和相似样例"
knowledge_query.py preflight --task "build.sh 编译失败，报错里出现 __ubuf__ half 参数类型不匹配"
knowledge_query.py search --query "OpDef dtype aclnn" --scope reference/
knowledge_query.py pipeline --recall bm25,tagtype --rerank bm25f --query "tiling parallel partitioning core number"
knowledge_query.py preflight --task "在 910C(A3) 平台实现算子，确认可用 API"
```

Bad:
```
knowledge_query.py search --query "完整任务描述加所有背景加所有猜测一次性塞进一个很长的自然语言 query"
```
错误模式：把 `Transpose UB`、`DeInterleave`、`DataCopy pair swap` 这类不同 API 猜测塞进同一次 `search` 的多个 query。

## 检索原则（与工具对应）
- **检索是多步推理**：默认流程是 `preflight -> get/read_first -> sufficiency check -> neighbors/overview/二次检索 -> answer`。除非用户只要求列出检索结果，否则不要在未读卡片时直接回答。
- **证据要过相关性校验**：strong 可作为主证据；medium 需要互相支撑；weak 或带明显 `risks` 的卡片只能作为线索，不能单独支撑 API 约束、实现方案、诊断结论或性能判断。
- **先收窄再召回**：用 `--scope/--kind/--bundle/...` 缩小范围，别盲目调大 `-k`。
- **一次 search 一个意图**：`search` 默认拒绝多个 `--query`，防止把候选 API、领域 hint 和主算子意图混排。确实需要 legacy 多 query 求和时，显式加 `--allow-multi-query` 并说明原因。
- **召回求全**：词法用 `bm25`(别名默认开，中英/术语自动扩展)；结构用 `tagtype`(tag/type/paradigm)；关系用 `graph`。可 `pipeline` 多路撒网。
- **重排默认确定性**：`bm25f`/`tagidf`/`quality`(可复现)。**模型路线 opt-in 且非确定，但可真跑、按指纹缓存复现**：`dense`(Embedding API，默认；无 Key 时用 `--backend hashing` 零依赖离线)、`reranker`/`llm-judge`(Claude Code SDK，一次 listwise 判定 + 缓存)。`llm-judge` 三路：`--verdicts` 预算 / 无 verdicts 自动调 SDK / `prepare-judge` 上下文判定。未配置后端→结构化 `not_configured`(码3)；运行失败→`model_runtime_error`(码2)；绝不裸 traceback。
- **默认排 stub**（`--status active`）；只要已验证卡时用 `--status verified`，找待补卡才用 `--status stub/all`。
- **图谱多跳是核心**：选卡后 `neighbors` 沿 typed edge 取样例/指南/声明头/相关，给**连通的知识面**，不只单卡。
- **先片段再整卡**：`search/recall/pipeline` 输出 `snippet`；必要时 `get --section/--max-chars` 控制上下文，最终引用各卡 `resource` 上游永久链接（field-notes 无 @sha 属正常）。
- **verify 分级**：日常检索健康检查用 `verify --level index`；治理 frontmatter 时再用 `--level schema` 或 `--level strict --limit N`；CI 或脚本消费时加 `--json`。
- **检索回归**：修改检索逻辑后运行 `eval --fail-under 1.0`。默认用 `eval/retrieval_cases.json` 中的小型用例集，输出 recall@k、suggested_hit_rate 和 MRR。

## Route selection
- API/符号/头文件语义：`preflight` 后用 `search` 或 `pipeline --recall bm25,tagtype --scope reference/`，再 `grep` 精确复核符号。
- 相似样例：优先 `pipeline --recall bm25,tagtype --rerank bm25f --scope ops/`，再 `neighbors` 查关联指南和 runbook。
- 性能/tiling/优化：优先 `pipeline --recall bm25,tagtype --rerank quality --scope runbooks/`，结果不足时扩到 `ops/`。
- 报错/日志诊断：先 `preflight`，对错误码、宏、关键短语跑 `grep`，再 `get` 命中的诊断卡或相似样例。
- 迁移/版本差异：先限定 `--kind guide` 或 `--type migration_guide`，再用 `neighbors` 查相关 API 和样例。

## 检索模式（recipes，可扩展）
`modes/` 下每个模式是一条定制路线（召回+重排+scope 的 worked-recipe）：
- [`modes/similar_examples.md`](modes/similar_examples.md) — 相似样例（ops/，tag/type/paradigm）
- [`modes/base_api.md`](modes/base_api.md) — 基础 API（reference/，bm25+kind）
- [`modes/optimization_practice.md`](modes/optimization_practice.md) — 优化实践（runbooks/+examples，quality 重排）
- 新增模式：在 `modes/` 丢一份 `.md`（模板见 `modes/README.md`）。**也可不用模式、自定义组合。**

## 输出 / 引用
答案给出依据卡的 doc-id + `resource` @sha；`local_path` 只用于本地读取，不替代对外引用。`relevance` 只表示候选相关性，最终证据强弱由 Agent 读卡后确认。标注 confidence（如 ops `confidence: provisional/verified`、runbooks `quality_score/severity`）。检索证据不足时明确说明已查过哪些 query/grep。
