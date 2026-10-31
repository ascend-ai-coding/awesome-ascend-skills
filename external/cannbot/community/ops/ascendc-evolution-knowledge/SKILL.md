---
name: external-cannbot-community-ops-ascendc-evolution-knowledge
description: Domain knowledge base for AscendC evolution optimization covering hardware
  architecture, algorithm insights, API pitfalls, optimization patterns, proven solutions,
  and the OKF self-evolution knowledge base (discovered strategies / proven solutions
  / failure patterns) for A3 (910B) architecture. 当进行 AscendC 进化优化需要查询硬件架构、算法洞察、API
  陷阱、优化模式、已验证方案或 OKF 自演进知识库（X 系列策略 / 跨算子方案 / 失败模式）时使用。
original-name: ascendc-evolution-knowledge
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# Evolution Knowledge Base

AscendC 进化优化领域知识库，分两层：

- **经典层 `references/a3/`**：人工策展的硬件架构、算法洞察、API 陷阱、优化模式与 proven solutions（只读为主）
- **OKF 自演进层 `references/okf/`**：进化过程中自动沉淀的策略/方案/失败模式卡（结构化 frontmatter，运行期首选读写目标）

当前支持 A3 (Ascend 910B) 架构。

## 文件布局

```
references/
  a3/                           # 经典知识库（Ascend 910B / A3 架构，人工策展）
    guide.md                    # 顶层快速入口
    hardware/
      guide.md                  # 瓶颈诊断启发式、tiling 公式
      ascend910b_arch.md        # 架构详情：UB=196KB, 40核, DMA 模型
    algorithm_insights/
      guide.md                  # 算法优化原则
      attention_family.md       # MHA/GQA/MQA 优化
      reduction_ops.md          # Norm/Softmax: two-pass, Welford
      elementwise_fusion.md     # 算子融合
    ascendc_api/
      guide.md                  # Top 5 致命陷阱（快速诊断）
      common_pitfalls.md        # 15 个常见陷阱详解
    optimization_patterns/
      guide.md                  # 决策树：bandwidth vs algorithm vs register
      double_buffering.md       # 20-80% 增益
      tiling_strategies.md      # 10-50% 增益
      causal_block_skip.md      # 20-50% (attention)
      pipeline_overlap.md       # 5-30% 增益
      memory_coalescing.md      # 10-40% (strided access)
    proven_solutions/
      INDEX.md                  # 已验证方案索引（compat 层，可追加写入）
  okf/                          # OKF 自演进知识库（进化沉淀，带结构化 frontmatter）
    index.md
    strategies/discovered/      # X 系列策略（discovered_strategy，OKF 增强版；
                                # 原始发现在 cannbot-knowledge 仓 optimizations/evolution_strategies/discovered/）
    runbooks/proven-solutions/  # 跨算子已验证方案（proven_solution，6 条）
    runbooks/failure-patterns/  # 失败模式（failure_pattern，用于 anti 提示）
```

> OKF 卡片的机器索引在 `../ascendc-evolution-strategies/references/INDEX.json`
> （`cards` 中 `source_key` 前缀 `ascendc-evolution-knowledge#okf/`，由 `generate_index_json.py` 的 `discover_okf()` 扫描生成）。

## 渐进式查询协议

知识库采用**渐进式披露**（progressive disclosure）：先读 guide.md 快速参考，按需深入。

### OKF 层优先（推荐）

运行期读取**先走 OKF**（结构化可检索），未命中再回落经典 a3：

1. **按算子族/瓶颈查 OKF**：用本 skill 的查询脚本（CLI 详见「查询 CLI」节）
   ```bash
   python3 scripts/query.py \
       --op-family attention --bottleneck compute_bound --status active
   ```
   - `--kind` 可限定 `discovered_strategy` / `proven_solution` / `failure_pattern`
   - 命中即读对应 OKF 卡全文，纳入策略选择
   - **证据门槛（sufficiency rule）**：至少 1 条 `cross_op_verified` 或 2 条 `single_op` 支撑结论；`stub` 卡只能当线索
2. **失败反查**：诊断失败时按 `--failed-combination` 查 failure_pattern，命中复用 Resolution

### 经典层按工作流阶段查询

| 阶段 | 必读 | 按需 |
|---|---|---|
| Init (世界模型初始化) | `a3/hardware/guide.md`, `a3/optimization_patterns/guide.md` | `a3/algorithm_insights/{family}.md` |
| Sub-agent 代码生成前 | `a3/ascendc_api/guide.md` (Top 5 致命陷阱) | `a3/optimization_patterns/*.md` (按策略类型) |
| Refine (新颖性评估) | `a3/proven_solutions/INDEX.md` | — |

### 按 Agent 角色查询

| Agent | 必读 | 可选 |
|---|---|---|
| ops-evo / lingxi-evo | OKF（按 op_family 查）+ hardware + optimization_patterns guide.md | algorithm_insights (匹配算子族) |
| ops-partial / lingxi-partial | ascendc_api/guide.md | optimization_patterns (匹配策略类型) |

## 检索优先级链

**OKF 知识库 → a3 经典知识库 → ascendc-evolution-strategies → proven_solutions → WebSearch**

（OKF 为进化沉淀的结构化卡，优先级最高；a3 为人工策展参考；strategy 库为 P/D/A/R 系列策略卡片。）

## 查询 CLI（OKF 结构化检索）

`scripts/query.py` 是 OKF 卡片的**只读结构化检索入口**（纯 stdlib、确定性、无外部依赖）。

**数据源**：`../ascendc-evolution-strategies/references/INDEX.json`（由 `../ascendc-evolution-strategies/scripts/generate_index_json.py` 的 `discover_okf()` 扫描 `references/okf/` 生成，`source_key` 前缀 `ascendc-evolution-knowledge#okf/`）。

**检索维度**（INDEX.json 中 OKF 卡的结构化字段）：

- `kind`：`discovered_strategy` | `proven_solution` | `failure_pattern`
- `op_families` / `bottlenecks`：来自 vocabulary（如 attention / compute_bound）
- `phenomenon`：失败现象（compile_error / precision_failure / perf_regression）
- `tags`：自由标签
- `confidence`：single_op | cross_op_verified
- `status`：stub | active | deprecated | retracted

**命令**：

```bash
# 按算子族 + 瓶颈检索（Init/SELECT 海选）
python3 scripts/query.py \
    --kind discovered_strategy --op-family attention --bottleneck compute_bound --status active
# 按失败组合反查（4.4.2 失败诊断）
python3 scripts/query.py \
    --kind failure_pattern --phenomenon precision_failure --failed-combination P22,A1
# 全文关键词检索（轻量 grep）
python3 scripts/query.py --keyword "GQA decode"
```

**证据强度规则（sufficiency rule）**：查询结果须满足以下任一条件才能作为决策依据：

- 至少 1 条 `confidence: cross_op_verified` 且 `status: active` 的条目
- 或至少 2 条 `confidence: single_op` / `status: active` 的条目互相支撑
- `status: stub` 的条目不能单独支撑结论，只能作为调查线索（查询默认 `--status active` 排除 stub，需显式 `--status all` 才查得到）

**输出**：JSON `{query, results[], sufficiency}`；`results[].path` 为 OKF 卡 doc 路径，命中即读卡全文纳入决策。

**与 X 系列原始发现的关系**：优先检索 OKF 增强版（`okf/strategies/discovered/`，带结构化字段）；OKF 缺失某 X ID 时回退 cannbot-knowledge 仓 `knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/`（原始发现主副本，OKF frontmatter，经 INDEX.json `discovered` 数组 doc_repo 标记定位）；新 X 写入知识仓 discovered/（OKF 契约）+ 本 skill OKF 增强版，并重建索引。

## 写入协议

- **OKF 层（首选）**：进化产生的策略/方案/失败模式一律经本 skill 的维护协议写入
  `references/okf/`（查重 → frontmatter 校验 → 置信度提升 → **重建 INDEX.json**）。
  触发点：lingxi-evo/ops-evo 的 **4.4.2 失败诊断**（新 failure_pattern）与 **4.5.3 策略提炼**（新 disc_X + proven，写知识仓 discovered + OKF）。
- **经典层**：`a3/proven_solutions/INDEX.md` 在策略提炼出跨算子通用方案时追加条目（compat 层）；其他文件为只读参考。
- **新发现的 X 双写**：cannbot-knowledge 仓 `.../evolution_strategies/discovered/disc_x{n}.md`（文件名小写，OKF frontmatter 契约）**和** OKF `okf/strategies/discovered/disc_X{n}.md` 都写，并重建索引（知识仓侧另须过其 check.sh）。

### 维护协议（OKF 写入）

写入由 agent 按下列协议直接写 OKF 卡（知识仓 discovered + OKF），无迁移脚本；历史数据一次性迁移已完成。

1. **查重**：discovered_strategy 按 `id`、proven_solution 按 `title`/`key`、failure_pattern 按 `key` 查重；命中同概念条目 → 增量 update（保留已有内容、补充新来源/证据），不新建、不整体覆盖。
2. **新建条目路径**：
   - discovered_strategy → `okf/strategies/discovered/disc_X{n}.md`
   - proven_solution → `okf/runbooks/proven-solutions/{kebab-title}.md`
   - failure_pattern → `okf/runbooks/failure-patterns/F{n}-{kebab-key}.md`
3. **frontmatter 规范**：`schema_version: okf.v1` + `kind` + 类型特有字段（discovered_strategy 带 `id/origin/confidence/status/op_families/bottlenecks/tags/discovered_round/base_speedup`；failure_pattern 带 `key/phenomenon/severity/failed_combination/operator_type/occurrences/created_at/updated_at`；proven_solution 带 `strategy_ids`）。
4. **置信度提升**：discovered_strategy 的 `reuse_stats.ops` ≥2 个算子且 `passed>0` → `cross_op_verified`；failure_pattern `occurrences≥2` 且跨 ≥2 算子 → `active`；proven_solution `Source` ≥2 算子 → `cross_op_verified`。
5. **矛盾处理**：不静默覆盖，正文追加 `> ⚠️ 矛盾标记：新来源 XXX 在 Y 场景下结论相反，待进一步验证` + 日期/来源；严重冲突新建 `status: stub` 候选条目，原条目保持不变。
6. **索引重建**：每次写入/更新后必须 `python3 ../ascendc-evolution-strategies/scripts/generate_index_json.py`（扫描 `references/okf/` 合并进 INDEX.json）。

> 不要人工直接写 OKF 文件；所有写入通过本 skill 维护协议或等价 agent 流程（4.4.2 / 4.5.3 / 每轮 refine 后）完成，保证 frontmatter 一致性与索引同步。

## 架构选择

- 目标芯片 Ascend 910B → 使用 `references/a3/`
- 进化沉淀 → `references/okf/`（结构化卡）+ 写入 `references/a3/proven_solutions/INDEX.md`（compat 登记）
