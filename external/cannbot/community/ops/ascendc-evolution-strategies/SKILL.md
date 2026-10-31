---
name: external-cannbot-community-ops-ascendc-evolution-strategies
description: AscendC kernel optimization strategy library with 250 strategies (161
  curated P/D/A/R + 89 discovered X), supporting tiered retrieval, compatibility checking,
  and autonomous strategy discovery. 当需要为 AscendC kernel 选择优化策略、按瓶颈/算子族筛选策略、检查策略兼容性或登记进化发现的新策略时使用。
original-name: ascendc-evolution-strategies
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# Evolution Strategies

AscendC 内核优化策略库，包含 250 策略（161 精选 + 89 进化发现），覆盖数据类型 (D)、性能 (P)、精度 (A)、A5 架构 (R)、进化发现 (X) 五大系列。

## 文件布局

```
references/
  scenario-routing/           # [选策略入口] 族判定 + 各族优化工作流（先于 index 查表）
    README.md                 # 族判定表（结构判据，不靠算子名）
    _common-prerequisites.md  # 通用前置判定链（C1-C5，所有族共享）
    flashattention.md / matmul.md / convolution.md / reduction.md / ...  # 族工作流
  strategy-index.md           # 策略卡注册表 + 补充反查（快速查表 / 瓶颈查表）
  strategy-compatibility.md   # 互斥表、依赖关系、推荐组合
  INDEX.json                  # 程序化索引（generate_index_json.py 生成）
  cards/                      # 仅保留 perf_shape_spec_01.md（分支化优化必读）+ SCHEMA.md；
                              # 161 张策略卡正文已迁至 cannbot-knowledge 仓
                              # knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/
                              # （原文件名小写化平铺，如 p1_double_buffer.md）
  preconditions/              # 适用性硬门控 YAML（每策略一份）+ strategy_meta 块
                              # （卡片机器元数据 bottlenecks/op_families/complexity 等的权威来源）
  families/                   # 算子族指南（attention / flashattention / matmul / reduction 等 8 族）
```

X 系列（进化发现策略）正文已迁至 cannbot-knowledge 仓
`knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/`
（OKF frontmatter，文件名小写 disc_x{n}.md）；INDEX.json 的 `discovered` 数组
带 `doc_repo: cannbot-knowledge` 标记指向该仓。

## 知识仓绑定与本地知识库（统一说明）

三仓分离后的知识根解析、`init.sh` 用法、各脚本参数、local-first 流转与验收方法，
统一见 [`references/knowledge-repo-integration.md`](references/knowledge-repo-integration.md)。
SKILL.md 只保留协议摘要；脚本细节以该文档为准。

关键约定：

- 知识根解析优先级：`--knowledge-root` > `CANNBOT_KNOWLEDGE_ROOT` > `.cannbot/knowledge.env` > 兄弟检出；
  显式来源无效会 **fail-loud**，不再静默换仓。
- 自进化发现先写本地 `.cannbot/knowledge-local/staging/`，审核后 `promote`，再 `pr` 同步知识仓。
- 验收：`scripts/knowledge_smoke.py`（知识是否被加载）+ `scripts/check_three_repo_contract.py`（三方适配）。

## 策略分层检索协议

### L0 (Universal) — 所有算子适用

26 个基础策略：D1-D5, P1-P13, A1-A8

**必读**: 先读 `references/scenario-routing/README.md` 按结构判据归族，打开对应族 workflow 按阶段选卡；`references/strategy-index.md` 的查表仅作补充反查。

### L1 (Advanced) — 按算子类别选择

进阶策略：P14-P147（编号含缺号，按族见 scenario-routing 各 workflow）

根据算子类别（CV fusion, attention, matmul, quantization, reduction, sort等）选择适用的 L1 策略。参考 `strategy-index.md` 中的 "When to Apply" 列。

特别关注标量计算向量化系列 (P94-P98)：对任何新生成的 kernel 优先检查是否含有标量 for 循环，这是单次优化中收益最大的模式。

### L2 (Discovered) — 进化发现策略

X 系列（disc_x1.md, disc_x2.md, ...），由进化过程动态发现并注册，正文在
cannbot-knowledge 仓 `optimizations/evolution_strategies/discovered/`。

## 策略兼容性检查

读取 `references/strategy-compatibility.md` 检查：

1. **互斥关系**: 部分策略不能同时使用（如 P1 vs P22/P19/P20）
2. **依赖关系**: 部分策略需要前置策略（如 P1 requires P5+P7）
3. **UB 预算**: Light (<50%) / Medium (50-75%) / Heavy (>75%)
4. **推荐组合**: 7 个经过验证的黄金组合

## 策略发现协议（local-first：先写本地库，再决定是否 PR）

当 open_exploration 节点取得显著提升（speedup > best_score_before * 1.10）时：

1. 读取 `references/strategy-index.md` 检查新发现的优化方法是否已有覆盖
2. 如果是全新策略：
   - 解析知识仓根（统一走 `scripts/knowledge_root.py`：CLI `--knowledge-root` >
     `CANNBOT_KNOWLEDGE_ROOT` > 项目 `.cannbot/knowledge.env` > 兄弟检出；**显式来源无效会 fail-loud**），
     `Glob <root>/knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x*.md` 获取当前最大编号
   - 用 `python3 scripts/knowledge_local.py add --project <项目根> --id X{n+1} --file <草稿>`
     写入本地库 `.cannbot/knowledge-local/staging/`（OKF frontmatter 须满足
     type/title/description/tags/status/sources/created_at/updated_at/platforms）；
     **运行期不直接写知识仓、不改 `references/strategy-index.md`**
   - 人工/CI 审核后 `knowledge_local.py promote --id X{n+1}`，再由
     `knowledge_local.py pr --id X{n+1}` 生成（或 `--execute` 执行到校验为止）同步到知识仓、
     重建索引、跑 `check.sh`、提交 PR 的步骤
3. 知识仓 PR 合并前须通过其 `check.sh`（frontmatter lint + 索引同步）

> **可变性说明**: 策略卡正文（P/D/A/R 系列）与进化发现正文（X 系列）均已迁至 cannbot-knowledge 仓；本地 `preconditions/` 的 `strategy_meta` 块为 P/D/A/R 卡的机器元数据。X 系列为运行时可写（写知识仓）。`strategy-index.md` 仅允许追加 X 系列条目。

## A5 架构扩展

当目标芯片为 Ascend 950 (351x/Regbase) 时，可查阅 cannbot-knowledge 仓 `optimizations/evolution_strategies/` 下 R 系列卡片（R1-R8，aliases 即策略 ID）。⚠️ R1-R8 当前全部标记 `status: deprecated`（注明使用了 AscendC 不存在的虚构 API），仅作方向留档，落地前必须以 strategy-index.md / INDEX.json 中的 active 状态为准：

- R1: VF fusion（合并多个 VF 函数，减少 Load/Store）
- R2: Register reuse（减少 RegTensor 溢出）
- R3: Dual-issue（指令调度优化）
- R4: Hardware loop（硬件循环构造优化）
- R5: Unaligned memory optimization
- R6: Low-latency tree reduction
- R7: SIMD/SIMT hybrid programming
- R8: Mutex synchronization
