---
name: external-cannbot-plugins-community-ops-perf-evolution-skills-evolution-knowledge
description: Domain knowledge base for AscendC evolution optimization covering hardware
  architecture, algorithm insights, API pitfalls, optimization patterns, and proven
  solutions for A3 (910B) architecture. 当进行 AscendC 进化优化需要查询硬件架构、算法洞察、API 陷阱、优化模式或已验证方案时使用。
original-name: evolution-knowledge
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# Evolution Knowledge Base

AscendC 进化优化领域知识库，覆盖硬件架构、算法洞察、API 陷阱、优化模式和验证方案。当前支持 A3 (Ascend 910B) 架构。

## 文件布局

```
references/
  a3/                           # Ascend 910B (Membase/A3) 知识
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
      INDEX.md                  # 已验证方案索引（可追加写入）
```

## 渐进式查询协议

知识库采用**渐进式披露**（progressive disclosure）：先读 guide.md 快速参考，按需深入。

### 按工作流阶段查询

| 阶段 | 必读 | 按需 |
|---|---|---|
| Init (世界模型初始化) | `a3/hardware/guide.md`, `a3/optimization_patterns/guide.md` | `a3/algorithm_insights/{family}.md` |
| Sub-agent 代码生成前 | `a3/ascendc_api/guide.md` (Top 5 致命陷阱) | `a3/optimization_patterns/*.md` (按策略类型) |
| Refine (新颖性评估) | `a3/proven_solutions/INDEX.md` | — |

### 按 Agent 角色查询

| Agent | 必读 | 可选 |
|---|---|---|
| ops-evo / lingxi-evo | hardware + optimization_patterns guide.md | algorithm_insights (匹配算子族) |
| ops-partial / lingxi-partial | ascendc_api/guide.md | optimization_patterns (匹配策略类型) |

## 检索优先级链

knowledge_base → evolution-strategies → proven_solutions → WebSearch

## 写入协议

- `a3/proven_solutions/INDEX.md`: 当策略提炼出通用方案时，追加条目
- 其他文件为只读参考

## 架构选择

- 目标芯片 Ascend 910B → 使用 `references/a3/`
