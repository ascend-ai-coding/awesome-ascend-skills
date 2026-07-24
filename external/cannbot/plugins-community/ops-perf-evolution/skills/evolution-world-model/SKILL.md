---
name: external-cannbot-plugins-community-ops-perf-evolution-skills-evolution-world-model
description: World model decision tree tools and reference documentation for evidence-driven
  AscendC kernel evolution, providing CLI operations (select, validate, summary, deep-profiling)
  and schema/operations reference. 当进行证据驱动的 AscendC 内核进化优化，需要初始化、选择、验证、更新世界模型决策树或查询其
  schema/操作规范时使用。
original-name: evolution-world-model
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# Evolution World Model

世界模型决策树工具和参考文档，用于证据驱动的 AscendC 内核进化优化。提供 CLI 工具管理持久化 JSON 决策树，支持节点选择、验证、摘要生成和深度 profiling 分析。

## 文件布局

```
references/
  schema.md                  # world_model.json 完整 JSON Schema 定义
  operations.md              # 四大操作：Init / Select / Refine / Analyze
  state_schema.md            # state.json 状态机定义

scripts/
  wm_ops.py                  # CLI 工具（选择/验证/摘要/深度分析）
  state_ops.py               # 运行时状态机管理
  profiling_evidence.py      # 瓶颈分析 → 策略映射
  session_anchor.py          # Session 身份锚定
  check_round_artifacts.py   # 产物完整性检查
  solution_db.py             # JSONL 谱系追踪
  transcript_audit.py        # 子 Agent 审计
```

## CLI 工具 (wm_ops.py)

### 子命令

```bash
# 节点选择：按 utility 评分排序，返回 top-N 节点
python3 plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/wm_ops.py select \
    --path {world_model.json} --n {parallel_num}

# 不变量验证：检查 parent 引用、状态合法性、延续性等
python3 plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/wm_ops.py validate \
    --path {world_model.json}

# 摘要生成：生成精简文本摘要用于 prompt 注入
python3 plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/wm_ops.py summary \
    --path {world_model.json} [--max-chars 1200]

# 深度 profiling：运行指令级分析，写入 profiling_evidence 到节点
python3 plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/wm_ops.py deep-profiling \
    --wm-path {world_model.json} --node-id {id} --work-dir {dir} --op-name {name} \
    [--merge-children]

# Refine：轮次结束后更新世界模型
python3 plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/wm_ops.py refine \
    --wm-path {world_model.json} --round {r} --results-dir {dir} \
    --parallel-map '{map}' --task-type {type}

# Diagnose：失败节点诊断
python3 plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/wm_ops.py diagnose \
    --wm-path {world_model.json} --node-id {id} \
    --failure-type {impl_error|strategy_infeasible} --failure-reason "..."

# Session 锚定
python3 plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/wm_ops.py session \
    --wm-path {world_model.json} --session-id {id} --evo-dir {dir} --op-name {name}

# Baseline evidence 挂载
python3 plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/wm_ops.py attach-baseline-evidence \
    --wm-path {world_model.json} --baseline-eval {baseline_evaluation.json}
```

### Utility 评分公式

```
utility = 3.0 * parent_score
        + 2.5 * (5 - difficulty)
        + 0.75 * depth
        + w_root_explore    # +2.0 if parent is root
        + w_evidence         # +1.5 if parent has profiling_evidence
```

## 参考文档

### Schema (references/schema.md)

定义 `world_model.json` 的完整结构：
- 顶层字段：kernel_summary, baseline_performance, decision_tree, open_questions, stagnation_count, best_score, hw_params, discovered_strategies
- 节点字段：id, parent_id, strategy_combination, mode, status, score, difficulty, solution_ref, profiling_insight, profiling_evidence, children

### Operations (references/operations.md)

四大操作的推理框架：
1. **Init**: 分析算子特征 → 形成瓶颈假设 → 设计初始优化方向
2. **Select**: utility 排序 → 保证 optimization_type 多样性 → 槽位分配
3. **Refine**: 评估结果写入 → 子节点生成/封印 → 停滞检测
4. **Analyze**: 回顾本轮 + 历史 → 更新 open_questions (5 条假设)

### State Schema (references/state_schema.md)

定义 `state.json` 状态机：
- 阶段：init → shared_prep → wm_init → round_select → round_generate → round_refine → round_react → round_checkpoint → finalize → report → done
- 漂移检测：stagnation_count ≥ 2 → drift_status=replan_required
- Hook 规则：11 条不变量检查

## 模块依赖

- `wm_ops.py` → imports `profiling_evidence.py`（merge_strategies_with_evidence）
- `state_ops.py` 独立运行，管理运行时状态
- `session_anchor.py` 独立运行，管理 session 身份锚定
- 所有脚本在同一 scripts/ 目录下，通过 sibling import 引用
