# lingxi-partial 子agent Prompt 模板

供 lingxi-evo 步骤 4.3 中启动 lingxi-partial 子agent 时使用。
主 agent 读取此模板，按变量填充规则填充后作为 prompt 传给子 agent。

---

```
[TASK]
Optimize AscendC kernel for {op_name}

Operator: {op_name} — {op_description}
Output: output/{op_name}_evo_{timestamp}/round_{r}/parallel_{p}

Shared files (DO NOT regenerate or modify): model.py, design/, <op_name>.json, <op_name>.json.bak

[MANDATORY OPTIMIZATION DIRECTION]
你被分配的优化方向是:
  方向描述: {node_description}
  策略组合: {strategy_combination}
  模式: {mode}

[WARNING] 你必须严格按此方向实现优化。
- 禁止偏离到其他方向（如仅调整 tiling 阈值等简单参数修改不算"实现策略"）
- 你的修改必须体现上述方向描述中的核心优化思路
- 如果此方向在技术上不可行，在 implementation_note 中说明原因，而不是静默切换到其他方向
- 禁止与兄弟变体做同一件事：若你的策略组合与下方任一兄弟的 sig 重叠超过 60%，必须在 implementation_note 写明差异化实现点。

同轮其他变体的方向（禁止重复）:
{other_variants_summary}

[Profiling Context]
Baseline:  bn={baseline_bottleneck_type} | recommended={baseline_suggested_strategies} | anti={baseline_anti_strategies}

[Alignment — 实现代码前必检]
本变体 opt_type={optimization_type}；strategy_combination={strategy_combination}
- 若 strategy_combination 含 Baseline anti 中任一策略 → 必须移除，或在 implementation_note 第一行写明例外理由
- 若 opt_type 与 Baseline bn 相悖（示例：bn=compute_bound 但 opt_type=bandwidth）→ 在 implementation_note 写明"故意背离"的理由
- 若 Baseline 行任一字段为 `N/A`，说明根级 baseline_evidence 未挂载，跳过对齐检查

Steps:
1. AscendC Translation - Translate TileLang design to AscendC kernel (ascendc-translator skill)
2. Degeneration check - validate_ascendc_impl.py + evaluate_ascendc.sh. See [Optimization Approach] for strategy application.
3. Functional verification - evaluate_ascendc.sh (accuracy) + lingxi_perf_driver.py (device-side perf)
   On failure: Conductor analysis (A/B/C classification) → targeted fix → retry (max 3 iterations)
4. Local Refinement (MANDATORY):
   If Max Improve Rounds > 0 AND compilation_success=true AND precision_passed=true → execute inner refinement loop (see [Config])
   Otherwise → write "local_refinement_rounds": 0 into evaluation_results.json
   DO NOT return without completing step 4.
5. **Save `implementation_note.txt` (≥100 字符，mandatory)**：
   **路径强约定**：`output/{op_name}_evo_{timestamp}/round_{r}/parallel_{p}/implementation_note.txt`
   （直接放在 parallel_{p}/ 顶层，**不**放在 kernel/ 等子目录内 — 事后审计/ledger 工具按顶层路径查找）。
   描述应用了哪些策略 + 参考了哪些 Playbook 段落 + 实际做了哪些改造（不仅是常量）+ 为什么这样改 + 约束 / Trade-off。
   **R12 hard block 会硬检查**：缺文件或 < 100 字符 → SubagentStop hook 阻塞退出，工作不算完成；ledger 也会读它做事后追溯。
   自由 narrative 即可，不强制 ## Strategy / ### Step 格式。

[WARNING] ANTI-TRICK POLICY: Do NOT modify model.py. Do NOT skip/simplify computation in model_new_ascendc.py. Do NOT hardcode test shapes. Violations = invalid variant.

[Optimization Approach]
If Mode=open_exploration:
  DO NOT read strategy files. Reason from first principles.
  a. Read best kernel: output/{op_name}_evo_{timestamp}/{best_solution_ref}/kernel/
     (if best_solution_ref is empty, start from TileLang design)
  b. Analyze [Hardware Specs] below
  c. Identify unaddressed bottleneck. Design novel AscendC optimization.
     (Consider: memory access patterns, compute scheduling, UB utilization, instruction selection, algorithmic restructuring)
  d. Implement COMPLETE new kernel (replace, don't patch)
  e. In implementation_note: state novel technique and rationale

If Mode=profiling_driven:
  DO NOT read strategy files. The node description contains specific profiling bottleneck data — use it as your optimization target.
  a. Read parent kernel at: output/{op_name}_evo_{timestamp}/{parent_solution_ref}/kernel/
  b. Parse the profiling bottleneck from [CONTEXT] node description (bottleneck type, pipeline ratios, bubble classes).
  c. Design a TARGETED optimization specifically addressing the diagnosed bottleneck. You are NOT limited to the strategy library.
  d. Apply optimization by modifying parent kernel (patch, don't replace from scratch)
  e. In implementation_note: state what profiling bottleneck was targeted and how

If Mode=strategy_guided (default):
  The [MANDATORY OPTIMIZATION DIRECTION] already contains the strategy direction and key points.
  策略相关资源（card 主体内容 + Playbook SOP）已由主 agent 通过 source_key 程序化加载，见下方 [STRATEGY RESOURCES] 段；不需要再去 plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/ 文件系统读取。

  若 [STRATEGY RESOURCES] 段为空 → 直接以 open_exploration 模式运行：从第一性原理 + Profiling Context 出发设计优化，**禁止去读 cards/P*_*.md**。
  If parent_solution_ref non-empty:
    Read parent kernel at: output/{op_name}_evo_{timestamp}/{parent_solution_ref}/kernel/
    Use as optimization starting point.
  {meta_prompt}

[STRATEGY RESOURCES]  ← v3.2 新增段，由主 agent 用 query_strategies / load_playbook 程序化注入
{strategy_resources_block}

[PRECONDITIONS — 已硬过滤]  ← v3.2 新增段
本变体的策略组合已通过 Preconditions 硬门控（详见 wm_ops filter-candidates 在节点 filtered_by 字段的记录）。
- 如果你怀疑某个策略不适用，**不要自行跳过**：在 implementation_note 末尾报告"该策略 PX 实际不可应用，理由 Y"，由主 agent 在下轮 refine 时决定是否扩大 Preconditions
- 不要再质疑适用性 — 已通过的就是适用的

[PLAYBOOK EXECUTION]  ← v3.2 新增段（强约束 R11）
若 [STRATEGY RESOURCES] 中含 Playbook 段（标记为 "## Playbook: PX_*"），必须**严格按 Step 1-5 SOP 执行**，并在 implementation_note 中按以下结构写作：

```markdown
## Strategy: P1 - Double Buffer        ← 每个采纳的有 Playbook 的策略一段

### Playbook Step 1: 定位关键结构
<grep 输出 + 文件行号>

### Playbook Step 2: 改造计划表
| 元素 | 当前值 | 目标值 | 位置 |

### Playbook Step 3: 代码改造
形态识别：α / β / γ
<改造对照 + Variant Notes>

### Playbook Step 4: 约束复核
<UB / L1 / 边界计算>

### Playbook Step 5: 编码并自检
<5 条 grep 命令实际运行结果，全部通过>
```

⚠️ R11 强约束：SubagentStop hook 会自动 audit implementation_note，缺失任一 Strategy 的任一 Step 1-5 → 阻塞退出。
- 老形态（无 ## Strategy: PX header）暂兼容
- 没有 Playbook 的策略不强制 Step 段落

[CONTEXT]
Node: {node_id} | Parent: {parent_solution_ref} | Best: {best_solution_ref}

{inspirations_text}

Kernel: {kernel_summary}
Score: global best {best_score}x | this variant from {parent_score}x ({parent_solution_ref or "baseline"})

[HISTORY]
{open_questions_rendered}

[Config]
Refinement: max={max_improve_rounds}, stagnation={improve_stagnation_window}
Hardware: {hw_params_one_liner}

Log progress after each step. Save all outputs to output directory. Return evaluation_results.json.
```
