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

Bottleneck-targeted guidance (priority order matters):
  memory_bound → increase tile size, enable double buffer, merge DMA transfers
  scalar_bound / scalar_compute / scalar_loop_around_vector / scalar_reduce_after_vector / scalar_matvec_driver (aiv_scalar_ratio > 40%) → FIRST apply scalar-to-vector strategies P94/P95/P99/P100; only if ratios remain high then consider family-specific tiling/core changes
  compute_bound → algorithm optimization, eliminate redundant compute
  mte2_stall → redesign data layout, enlarge transfer blocks, batch small DMAs
  tiling_imbalance → rebalance core workload, adjust block_dim partitioning
  no_overlap → restructure CopyIn/Compute/CopyOut stage boundaries for pipeline overlap

Special rule for scalar subtypes: When profiling shows aiv_scalar_ratio > 40%, inspect outer loops for:
  - scalar_loop_around_vector: for-loop contains Vector API calls (Axpy/Muls/ReduceSum)
  - scalar_reduce_after_vector: GetValue from Vector Reduce result into scalar array, then scalar aggregate
  - scalar_matvec_driver: GetValue per row to scale matrix rows (qRot / Pi × q patterns)
Do not stop at "the kernel already uses ReduceSum/Axpy/Muls". These are exactly the scalar-shell-around-vector anti-patterns (scalar loop wrapping Vector API calls).

Steps:
1. AscendC Translation - Translate TileLang design to AscendC kernel (ascendc-translator skill)
2. Degeneration check - validate_ascendc_impl.py + evaluate_ascendc.sh. See [Optimization Approach] for strategy application.
3. Functional verification - evaluate_ascendc.sh (accuracy) + 性能评测用 ops-profiling `msprof_profile_run.sh --quick --output-dir={output_dir} --device={npu}`（产出 performance.json / performance.log / perf_report.md）
   [禁止] 禁止手写自定义计时脚本（自造 measure_perf.py / event_timing.py / op_only_timing.py 等）替代标准评测——性能数据一律以 ops-profiling quick 模式产出的 performance.json / performance.log 为准
   [评估基准约束] evaluation_results.json 的 comparison 必须含 `ref_base: "pytorch"`；`base_time_ms` 必须是 `shared/model.py`（PyTorch 参考实现）测得的时间，`gen_time_ms` 是变体内核时间。**禁止**用父内核/基线内核作 base（否则 speedup 变成"变体 vs 父内核"，污染世界模型 score，导致最优判定错误）。缺 `ref_base` 或 base_time_ms 非 PyTorch 时间视为评估不合格。
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

[WARNING] ANTI-TRICK POLICY: Do NOT modify model.py. Do NOT skip/simplify computation in model_new_ascendc.py. Do NOT hardcode test shapes. Do NOT reduce/sample/truncate the test-case JSON or substitute a subset — evaluation MUST run on the FULL case set (content hash equal to <op_name>.json.bak). Violations = invalid variant.

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

If Mode=workflow_guided:
  本变体是归族 workflow 分支的专门变体。必须严格按下方 [WORKFLOW_REF] 中的 stage_order 阶段顺序优化，不得跳过任一阶段；[WORKFLOW_REF] 中若含 `traits`（正交 trait 列表，如 causal/sparse-blocktable/paged-decode/phase=bwd），按其对应的 workflow trait 追加表叠加对应策略（不改变主链顺序）：
  a. Read the family workflow file at: {workflow_file_path}（如 .claude/skills/ascendc-evolution-strategies/references/scenario-routing/convolution.md）
  b. 阶段 0 — 通用前置：过 _common-prerequisites.md C1-C5（命中用卡、未命中跳过）
  c. 阶段 1 — 分支专用路径：按 {branch} 分支的「判定条件 → 专用路径」实现（如 conv C-2 → P128 hi/lo bf16 3-GEMM、C-6 → im2col+Cube GEMM）
  d. 阶段 2 — 族内收尾：如 workflow 转 GEMM 主干，按 matmul.md M2/M3/M7 阶段收尾（多核切分、L2 复用、CV 流水）
  e. 阶段 3 — 门控纪律：遵守 workflow 的「门控纪律」段（如硬件 img2col 展开 k 序须 one-hot 探针实测、W<24 禁 LoadData3D、留环境变量开关）
  f. In implementation_note: 以 ## Workflow: {workflow_file} {branch} 开头，逐阶段记录做了什么、命中了哪些判定、跳过了哪些（含理由）
  策略相关资源（card + Playbook SOP）见下方 [STRATEGY RESOURCES] 段；workflow 分支专用路径的策略 Playbook 在其中。

If Mode=strategy_guided (default):
  The [MANDATORY OPTIMIZATION DIRECTION] already contains the strategy direction and key points.
  策略相关资源（card 主体内容 + Playbook SOP）已由主 agent 通过 source_key 程序化加载，见下方 [STRATEGY RESOURCES] 段；不需要再去 cannbot-knowledge 仓文件系统读取。

  若 [STRATEGY RESOURCES] 段为空 → 直接以 open_exploration 模式运行：从第一性原理 + Profiling Context 出发设计优化，**禁止去读策略卡原文**。
  If parent_solution_ref non-empty:
    Read parent kernel at: output/{op_name}_evo_{timestamp}/{parent_solution_ref}/kernel/
    Use as optimization starting point.
  {meta_prompt}

[WORKFLOW_REF]  ← 归族 workflow 分支保底节点专用段（v3.2.1 新增）
{workflow_ref_block}
（非 workflow_guided 节点此段为空，跳过）

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

[TILING DESIGN CHECKLIST]
若算子属于 ascendc-tiling-design 已覆盖族，在实现优化前必须逐项确认对应类别的 Tiling 设计要点（插件本地副本：`community/ops/ascendc-evolution-strategies/references/tiling-design/{family}/`，可自行修改维护）：

- **MatMul 类** (P104-P108)
  - [ ] Host tiling 使用 `MatmulApiTiling::GetTiling(TCubeTiling&)`，正确覆写 singleCoreM/N/Ka/Kb、baseM/N/K、usedCoreNum
  - [ ] 当 `totalBlock < aicCoreNum` 时执行多核切分扩展，保持 `ALIGNED_H=16` 且不低于 baseM/baseN
  - [ ] Kernel 端按 `(mIdx==mTotalCnt-1 ? mBaseTail : singleCoreM)` 调用 `SetSingleShape`，尾块有越界守卫
  - [ ] `TCubeTiling` 以阻塞拷贝方式写入 TilingHeader；CMake 链接 `torch torch_python c10`
  - [ ] 平台路由：按芯片/平台选择 `MatmulImpl` / `Blaze` / `tensor_api` 实现

- **FlashAttention 类** (P115)
  - [ ] 按 Roofline 估算确定 tile 大小，优先保证 Q/K 块在 L1/UB 复用
  - [ ] 多核切分覆盖 batch/head/seq 维度，softmax 变体正确选择 online-softmax 或分块归约路径
  - [ ] CopyIn/Compute/CopyOut 流水边界清晰，避免 Vector/Cube 单元空泡

- **Reduction 类** (P109-P110)
  - [ ] 合轴后得到 A/R 交替序列，按 A0 判断走 AR（尾轴）或 ARA（非尾轴）模式
  - [ ] 多轴归约逐级处理，每步输出 shape 降维；`alignedCols` 按 32B 对齐
  - [ ] UB accumulator 复用：减少中间 GM 回写，注意 `rLength` vs `rLengthAlign` 用途

- **Sort 类** (P111)
  - [ ] 采用两级归并（Sort32 + MrgSort）或 `alg-two-level-mrgsort` 推荐流程
  - [ ] 行长度按 64 元素倍数填充，索引 buffer 大小与对齐满足 API 要求
  - [ ] 批量行并行时 `repeat` 参数正确，避免跨行数据污染

- **Elementwise 类** (P112)
  - [ ] 选择向量化 tile size（如 fp16 128B 对齐），连续内存访问优先
  - [ ] 尾块使用 DataCopyPad 或 masked vector 处理，避免标量回退
  - [ ] 多核按输出张量均匀切分，避免冗余 DMA

- **Broadcast 类** (P113)
  - [ ] 识别 broadcast 轴，对齐小维度到 32B，使用 vectorized 连续搬运
  - [ ] 输入形状不同时优先合并 strides，避免逐元素标量索引
  - [ ] 输出 tile 与 broadcast 源 tile 的 offset 计算保证对齐

- **Conversion 类** (P114)
  - [ ] 建立输入→输出轴映射，按 32B 对齐分块搬运
  - [ ] ND transpose 使用多维 tiling，非连续维度通过 DataCopyPad 处理
  - [ ] 当前仅 Transpose 部分支持；其他 conversion 算子按 elementwise/broadcast 处理

完成 checklist 后，在 `implementation_note.txt` 中说明本算子属于哪一族、应用了哪些 Tiling 要点。

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
