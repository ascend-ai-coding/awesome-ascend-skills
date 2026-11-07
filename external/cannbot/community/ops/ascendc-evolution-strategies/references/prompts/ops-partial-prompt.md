# ops-partial 子agent Prompt 模板

供 ops-evo 步骤 4.3.2 中启动 ops-partial 子agent 时使用。
主 agent 读取此模板，按 ops-evo.md 中的"Prompt 变量填充规则"填充 `{变量}` 后作为 prompt 传给子 agent。

---

```
[TASK]
Modify, build, and evaluate AscendC kernel code for {op_name} operator in ops repository.

Operator: {op_description_from_def_cpp}
Source: {output_dir}/shared/original/
Output: {output_dir}/round_{r}/parallel_{p}/modified_files/

[MANDATORY OPTIMIZATION DIRECTION]
你被分配的优化方向是:
  方向描述: {node_description}
  策略组合: {strategy_combination}
  模式: {mode}

[WARNING] 你必须严格按此方向实现优化。
- 禁止偏离到其他方向（如仅调整 tiling 阈值、仅改 BUFFER_NUM 等简单参数修改不算"实现策略"）
- 你的修改必须体现上述方向描述中的核心优化思路
- 如果此方向在技术上不可行，在 implementation_note.txt 中说明原因，而不是静默切换到其他方向
- 禁止与兄弟变体做同一件事：若你的策略组合与下方任一兄弟的 sig 重叠超过 60%，必须在 implementation_note.txt 写明差异化实现点。

同轮其他变体的方向（禁止重复）:
{other_variants_summary}

[Optimization Approach]
If Mode=open_exploration:
  DO NOT read strategy files. Reason from first principles.
  Analyze bottleneck and profiling data, design novel optimization approach.

If Mode=profiling_driven:
  DO NOT read strategy files. The node description contains specific profiling bottleneck data — use it as your optimization target.
  a. Read parent kernel code and understand current implementation
  b. Parse the profiling bottleneck from node description (e.g., "MTE2 stall 62%", "scalar compute blocking 40%")
  c. Design a TARGETED optimization specifically addressing the diagnosed bottleneck
  d. Modify kernel/tiling files with targeted patches (don't rewrite from scratch)
  e. In implementation_note: state what profiling bottleneck was targeted and how

If Mode=strategy_guided:
  The [MANDATORY OPTIMIZATION DIRECTION] already contains the strategy direction and key points.
  策略相关资源（card 主体内容 + Playbook SOP）已由主 agent 通过 source_key 程序化加载，见下方 [STRATEGY RESOURCES] 段；不需要再去 cannbot-knowledge 仓文件系统读取（避免 token 浪费 + 跨 session 重读）。

  若 [STRATEGY RESOURCES] 段为空 → 直接以 open_exploration 模式运行：从第一性原理 + Profiling Context 出发设计优化，**禁止去读策略卡原文**。

[STRATEGY RESOURCES]  ← v3.2 新增段，由主 agent 用 query_strategies / load_playbook 程序化注入
{strategy_resources_block}

[PRECONDITIONS — 已硬过滤]  ← v3.2 新增段
本变体的策略组合已通过 Preconditions 硬门控（详见 wm_ops filter-candidates 在节点 filtered_by 字段的记录）。
- 如果你怀疑某个策略不适用，**不要自行跳过**：在 implementation_note.txt 末尾报告"该策略 PX 实际不可应用，理由 Y"，由主 agent 在下轮 refine 时决定是否扩大 Preconditions
- 不要再质疑适用性 — 已通过的就是适用的

[PLAYBOOK EXECUTION]  ← v3.2 新增段（强约束 R11）
若 [STRATEGY RESOURCES] 中含 Playbook 段（标记为 "## Playbook: PX_*"），必须**严格按 Step 1-5 SOP 执行**，并在 implementation_note.txt 中按以下结构写作：

```markdown
## Strategy: P1 - Double Buffer        ← 每个采纳的有 Playbook 的策略一段

### Playbook Step 1: 定位关键结构
<grep 输出 + 文件行号记录>

### Playbook Step 2: 改造计划表
| 元素 | 当前值 | 目标值 | 修改位置 |
| ... | ... | ... | ... |

### Playbook Step 3: 代码改造
形态识别：α / β / γ
<改造对照 + Variant Notes 说明>

### Playbook Step 4: 约束复核
<UB 占用 / L1 预算 / 边界检查的具体计算>

### Playbook Step 5: 编码并自检
<5 条 grep 命令的实际运行结果，全部期望值通过>
```

⚠️ R11 强约束：SubagentStop hook 会自动 audit 本文件，缺失任一 Strategy 的任一 Step 1-5 → 阻塞退出。
- 老形态（无 ## Strategy: PX header）暂兼容，但新 partial 应按上述格式
- 没有 Playbook 的策略不强制 Step 段落
- "Playbook Step" 必须是中文/英文标点之间合法的 markdown header (### 开头)

[TILING DESIGN CHECKLIST]
若算子属于 ascendc-tiling-design 已覆盖族，在实现优化前必须逐项确认对应类别的 Tiling 设计要点：

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

[Profiling Context]
Baseline:  bn={baseline_bottleneck_type} | recommended={baseline_suggested_strategies} | anti={baseline_anti_strategies}
Parent:    bn={bottleneck} | pipeline={profiling_one_liner} | recommended={recommended_strategies}
{若有 profiling_evidence:}
Bubble Analysis: D-class={d_class_pct}% C-class={c_class_pct}%
Primary stall: {primary_bottleneck}
DMA efficiency: MTE2 short={mte2_short_pct}% MTE3 short={mte3_short_pct}%
Pattern: {pattern_type} | Overlap: {overlap_status}
{/若}

[Alignment — 实现代码前必检]
本变体 opt_type={optimization_type}；strategy_combination={strategy_combination}
- 若 strategy_combination 含 Baseline anti 中任一策略 → 必须移除，或在 implementation_note.txt 第一行写明例外理由
- 若 opt_type 与 Baseline bn 相悖（示例：bn=compute_bound 但 opt_type=bandwidth）→ 在 implementation_note.txt 写明"故意背离"的理由
- 若 Baseline 行任一字段为 `N/A`，说明根级 baseline_evidence 未挂载，跳过对齐检查

Bottleneck-targeted guidance:
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

Hardware: {hw_params_one_liner}

Steps:
1. Read original kernel: shared/original/ (op_kernel/*.cpp, *.h, op_host/*_tiling.h, *_tiling.cpp)
2. If parent_solution_ref non-empty: read parent variant from {output_dir}/{parent_solution_ref}/modified_files/
3. Apply optimization per [MANDATORY OPTIMIZATION DIRECTION]
4. Save outputs (3 files, **all mandatory**) — 注意**两个不同目录**:

   **在 `{output_dir}/round_{r}/parallel_{p}/` 顶层**（直接放 parallel_{p}/ 下）:
   - **`implementation_note.txt`** — ≥100 字符的自由 narrative，描述：
     · 应用了哪些策略（如 P1+P5）+ 参考了哪些 Playbook 段落（如 P1 Step 3B）
     · 实际做了哪些改造（不仅是常量改动）+ 为什么这样改
     · 遇到的约束 / Trade-off / 待优化点
     **R12 hard block 会硬检查**：缺文件或 < 100 字符 → SubagentStop hook 阻塞退出。
     **路径强约定**：`{output_dir}/round_{r}/parallel_{p}/implementation_note.txt`，
     **不要**放在 `modified_files/` 子目录内（事后审计/ledger 工具按顶层路径查找）。
     自由 narrative 即可，不强制 ## Strategy / ### Step 格式。

   **在 `{output_dir}/round_{r}/parallel_{p}/modified_files/` 子目录**:
   - `modified_files/*.cpp, *.h` — 改过的算子源码（保持原始相对结构 op_kernel/*.cpp 等）
   - `modified_files/code_changes.json` — 结构化变更摘要
6. Apply modified files to worktree and build
7. Evaluate with eval-lock (baseline vs evolved on bound device)

Modification Constraints (STRICTLY ENFORCED):
- op_kernel/*.cpp, *.h: Free to modify (do not change call signatures)
- op_host/*_tiling.h, *_tiling.cpp: Allowed to modify (do not change field names)
- op_host/*_def.cpp: Restricted (compile options/optimization attributes ONLY; NEVER change inputs/outputs/type constraints)

[BUILD & EVAL CONTEXT]
Worktree Path: {worktree_base}/round_{r}_p{p}
Worktree Repo Root: {worktree_base}/round_{r}_p{p}
Worktree Op Path: {worktree_base}/round_{r}_p{p}/{OP_PATH_RELATIVE}
Z-Search Root: {z_search_root}
Install Path: $(pwd)/output/{op_name}_ops-evo_{timestamp}/round_{r}/parallel_{p}/evolved
Baseline Install Path: $(pwd)/output/{op_name}_ops-evo_{timestamp}/baseline
Device ID: {session_device_id}
Eval Lock Path: {eval_lock_path}
Baseline Cache: $(pwd)/output/{op_name}_ops-evo_{timestamp}/baseline_evaluation.json
Call Spec: $(pwd)/output/{op_name}_ops-evo_{timestamp}/shared/call_spec.json
Op Name: {op_name}
Build Op Name: {build_op_name}
SOC: {soc}
Task Type: {task_type}
Eval Lock Timeout: 300

[HISTORY]
{open_questions_rendered}

Output all modified files to: {output_dir}/round_{r}/parallel_{p}/modified_files/
```
