# 世界模型操作指南

本文件定义了 `lingxi-evo` / `ops-evo` Agent 在进化过程中对 `world_model.json` 执行的四种核心操作。

**执行方式**：确定性逻辑由 `wm_ops.py` 脚本执行，LLM 推理由 Agent 补充。

| 操作 | 脚本命令 | Agent 补充 |
|------|---------|-----------|
| Init | Agent 推理（无脚本） | 算子分析 + 节点设计 |
| Select | `wm_ops.py select` | 无 |
| Refine | `wm_ops.py refine` | 失败诊断（`wm_ops.py diagnose`）+ Analyze |
| Analyze | Agent 推理（无脚本） | 更新 open_questions |

---

## 操作一：Init（初始化）

**时机**：步骤3（共享文件准备）完成后，进化轮次开始前执行一次。

**输入**：
- `output/{op_name}_evo_{timestamp}/shared/{op_name}_op_desc.json`
- `output/{op_name}_evo_{timestamp}/shared/{op_name}_functional.py`
- 基线评测结果（Mode B 专有）
- `plugins-community/ops-perf-evolution/skills/evolution-strategies/references/strategy-index.md`
- `parallel_num` 参数

**推理过程**（Agent 自身思考）：

```
1. 算子特性分析：
   - 这是内存密集型还是计算密集型算子？
     （判断依据：元素级操作→内存密集；矩阵乘→计算密集）
   - 数据类型：FP16/BF16/FP32？是否需要精度特殊处理？
   - 形状特征：输入shape固定还是可变？是否有对齐问题（非32字节倍数）？
   - 复杂度：是否有归约操作、广播、特殊数学函数？

2. 瓶颈假设（根据算子特性形成 open_questions）：
   - 内存密集型算子 → 首要假设是带宽利用率不足
   - 计算密集型算子 → 首要假设是计算流水线停顿
   - 存在尾块 → 假设尾块处理影响整体效率
   - FP16输入 → 假设精度问题可能影响正确性

3. 初始优化方向设计（每个方向 = 一个决策树节点）：
   - 数量：parallel_num × 2 个（确保前两轮不缺 open 节点）
   - 多样性要求：不同节点的 strategy_combination 不得完全相同
   - 覆盖范围：
     * 至少 2 个性能方向（P系列：双缓冲、分块、向量化等）
     * 1 个组合方向（P+P 组合）
     * 1 个精度/类型方向（A/D系列，如适用）
     * 1 个高难度深度方向（difficulty=4，复杂组合）
     * 1 个保守基础方向（difficulty=2，单一策略）
   - 参考 strategy-index.md 中的 "When to Apply" 列，识别匹配该算子类型的策略
   - 类型覆盖要求（用于 Select 多样性保底）：
     * 每个节点必须设置 optimization_type 字段
     * bandwidth/tiling/algorithm 三类各至少有 1 个节点
     * 推导规则：策略中 P1/P7/P10/P11 为主 → bandwidth；
       P2/P4/P5/P8 为主 → tiling；其余或 open_exploration → algorithm
     * D/A 系列策略不影响类型标签（它们是精度约束）

4. 难度估计原则：
   - difficulty=1: 单一简单策略（如仅P7对齐）
   - difficulty=2: 单一中等策略（如P1双缓冲）
   - difficulty=3: 两策略组合（如P1+P7）
   - difficulty=4: 三策略组合或复杂单策略（如P2+P4+P10）
   - difficulty=5: 已知失败（评测后自动设置，初始化时不使用）
```

**输出**：按照 `schema.md` 格式写入 `world_model.json`。

**兜底**：若 JSON 写入失败，向用户报告警告，设置运行时标志 `world_model_active = false`，后续使用 tiered sampling 兜底。

---

## 操作二：Select（动作选择）

**时机**：每轮进化开始时，创建并行目录前执行。

**输入**：`world_model.json`，`parallel_num`

**推理过程**（Agent 自身思考）：

```
1. 读取所有 status="open" 的节点列表

2. 对每个 open 节点计算效用分（统一公式，与 wm_ops.py 一致）：
   parent_node = decision_tree.nodes[node.parent_id]
   parent_score = parent_node.score（若为null则用1.0）

   w_root_explore = 2.0（若 parent_id == "root"）或 0.0（其他）
   w_evidence = 1.5（若父节点有 profiling_evidence）或 0.0
   w_expected_gain = min(3.0, 2.0 × log1p(expected_gain))，由 compute_expected_gain 算

   utility = 3.0 × parent_score          // 父节点表现越好，子节点越值得探索
           + 2.5 × (5 - difficulty)       // 难度越低，优先级越高
           + 0.75 × depth                 // 适当鼓励深度探索（已证明方向继续挖掘）
           + w_root_explore               // 根节点直接子节点享有探索加成
           + w_evidence                   // 父节点有深度 profiling 数据时加成
           + w_expected_gain              // v3.2 Stage 3 前瞻收益：策略能解决多少父瓶颈

   参数含义：
   | 参数 | 取值 | 说明 |
   |------|------|------|
   | parent_score | float ≥ 1.0 | 父节点的 speedup 评分，null 时用 1.0 |
   | difficulty | 1-5 整数 | 实现难度，越低优先级越高 |
   | depth | int ≥ 0 | 节点在决策树中的深度 |
   | w_root_explore | 2.0/0.0 | 根节点的直接子节点（parent_id=="root"）享有 +2.0 加成，保证第一层方向充分探索后才深化子分支 |
   | w_evidence | 1.5/0.0 | 父节点有 profiling_evidence（指令级深度分析）时加成，因为更精确的瓶颈数据使子节点策略推荐更可靠 |
   | w_expected_gain | 0.0~3.0 | v3.2 Stage 3 前瞻奖励：基于父 facts × 父 diagnosis.labels × 子 strategy_combination 反查 INDEX.json triggers，按 Amdahl 上限估"若策略完全消除瓶颈可获得的 delta speedup"，再经 log1p 衰减 + cap 3.0 转为权重。父无 facts/diagnosis 或子无 strategy 时 0；不引入回归风险，公式自然退化 |

3. 按 utility 降序排列所有 open 节点

**执行方式**（优先调用确定性脚本）：
```bash
SELECTIONS=$(python3 plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/wm_ops.py select \
  --path output/{op_name}_evo_{timestamp}/world_model.json \
  --n {parallel_num})
```
若脚本不可用，按上述公式手工计算。

4. 槽位分配（保底 + 放权，共 parallel_num 个槽位）：

   保底轮（确保类型覆盖）：
   - 对 bandwidth、tiling、algorithm 三类，各取该类型内效用分最高的 1 个 open 节点（strategy_guided 组）
   - 若某类型无 open 节点则跳过
   - 保底节点最多占 min(3, sg_slots) 个槽位

   剩余槽位（效用分驱动）：
   - 从未选中的 strategy_guided 节点中按效用分降序填充
   - 不再有类型约束

   **分支多样性约束**（防止单链深挖）：
   - 计算所有 open strategy_guided 节点中不同 parent_id 的数量 → `num_active_branches`
   - 每个 parent_id 最多贡献 `ceil(n / num_active_branches)` 个槽位
   - 保底轮和剩余槽位轮均受此约束
   - 若约束导致槽位不足，做无约束二次填充（保证不少返回节点）
   - 此约束解决的问题：passed 节点生成的子节点因 `3.0 × parent_score` 主导项获得最高 utility，
     在无约束时会占满所有 strategy_guided 槽位，导致其他分支被饿死

   open_exploration 专用槽位（不变）：
   - 槽位 (parallel_num-1) 保留给 open_exploration 组效用分最高的节点
   - 若无可用则由 strategy_guided 补位

   若 parallel_num == 1：取全局效用分最高节点，不区分类型

   - 若 open 节点总数不足：用自由探索占位节点补齐（strategy_combination=[], difficulty=2,
       description="自由探索，基于已有经验选择多样化策略"）

5. 将选中节点的 status 更新为 "in_progress"，写回 world_model.json

6. 输出选择结果（供 4.2 构建子agent提示词使用）：
   [{
     "parallel_index": 0,
     "node_id": "n1",
     "description": "双缓冲流水线 + 32字节对齐",
     "strategy_combination": ["P1", "P7"],
     "parent_solution_ref": null  // 父节点的solution_ref（如有）
   }, ...]
```

**兜底**：若读取 world_model.json 失败，返回空选择列表，子agent自由选择策略。

---

## 操作三：Refine（结果更新）

**时机**：每轮所有子agent完成后，收集结果后立即执行。

**执行方式**：确定性逻辑由 `wm_ops.py refine` 脚本完成，LLM 推理部分由 Agent 补充。

### 步骤1：脚本化更新（必须执行）

```bash
python3 plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/wm_ops.py refine \
    --wm-path {world_model_path} \
    --round {r} \
    --results-dir output/.../round_{r}/ \
    --parallel-map '{"0":"n1","1":"n2","2":"x0"}' \
    --task-type {task_type}
```

脚本自动完成：
- 读取每个变体的 `evaluation_results.json`
- 更新节点 `status` / `score` / `solution_ref`
- 从 `evolved.pipeline` 提取 `profiling_insight`（bottleneck 推断）
- 瓶颈迁移检测（`bottleneck_shift`）
- 按 mode 规则生成子节点（strategy_guided: 2或1, open_exploration: 1, profiling_driven: 条件生成）
- 动态阈值停滞计数（基于 `measurement_quality`）
- 输出 `pending_diagnosis.json`（失败节点列表）和 `round_summary`

### 步骤2：失败诊断（Agent LLM 补充）

对 `pending_diagnosis.json` 中的每个失败节点，Agent 读取 `implementation_note.txt` 推理 `failure_type`，然后调用：

```bash
python3 plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/wm_ops.py diagnose \
    --wm-path {world_model_path} \
    --node-id {node_id} \
    --failure-type {impl_error|strategy_infeasible} \
    --failure-reason "{一句话原因}"
```

诊断决策树（Agent 推理）：
- Layer 1 — 编译层：API 错误/参数类型 → `impl_error`；硬件不支持 → `strategy_infeasible`
- Layer 2 — 精度层：数量级错误 → `impl_error`；代码与策略吻合但精度不满足 → `strategy_infeasible`

脚本自动处理：`impl_error` 且 retry<2 → 生成修复子节点；`strategy_infeasible` 或 retry>=2 → 封锁节点(difficulty=5)。

### 步骤3：验证

```bash
python3 plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/wm_ops.py validate --path {world_model_path}
```

### 软剪枝（Soft Prune，自动执行）

Refine 和 Diagnose 完成后自动执行软剪枝：
- 遍历所有 `status="open"` 节点，沿 parent 链向上查找
- 若遇到已封锁祖先（`status=failed` 且 `difficulty>=5`），则该 open 节点被软剪枝（`difficulty` 设为 5）
- 遇到健康祖先（`status=passed/completed`）时停止查找，不会误剪
- 不删除节点，保持 `parent_id` 引用完整性

也可手动执行：
```bash
python3 plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/wm_ops.py prune --path {world_model_path}
```

**兜底**：若 refine 脚本失败，执行 tiered sampling（分层采样），不阻断进化流程。

---

## 操作四：Analyze（假设更新）

**时机**：Refine 完成后立即执行。

**输入**：更新后的 `world_model.json`，本轮所有评测结果

**推理过程**（Agent 自身思考）：

```
1. 读取 world_model.json

2. 基于本轮及历史评测结果，对 open_questions 进行推理：

   分析模式（从评测结果中归纳规律）：
   - 哪些策略组合取得了最好的效果？
     → "策略X在该算子上表现良好，应优先在后续变体中使用"
   - 哪些策略组合失败或效果不佳？
     → "策略Y导致编译失败，后续子节点应避免单独使用Y"
   - 已通过节点之间是否有共同特征？
     → "所有通过节点都包含双缓冲，这可能是关键因素"
   - 是否识别出新的瓶颈假设？
     → "P10（向量化）提升有限，可能不是带宽瓶颈，而是计算瓶颈"

3. 将归纳出的假设写入 open_questions（最多5条，用简洁的陈述句）：
   - 每条不超过50字
   - 直接可指导下一轮子节点生成的具体建议优先
   - 删除已被证伪的旧假设

4. 写回 world_model.json
```

**注意**：Analyze 的结果（open_questions）会在下一轮 Refine 的子节点生成阶段被参考，形成"假设→验证→新假设"的循环。

---

## 操作时序图

```
Step 3.5:  [Init] → 创建初始世界模型，生成 N×2 个 open 节点

Round 1:
  [Select] → 选 parallel_num 个 open 节点，标记 in_progress
  [并行生成] → lingxi-partial × parallel_num（每个变体有明确策略）
  [收集结果]
  [Refine]      → 更新节点状态，生成通过节点的子节点
  [CSV Profile] → 每个 passed 节点做 CSV 级瓶颈诊断，检测瓶颈迁移
  [Deep Profile] → 条件触发：瓶颈迁移/CSV盲区/停滞破局时做指令级空泡分析
  [Analyze]     → 更新 open_questions

Round 2:
  [Select] → 从 open 节点（含 Round1 生成的子节点）选 parallel_num 个
  [并行生成] → lingxi-partial × parallel_num
  [收集结果]
  [Refine]      → 继续更新树
  [CSV Profile] → 更新瓶颈诊断，检测相对上轮的瓶颈迁移
  [Deep Profile] → 条件触发（含与上次深度分析的 diff 对比）
  [Analyze]     → 继续更新假设
  ...

终止条件满足 → 输出最优解 + 保存 world_model.json 快照
```

---

## 效用函数参数调整建议

在默认效用函数 `utility = 3.0×parent_score + 2.5×(5-difficulty) + 0.75×depth + w_root_explore` 中：

| 场景 | 建议调整 |
|------|---------|
| 探索阶段（前1-2轮）| depth 权重降为 0（避免过早深度优先） |
| 剥削阶段（best_score已很高）| parent_score 权重提升到 4.0 |
| 停滞时（stagnation_count>0）| 强制选择 depth 最浅的节点（重新探索） |

---

## 停滞阈值配置

```
stagnation_window = 2   // 默认值
                        // 根据 max_rounds 自动调整：
                        // max_rounds <= 3 → stagnation_window = 1
                        // max_rounds == 4 → stagnation_window = 2
                        // max_rounds >= 5 → stagnation_window = 3

// 双计数器共享同一阈值，任意一个触发即停止（OR 逻辑）：
//   stagnation_count ≥ stagnation_window      → 全局停滞（本轮最佳 vs best_score×1.02）
//   stagnation_count_vs_base ≥ stagnation_window → 分支停滞（本轮最佳 vs 父节点得分）
```
