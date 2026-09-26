# 世界模型 JSON Schema 定义

本文件定义了世界模型的完整 JSON 数据结构。世界模型以文件形式持久化于磁盘：

```
output/{op_name}_evo_{timestamp}/world_model.json
```

---

## 顶层结构

```json
{
  "kernel_summary": "<字符串> 算子功能、计算模式、已知约束的一段描述（1-3句话）",

  "session": {
    "session_id": "ai_infra_sparse_flash_attention_gqa_ops-evo_20260430_202933",
    "start_time": "2026-04-30T20:29:33+0800",
    "requested_rounds": 5,
    "actual_rounds_completed": 3,
    "evo_dir": "output/ai_infra_sparse_flash_attention_gqa_ops-evo_20260430_202933",
    "op_name": "ai_infra_sparse_flash_attention_gqa"
  },
  // session: session 级身份锚定，防止 agent 失忆后误用历史目录
  //   session_id: 唯一标识，格式为 {op_name}_ops-evo_{timestamp}
  //   start_time: ISO8601 时间戳
  //   requested_rounds: 用户请求的最大轮数
  //   actual_rounds_completed: 实际已完成的轮数（由 refine 自动更新）
  //   evo_dir: 本次进化的根目录绝对路径
  //   op_name: 算子名称

  "baseline_performance": {
    "speedup": 1.0,       // 基线加速比（描述模式默认1.0，基线模式填入实际测量值）
    "time_ms": null       // 基线执行时间（ms），可为null
  },

  "decision_tree": {
    "nodes": {
      "<node_id>": { ... }   // 节点字典，key为节点ID
    }
  },

  "open_questions": [
    "<字符串> 当前未解答的性能假设或待验证的优化方向"
  ],

  "stagnation_count": 0,         // 连续无显著提升的轮数计数器（vs 全局 best_score×1.02）
  "stagnation_count_vs_base": 0, // 连续本轮最佳未超越父节点得分的轮数计数器（K-Search 风格）
  "best_score": 1.0,             // 迄今为止所有已评测节点中的最高 speedup 值
  "world_model_active": true,    // false 表示世界模型初始化失败，已回退到 tiered sampling
  "solution_db_path": null,      // solution_db.jsonl 的相对路径（血统追踪，可选）
  "hw_params": null,
  "discovered_strategies": [],
  "baseline_evidence": null
  // 基线 profiling 的硬映射证据（可选，根级字段）。
  // 由 wm_ops.py attach-baseline-evidence 在跑完 baseline_evaluation 后写入；
  // 结构与节点的 profiling_evidence 完全一致（见节点结构）。
  // 非 null 时，用于：
  //   1) select_nodes 中触发 baseline 对齐惩罚（见 compute_utility w_baseline_mismatch）
  //   2) 注入 ops-partial / lingxi-partial prompt 的 [Profiling Context] 作为 Baseline 行
  // null 表示 baseline pipeline 不可用或 attach-baseline-evidence 未被调用，
  // 系统回退到不做对齐惩罚、prompt 中跳过 Baseline 行的行为
  // open_exploration 节点发现并提炼的新策略 ID 列表（X 前缀，区别于人工编写的 P/A/D 系列）
  // 示例：["X1", "X2"]；这些策略已写入策略库，可被后续 strategy_guided 节点引用
  // 目标芯片硬件规格（由 npu-arch skill 在步骤3.5-HW 写入）
  // 非 null 时结构：{ chip_model, ub_size_bytes, core_num, peak_bw_gbps,
  //                   peak_vector_tflops_per_core, alignment_bytes, max_tile_fp16_double_buf }
  // null 表示查询失败或芯片型号不支持，依赖 hw_params 的后续步骤将静默跳过
}
```

---

## 节点结构（decision_tree.nodes 中的每个节点）

```json
{
  "id": "<字符串> 节点唯一标识符（如 'root', 'n1', 'n2', 'n1_1'）",

  "parent_id": "<字符串> 父节点ID，根节点为 null>",

  "description": "<字符串> 该节点的优化方向描述（1-2句话，说明用什么策略解决什么瓶颈）",

  "strategy_combination": ["P1", "P7"],
  // 适用于此节点的策略ID列表（对应 plugins-community/ops-perf-evolution/skills/evolution-strategies/references/strategy-index.md 中的ID）
  // 根节点和自由探索节点为 []

  "mode": "strategy_guided",
  // 节点探索模式，取值：
  //   "strategy_guided"   — 默认：使用 strategy_combination 中的策略，或从策略库自由选择
  //   "open_exploration"  — 开放探索：禁止读取策略库，LLM 从最优内核代码和 profiling 自主推理新优化方向
  //   "profiling_driven"  — Profiling驱动：基于父节点的 profiling_insight/profiling_evidence 中的具体瓶颈诊断，
  //                         由 LLM 自主设计针对性优化方案。不限于策略库，但必须针对诊断出的具体瓶颈。
  //                         与 open_exploration 的区别：profiling_driven 有明确的瓶颈靶点（来自profiling数据），
  //                         而 open_exploration 是完全自由的第一原理推理。
  //                         strategy_combination 字段保留为空数组 []，优化方向完全由 description 中的
  //                         profiling 瓶颈描述指导。

  "optimization_type": "bandwidth",
  // 性能优化类型标签，取值: "bandwidth" | "tiling" | "algorithm" | "register_opt" | "vf_fusion" | "instruction_sched"
  // - bandwidth: 带宽/搬运优化（P1双缓冲, P7对齐, P10向量化搬运, P11尾块, R5非对齐优化）
  // - tiling: 分块/并行优化（P2自适应分块, P4多核均衡, P5流水线同步, P8 UB分区）
  // - algorithm: 算法级重构（P13高层API, open_exploration, R6低延迟归约, R7 SIMD/SIMT混合）
  // - register_opt: [A5专用] 寄存器优化（R2寄存器复用、减少RegTensor数量避免spill/fill）
  // - vf_fusion: [A5专用] VF函数融合（R1多VF合并、减少冗余Load/Store）
  // - instruction_sched: [A5专用] 指令调度优化（R3双发射、R4 Hardware Loop规范、R8 Mutex同步）
  // 由 Init 和 Refine 自动推导，用于 Select 阶段的多样性保底约束
  // D/A 系列策略不参与类型推导（它们是精度约束，非性能方向）
  // A5 (351x/Regbase) 架构新增 register_opt/vf_fusion/instruction_sched 三个类型

  "status": "open",
  // 节点状态，取值：
  //   "open"        — 尚未执行，可被选择
  //   "in_progress" — 当前轮次已选中，正在执行
  //   "passed"      — 评测通过（编译成功 + 精度正确）
  //   "failed"      — 评测失败（编译失败 或 精度不通过）
  //   "completed"   — 特殊状态，仅用于根节点（作为起始点使用）

  "score": null,
  // 节点的 speedup 值（评测通过后填入，失败节点保持 null）

  "difficulty": 2,
  // 实现难度估计（1-5整数），失败节点设为5
  // 用于效用函数计算，难度越高优先级越低

  "depth": 1,
  // 节点在决策树中的深度（根节点为0，子节点递增）

  "solution_ref": null,
  // 评测通过后填入该变体的输出路径（如 "round_1/parallel_0"）
  // 失败或未执行节点为 null

  "parent_code_ref": null,
  // 父节点实现的 kernel 路径（由 wm_ops.py 从 solution_db 填充，可选）
  // 格式示例："round_1/parallel_0/{op_name}Custom/op_kernel/{op_name}_custom.cpp"

  "children": [],
  // 子节点ID列表（评测通过且有实质提升时生成子节点，子节点ID追加到此列表）

  "failure_type": null,
  // 失败节点的失败原因分类（仅 status="failed" 时有意义，其他节点为 null）：
  //   "impl_error"         — 实现错误：策略方向本身可行，但子agent代码写错了
  //                         （如：A6 Rsqrt未做NR迭代、Welford未用FP64累积）
  //                         → 该方向仍值得重试，difficulty 不设为5
  //   "strategy_infeasible" — 策略不可行：策略本身与该算子的精度/约束不兼容
  //                         （如：近似误差量级合理但仍超阈值）
  //                         → 封锁该方向，difficulty 设为5

  "failure_reason": null,
  // 一句话说明失败诊断依据（供 Analyze 和子节点 description 引用）
  // 示例："rmse=25.79 >> 1，判断为 Rsqrt 未做 NR 迭代所致（impl_error）"

  "retry_count": 0
  // 该策略方向已重试次数（impl_error 节点的修复子节点继承 parent.retry_count + 1）
  // retry_count >= 2 时即使 impl_error 也不再生成子节点（避免无限重试）

  "profiling_insight": null,
  // 性能瓶颈分析（可选字段，精度通过后由 lingxi-evo 步骤 4.4.P 填入）
  // 非 null 时结构：{ "bottleneck": "memory_bound", "recommended_strategies": ["P1","P2"],
  //                   "profiling_one_liner": "MTE2:45% | Vec:37% | ..." }
  // null 表示尚未分析或 profiling 数据不存在

  "profiling_evidence": null
  // 指令级深度空泡分析（可选字段，由 lingxi-evo 步骤 4.4.P2 在触发条件满足时填入）
  // 数据来源：ops-profiling skill 的 msprof_perf_summary.py --diagnose 分析结果
  // 非 null 时结构：
  // {
  //   "bottleneck_type": "mte2_stall",          // 瓶颈分类: mte2_stall / mte3_stall / tiling_imbalance / scalar_loading / scalar_compute / compute_bound / near_optimal / no_overlap / partial_overlap / undersize_transfer / icache_miss / bus_contention
  //   "d_class_pct": 62.5,                      // D类空泡占比（跨流水线同步等待）
  //   "c_class_pct": 15.3,                      // C类空泡占比（标量参数加载阻塞）
  //   "imbalance_ratio": 1.12,                  // 跨核负载不均衡比（>1.3 为不均衡）
  //   "primary_bottleneck": "MTE2",             // D类空泡的主等待流水线
  //   "suggested_strategies": ["P1", "P10"],    // 基于空泡分析推荐的策略
  //   "anti_strategies": ["P3"],                // 应避免的策略
  //   "description": "MTE2 数据加载等待是主瓶颈...",
  //   "top_recommendation": "启用双缓冲隐藏MTE2搬运延迟",
  //   "pattern_type": "periodic",               // 空泡模式: periodic / sporadic / cold_start_dominant / tail_dominant (T7)
  //   "overlap_status": "no_overlap",           // 流水线重叠状态: no_overlap / partial_overlap / good_overlap (T8)
  //   "dominant_subtype": "D_MTE2_WAIT",        // 主导空泡子类型 (二级分类)
  //   "dma_efficiency": {                       // DMA 搬运效率 (T9)
  //     "mte2_short_pct": 35.0,                 // MTE2 短搬运占比
  //     "mte3_short_pct": 20.0                  // MTE3 短搬运占比
  //   }
  // }
  // null 表示深度分析未触发、trace 数据不存在或分析失败
  // 与 profiling_insight 的关系：profiling_insight 是 CSV 级快速诊断（每轮必做），
  // profiling_evidence 是 trace 级深度分析（仅在特定条件触发时执行），两者互补
}
```

---

## 节点 ID 命名规范

- 根节点：`"root"`
- 第一层子节点：`"n1"`, `"n2"`, `"n3"`, ...
- 深层子节点：`"n1_1"`, `"n1_2"`, `"n2_1"`, ...（父节点ID + 下划线 + 序号）

---

## 完整示例（初始化后的世界模型）

```json
{
  "kernel_summary": "FastGELU算子，对输入张量逐元素应用GELU激活函数。计算密集度低，主要为内存带宽瓶颈。支持FP16/BF16，形状可变，存在尾块对齐问题。",

  "baseline_performance": {
    "speedup": 1.0,
    "time_ms": null
  },

  "decision_tree": {
    "nodes": {
      "root": {
        "id": "root",
        "parent_id": null,
        "description": "基线内核，未应用优化策略",
        "strategy_combination": [],
        "status": "completed",
        "score": 1.0,
        "difficulty": 1,
        "depth": 0,
        "solution_ref": null,
        "children": ["n1", "n2", "n3", "n4", "n5", "n6"]
      },
      "n1": {
        "id": "n1",
        "parent_id": "root",
        "description": "双缓冲流水线 + 32字节对齐，通过计算与数据搬运重叠提升吞吐量",
        "strategy_combination": ["P1", "P7"],
        "status": "open",
        "score": null,
        "difficulty": 2,
        "depth": 1,
        "solution_ref": null,
        "children": []
      },
      "n2": {
        "id": "n2",
        "parent_id": "root",
        "description": "自适应分块 + 多核负载均衡，针对不同输入形状自动选择最优分块策略",
        "strategy_combination": ["P2", "P4"],
        "status": "open",
        "score": null,
        "difficulty": 3,
        "depth": 1,
        "solution_ref": null,
        "children": []
      },
      "n3": {
        "id": "n3",
        "parent_id": "root",
        "description": "向量化数据搬运 + 尾块GatherMask处理，最大化内存带宽利用率",
        "strategy_combination": ["P10", "P11"],
        "status": "open",
        "score": null,
        "difficulty": 3,
        "depth": 1,
        "solution_ref": null,
        "children": []
      },
      "n4": {
        "id": "n4",
        "parent_id": "root",
        "description": "混合精度架构 + FP32中间计算，FP16输入使用FP32中间值保证精度",
        "strategy_combination": ["D1", "A1"],
        "status": "open",
        "score": null,
        "difficulty": 2,
        "depth": 1,
        "solution_ref": null,
        "children": []
      },
      "n5": {
        "id": "n5",
        "parent_id": "root",
        "description": "双缓冲 + 自适应分块组合，综合提升流水线效率和分块适应性",
        "strategy_combination": ["P1", "P2"],
        "status": "open",
        "score": null,
        "difficulty": 3,
        "depth": 1,
        "solution_ref": null,
        "children": []
      },
      "n6": {
        "id": "n6",
        "parent_id": "root",
        "description": "UB内存分区优化 + 流水线同步，精细化UB使用减少等待",
        "strategy_combination": ["P8", "P5"],
        "status": "open",
        "score": null,
        "difficulty": 4,
        "depth": 1,
        "solution_ref": null,
        "children": []
      }
    }
  },

  "open_questions": [
    "该算子是内存带宽瓶颈还是计算密集型？需要通过profiling确认",
    "不同输入形状（大shape vs 小shape）对最优分块策略的影响程度？",
    "尾块对齐问题是否对整体性能有显著影响？"
  ],

  "stagnation_count": 0,
  "stagnation_count_vs_base": 0,
  "best_score": 1.0,
  "world_model_active": true,
  "hw_params": null,
  "discovered_strategies": [],
  "baseline_evidence": null
}
```

---

## 完整示例（两轮进化后的世界模型）

```json
{
  "kernel_summary": "FastGELU算子，内存带宽瓶颈已确认，双缓冲效果显著。尾块对齐影响较小。",

  "baseline_performance": {
    "speedup": 1.0,
    "time_ms": 0.5
  },

  "decision_tree": {
    "nodes": {
      "root": {
        "id": "root", "parent_id": null, "description": "基线内核",
        "strategy_combination": [], "status": "completed", "score": 1.0,
        "difficulty": 1, "depth": 0, "solution_ref": null,
        "children": ["n1", "n2", "n3", "n4", "n5", "n6"]
      },
      "n1": {
        "id": "n1", "parent_id": "root",
        "description": "双缓冲流水线 + 32字节对齐",
        "strategy_combination": ["P1", "P7"],
        "status": "passed", "score": 2.3, "difficulty": 2, "depth": 1,
        "solution_ref": "round_1/parallel_0",
        "children": ["n1_1", "n1_2"]
      },
      "n1_1": {
        "id": "n1_1", "parent_id": "n1",
        "description": "在n1基础上增加向量化数据搬运，进一步提升带宽利用率",
        "strategy_combination": ["P1", "P7", "P10"],
        "status": "open", "score": null, "difficulty": 3, "depth": 2,
        "solution_ref": null, "children": []
      },
      "n1_2": {
        "id": "n1_2", "parent_id": "n1",
        "description": "在n1基础上增加多核负载均衡，分摊计算压力",
        "strategy_combination": ["P1", "P7", "P4"],
        "status": "open", "score": null, "difficulty": 3, "depth": 2,
        "solution_ref": null, "children": []
      },
      "n2": {
        "id": "n2", "parent_id": "root",
        "description": "自适应分块 + 多核负载均衡",
        "strategy_combination": ["P2", "P4"],
        "status": "passed", "score": 1.7, "difficulty": 3, "depth": 1,
        "solution_ref": "round_1/parallel_1",
        "children": []
      },
      "n3": {
        "id": "n3", "parent_id": "root",
        "description": "向量化数据搬运 + 尾块GatherMask处理",
        "strategy_combination": ["P10", "P11"],
        "status": "failed", "score": null, "difficulty": 5, "depth": 1,
        "solution_ref": null, "children": []
      }
    }
  },

  "open_questions": [
    "双缓冲（P1）是关键优化，所有变体都应包含它作为基础",
    "尾块处理（P11）导致编译失败，暂时避免单独使用P11",
    "n1分支值得深度探索：在P1+P7基础上叠加P10或P4的效果？"
  ],

  "stagnation_count": 0,
  "stagnation_count_vs_base": 0,
  "best_score": 2.3,
  "world_model_active": true,
  "hw_params": {
    "chip_model": "910B3",
    "ub_size_bytes": 196608,
    "core_num": 40,
    "peak_bw_gbps": 57.6,
    "peak_vector_tflops_per_core": 0.2304,
    "alignment_bytes": 32,
    "max_tile_fp16_double_buf": 16384
  },
  "discovered_strategies": ["X1"]
}
```

---

## 字段约束

| 字段 | 类型 | 约束 |
|------|------|------|
| `status` | string | 必须是 open/in_progress/passed/failed/completed 之一 |
| `score` | float 或 null | 通过后为正数，失败或未执行为 null |
| `difficulty` | int | 1-5 整数，失败节点强制设为 5 |
| `depth` | int | 非负整数，根节点为 0 |
| `strategy_combination` | string[] | 元素必须是 strategy-index.md 中存在的策略 ID |
| `failure_type` | string 或 null | 仅失败节点有值："impl_error" 或 "strategy_infeasible" |
| `failure_reason` | string 或 null | 一句话诊断说明，失败节点必填 |
| `retry_count` | int | 非负整数，impl_error 修复子节点继承 parent+1；≥2 时不再生成子节点 |
| `stagnation_count` | int | 非负整数，连续无提升轮数（vs best_score×1.02） |
| `stagnation_count_vs_base` | int | 非负整数，连续本轮最佳未超越父节点得分的轮数 |
| `best_score` | float | 始终 ≥ 1.0（至少等于基线） |
| `hw_params` | object 或 null | null 表示硬件查询失败；非 null 时含 chip_model、ub_size_bytes、core_num、peak_bw_gbps 等字段 |
| `mode` | string | 节点字段，必须是 strategy_guided、open_exploration 或 profiling_driven 之一；缺省视为 strategy_guided |
| `optimization_type` | string | 必须是 bandwidth/tiling/algorithm 之一 |
| `discovered_strategies` | string[] | 顶层字段，X 前缀策略 ID 列表；空列表表示本次运行尚未发现新策略 |
| `session` | object 或 null | session 身份锚定，非 null 时含 session_id、start_time、requested_rounds、actual_rounds_completed、evo_dir、op_name |
| `profiling_insight` | object 或 null | 节点可选字段；null 表示数据缺失或分析失败；非 null 时含 bottleneck、recommended_strategies、profiling_one_liner 三个子字段 |
| `profiling_evidence` | object 或 null | 节点可选字段；指令级深度空泡分析结果；null 表示未触发或分析失败；非 null 时含 bottleneck_type、d_class_pct、c_class_pct、imbalance_ratio、suggested_strategies、anti_strategies、pattern_type、overlap_status、dominant_subtype、dma_efficiency 等子字段 |
| `baseline_evidence` | object 或 null | 顶层可选字段；根级基线 profiling 证据（由 attach-baseline-evidence 写入）；结构同 `profiling_evidence`；null 表示基线 pipeline 不可用或该子命令未被调用；非 null 时必须 `bottleneck_type ∈ BOTTLENECK_STRATEGY_MAP` 的合法键 |
