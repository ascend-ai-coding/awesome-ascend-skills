# Meta Prompts for Operator Generation

本目录包含用于指导CAKE2算子生成的优化策略metaprompts。策略来源：43个生产级算子的专家实现 + 6个CV融合算子的IdeaPool专家实现分析。

## 目录结构

```
meta_prompts/
├── strategy_index.md          ← 主索引表（注入为 {meta_prompt}）
├── strategies/
│   ├── dtype_01_mixed_precision.md
│   ├── dtype_02_template_kernel.md
│   ├── dtype_03_tilingkey_dispatch.md
│   ├── dtype_04_fp8_int4_conversion.md
│   ├── dtype_05_bf16_special_handling.md
│   ├── perf_01_double_buffer.md
│   ├── perf_02_adaptive_tiling.md
│   ├── perf_03_small_d_optimization.md
│   ├── perf_04_load_balance.md
│   ├── perf_05_pipeline_sync.md
│   ├── perf_06_multi_algo_selection.md
│   ├── perf_07_data_alignment.md
│   ├── perf_08_ub_memory_mgmt.md
│   ├── perf_09_deterministic_output.md
│   ├── perf_10_vectorized_copy.md
│   ├── perf_11_tail_block_handling.md
│   ├── perf_12_broadcast_mask.md
│   ├── perf_13_special_algorithms.md
│   ├── perf_14_cv_pipeline_preload.md    ← CV融合/高级调优 (P14-P18)
│   ├── ...
│   ├── perf_18_l1_7buf_resident_partition.md
│   ├── perf_19_custom_pingpong_double_buffer.md  ← 高级数据搬运 (P19-P45)
│   ├── ...
│   ├── perf_45_topk_index_cache_iterator.md
│   ├── perf_46_matmulimpl_highlevel_api.md  ← CV融合算子架构模式 (P46-P52, IdeaPool)
│   ├── ...
│   ├── perf_52_l2_cache_hint.md
│   ├── acc_01_fp32_intermediate.md
│   ├── acc_02_welford_algorithm.md
│   ├── acc_03_rounding_mode.md
│   ├── acc_04_pipeline_barrier.md
│   ├── acc_05_softmax_stability.md
│   ├── acc_06_high_precision_rsqrt.md
│   ├── acc_07_index_boundary.md
│   ├── acc_08_quant_precision.md
│   └── _unmatched.md          ← 未分类策略（供审查）
└── README.md
```

## 使用方法

### Agent工作流（推荐）

在dsl_lowering步骤中：

1. 读取 `plugins-community/ops-perf-evolution/skills/evolution-strategies/references/strategy_index.md`
2. 选择所有适用于当前算子的策略ID
3. 读取每个引用的 `strategies/XXX.md` 详情文件
4. 在 `tiling_pass`、`init_pass`、`process_pass`、`process_nonaligned_pass` 中应用选定的模式

### 按算子类型快速参考

| 算子类型 | 推荐策略 |
|---------|---------|
| LayerNorm / RMSNorm | D1, D2, P1, P2, P3, A1, A2, A6 |
| 量化算子 | D1, D4, D3, P2, P4, A3, A8 |
| 逐元素操作 | D1, D2, P1, P2, A1, A4 |
| Softmax / 注意力 | D1, P1, P5, A3, A4, A5 |
| Pooling / Gather | D1, P1, P2, P11, A7 |
| 优化器算子 | D1, P1, P2, P9, A1 |
| 广播/掩码算子 | D1, P1, P12, A5 |
| 索引/散射算子 | D1, P1, P7, A7 |
| 特殊/复杂算子 | D2, D3 |
| 排序/选择算子 (NMS, TopK) | D1, P1, P7, **P96**, **P97** |
| 归约算子 (GroupNorm, LayerNorm) | D1, D2, P1, P2, P3, A1, A2, A6, **P94**, **P95**, **P98** |
| CV Matmul (Cube+Vector) | D1, D3, P1, P4, P7, A3, P14, P17, P46, P47, P49, P51, P52 |
| CV FFN (MoE) | D1, D3, P1, P4, A1, P46, P48, P49, P50 |
| Flash Attention | D1, P1, P5, A4, A5, P14, P16, P17, P18, P29, P38 |

## 策略概览

### 多数据类型支持 (D1-D5)

| ID | 策略 | 适用场景 |
|----|------|---------|
| D1 | 混合精度架构 | 任何FP16/BF16输入的算子 |
| D2 | 模板化内核类型分发 | 编译时多类型内核 |
| D3 | TilingKey驱动类型分发 | Host端多类型选择 |
| D4 | FP8/INT4量化转换 | 量化输出算子 |
| D5 | BF16/多平台特殊处理 | BF16输出或多硬件平台 |

### 性能优化 (P1-P52)

| ID | 策略 | 适用场景 |
|----|------|---------|
| P1 | 双缓冲机制 | 任何内存密集型内核 |
| P2 | 自适应分块策略 | 可变行/列维度 |
| P3 | 小D多行合并优化 | Hidden size ≤ 640 |
| P4 | 多核负载均衡 | 数据分布不均匀 |
| P5 | 流水线同步控制 | 所有双缓冲内核 |
| P7 | 32B对齐与DataCopyPad | 非对齐输入 |
| P8 | UB内存分区管理 | 多UB张量内核 |
| P9 | 确定性输出 | 训练算子 |
| P10 | 向量化数据拷贝 | CopyIn/CopyOut优化 |
| P11 | 尾块处理 | Pooling/Gather |
| P12 | 广播与掩码操作 | 带Mask输入、广播维度 |
| P94 | 标量归约向量化 (ReduceSum/ReduceMax) | 所有含 sum/max 标量循环的算子 |
| P95 | 标量逐元素向量化 (Adds/Muls 链) | 所有含 y[i]=f(x[i]) 标量循环的算子 |
| P96 | 硬件排序引擎 (Sort32/MrgSort) | NMS, TopK, 任何含排序/选择的算子 |
| P97 | 零拷贝子张量切片 | 大 tensor 子集批量向量运算 |
| P98 | 混合标量/向量自适应调度 | 可变 workload 大小的 reduction/norm |
| P99 | BroadCast + 批量向量运算 | 小向量广播到矩阵做逐行 Muls/Adds |
| P100 | GatherMask 替代 SetValue/GetValue | 固定间隔取元素、解交织、RoPE 奇偶分拆 |

### CV融合/高级调优 (P14-P18) — Layer L1

| ID | 策略 | 适用场景 |
|----|------|---------|
| P14 | CV流水预发射 | Flash Attention 3-slot task cache |
| P16 | 代价模型分核 | Alignment-aware cost model |
| P17 | CV并行基本块设计 | Cube+Vector 并行块 |
| P18 | L1七缓冲常驻分区 | Flash Attention L1 分区 |

### 高级数据搬运 (P19-P45) — Layer L1

| ID | 策略 | 适用场景 |
|----|------|---------|
| P19-P22 | 队列管理与双缓冲 | 自定义Ping-Pong、三缓冲、TQueBind |
| P24-P26 | DataCopy参数与对齐 | ND2NZ、填充对齐、步长转置 |
| P28-P30 | 同步与流水控制 | HardEvent、跨核同步、密集PipeBarrier |
| P32-P33 | 特殊搬运模式 | 散射块步长、聚集偏移表 |
| P34-P45 | 缓冲驻留与复用 | 权重常驻、累加器、Workspace复用 |

### CV融合算子架构模式 (P46-P52) — Layer L1, 来源: IdeaPool

| ID | 策略 | 适用场景 |
|----|------|---------|
| P46 | MatmulImpl高阶API | 替代手写matmul流水线 |
| P47 | 对角线块调度 | 多核M×N分块L2优化 |
| P48 | 多量化模式编译期分发 | A8W8/A4W4/A16W8/A8W4 MSD |
| P49 | 硬件加速反量化 | AscendDequant硬件融合 |
| P50 | SwiGLU融合流水线 | MoE FFN融合 (narrow scope) |
| P51 | 动态AIC/AIV核配比 | K维阈值切换 |
| P52 | L2 Cache Hint优化 | 多核竞争L2禁用 |

### 精度优化 (A1-A8)

| ID | 策略 | 适用场景 |
|----|------|---------|
| A1 | FP32中间计算 | BF16/FP16精度要求 |
| A2 | Welford数值稳定算法 | LayerNorm, BatchNorm |
| A3 | 舍入模式控制 | 任何Cast到低精度 |
| A4 | SetFlag/WaitFlag事件同步 | 跨管道数据依赖 |
| A5 | 数值安全与特殊值处理 | Softmax, NaN/Inf风险 |
| A6 | 高精度rsqrt (Newton-Raphson) | 需要rsqrt的规范化算子 |
| A7 | 索引与边界安全处理 | Gather/Scatter, 用户控制索引 |
| A8 | 量化专用精度处理 | 量化输出、自定义浮点格式 |

## 更新维护

使用 `build_strategy_index.py` 从IdeaPool重新生成：

```bash
python3 build_strategy_index.py
```

该脚本会：
1. 读取所有 `IdeaPool/*/expert_ideas.json`
2. 按策略主题分组（关键字匹配，跨类别）
3. 在每个组内检测变体（Jaccard相似度）
4. 生成61个策略详情文件和主索引表

## 进化发现策略 (P94-P98) — 标量计算向量化系列

来源：NMS (30_NMS)、GroupNorm (11_GroupNorm_evo)、TopK (TopK_evo_0629) 三次进化优化，跨算子验证。

| ID | 策略 | 适用场景 |
|----|------|---------|
| P94 | 标量归约 → ReduceSum/ReduceMax | 所有含 sum/max 标量循环的算子 |
| P95 | 标量逐元素 → Adds/Muls 向量链 | 所有含 `y[i]=f(x[i])` 标量循环的算子 |
| P96 | 硬件 Sort32/MrgSort 替代标量排序 | NMS, TopK, 任何含排序/选择的算子 |
| P97 | 零拷贝子张量切片 + 批量向量运算 | 需要从大 tensor 取子集做批量运算的算子 |
| P98 | 混合标量/向量路径自适应调度 | 有可变 workload 大小的 reduction/norm 算子 |
| P99 | BroadCast + 批量向量运算 | 小向量广播到矩阵做逐行缩放 |
| P100 | GatherMask 替代 SetValue/GetValue | 解交织、RoPE 奇偶分拆、固定间隔取元素 |

### 协同关系

```
P94 (标量归约向量化) + P68 (低延迟归约) → 最优归约性能
P95 (逐元素向量化) + P69 (UB 融合链) → 最小化 GM 往返
P96 (硬件排序) + P1 (双缓冲) + P7 (32B对齐) → TopK/NMS 完整方案
P97 (零拷贝切片) + P95 (向量链) → NMS IoU 完整方案
P98 (混合调度) + P94 + P95 → 全 workload 覆盖无退化
```

## 质量保证

- **来源**: 43个生产级算子 + 6个CV融合算子的专家实现
- **覆盖率**: 431/431条策略（100%）已分类
- **未匹配**: 0条策略（全部分类成功）

---

**版本**: v5.0 (策略扩展版)
**最后更新**: 2026-04-09
**来源**: 43个生产级算子 + 6个CV融合算子的专家实现
