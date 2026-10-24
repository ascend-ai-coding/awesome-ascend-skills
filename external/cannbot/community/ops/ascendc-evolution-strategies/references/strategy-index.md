# AscendC Kernel Optimization Strategy Index

Source: 43 production-grade operators + 6 CV fused operator expert implementations (IdeaPool). Select all applicable strategies for your operator, then **Read each referenced detail file** before writing AscendC code.

> Detail File 列路径为 **cannbot-knowledge 仓相对路径**（`knowledge/` 前缀），按兄弟检出 `../cannbot-knowledge` 或环境变量 `CANNBOT_KNOWLEDGE_ROOT` 解析；Playbook 也可用 `scripts/load_playbook.py --strategy-ids <ID>` 程序化加载（自动剥离 frontmatter）。策略卡机器元数据（bottlenecks/op_families/complexity/conflicts/synergizes）的权威来源为 `preconditions/{ID}.yaml` 的 `strategy_meta` 块。

## 编号说明
策略卡共 161 张：P 系列（性能，P1-P147）/ D 系列（数据类型，D1-D5）/ A 系列（精度，A1-A8）/ R 系列（A5 架构，R1-R8）。P 系列编号存在间隙（P15, P23, P27, P31, P36, P39, P141），为预留位。
所有有效策略 ID 均在下方表格中列出。Agent 在 strategy_combination 中只应使用下方表格中的 ID。

## Layer 定义
- **L0 (Universal)**: 适用于几乎所有算子的通用策略（D1-D5, P1-P12, A1-A8）
- **L1 (Advanced)**: 适用于特定类别算子的高级策略（P14-P126）
- **L2 (Discovered)**: 进化过程中发现的新策略（X1+）

## Tags 体系
- `datapath`: 通用数据搬运通路优化
- `cv_fusion`: Cube+Vector 融合算子
- `flash_attention`: Flash Attention 系列
- `quantization`: 量化相关
- `moe`: MoE 模型特定
- `matmul`: 矩阵乘相关
- `manual`: 需要手动调优

---

## 多数据类型支持策略 (D1-D5) [Layer: L0]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| D1 | Mixed precision architecture | L0 | datapath | Any op with FP16/BF16 input | knowledge/contrib/ascendc/runbooks/performance/d1_mixed_precision.md |
| D2 | Template kernel type dispatch | L0 | datapath | Compile-time multi-type kernel | knowledge/contrib/ascendc/runbooks/performance/d2_template_kernel.md |
| D3 | TilingKey-driven type dispatch | L0 | datapath | Host-side multi-type selection | knowledge/contrib/ascendc/runbooks/performance/d3_tilingkey_dispatch.md |
| D4 | FP8/INT4 quantization conversion | L0 | quantization | Quantization output operators | knowledge/contrib/ascendc/runbooks/performance/d4_fp8_int4_conversion.md |
| D5 | BF16-specific handling | L0 | datapath | BF16 output with rounding control | knowledge/contrib/ascendc/runbooks/performance/d5_bf16_special_handling.md |

## 性能优化策略 — 通用 (P1-P12) [Layer: L0]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P1 | Double buffering (BUFFER_NUM=2) | L0 | datapath | Any memory-bound kernel | knowledge/contrib/ascendc/runbooks/performance/p1_double_buffering.md |
| P2 | Adaptive tiling (Split-N vs Split-D) | L0 | datapath | Variable row/column dimensions | knowledge/contrib/ascendc/runbooks/performance/p2_adaptive_tiling.md |
| P3 | Small-D multi-row merging | L0 | datapath | Hidden size ≤ 640 or small D | knowledge/contrib/ascendc/runbooks/performance/p3_small_d_optimization.md |
| P4 | Multi-core load balancing | L0 | datapath | Uneven data distributions | knowledge/contrib/ascendc/runbooks/performance/p4_load_balance.md |
| P5 | Pipeline sync (PipeBarrier/events) | L0 | datapath | All double-buffered kernels | knowledge/contrib/ascendc/runbooks/performance/p5_pipeline_sync.md |
| P7 | 32B alignment + DataCopyPad | L0 | datapath | Non-aligned input shapes | knowledge/contrib/ascendc/runbooks/performance/p7_data_alignment.md |
| P8 | UB memory partitioning | L0 | datapath | Kernels with multiple UB tensors | knowledge/contrib/ascendc/runbooks/performance/p8_ub_memory_partitioning.md |
| P9 | Deterministic output (workspace) | L0 | datapath | Training ops needing reproducibility | knowledge/contrib/ascendc/runbooks/performance/p9_deterministic_output.md |
| P10 | Vectorized data copy | L0 | datapath | CopyIn/CopyOut optimization | knowledge/contrib/ascendc/runbooks/performance/p10_vectorized_copy.md |
| P11 | Tail block handling (GatherMask) | L0 | datapath | Pooling/gather with uneven splits | knowledge/contrib/ascendc/runbooks/performance/p11_tail_block_handling.md |
| P12 | Broadcast & mask operations | L0 | datapath | Operators with mask inputs, broadcasting | knowledge/contrib/ascendc/runbooks/performance/p12_broadcast_mask.md |

## 性能优化策略 — CV融合/高级调优 (P14-P18) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P14 | CV pipeline preload (3-slot task cache) | L1 | cv_fusion, flash_attention | Flash Attention Cube-Vector 流水预发射 | knowledge/contrib/ascendc/runbooks/performance/p14_cv_pipeline_preload.md |
| P16 | Cost-driven core partition | L1 | cv_fusion, manual | Alignment-aware 代价模型分核 | knowledge/contrib/ascendc/runbooks/performance/p16_cost_driven_core_partition.md |
| P17 | CV parallel block sizing | L1 | cv_fusion | CV 并行基本块设计 | knowledge/contrib/ascendc/runbooks/performance/p17_cv_parallel_block_sizing.md |
| P18 | L1 7-buffer resident partition | L1 | cv_fusion, flash_attention | L1 七缓冲常驻分区 | knowledge/contrib/ascendc/runbooks/performance/p18_l1_7buf_resident_partition.md |

## 性能优化策略 — 高级数据搬运 (P19-P45) [Layer: L1]

### 队列管理与双缓冲 (P19-P22)

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P19 | Custom ping-pong double buffer | L1 | datapath | 自定义 Ping-Pong 双缓冲管理 | knowledge/contrib/ascendc/runbooks/performance/p19_custom_pingpong_double_buffer.md |
| P20 | Triple buffer rotation | L1 | datapath | 三缓冲轮转（搬入/计算/搬出完全重叠） | knowledge/contrib/ascendc/runbooks/performance/p20_triple_buffer_rotation.md |
| P21 | Matrix 2×2 buffer policy | L1 | datapath | 2×2 矩阵缓冲策略 | knowledge/contrib/ascendc/runbooks/performance/p21_matrix2x2_buffer_policy.md |
| P22 | TQueBind bidirectional reuse | L1 | datapath | TQueBind 双向复用 | knowledge/contrib/ascendc/runbooks/performance/p22_tquebind_bidirectional_reuse.md |

### DataCopy 参数与对齐 (P24-P26)

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P24 | ND2NZ format copy | L1 | datapath | ND→NZ 格式转换搬运 | knowledge/contrib/ascendc/runbooks/performance/p24_nd2nz_format_copy.md |
| P25 | DataCopy pad alignment | L1 | datapath | DataCopy 填充对齐优化 | knowledge/contrib/ascendc/runbooks/performance/p25_datacopy_pad_alignment.md |
| P26 | Stride fused transpose | L1 | datapath | 步长融合转置 | knowledge/contrib/ascendc/runbooks/performance/p26_stride_fused_transpose.md |

### 同步与流水控制 (P28-P30)

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P28 | HardEvent producer-consumer | L1 | datapath | HardEvent 生产者-消费者同步 | knowledge/contrib/ascendc/runbooks/performance/p28_hardevent_producer_consumer.md |
| P29 | Cross-core AIC-AIV sync | L1 | cv_fusion | 跨核 AIC-AIV 同步 | knowledge/contrib/ascendc/runbooks/performance/p29_crosscore_aic_aiv_sync.md |
| P30 | Dense PipeBarrier vector | L1 | datapath | 密集 PipeBarrier 向量同步 | knowledge/contrib/ascendc/runbooks/performance/p30_dense_pipebarrier_vector.md |

### 特殊搬运模式 (P32-P33)

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P32 | Scatter block stride | L1 | datapath | 散射块步长搬运 | knowledge/contrib/ascendc/runbooks/performance/p32_scatter_block_stride.md |
| P33 | Gather offset table | L1 | datapath | 聚集偏移表搬运 | knowledge/contrib/ascendc/runbooks/performance/p33_gather_offset_table.md |

### 缓冲驻留与复用 (P34-P45)

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P34 | Weight load-once reuse | L1 | datapath | 权重一次加载多次复用 | knowledge/contrib/ascendc/runbooks/performance/p34_weight_load_once_reuse.md |
| P35 | TBuf resident accumulator | L1 | datapath | TBuf 常驻累加器 | knowledge/contrib/ascendc/runbooks/performance/p35_tbuf_resident_accumulator.md |
| P37 | Grad accumulator cross-batch | L1 | datapath | 梯度跨 batch 累加器 | knowledge/contrib/ascendc/runbooks/performance/p37_grad_accumulator_cross_batch.md |
| P38 | Softmax state resident | L1 | flash_attention | Softmax 状态常驻 buffer | knowledge/contrib/ascendc/runbooks/performance/p38_softmax_state_resident.md |
| P40 | Workspace double buffer | L1 | datapath | Workspace 双缓冲 | knowledge/contrib/ascendc/runbooks/performance/p40_workspace_double_buffer.md |
| P41 | Gamma/scale resident | L1 | datapath | Gamma/scale 常驻 buffer | knowledge/contrib/ascendc/runbooks/performance/p41_gamma_scale_resident.md |
| P43 | Grad ping-pong deferred scatter | L1 | datapath | 梯度 Ping-Pong 延迟散射 | knowledge/contrib/ascendc/runbooks/performance/p43_grad_pingpong_deferred_scatter.md |
| P44 | Transpose buffer resident | L1 | datapath | 转置 buffer 常驻 | knowledge/contrib/ascendc/runbooks/performance/p44_transpose_buffer_resident.md |
| P45 | TopK index cache iterator | L1 | datapath | TopK 索引缓存迭代器 | knowledge/contrib/ascendc/runbooks/performance/p45_topk_index_cache_iterator.md |
## 性能优化策略 — CV融合算子架构模式 (P46-P52) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P46 | MatmulImpl high-level API | L1 | cv_fusion, matmul | 替代手写 matmul 流水线 | knowledge/contrib/ascendc/runbooks/performance/p46_matmulimpl_highlevel_api.md |
| P47 | Diagonal block scheduling | L1 | cv_fusion, matmul | 多核 M×N 分块 L2 cache 优化 | knowledge/contrib/ascendc/runbooks/performance/p47_diagonal_block_scheduling.md |
| P48 | Multi-quantization mode dispatch | L1 | quantization, matmul | 多量化模式编译期分发 | knowledge/contrib/ascendc/runbooks/performance/p48_multi_quant_dispatch.md |
| P49 | Hardware-accelerated dequantization | L1 | quantization | AscendDequant 硬件反量化 | knowledge/contrib/ascendc/runbooks/performance/p49_hw_dequantization.md |
| P50 | SwiGLU fusion pipeline | L1 | moe, cv_fusion | MoE FFN SwiGLU 融合 (narrow scope) | knowledge/contrib/ascendc/runbooks/performance/p50_swiglu_fusion.md |
| P51 | Dynamic AIC/AIV core ratio | L1 | cv_fusion, matmul | K 维阈值动态核配比 | knowledge/contrib/ascendc/runbooks/performance/p51_dynamic_aic_aiv_ratio.md |
| P52 | L2 Cache Hint optimization | L1 | cv_fusion, matmul | 多核竞争场景 L2 禁用 | knowledge/contrib/ascendc/runbooks/performance/p52_l2_cache_hint.md |

## 性能优化策略 — 片上缓存与常驻复用 (P53-P56) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P53 | L1 resident reuse | L1 | cv_fusion, flash_attention | Q 矩阵 L1 常驻跨迭代复用，消除重复搬运 | knowledge/contrib/ascendc/runbooks/performance/p53_l1_resident_reuse.md |
| P54 | Launch overhead optimization | L1 | datapath | 减少 kernel launch 开销，合并小 kernel | knowledge/contrib/ascendc/runbooks/performance/p54_launch_overhead_optimization.md |
| P55 | Block reduce-max optimization | L1 | datapath | 块级 ReduceMax 优化，二分累加 | knowledge/contrib/ascendc/runbooks/performance/p55_block_reduce_max_optimization.md |
| P56 | DataCopy batch output | L1 | datapath | 批量 DataCopy 输出，减少搬运次数 | knowledge/contrib/ascendc/runbooks/performance/p56_datacopy_batch_output.md |

## 性能优化策略 — FlashAttention/Decode 专用 (P57-P58) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P57 | FlashDecode G-partition reduction | L1 | flash_attention | FlashDecode G 维分区归约 | knowledge/contrib/ascendc/runbooks/performance/p57_flashdecode_g_partition.md |
| P58 | TND load-balanced partition | L1 | flash_attention | T/N/D 维负载均衡分区 | knowledge/contrib/ascendc/runbooks/performance/p58_tnd_load_balance_partition.md |

## 性能优化策略 — 数据搬运与格式优化 (P59-P66) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P59 | Output transpose fusion | L1 | datapath | 输出转置融合，减少额外搬运 | knowledge/contrib/ascendc/runbooks/performance/p59_output_transpose_fusion.md |
| P60 | KV NZ format optimization | L1 | matmul, flash_attention | KV cache NZ 格式优化 | knowledge/contrib/ascendc/runbooks/performance/p60_kv_nz_format_optimization.md |
| P61 | L2 cache optimization | L1 | flash_attention | 关闭双页表开启 L2 Cache，KV 数据缓存 | knowledge/contrib/ascendc/runbooks/performance/p61_l2_cache_optimization.md |
| P63 | Iterate async interface | L1 | matmul | 异步迭代接口，隐藏搬运延迟 | knowledge/contrib/ascendc/runbooks/performance/p63_iterate_async_interface.md |
| P64 | GM address conflict avoidance | L1 | datapath | GM 地址冲突规避，优化多核访存 | knowledge/contrib/ascendc/runbooks/performance/p64_gm_address_conflict_avoidance.md |
| P65 | UB bank conflict avoidance | L1 | datapath | UB bank 冲突规避，优化地址分配 | knowledge/contrib/ascendc/runbooks/performance/p65_ub_bank_conflict_avoidance.md |
| P66 | GM 512B alignment bandwidth | L1 | datapath | GM 512B 对齐带宽优化 | knowledge/contrib/ascendc/runbooks/performance/p66_gm_512b_alignment.md |

## 性能优化策略 — Vector 计算效率与标量向量化 (P67-P69, P94-P103) [Layer: L1]

P67-P69 为 Vector 计算效率基础策略（Counter 模式、低延迟归约、UB 融合链）。P94-P101 为标量→向量专项替代策略，覆盖归约（P94）、逐元素运算（P95）、排序（P96）、零拷贝切片（P97）、自适应调度（P98）、Broadcast 批量（P99）、GatherMask 标量替代（P100）、GatherMask 列分量解交织（P101）八种反模式。P102-P103 为架构级策略：当标量/向量在**模拟矩阵运算**（matmul/dot/三角求解）时，向量化收益封顶，需 MIX CV 融合重写（P102）及 Cube Neumann 三角求逆（P103）。

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P67 | Vector counter mode | L1 | datapath | Vector counter 模式，Scalar 控制流优化 | knowledge/contrib/ascendc/runbooks/performance/p67_vector_counter_mode.md |
| P68 | Low-latency reduction | L1 | datapath | 低延迟归约指令组合 (BlockReduceSum+WholeReduceSum) | knowledge/contrib/ascendc/runbooks/performance/p68_low_latency_reduction.md |
| P69 | UB fused vector chain | L1 | datapath | UB 融合连续 Vector 计算链，中间结果保留 UB | knowledge/contrib/ascendc/runbooks/performance/p69_ub_fused_vector_chain.md |
| P94 | Scalar reduction → Vector Reduce API (ReduceSum/ReduceMax/WholeReduceSum) | L1 | datapath | 逐元素标量归约循环（for(i) sum+=x[i]），Reduce/Softmax/Norm 算子 | knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p94_scalar_reduction_to_vector.md |
| P95 | Scalar element-wise → Adds/Muls vector chain | L1 | datapath | 逐元素标量算术循环（for(i) y[i]=f(x[i])），Elementwise/Norm 算子 | knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p95_scalar_elementwise_to_vector.md |
| P96 | Hardware sort engine (Sort32/MrgSort) replacement | L2 | datapath | 标量冒泡/选择排序或 TopK 选择，Sort/TopK/NMS 算子 | knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p96_hardware_sort_replacement.md |
| P97 | Zero-copy sub-tensor slicing + batch vector ops | L1 | datapath | 从 UB tensor 按偏移取子集做批量运算，需避免逐元素 gather | knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p97_zero_copy_subtensor.md |
| P98 | Hybrid scalar/vector adaptive dispatch (per-tile workload) | L1 | datapath | 小 tile 向量化收益为负（EnQue/DeQue 开销>标量循环），需自适应切换 | knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p98_hybrid_scalar_vector_dispatch.md |
| P99 | BroadCast + batch vector op replaces row-wise scalar loop | L1 | datapath | 小向量广播到大矩阵做逐元素运算，避免逐行 Muls/Adds dispatch 开销 | knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p99_broadcast_batch_vector_ops.md |
| P100 | GatherMask replaces element-wise SetValue/GetValue | L1 | datapath | 按固定间隔从连续 tensor 取元素重组（解交织、RoPE 奇偶分拆等） | knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p100_gathermask_replacement.md |
| P101 | GatherMask replaces host-side column split (`.select().contiguous()`) | L1 | datapath | interleaved (N,4)→4 路分量分离，消除 Host 侧列拆分和多路 DMA | knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p101_gathermask_host_column_split.md |
| P102 | Scalar/vector matmul emulation → MIX CV fusion rewrite | L1 | cv_fusion, matmul, attention, special | 标量/向量循环模拟矩阵运算（嵌套 dot、伪 matmul、三角求解），维度 ≥64 → 架构级重写（数量级收益） | knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p102_scalar_matmul_to_cv_fusion_rewrite.md |
| P103 | Cube Neumann product triangular solve | L1 | cv_fusion, attention, special | MIX 架构下标量 forward substitution 串行链 → (I-X)⁻¹=Π(I+X^{2^i}) 11 Mmad 替代 | knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p103_cube_neumann_triangular_solve.md |

## 性能优化策略 — Matmul/量化专用 (P70-P72) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P70 | FixPipe buffer inline quantization | L1 | matmul, quantization | FixPipe buffer 内联量化 | knowledge/contrib/ascendc/runbooks/performance/p70_fp_buffer_inline_quant.md |
| P71 | Matmul IBShare L1 sharing | L1 | matmul | Matmul IBShare L1 共享，减少重复搬运 | knowledge/contrib/ascendc/runbooks/performance/p71_matmul_ibshare_l1.md |
| P72 | Matmul Split-K | L1 | matmul | K 轴切分多核并行，AtomicAdd 累加 | knowledge/contrib/ascendc/runbooks/performance/p72_matmul_split_k.md |

## 性能优化策略 — CV 并行与核配比 (P73-P75) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P73 | CV parallel AIC:AIV ratio | L1 | cv_fusion | AIC:AIV 核配比与多 Workspace 流水 | knowledge/contrib/ascendc/runbooks/performance/p73_cv_parallel_aic_aiv_ratio.md |
| P74 | Selective L2 cache disable | L1 | datapath | 选择性 L2 cache 禁用，减少多核竞争 | knowledge/contrib/ascendc/runbooks/performance/p74_selective_l2_cache_disable.md |
| P75 | Dual AIV workload split | L1 | cv_fusion | 双 AIV M/S1 轴负载拆分 | knowledge/contrib/ascendc/runbooks/performance/p75_dual_aiv_workload_split.md |

## 性能优化策略 — Sparse Attention 专用 (P76-P78) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P76 | V_TEMPLATE KV pre-merge workspace | L1 | flash_attention | KV 预合并 Workspace 管理 | knowledge/contrib/ascendc/runbooks/performance/p76_v_template_kv_premerge.md |
| P77 | L1 chunk iterator sparse aggregation | L1 | flash_attention | L1 分块迭代器稀疏自适应聚合 | knowledge/contrib/ascendc/runbooks/performance/p77_l1_chunk_iterator_sparse_aggregation.md |
| P78 | On-chip cache matmul acceleration | L1 | matmul | 片上缓存加速 Matmul | knowledge/contrib/ascendc/runbooks/performance/p78_onchip_cache_matmul_acceleration.md |

## 性能优化策略 — 高级流水与融合 (P79-P84) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P79 | Grad Cube2+Cube3 fusion | L1 | matmul | 梯度 Cube2+Cube3 融合 | knowledge/contrib/ascendc/runbooks/performance/p79_grad_cube23_fusion.md |
| P80 | Sink task pipeline injection | L1 | flash_attention | Sink task 流水注入与 S2 block skip | knowledge/contrib/ascendc/runbooks/performance/p80_sink_task_pipeline_injection.md |
| P81 | Resident UB/TBuf reuse | L1 | datapath | 小参数/状态常驻 UB/TBuf，跨迭代复用 | knowledge/contrib/ascendc/runbooks/performance/p81_resident_ub_tbuf_reuse.md |
| P83 | L1 resident reuse partition | L1 | matmul | L1 常驻复用多缓冲分区 | knowledge/contrib/ascendc/runbooks/performance/p83_l1_resident_reuse_partition.md |
| P84 | Vector compute efficiency patterns | L1 | datapath | Counter 模式+低延迟指令+UB 融合链 | knowledge/contrib/ascendc/runbooks/performance/p84_vector_compute_efficiency_patterns.md |

## 性能优化策略 — 缓冲区复用与 Matmul 调优 (P85-P88) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P85 | On-chip buffer zone reuse | L1 | datapath | 片上 buffer 分区复用 | knowledge/contrib/ascendc/runbooks/performance/p85_on_chip_buffer_zone_reuse.md |
| P86 | Matmul API transport efficiency tuning | L1 | matmul | Matmul 内部数据通路效率调优 | knowledge/contrib/ascendc/runbooks/performance/p86_matmul_api_transport_efficiency_tun.md |
| P87 | Manual mmad pipeline timing control | L1 | matmul, manual | 手动 Mmad 流水时序控制 | knowledge/contrib/ascendc/runbooks/performance/p87_manual_mmad_pipeline_timing_control.md |
| P88 | Multiphase compute phase decomposition | L1 | cv_fusion | 多阶段计算拆分，独立 TPipe 资源重分配 | knowledge/contrib/ascendc/runbooks/performance/p88_multiphase_compute_phase_decomposit.md |

## 性能优化策略 — UB 共享与 Matmul 微调 (P89-P93) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P89 | UB high-level API temp buffer sharing | L1 | datapath | SoftMax 等高阶 API 与计算阶段 TBuf 分时复用 | knowledge/contrib/ascendc/runbooks/performance/p89_ub_api_temp_buffer_sharing.md |
| P90 | Conditional PipeBarrier by matrix size | L1 | matmul, flash_attention | 手动 Matmul L0C 累加模式按 fractal 数动态同步 | knowledge/contrib/ascendc/runbooks/performance/p90_conditional_pipebarrier.md |
| P91 | Softmax gradient pre-computation phase | L2 | flash_attention | FA 反向 dY*O ReduceSum 预计算为独立 VecSfmg 阶段 | knowledge/contrib/ascendc/runbooks/performance/p91_softmax_grad_precompute.md |
| P92 | Matmul base block compute-to-memory ratio | L1 | matmul, cv_fusion | 优化 baseM/baseN/baseK 最大化 Cube 计算访存比 | knowledge/contrib/ascendc/runbooks/performance/p92_matmul_block_compute_ratio.md |
| P93 | Matmul K-axis staggered memory access | L1 | matmul | 多核 Matmul K 轴错峰访问缓解 GM 地址冲突 | knowledge/contrib/ascendc/runbooks/performance/p93_matmul_k_stagger_access.md |

## 性能优化策略 — Ascend C Tiling 系统化设计 (P104-P115) [Layer: L1]

这些策略来自 `ascendc-tiling-design` skill 的系统化 Tiling 方法论，聚焦现有 P 系列未覆盖的 Tiling 设计维度（多核扩展、L2 切分、Batch 合并、Serpentine、平台路由、归约/排序/逐元素/广播/转置/FlashAttention 的 Tiling 流程）。

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P104 | Matmul multi-core tiling expansion | L1 | matmul, tiling | `MatmulApiTiling::GetTiling` 返回 `totalBlock < aicCoreNum`，需手动缩小 singleCoreM/N 以充分利用多核 | knowledge/contrib/ascendc/runbooks/performance/p104_matmul_multi_core_tiling_expansion.md |
| P105 | Matmul L2 cache tile sizing | L1 | matmul, tiling | 大算子启用 L2 级切分，设置 `mTileBlock×nTileBlock` 并配合 Serpentine 反向遍历 | knowledge/contrib/ascendc/runbooks/performance/p105_matmul_l2_cache_tile_sizing.md |
| P106 | Matmul batch-merge-to-M tiling | L1 | matmul, tiling | `batchB==1 && !transA` 时将 batch 维度合并到 M，提升数据局部性 | knowledge/contrib/ascendc/runbooks/performance/p106_matmul_batch_merge_to_m.md |
| P107 | Matmul serpentine traversal scheduling | L1 | matmul, tiling | L2 Tile 内 base 块 serpentine 反向遍历，提升 A/B 数据复用 | knowledge/contrib/ascendc/runbooks/performance/p107_matmul_serpentine_traversal.md |
| P108 | Matmul platform routing | L1 | matmul, tiling | 根据 NpuArch 选择 `MatmulImpl`+`MatmulApiTiling`（DAV_2201）或 Blaze/tensor_api（DAV_3510） | knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p108_matmul_platform_routing.md |
| P109 | Reduction multi-axis tiling | L1 | reduction, tiling | 多轴归约场景下的轴合序、AR/ARA 分阶段 Tiling 设计 | knowledge/contrib/ascendc/runbooks/performance/p109_reduction_multi_axis_tiling.md |
| P110 | Reduction UB accumulator reuse | L1 | reduction, tiling | 归约累加器在 UB 中常驻复用，减少中间结果搬运 | knowledge/contrib/ascendc/runbooks/performance/p110_reduction_ub_accumulator_reuse.md |
| P111 | Sort two-level merge tiling | L1 | sort, tiling | Sort/TopK 多级归并 Tiling，边界填充与索引 buffer 规划 | knowledge/contrib/ascendc/runbooks/performance/p111_sort_two_level_merge_tiling.md |
| P112 | Elementwise vectorized tile sizing | L1 | elementwise, tiling | 逐元素算子的向量化 tile 大小选择、内存合并与多核分发 | knowledge/contrib/ascendc/runbooks/performance/p112_elementwise_vectorized_tile_sizing.md |
| P113 | Broadcast aligned tiling | L1 | broadcast, tiling | 广播算子的对齐 Tiling、向量化加载与内存合并 | knowledge/contrib/ascendc/runbooks/performance/p113_broadcast_aligned_tiling.md |
| P114 | Transpose ND tiling with alignment | L1 | conversion, tiling | ND 转置的轴映射、对齐填充与 32B/512B 对齐搬运 | knowledge/contrib/ascendc/runbooks/performance/p114_transpose_nd_tiling_with_alignment.md |
| P115 | FlashAttention tiling routing | L1 | flash_attention, tiling | FlashAttention 类算子的 Tiling 模式路由（标准/GQA/MLA/paged 等） | knowledge/contrib/ascendc/runbooks/performance/p115_flashattention_tiling_routing.md |

## 性能优化策略 — GQA/FlashAttention 实战优化 (P116-P119) [Layer: L1-L2]

这些策略覆盖 attention 类算子（尤其 GQA decode）的四个结构性优化点：G→M 维合并、host 端 matmul tiling 配方、mask 预生成剥离、online softmax 改写。

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P116 | GQA grouped-M head merge | L1 | flash_attention, gqa, tiling | GQA decode（Sq≤4, G>1）时把 G 个 query head 零拷贝重解释进 matmul M 维，消除 G 个 M=1 小 matmul | knowledge/contrib/ascendc/runbooks/performance/p116_gqa_grouped_m_head_merge.md |
| P117 | MatmulApiTiling host recipe | L1 | attention, matmul, tiling | QK^T/PV 双 BMM 链 host 端接入 MatmulApiTiling：SetBufferSpace + d∈(64,128] SetFixSplit + depthA1 2→3 | knowledge/contrib/ascendc/runbooks/performance/p117_matmul_api_tiling_host_recipe.md |
| P118 | Mask precompute parallel AIV | L1 | flash_attention, mask | causal mask 下沉独立 AIV kernel 全核并行预生成；按需 dense right-down 替代通用 2048×2048 压缩表 | knowledge/contrib/ascendc/runbooks/performance/p118_mask_precompute_parallel_aiv.md |
| P119 | Two-pass to online softmax | L2 | flash_attention, softmax | 两遍/逐 token softmax 改写为 online 单遍 KV 流式（running max/sum/acc rescale），K/V GM 读取减半 | knowledge/contrib/ascendc/runbooks/performance/p119_two_pass_to_online_softmax.md |

## 性能优化策略 — MHA Cube 化实战优化 (P120-P123) [Layer: L1-L2]

这些策略覆盖纯向量 attention 到 cube 流水的完整改造链：mmad 重写 → CV 流水 → tile 大块化 → causal 跳过。四张卡存在依赖顺序：先 P120 建立 cube 路径，再 P121/P122 铺流水与大块化，P123 随时可叠加。

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P120 | FA vec-dot to cube mmad | L2 | flash_attention, cv_fusion | QK^T/PV 由 Mul+ReduceSum/标量乘加模拟时重写为 Cube mmad：K ColumnMajor 免转置 + Q 常驻 L1 + P 回写输入 dtype | knowledge/contrib/ascendc/runbooks/performance/p120_fa_vec_dot_to_cube_mmad.md |
| P121 | FA SplitFuse CV pipeline | L2 | flash_attention, cv_fusion | FA 四阶段串行/SyncAll 时改写为 AIC QK+PV / AIV softmax+rescaleO 流水：GM workspace 多槽 + 三 flag 握手 + PRE_LAUNCH 预发射 | knowledge/contrib/ascendc/runbooks/performance/p121_fa_splitfuse_cv_pipeline.md |
| P122 | KV stack large tile + L1 adaptive | L1 | flash_attention, matmul | KV tile ≤128 或 L1 划分硬编码时：KV 堆叠 512 摊薄握手 + nDyn/kDyn 按 L1 预算公式自适应 | knowledge/contrib/ascendc/runbooks/performance/p122_kv_stack_large_tile_l1_adaptive.md |
| P123 | Causal KV skip + boundary mask | L1 | flash_attention, mask | causal/sparse 全量循环 + 逐元素 mask 兜底时：noSkipKvS 循环界裁剪 + 边界 tile mask 特化 + dtype×mask 编译期 4 入口分发 | knowledge/contrib/ascendc/runbooks/performance/p123_causal_kv_skip_boundary_mask.md |

## 性能优化策略 — MLA Decode 实战优化 (P124-P126) [Layer: L1-L2]

这三张卡覆盖 MLA decode 的三个增量优化点：K 维拼接、任务边界流水、变长调度。依赖顺序：P124 要求 P120 已建立 cube 路径；P125 要求 P121 已铺 CV 流水；P126 独立可先行。

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P124 | MLA nope+rope concat GEMM K | L1 | flash_attention | QK^T 的 nope/rope 两段独立计算再相加时：拼成单 GEMM K 维（embed-split 128 分段流水 + Q 拼接常驻 L1 + 单 L0C 累加一次写回） | knowledge/contrib/ascendc/runbooks/performance/p124_mla_nope_rope_concat_gemm_k.md |
| P125 | Cross-task cube prefetch | L2 | flash_attention, cv_fusion | CV 流水多任务 kernel 任务边界断流时：KV 循环尾部一拍预读下一任务 tiling 并预发射其首块 QK^T（含 Q→L1），pingpongIdx 跨任务连续 | knowledge/contrib/ascendc/runbooks/performance/p125_cross_task_cube_prefetch.md |
| P126 | GM tiling table task schedule | L1 | flash_attention | decode 变长 kvSeqlen 仍按 totalQ 均匀切分 + 除法反解坐标时：host 拍平任务表（kvSeqlen 降序）+ kernel process 轮询领取，一次 launch 跑全 batch | knowledge/contrib/ascendc/runbooks/performance/p126_gm_tiling_table_task_schedule.md |

## 性能优化策略 — Conversion 模式特化与 Conv 实战优化 (P127-P131) [Layer: L1-L2]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P127 | Conversion pattern specialization routing | L1 | conversion, transpose | permute/transpose/split 等纯布局变换算子：R1 view 短路→R2 维合并→R3 退化路由→R4 模式专用 kernel→R5 fallback；禁止单一 generic gather 主路径 | knowledge/contrib/ascendc/runbooks/performance/p127_conversion_pattern_specialization_routing.md |
| P128 | FP32-via-BF16 hi/lo 3-GEMM | L2 | conv, matmul, fp32 | Cube 不吃 fp32 时：hi+lo 双 bf16 分量 3-GEMM 保精度（~1e-4）换 Cube 吞吐（ex-X17） | knowledge/contrib/ascendc/runbooks/performance/p128_fp32_via_bf16_hilo_3gemm.md |
| P129 | Fused img2col LoadData3D | L2 | conv, im2col | conv kHkW>1：A 侧免 GM 物化随路展开；特征图超 L1 用 band 窗口化（ex-X18） | knowledge/contrib/ascendc/runbooks/performance/p129_fused_img2col_loaddata3d.md |
| P130 | 1x1 Conv flipped GEMM | L1 | conv, pointwise | 1×1 卷积：原始布局即 GEMM 操作数，免 wrepack/im2col 两级发射（ex-X19） | knowledge/contrib/ascendc/runbooks/performance/p130_1x1conv_flipped_gemm.md |
| P131 | Weight repack LRU cache | L1 | conv, matmul, host-cache | 推理式重复调用：host 侧缓存权重 repack 结果，命中跳过 repack kernel 发射（ex-X20） | knowledge/contrib/ascendc/runbooks/performance/p131_weight_repack_lru_cache.md |

## 性能优化策略 — SSM/Scan 实战优化 (P132-P136) [Layer: L1-L2]

覆盖无 QK^T、无 softmax 的线性递归/chunk-scan 算子（GDN/KDA delta rule、GLA 门控线性注意力、Mamba2 等）。主链顺序：P132 三阶段分解 → P133 块内 CV 流水 → P134 任务划分 → P135 求逆分治（仅 GDN/KDA）→ P136 数值/UB。参考实现：chunk_gated_delta_rule。

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P132 | 线性递归 chunk-scan 三阶段分解 | L2 | ssm, linear_attention, recurrent, chunk-scan | 逐 token 串行递推：切成 C=64 chunk，块内预计算（全并行）→ 块间状态递推（仅 (Dv,Dk) 串行）→ 输出投影，O(seq) 串行压到 O(seq/C) | knowledge/contrib/ascendc/runbooks/performance/p132_chunk_scan_three_stage_decomposition.md |
| P133 | 块内预计算 AIC/AIV CV 流水 | L2 | ssm, cv_fusion, chunk-scan | Stage1 多级 matmul+向量预处理用 SyncAll 同步：cube 算 KKT/求逆/kcumdecay，vector 算 cumsum/exp/前代，6-flag 定向握手替代 | knowledge/contrib/ascendc/runbooks/performance/p133_ssm_stage1_cv_pipeline.md |
| P134 | chunk-group 任务划分 | L1 | ssm, chunk-scan, multi-core | 全序列一次分核导致跨核状态依赖退化串行：group 间串行 + group 内 (nv,chunk) 多核并行，maxGroupLength=P_NUM×核×C | knowledge/contrib/ascendc/runbooks/performance/p134_ssm_chunk_group_partitioning.md |
| P135 | delta-rule 矩阵求逆分治 | L2 | ssm, delta-rule, matmul | (I-A)^{-1} 整块串行求逆：对角子块向量前代 + 非对角 X10=X11·M10·X00 cube MM，O(C³) 求逆并行化（GLA 用 cumprod 无求逆） | knowledge/contrib/ascendc/runbooks/performance/p135_delta_rule_inverse_divide_conquer.md |
| P136 | SSM 中间量 bf16 量化 + 状态常驻 | L1 | ssm, ub-memory, bf16 | 中间量全 fp32/状态每 chunk 搬 GM：块间传递全程 bf16（求逆/衰减核心留 fp32）+ state (Dv,Dk) UB 常驻 + head grouping Nv//Nk | knowledge/contrib/ascendc/runbooks/performance/p136_ssm_intermediate_bf16_quant.md |

## 性能优化策略 — SparseFlashAttention 实战优化 (P137-P140) [Layer: L1-L2]

覆盖 token 级稀疏（top-k）与 MLA 的 SFA 优化。来源：Z-Search cannbench-evo 实测沉淀。P137 是稀疏主路径（紧凑化走 Cube）；P138/P139 是稳定性加固（L0 容量守卫 / 多槽零初始化）；P140 是 MLA M-tile 放大（依赖 P138）。

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P137 | 稀疏 top-k 紧凑化走 Cube | L2 | flash_attention, sparse, cube | token 级稀疏（topK）被误认"不规则索引无法用 Cube"：AIV 标量 gather 搬成 workspace 紧凑 tile（有效行前置 + 尾部零填充）再喂 AIC mmad，稀疏只加搬运契约不改 Cube 结论（17.28x） | knowledge/contrib/ascendc/runbooks/performance/p137_sparse_topk_compact_gather_to_cube.md |
| P138 | host tiling L0 容量守卫 | L1 | flash_attention, matmul, stability | Cube tiling 用写死的 L0 尺寸 → 更小 L0 芯片 InitBuffer 溢出随机 SIGSEGV：`GetCoreMemSize(L0_A/L0_C)` 运行时查询 + groupTile 自适应减半（跨硬件稳定性） | knowledge/contrib/ascendc/runbooks/performance/p138_host_tiling_l0_capacity_guard.md |
| P139 | 多槽流水 workspace 全槽零初始化 | L1 | flash_attention, stability | 预发射多槽流水只初始化 slot0 → 后续槽 padded 行残留 NaN/inf 偶发崩溃：零初始化遍历全部槽 + AIC 直读的 score/prob padded 区（防 NaN 稳定性） | knowledge/contrib/ascendc/runbooks/performance/p139_multi_slot_workspace_zero_init.md |
| P140 | MLA（kvHeads=1）M-tile 放大 | L1 | flash_attention, mla, tiling | M-tile 上限 = 共享 K/V 的查询行数：GQA 只有 groupSize（小）硬拉无益，MLA（kvHeads=1）可到 qHeads 放大一个数量级，UB/L0 自适应（配合 P138）（1.95x） | knowledge/contrib/ascendc/runbooks/performance/p140_mla_large_m_tile.md |

## 性能优化策略 — MHA/Attention 实战优化补充 (P142-P143) [Layer: L1-L2]

来自 17_MultiHeadAttention 进化实测沉淀：UB 常驻 O 累加器砍掉 per-KV-tile GM 往返；host 端 launch 合并 + per-shape tiling 缓存压掉中小 shape 的 host 固定开销。

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P142 | Attention UB 常驻 O 累加器 | L2 | attention, flash_attention, ub-resident, o-accumulator, kv-tile, gm-roundtrip | Flash Attention 每 KV tile 更新 O 时把 O_tile 常驻 UB，跨 tile rescale+累加，循环结束再一次性写 GM；砍掉 per-tile MTE2/MTE3 往返。前提：O_tile(M_tile×d) 能放进 UB；与 GM workspace-ring（X38）互斥 | knowledge/contrib/ascendc/runbooks/performance/p142_attention_ub_o_accumulator.md |
| P143 | Attention host 端 launch 合并 + per-shape tiling 缓存 | L2 | attention, host-overhead, launch-merge, tiling-cache, split-kernel, h2d-elimination | 910B2C 小 shape/fp32 路径 attention 常被 host 固定开销主导：合并同前向内的 split kernel launch、per-shape 缓存 tiling 与 workspace、用 at::empty 替代 zeros | knowledge/contrib/ascendc/runbooks/performance/p143_attention_host_launch_merge.md |

## CV-Attn（CP8 门控广播）专项策略（P144-P147）

conv-based CV-Attn（Triplet/BAM/CBAM：global pool → conv 2→1 → BN(training) → gate 广播）的「跑通 + 到 1x」充分集。来源：64_TripletAttention_v2（1.28x / 50-50）+ 36_BAM_evo（0.98x）跨算子验证。路由入口：scenario-routing/flashattention.md `cv-attn` 分支。P144-P147 一轮串完即可达 1x；随后按 profiling 证据追加 P104/X58（多核）或 P128/P129（Cube conv）。

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P144 | host 端 zpool/BN-stats 计算 + 标量传参 | L1 | attention, host-compute, scalar | 含 global pool + BN(training)，host 原本做 D2H 回读 + stream sync；910B2C RA ReduceSum >16 列不可靠 | knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p144_host_pool_bnstats_scalar_pass.md |
| P145 | 多 launch 免同 launch GM 写后读 | L1 | attention, architecture | 910B2C 同 launch 内 MTE3→MTE2 写后读不可靠（npuSynchronize 崩溃/数值错乱）；跨相位数据走独立 launch | knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p145_multilaunch_no_same_launch_gm_rw.md |
| P146 | 910B2C 32B 对齐/标量通道加固 | L1 | attention, alignment, scalar | 小 H×W 算子的 blockLen 32B 对齐、单 block 读不可靠、列读 strided、广播 8× 缓冲、标量通道 async | knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p146_910b2c_alignment_scalar_hardening.md |
| P147 | NPU conv 12-sig-bit 舍入模拟 | L1 | attention, precision, conv | 自写向量卷积/host 预计算 conv 输入时，K=3 大 case 相对误差 ~1e-3 超桶：对激活+核套 RoundTo12Sig | knowledge/contrib/ascendc/optimizations/evolution_strategies/cards/p147_npu_conv_12sig_rounding_emulation.md |
## 精度优化策略 (A1-A8) [Layer: L0]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| A1 | FP32 intermediate computation | L0 | datapath | BF16/FP16 with precision requirement | knowledge/contrib/ascendc/runbooks/performance/a1_fp32_intermediate.md |
| A2 | Welford numerically stable mean/var | L0 | datapath | LayerNorm, BatchNorm, RMSNorm | knowledge/contrib/ascendc/runbooks/performance/a2_welford_algorithm.md |
| A3 | Rounding mode control (CAST_*) | L0 | datapath | Any Cast to lower precision | knowledge/contrib/ascendc/runbooks/performance/a3_rounding_mode.md |
| A4 | SetFlag/WaitFlag event sync | L0 | datapath | Data dependencies across pipes | knowledge/contrib/ascendc/runbooks/performance/a4_pipeline_barrier.md |
| A5 | Softmax numerical stability | L0 | datapath | Softmax, attention score ops, NaN/Inf risk | knowledge/contrib/ascendc/runbooks/performance/a5_softmax_stability.md |
| A6 | High-precision rsqrt (Newton-Raphson) | L0 | datapath | Normalization ops needing rsqrt | knowledge/contrib/ascendc/runbooks/performance/a6_high_precision_rsqrt.md |
| A7 | Index & boundary safety | L0 | datapath | Gather/scatter, user-controlled indices | knowledge/contrib/ascendc/runbooks/performance/a7_index_boundary.md |
| A8 | Quantization-specific precision | L0 | quantization | Quantization output, dequantization with bias | knowledge/contrib/ascendc/runbooks/performance/a8_quant_precision.md |

## 探索发现策略 (Discovered Strategies) [Layer: L2]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| X1 | Inline Target Capture | L2 | — | Reduce + single-element lookup ops | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x1.md |
| X2 | Big-Tensor-On-Device, Index-Only-Over-Bus | L2 | gather, routing, bus-bound | Host-coordinated gather/scatter where data tensor ≫ index tensor | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x2.md |
| X3 | End-to-End Dtype Passthrough | L2 | elementwise, foreach, multi-dtype, bandwidth-bound | Multi-dtype elementwise where wrapper upcasts fp16/bf16→fp32 and inflates HBM traffic ~5x; sink dtype boundary into kernel | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x3.md |
| X4 | Concat-Fused Strided DMA + Two-Level Skv Reduction | L2 | decode-attention, latency-bound, GQA N_kv=1 | Collapse nope|rope concat into strided K-tile DMA + two-level vmul→vcadd→vcadd + one-pass FA-2; kills per-key DMA/sync latency (decode 0.11-0.27x→0.25-0.79x) | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x4.md |
| X5 | GQA-Decode GEMV→GEMM (M=N_q) | L2 | decode-attention, compute-domain, GQA N_kv=1, cube | For GQA decode/MTP (S∈{1,2}, N_kv=1) take decode M=N_q(128/64) so QK^T=Q[N_q,D_qk]@K^T and PV=P[N_q,S_kv]@V become dense cube GEMM instead of N_q vec GEMVs; stream K once (NC=256), V once (sub-col nd2nz), N-tile PV to L0C; decode 11-33x (geomean 0.57x→3.53x) | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x5.md |
| X6 | Merge Partial + Reduce Kernel for Split-KV FlashDecode | L2 | decode-attention, split-kv, kernel-fusion | Split-KV 场景 partial 与 reduce 两个 kernel 融合，消除 reduce kernel 的独立 launch/stream 开销（链路 149.58→99.53us，-33.5%） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x6.md |
| X7 | Token Block Size (TB) Tuning for Pure-Vector Attention | L2 | decode-attention, pure-vector, tiling | 纯 Vector attention 的 token block 尺寸调优（同窗口 mean_asc -4.2%，case 级最高 -42.6%） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x7.md |
| X8 | Host-Side Precomputation of Constant Offset Tables | L2 | host-precompute, offset-table, latency-bound | 常量偏移表（如 Pi 转置）移到 Host 预计算，kernel 内零重复计算（mean_asc 149.58→125.64us，-16.0%） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x8.md |
| X9 | Deferred Scaling (反量化仿射代数重排/延迟缩放) | L2 | dequant, affine, vector-efficiency | 反量化仿射结构代数重排，延迟缩放减少逐元素乘加链（单核 aiv_time -33%，task_duration 18.23→12.91us） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x9.md |
| X10 | Device-side Grouped Scheduling + NZ Plane Weight Cache | L2 | moe, matmul, cube, grouped | Grouped matmul（group_list cumsum）Cube 化：AIC 设备侧自建分组调度（empty group 跳过），weight NZ plane 静态缓存 + m-fastest tile 排序多核共享 strip，fixpipe NZ 步长=orgM×16 原始值（level4_grouped_matmul_swiglu_quant 0.0026x→4.82x vs PyTorch） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x10.md |
| X11 | ND-drain + L0B-feed-driven baseM Scaling | L2 | matmul, quantization, cube, c220 | 量化 matmul Cube 利用率修复：多 fragment NZ drain 为芯片缺陷（与 orgM 对齐无关）→ CType 路由 ND；瓶颈是 L0B-fill:mad 比而非 L2 refetch → baseM 扩至 L0C 上限（int32 128@256）；重叠尾 strip 解 singleM<baseM/singleN≤16 崩溃；AIV 反量化 strip 化（strip 所有权+per-strip 向量+TQue 双缓冲+单条 2D DataCopyPad）（level3_quant_matmul 0.1999x→0.8274x vs 官方 QuantBatchMatmulV3） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x11.md |
| X12 | Pure-VEC Streaming Weight-Quant GEMV (小 M 分流) | L2 | weight-quant matmul, GEMV, small-M, c220 | 量化 GEMV 纯 VEC 流式路径：scale 提出 K 循环 + mixed-precision Axpy + 确定性两阶段 K-split（c220 SetAtomicAdd 丢更新禁用原子加）+ occupancy makespan tiling；小 M(1-8) 3.3-8.8x，大 M 须分流 cube（level3_weight_quant_batch_matmul 0.98x geomean，小 M case 最高 8.77x） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x12.md |
| X13 | Shape 自适应双引擎分流 (M-Threshold Hybrid Routing) | L2 | weight-quant matmul, skinny-GEMM, GEMV+GEMM mixed shapes | Host 侧按 M 路由两条完整 kernel 路径（小 M 纯 VEC streaming 吃带宽极限 / 中大 M cube 吃 MAC），中间区间按 K*N 流量摊销固定开销做实测校准二级阈值；路由只换引擎不改数值（共享精确代数分解）（level3_weight_quant_batch_matmul VEC 0.98x / cube 1.51x → 混合 2.03x geomean） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x13.md |
| X14 | Mix-Kernel 跨核生产者/消费者融合 (Pairwise AIV-Producer/AIC-Consumer) | L2 | weight-quant matmul, multi-kernel pipeline, dequant-pass elimination, c220 | 硬件 int8-cube 路径证伪后，将独立 VEC dequant pass 与 cube kernel 融合为单 mix kernel：AIC ci 与 AIV(2ci,2ci+1) 配对按 strip 序生产/消费 write-once workspace，GM-flag 协议同步（128B 步长 + MTE3 DataCopyPad 发布 + __gm__ volatile 有界轮询 + DCCI）；launch 3→2、cast 与 cube 重叠（level3_weight_quant_batch_matmul 2.03x→2.44x，dequant kernel launch=0） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x14.md |
| X15 | 递归拆分-并行投影外提 (Recurrent Split: Hoist Parallel Projection) | L2 | rnn, lstm, gru, scan, recurrent, dataflow | 含串行循环算子将非递归投影（全部时间步 x@W_ih）合并为单次 host 侧大 GEMM 外提走 Cube，自定义内核只保留严格串行的递归残差（h@W_hh+门激活+状态更新），递归权重 UB 常驻+超规格回退；串行链计算量 O(seq×(I+H)×4H)→O(seq×H×4H)（LSTM 0.202x→0.852x mean） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x15.md |
| X16 | 循环算子 H-slice 多核切分 (Recurrent H-Slice Multi-Core Split) | L2 | rnn, lstm, gru, scan, recurrent, multi-core, small-batch | 时间维不可并行时按 hidden 维切多核：每核独占 [hStart,hEnd) 及 4 门控列、W_hh 切片 UB 常驻跨时间步复用、c/h 切片独立更新零核间通信；host tiling 按 shape 在 batch-parallel/H-slice 间调度（batchNum≤4 或 H≥256&&batchNum≤32 走 H-slice）；**禁忌：门控列为 {j,H+j,2H+j,3H+j} 跨步取，禁连续 [4hStart,4hEnd) 切**；与 X15 正交可组合（纯 kernel 0.335x→0.799x，移植进 X15 架构 0.852x→2.12x mean） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x16.md |
| X17 | FP32-via-BF16 Hi/Lo 拆分 3-GEMM | L2 | fp32, cube, split-precision, conv, matmul | 910B Cube 不吃 fp32 输入时：x≈hi+lo 双 bf16 分量 3-GEMM（hh+hl+lh，fp32 累加）+ 单趟 split-xtrans；±inf 语义天然保持；2-GEMM 已证精度不足否决；L0 队列深度按每 chunk tile 数定（双 tile 用 depth-4）；fp32 geomean 0→0.50 | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x17.md |
| X18 | Fused img2col LoadData3D 随路展开（含 band 窗口化） | L2 | conv, im2col, cube, bandwidth-bound | 卷积 A 侧免 GM 物化：NC1HWC0 常驻 L1 + img2col LoadData3D 随路展开进 L0A；xNz 超 L1 时 band 行带窗口化（padTop'/padBottom' 重算保 Ho 光栅恒等）；硬件 k 序 (ci1,kh,kw,ci0) 须 one-hot 探针确认；W<24 窄图/kExtension>64 门控回退；消 70% im2col 占比，大 case 0.32→0.59 | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x18.md |
| X19 | 1×1 Conv 翻转 GEMM（原始布局即操作数） | L2 | conv-1x1, pointwise, launch-bound | kH=kW=1/s=1/p=0 且 Ci%16==0、(H*W)%16==0 时：A=原始权重 [Co,Ci]、B=原始 x [Ci,H*W]，免 wrepack+im2col 两级发射，epilogue 通道行直写 NCHW；UB 中转解除对齐门控已证净负；实测 1.0~1.98× | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x19.md |
| X20 | Host 侧权重预打包结果 LRU 缓存 | L2 | host-cache, weight-repack, inference, conv, matmul | conv/matmul 等带权重 repack 步骤的算子，repack 输出是确定函数，推理式重复调用按 (data_ptr+sizes+dtype+version+布局) 做 LRU 缓存，命中跳过整个 repack 发射；条目持 weight 强引用防地址复用；每次调用省 5-15us | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x20.md |
| X21 | 910B2C Cube 化平台纪律 + 数据驱动 Kahan 精度路由 | L2 | conv3dbf, fp32, kahan, precision-routing | conv 类 fp32 算子 Cube 化时：平台纪律（GetCoreNum 保底、no_half_ufp16 指令表、fp16 3-GEMM）＋数据驱动精度路由（小数值走 fp16 直算、大数值走 Kahan 补偿），绕过 fp32-Cube 缺失并保精度 | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x21.md |
| X22 | Depthwise 直接卷积（绕过 im2col+GEMM） | L2 | depthwise, conv2d, vector-conv, im2col-bypass | depthwise(groups==Cin,cig==1) 用单 AIV fp32 向量卷积直接算逐通道 kh*kw dot：无 im2col/GEMM/fp16 分裂/GM workspace/Cube；Tile=(image,co-block8,oh-block) 保 32B 对齐，对齐 interior 向量化+标量 head/tail 独立 corr 缓冲消 S/V hazard；depthwise gen-time 提升 18-84x | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x22.md |
| X23 | Batched Spatial 全宽向量化 + 编译器驱动标量预取 | L2 | conv, cv_fusion, elementwise, s-pipe-latency-hiding | 滑窗卷积 kw==1/W 维逐点：把 P=floor(64/W) 个连续 oh 打包进 64-lane fp32 向量，每 (co,tap) 1 次权重标量读驱动 P*W 宽 Axpy（向量指令+标量读降 P 倍）；Boundary 零填充 + strided DataCopyPad；`#pragma unroll 4` 驱动编译器软件流水隐藏 ~43cy S-pipe 标量读延迟（⚠️手工多标量预取触发编译器误编译，禁用）；W%8==0 门控；geomean 0.36x→3.5x | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x23.md |
| X24 | AIC_ONLY 任务类型压设备 launch 地板（固定开销主导算子） | L2 | matmul, elementwise, reduction, aic-only, device-launch-floor | 微小 shape 固定开销主导算子（cube_util 极低、标量占比高）：任务类型从 MIX_AIC_1_2 改 `KERNEL_TYPE_AIC_ONLY` 去掉每 AIC 的 2 个空转 AIV，设备固定地板 ~7-8μs→~4.3-4.5μs；配 compact tiling 头（65→4 int32）；与手动 mmad 叠加 12.09→6.88μs（1.757x vs 基线）；上限：910B kernel launch 地板 ~4.3-5.6μs | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x24.md |
| X25 | 向量权重广播（Brcb + MulAddDst）替代 S-pipe 标量读 | L2 | conv, cv_fusion, elementwise, matmul, scalar-read-elimination | 标量权重驱动向量 FMA 且权重跨 batch 复用：权重 GM 预展开 [Co,K,8]（每权重 8 副本），每 (g,tap) 一次 V-pipe `Brcb` 广播成 64-lane 权重向量驱动 numBatches 条 `MulAddDst`，彻底消除 S-pipe GetValue 标量读（~43cy）；wvec 2×64-lane ping-pong 破写后读；多 batch 摊销最大（case28 5.9x）；numBatches=1 高 K 无摊销收益被 8x 权重载入抵消 | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x25.md |
| X26 | Col2im Tap-Batch 事件化（散射类 per-tap 流水重构） | L2 | conv, deconv, scatter, gather, datapath | 散射算子（col2im/deconv）逐输出行 tap 循环每 tap 一次 PIPE_ALL（~16/行）→ tap-batch：一次发出该行所有 tap 的 MTE2 加载 + 1 barrier，随后 V 管道 in-order 背靠背 Gather+Add + 1 barrier；barrier 降到 2/行；对齐 zero slot 在 V 管道初始化（S/V 无保序）、行级 B1/B4 保守保留 PIPE_ALL（MTE3 WAR）、禁忌 strided 多块 DataCopyPad（dav-2201 挂死）；col2im 316→277us | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x26.md |
| X27 | transpose-on-load 分形加载 + 全 shape 手动 mmad 二维 tile 网格（免 host 预转置） | L2 | matmul, manual-mmad, transpose-on-load, full-shape-grid, aic-only | transB 类 matmul：B 转置由 `LoadToL0B<LOAD2D> isBTranspose=true` 在 L1→L0B 分形加载时完成（免 host workspace 预转置 pass，910B2C 任意 M/N/K 正确，区别于 f3 LoadData3D 陷阱）；输出 C 按 gridM×gridN 16 对齐 tile、每 tile FULL K 单条 Mmad；并行度扩展启发式（per-tile 操作数>96KB 且 grid<24 拆分大维）；全 50 shape 无一 fallback，0.35→0.771x vs PyTorch（2.40x vs 基线），fp32[64,128]=6.3μs vs x24 host-transpose 版 11.2μs | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x27.md |
| X28 | 小 S prefill FA 自适应 KV chunk 粒度 | L2 | flash_attention, gqa, prefill, tiling | FA/GQA prefill 的 kvPad=align_up(S,256/128) 对小 S 过度 padding（4-85x 白算）时：按 S 自适应 nChunk（小 S 用 64，kvPad=align_up(S,64)）+ kernel 16 对齐尾块 partial chunk，消除 padding 冗余。GQA 上小/中 case attention device 时间 -2-4x，geomean 0.500->0.588（P8 score 0.976）| knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x28.md |
| X29 | Shift-Free Tap-Comb 稠密直算（stride=1 反卷积消 Gather） | L2 | conv, deconv, stride1, gather-elimination, dense-direct | stride=1 反卷积恒等变换：ConvTranspose2d ≡ 零填充输入+180°翻核标准卷积；comb[ci][kf2][row][0:woA]=P 左移 kf2 对齐槽，固定 (ci,kf2,kf) 一次对齐 Axpy 稠密累加，彻底消除 Gather 间接寻址/col2im 散射/G workspace 往返/AIC→AIV 交接；纪律 ph=ho+kh-1、DataCopyPad UB→GM blockLen≥32B(wo<8 用 woA 填充缓冲+host 视图)、向量化 comb 构建 dav-2201 507015 回退标量；case0 0.927x, case7 2.433x (1.83x kernel 级) | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x29.md |
| X30 | 头维-minor Q 布局免 heads-merge kernel（token-major 折叠 head 合并） | L2 | attention, flash_attention, mqa, gqa, kernel-elimination | 物化式 MHA/MQA/GQA 的 heads-merge kernel（O[BH*S,hdPad]→Om[M,H*hdPad] 转置+cast）是纯搬运；因注意力链对 Q 行序不敏感，Q 布局改 token-major(head-minor)（prep 从逐 head strided gather 变 token 粒度连续块拷贝 blockLen=H*hdPad），PV-GEMM 输出 O.view(M,H*hdPad) 天然即合并 heads 布局——heads-merge 整段删除（少 1 launch + O/Om GM 往返 + cast pass）；MQA 0.965→1.127x（+17%），配合 PV 输出直接 cast T 再省 cast pass | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x30.md |
| X31 | bounded logits softmax 去 max 减法（平移不变 + 数值有界时跳过 max-trick） | L2 | reduction, elementwise, attention, softmax | softmax(z) 平移不变，max 减法（WholeReduceMax+Brcb+Sub 约 3 pass+8 barrier/tile）仅防 exp 溢出（fp32 溢出点 88）；logits 实测有界 |z|<40 时删除 max 减法直接 Muls(invDiv)+Exp（pass 8->5, barrier ~23->~12），配合行求和 chunk 化 WholeReduceSum 全发出 + log-depth 树合并（barrier O(C)->O(log C)）；softmax 单 kernel -25%；注释记录 logits 上界证据 + 保留恢复开关 | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x31.md |
| X32 | fp16 hi/lo 投影 GEMM 精度替代 + 投影 GEMM 真实瓶颈识别（MTE2-bound） | L2 | fp32, projection-gemm, cube, precision, gqa | fp32 投影 GEMM 是 GQA prefill dominant 成本（case44 占 55%）但 **MTE2-bound 非 compute-bound**（mac 0.245/mte2 0.709）；P128 bf16 对投影近零输出 MERE 超 2^-13，fp16 hi/lo（~2^-21）是精度安全替代但单维度无净收益（cube 翻倍被内存瓶颈吞没）；正确杠杆是投影 GEMM L1 双缓冲（+3-14%）/大块化/消除 GM 往返 | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x32.md |
| X33 | attention workspace-ring 握手延迟 bound 的 Q-tile 放大 + 投影 GEMM group A-reuse | L2 | attention, gqa, prefill, workspace-ring, matmul, mte2 | GQA attention 各管道利用率<15% 是跨核 AIC<->AIV workspace-ring 握手延迟 bound：per-task Q tile 32->64->128 每次减半握手次数（case47 +24%、case35 +19%，attention 318->265us）；投影 GEMM MTE2-bound 用 by-group A-reuse（GROUP_BY=6，A over-read 63x->11x，Stage-A 781->450us，mte2_ratio 0.746->0.693）；配 X24 AIC_ONLY/AIV_ONLY + P1 L1 双缓冲（全量 0.598->0.657x，case2 fp32 1.21x）| knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x33.md |
| X34 | Col2im 相位化行归约（stride-s 反卷积散射重构） | L2 | deconv, col2im, scatter, gather, datapath | 反卷积 fallback 真正瓶颈是 col2im 散射（非 im2col 带宽）：GEMM 中间 G 仅 0.03-2MB、~9us 非瓶颈；把 col2im 从'逐输出行 undersize 加载+Gather+parity+PIPE_ALL(~274us)'改为'每相位相位内 K²/s² 条 G 行长全长对齐向量 Add 归约 → 相位平面 → 行粒度批量写回'；列移位经 DataCopyPad src offset+padding 落地（AIV 全 32B 对齐，绕开 odd-shift 对齐封锁）；tap 表预计算、一次向量 Gather 奇偶交织、多块加载门控 wi%8==0&&srcCol==0；case2 548→118us (4.7x)，全量 geomean +18% | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x34.md |
| X35 | MatmulImpl Host 固定开销消除 + 全覆盖串行多 tile | L2 | matmul, host-overhead, launch-bound, workspace, tiling-cache, multi-core | MatmulImpl 中小 shape launch-bound 时：workspace at::empty（位精确验证免清零）+ per-shape tiling 缓存 + workspace 复用 + kernel 核内串行多 tile 遍历（tile+=blockDim，End 单次，修 totalBlock>blockDim 漏算）；自适应 core re-grid（L0C 约束，目标 totalBlock∈[8,24]，大 shape 保留对齐 tile 避 Cube 掩码浪费）（MatmulTransA 0.272x→0.713x，2.6x） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x35.md |
| X36 | Zero-Host-Padding AIV Small-Matmul | L2 | matmul, aiv, host-padding, small-shape | matmul 系算子非 16 对齐小 shape（M*K≤256）时 host constant_pad_nd 的 pad kernel 主导官方 msprof 时长：host 不 pad 直接下发原始 shape + AIV 核内精确 matmul（fp32 累加 + 32B 对齐行展开 + M 行多核切分）；准入超阈值回退 Cube MatmulImpl（BatchMatmul 0.375x→0.61x，小 shape 0.15x→0.98x） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x36.md |
| X37 | FP32-via-FP16 Hi/Lo 4-GEMM Cube | L2 | fp32, fp16, cube, hi-lo, precision, flash_attention, matmul | Cube 不吃 fp32 且精度要求 fp32 allclose（atol≈2e-5）时：fp32 拆 fp16 hi/lo 双分量（11 尾数位，捕获 ~23bit）+ 4 次 fp16 Mmad（Ah@Bh+Ah@Bl+Al@Bh+Al@Bl，fp32 L0C 累加，含 lo·lo）达 ~9e-6 误差（bf16 3-GEMM ~1e-4 不够）；独立 AIV split 预 pass 用 8-deep ring UB 规避 MTE3→V WAR（WindowAttention fp32 geomean 0.1x→0.956x，整体 0.47x→0.98x） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x37.md |
| X38 | GM-Workspace-Ring KV-tile 宽度缩放 + Host 批量权重打包 | L2 | flash_attention, workspace-ring, kv-tile, host-packing, scalar_loading, cube-efficiency | FA 用 GM workspace ring 握手（AIC↔AIV 经 GM 中转 S/P/O）且每 KV tile 握手开销主导时：kvN 自适应（seq<=64 用 64，seq>64 用 128）减半 GM ring 往返 + 翻倍 MM1/MM2 cube GEMM，workspace/buffer 按 128 max 分配仅 slot/MM/fixpipe/softmax srcK 变运行时 kvN，sPad align_up(seq,kvN) 修越界；融合投影 host 侧 per-head 权重打包 copy-storm（48+ 次 narrow().copy_()，fp32 case 占墙钟 40%）批量化为 cat+pad+reshape+单次 strided copy，输出改 at::empty（FlashAttentionV2 mean 0.579→0.856，geomean 0.530→0.794） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x38.md |
| X39 | Host 全写 workspace 免零初始化（at::zeros→at::empty） | L2 | host-overhead, workspace, memset, memory-bound | host 每次 forward 分配大 workspace/输出 buffer 且 kernel 会写满全部区域（无读前未写）时，用 at::empty 替代 at::zeros 消除每次调用的百 MB~GB 级 memset 地板；配合动态核数 blockDim=min(coreNum,totalTasks) 小任务不拉空转核；红线：必须逐区域审计"写满后才读"，部分写/依赖零初始化的区域禁用（垃圾值 NaN）。与 P54 互补（P54 减 launch 次数，本策略减 memset 成本）（GqaSparseFwd: 大 case 1.35-2.3x，50/50 精度过） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x39.md |
| X40 | MatmulApiTiling 单核陷阱修复（singleN/singleM hint 前置多核化） | L2 | matmul, tiling, totalBlock-1, tiny-shape, multi-core | 小 shape GEMM 走 MatmulApiTiling::GetTiling 时，"整块塞得进单核"的 shape 被 API 返回 singleCoreM=M/singleCoreN=N（mTotalCnt=nTotalCnt=1 → totalBlock=1），全部 GEMM 只跑 1 个 AIC 核（24 核用 1）。修复：在 GetTiling **之前**设 tilingApi.singleN=128（或 per-M 分段 hint），让 API 自洽重排 singleCoreN/baseN/baseK/stepN/depthA1/depthB1 得到多 N-block 吃满核；严禁 GetTiling 之后手工缩 singleCoreM/N（依赖字段不重算 → AIC trap 507014）。per-M 规则：M≤64→singleN=32（24 N-block）；64<M≤128→singleM=64（12 blocks）；M>128→singleM=M/4 对齐 16（24 blocks）（FluxTimestepGuidanceProjectionEmbedding 910B2C 24AIC: 0.179x→1.461x vs PyTorch，8 倍） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x40.md |
| X41 | singleCoreN 对齐 baseN 倍数（量化 Matmul L0B 填充利用率修复） | L2 | matmul, quantization, tiling, l0b-utilization, alignment | 量化 matmul 大 shape 的 AIC 瓶颈是 L0B-fill:mad 比 / L2→L1 喂给率（aic_mte2_ratio≈0.92、mac_ratio≈0.45）而非 MAC 时：singleCoreN 非 baseN 整数倍（如 342=256+86）使每 block 末尾 base-N tile masked partial、浪费 ~1/3 L0B 填充；在 GetTiling+多核 shrink 后把 singleCoreN round-up 到最近 baseN 倍数（342→512），对齐后 totalBlock<核数则回退 baseN 保核心网格；不改 singleCoreM（避 X11 的 singleM<baseM AIC 崩溃）；对齐勿过头（2 倍对齐 block 数减半破坏负载均衡，实测反噬）。X11 聚焦 baseM、本策略聚焦 singleCoreN，两个独立杠杆（QuantMatmul 910B2C: case2 722→296μs -59%，asc kernel geomean -21.7%，10/10 bit-exact） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x41.md |
| X42 | 路由选中分支消除（one-hot selector host 读回，只算被选中分支） | L2 | attention, moe, conditional-compute, routing, one-hot, branch-elimination | 主计算前有廉价 router/selector 产生 one-hot 选择向量（AdaptiveAttention 候选层、MoE expert 路由）且输出=Σ sel_i·branch_i(x) 时，未选中分支的计算全是纯浪费：router kernel 后 host D2H 读回 selector（仅 B×n_branch 个 float；注意 CUBE_ONLY 核内 GetValue 读 GM 恒 0 不可用），按 batch 拍平 task→(batch,branch) 映射，只算被选中分支的权重/数据，跳过 n_branch-1 条完整计算链（33_AdaptiveAttention: 基线 geomean 0.187x→0.4494x 2.4x，大 shape 5-7.6x） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x42.md |
| X43 | Cube 输出 FixPipe 融合 Epilogue（C_TYPE=输出 dtype+SetBias，acc 不落 GM） | L2 | matmul, cv-fusion, epilogue, fixpipe, launch-reduction | matmul+向量后处理（bias/cast/轻量 elementwise，不含跨行归约）的融合算子，把独立 epilogue kernel 下沉进 Cube FixPipe 段：C_TYPE 直接设为输出 dtype（fp16/bf16）+ EnableBias/SetBias 在 FixPipe 内做 fp32 acc+bias 后 F322F16/BF16 写 GM，acc [M,N] 完全不落 GM、launch 数 -1、消除 acc 的 GM 写/读往返。SetBias 必须 per-N-tile 全局列偏移 SetBias(biasGlobal[nIdx*singleN])（grid-stride 切分下内置偏移不按列块走，否则多 N-tile 时 bias 加错列）；一个 ascendc_library 只能放一个 MIX cube kernel（CV 融合算子，两路独立发现：0.451x→0.620/0.633x +40%） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x43.md |
| X44 | fp32 Cube mmad 全量重写替代向量模拟 matmul（SetHF32(false) 全精度） | L2 | matmul, attention, cube, fp32, vector-simulated-matmul, precision | 基线把 matmul 实现为 AIV 向量标量模拟（16×16 tile 逐点 GetValue 累加）绕过 Cube 是数量级性能杀手：算子内全部 matmul 改 Matmul 高阶 API（ASCENDC_CUBE_ONLY+REGIST_MATMUL_OBJ+IterateAll）直接走 fp32 Cube，910B2C 上 SetHF32(false) 关闭 HF32 精度截断获全精度（MERE~1e-6；HF32(true) 仅 10-bit 尾数不够 fp32 rtol=1.2e-4）。工程红线：fp32 向量 op count 须 ≥8 且为 8 的倍数；非 2 幂 nPad 手写 halving 归约会越界，改 WholeReduceMax/Sum；uint8 位图 mask 须 256-bit block 对齐（attention 族多 matmul 流水: 0.0013x→4.149x vs PyTorch，小 shape 2-3x、大 shape 7-9x） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x44.md |
| X45 | 融合管线 GM 往返消除（split-into-GEMM-load + weight-stationary + attention 软件流水） | L2 | attention, gqa, gm-roundtrip, weight-stationary, l1-residency, software-pipelining | 大 fp32 投影 GEMM 为 MTE2(GM→L1)-bound（aic_mte2_ratio~0.68，cube/fixpipe idle）时，融合管线中间张量落 GM 再读回全是浪费，三个独立杠杆：① A 侧 split-into-GEMM-load——fp32→fp16 hi/lo 拆分 AIV 就地 UB 完成经 workspace ring 交 AIC，A tile 整 [64,kPad] 驻留 L1 跨 column-task 复用（须 bx 行块粒度派生，每 task 重拆会回归）；② B 侧 weight-stationary 混合网格（MGROUP=6）——B tile 每 k-step 只载一次跨 6 个 M 块复用，仅 mNum>MGROUP 且任务数≥核数才启用（WS 额外屏障对小 shape 反噬 8-13%），L0C 128KB 约束 MGROUP*64*64*4B（MGROUP=7 尾块不均回归）；③ attention 段软件流水（21_GroupedQueryAttention: 0.657→0.926x，A 侧 +1.3~10.3%、B 侧 c35 +12%） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x45.md |
| X46 | 原生 fp32 Cube 单 pass 投影/attention 替代 fp16 hi/lo 3-mmad（对齐 PyTorch fp32 参考路径） | L2 | discovered | 见 knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x46.md | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x46.md |
| X47 | 融合 attention 结构消除 + core_softmax 块向量化 + launch 合并 | L2 | discovered | 见 knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x47.md | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x47.md |
| X48 | 链式 GEMM M-block 归属调度（第一阶段输出复用） | L2 | discovered | 见 knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x48.md | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x48.md |
| X49 | 两阶段 GEMM 跨核生产/消费切分（N-split stage-split + GM-volatile flag） | L2 | discovered | 见 knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x49.md | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x49.md |
| X50 | 融合 decode MLA kernel — 5→2 kernel + (j,head-block) 并行 + KV tile 跨 head 复用 | L2 | discovered | 见 knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x50.md | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x50.md |
| X51 | 大 S 小 landmark 统一 cube-core 路由阈值扩展 | L2 | discovered | 见 knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x51.md | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x51.md |
| X52 | Host 数据通路消除（reshape-materialization 消除 + 条件 at::empty + 单 pass strided-gather transpose） | L2 | discovered | 见 knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x52.md | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x52.md |
| X53 | 批量 2D online softmax（跨 head-group 向量化，热循环 0 标量）+ 自适应 head-group tiling | L2 | discovered | 见 knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x53.md | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x53.md |
| X54 | (q, head-block) 任务切分 + 全宽 per-head 归约 + 双缓冲 tile gather + PV 行树归约 | L2 | discovered | 见 knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x54.md | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x54.md |
| X55 | Shared-im2col 双分支 Conv-on-Cube（一份 im2col 供两分支 + HF32-for-fp32 conv 路由） | L2 | discovered | 见 knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x55.md | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x55.md |
| X56 | STRIDE-SPARSE residue-block Cube attention（stride-mask 按 residue 分组只算有效列） | L2 | discovered | 见 knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x56.md | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x56.md |
| X57 | attn+outproj 单 kernel 链融合 + native proj BLOCK_M 64 头合并 GEMM | L2 | discovered | 见 knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x57.md | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x57.md |
| X58 | N=1 matvec 链归约突破（多相位拆分 + RA 二叉树 ReduceSum + Broadcast 消标量 + PipeBarrier 松弛） | L2 | discovered | 见 knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x58.md | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x58.md |
| X59 | Nibble 分形批转置（可分离偏移表+向量链整批+DMA合并） | L2 | quantization, transpose, conversion, datapath, bandwidth-bound | int4/fp4 nibble→ZN 分形的 kernel 内转换在无 shift 指令平台（310P）向量 op 密集+碎 DMA 主导（带宽效率<10%）时：连续窗口合并成批（nibble 提取链整批 8K 元素）、转置用单次大 gather+可分离偏移表 T[n][k]=P[n]+Q[k]（Duplicate+Add 外积向量化建表免 N*K 标量 SetValue）、MTE2 背靠背+MTE3 合并连续大块且与下批向量链重叠（quant_batch_matmul_v4 W4A8 310P: 5 shape 2.66-3.53x, geomean 3.43x, 精度 bit-exact） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x59.md |
| X60 | 向量链发射最小化（代数重写+calcount合并） | L2 | quantization, conversion, datapath, issue-bound | 向量链密集且每 op 跟屏障、Scalar busy 与 Vector busy 同高（发射率墙 ~4-5ns/指令）时：位代数恒等式重写数值链（伴随掩码 And 替代 floor 技巧链，21 call/13 屏障→9 call/5 屏障）+ N 次小 calcount 同类 call 合并为 1 个大 call（微指令流不变，省 call site 发射开销）。恒等式须代数可证 bit-exact；先查头文件排除 API 死路（m200 Gatherb NOT_SUPPORT、Gather 128 half/repeat）(quant_batch_matmul_v4 W4Trans 310P round2: T2 1.40x, Vector 848→616us Scalar 310→83us) | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x60.md |
| X61 | 硬件转置指令替代 vgather 通路（v4dtrans+strided copy 交织） | L2 | conversion, quantization, transpose, datapath, issue-bound | vgather 被 Gather repeat 上限钉死在发射率墙时查 platform_config Intrinsic 清单找专用转置指令: 310P3 无 v4trans 但有 v4dtrans/vtranspose, Transpose(NCHW2NHWC) 一次完成 [C,HW]->[HW,C] tile 转置(几何 dst[hw*C+c]=src[c*HW+hw], 上板首测); 每窗 2 次 Transpose(两路分量)+2 次 strided UB->UB DataCopy 行交织替代全部 vgather 微指令(4 发射单元/窗 vs 32 条); E/H 暂存覆盖流水线死 buffer, offset 表随 gather 删除 (quant_batch_matmul_v4 W4Trans 310P round3: T2 3.99x 累计, Vector 567→163us) | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x61.md |

---

## How to Use

1. **先归族**：打开 [`scenario-routing/README.md`](scenario-routing/README.md)，用结构判据确定算子族，打开对应族 workflow
2. 按 workflow 阶段顺序选卡（通用前置 → 族内阶段），本文档表格仅作补充反查
3. Read each referenced `playbooks/XXX.md` detail file (P102/P103 无 playbook，读对应 cards/ 卡片)
4. Apply the selected patterns in `tiling_pass`, `init_pass`, `process_pass`, `process_nonaligned_pass`
5. **Check compatibility**: Read `strategy-compatibility.md` to verify no mutual exclusions and all dependencies are met

### 族判定快速路由（选策略前先归族）

**已下沉为独立场景路由目录**：[`scenario-routing/README.md`](scenario-routing/README.md)。先按结构判据归族（Attention 系 FA/GQA/MLA / SSM-Scan 系 GDN-KDA/GLA / conv / matmul / quantization / reduction / sort / conversion / 小族 / elementwise），再打开对应族的 workflow 文件按阶段顺序选卡——不要在下表平铺选卡。下表仅作补充反查。

### 按算子类型快速查表 (Quick Reference by Operator Type)

| Operator Type | L0 Strategies | L1 Strategies |
|---------------|---------------|---------------|
| LayerNorm / RMSNorm / GroupNorm | D1, D2, P1, P2, P3, A1, A2, A6 | P19, P55, P68, P69, P81, P84, P94, P95, P98, P99 |
| Quantization ops | D1, D4, D3, P2, P4, A3, A8 | P48, P49, P70 |
| Element-wise (foreach) | D1, D2, P1, P2, A1, A4 | P67, P69, P84, P89, P95, P97, P100, P101, P112 |
| Softmax / Attention | D1, P1, P5, A3, A4, A5 | P14, P18, P38, P55, P68, P81, P94, P100, P119, P123 |
| Pooling / Gather | D1, P1, P2, P11, A7 | P33, P56, P59 |
| Optimizer ops | D1, P1, P2, P9, A1 | P81, P84, P94, P95 |
| Broadcast / Mask ops | D1, P1, P12, A5 | P67, P99, P113, P118 |
| Index / Scatter ops | D1, P1, P7, A7 | P32, P64, P100 |
| Special / Complex ops | D2, D3 | P85, P88, P96, P102 |
| CV Matmul (Cube+Vector) | D1, D3, P1, P4, P7, A3 | P14, P17, P46, P47, P49, P51, P52, P63, P70, P71, P72, P73, P78, P79, P83, P86, P87, P90, P92, P93, P102, P103, P104, P105, P106, P107, P108, P117, P120 |
| CV FFN (MoE) | D1, D3, P1, P4, A1 | P46, P48, P49, P50, P73, P75, P88 |
| Flash Attention | D1, P1, P5, A4, A5 | P14, P16, P17, P18, P29, P38, P53, P57, P58, P60, P61, P76, P77, P80, P90, P91, P115-P126, P142（UB 常驻 O，与 X38 二选一）, P143（host launch 合并，中小 shape）（**按 scenario-routing/flashattention.md 的分支路由 + 基础链 + trait 追加选用，勿平铺**） |
| SSM/Scan（GDN-KDA/GLA） | D1, P1, A2, A5 | P132-P136（**按 scenario-routing/ssm-scan.md 分支选用**）；近邻 P109/P110、X15/X16 |
| Reduction (mean/sum/max/min) | D1, P1, P2, A1, A2 | P68, P69, P109, P110 |
| Sort / TopK | D1, P1, P2, A7 | P96, P111 |
| Transpose / Layout Conversion | D1, P1, P7, P10 | P127（模式特化路由）, P26, P114 |
| Convolution (conv1d/2d/3d, depthwise, ConvTranspose) | D1, P1, P7, A1 | P130（1×1）, P128（fp32）, P129（fused im2col）, P131（repack 缓存）——按 scenario-routing/convolution.md 模式分发 |
| Data Path (通用数据通路) | D1, D5, P1, P8, P10, P20, P25, P40, P54, P65, P66, P74 | — |

### Attention 系优化 SOP

**已迁入场景路由**：[`scenario-routing/flashattention.md`](scenario-routing/flashattention.md)（Step 2 分支路由 standard/gqa/mla → Step 3 基础链 S1 结构改写 → S2 CV 流水 → S3 内存搬运 → S4 mask/稀疏特化 → S5 调度均衡 → S6 通用收尾 → Step 4 正交 trait 追加）。
**SSM/Scan 系**（GDN-KDA/GLA）走 [`scenario-routing/ssm-scan.md`](scenario-routing/ssm-scan.md)（P132 三阶段分解 → P133 CV 流水 → P134 任务划分 → P135 求逆分治[仅 GDN-KDA] → P136 bf16 量化）。

### 按瓶颈类型查表 (Bottleneck-Driven Strategy Selection)

供世界模型 profiling_evidence 参考：

| Bottleneck Type | Primary | Secondary | Anti |
|-----------------|---------|-----------|------|
| mte2_stall | P1, P10 | P2, P7, P19, P18, P56, P59, P66, P92, P93, P101, P112, P113, P114, P118, P119, P124, P127, P129, P130 | P3 |
| mte3_stall | P1 | P8, P19, P40, P56, P59 | — |
| tiling_imbalance | P4, P2 | P11, P47, P51, P58, P72, P73, P75, P104, P106, P116, P126 | P3 |
| scalar_loading | P5, P67 | P2, P54, P90, P97, P99, P100, P101 | — |
| scalar_compute | P5, P67, P94, P95, P98, P99, P100, P102 | P84, P103, P111, P120, P126, P127 | — |
| compute_bound | D1, P46, P94, P95, P96, P102 | A1, P47, P68, P69, P79, P84, P91, P92, P103, P120, P123, P127, P128, P130, P131 | — |
| near_optimal | P98 | — | — |
| no_overlap | P1 | P10, P19, P63, P118, P119, P121, P122, P125 | — |
| partial_overlap | P1, P2 | P8, P18, P26, P53, P83, P90, P125 | — |
| undersize_transfer | P2, P10 | P7, P25, P45, P66 | — |
| icache_miss | — | — | — |
| bus_contention | P8, P65 | P1, P28, P64, P93 | — |
| l2_cache_thrash | P52, P74 | P61, P78, P105, P107 | — |
| ub_memory_pressure | P8, P85 | P81, P88, P89, P110, P111 | P20 |
| tiling_mismatch | P104, P108 | P105, P106, P107, P115 | — |
| cube_underutilization | P104, P106 | P102, P115, P116, P117, P122, P123, P124 | — |
| sync_overhead | P4, A4 | P111, P115 | — |
| vec_underutilization | P112 | P67, P69 | — |
| tail_block | P11 | P112, P113, P114 | — |

| X67 | MatmulImpl 中档 M per-(M,N) 三候选网格自适应分发 | L2 | discovered | 见 knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x67.md | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x67.md |
| X68 | decode MLA 精度机器放松（去 Dot2/Kahan）+ 两遍精确 p-rounding + KV-chunk 核切分 | L2 | discovered | 见 knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x68.md | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x68.md |
| X71 | 纯向量 AIV 注意力全重构 + 最小跨管同步纪律（AIC/MIX 注册不可用时） | L2 | discovered, pure-vector, aiv-only, sync-discipline, attention | AIC/MIX 注册不可用时把 attention 全链重构到 AIV 纯向量管，跨管同步最小化（pipebarrier 纪律）+ causal 跳过 + axpy 融合 + strided datacopy。103_HSTU round_1/parallel_2：baseline 0.4356x → 3.73x geomean vs PyTorch（kernel mean_asc 35094→5107us，~6.9x，50/50 精度 PASS） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x71.md |
| X72 | 掩码来源分析削减向量管计算（全 1 快路径 + 对角三角跳过 + S 广播融合） | L2 | discovered, mask-provenance, vector-pipe-reduction, attention | 按掩码来源三分：全 1 块走快路径、对角块三角跳过、S 广播融合，逐位等价（fp64 maxdiff=0）。103_HSTU round_3/parallel_2：3.99x → 4.57x geomean vs PyTorch（+14.5%，50/50 精度 PASS），与 X71/P94/P123 叠加 | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x72.md |
| X73 | 投影 GEMM exact-N 编译期 da 分发消除零填充列计算 + QKV 三面板共享单次 A-chunk 加载 | L2 | discovered | 见 knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x73.md | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x73.md |
| X74 | 流量分档异构核数 | 按总流量分档选 AIV 核数（小流量维持 40，大流量用满 48），解 launch 地板与 per-core MTE 上限的双约束 |
| X75 | 纯向量算子显式声明 AIV_ONLY task type | L1 | discovered | 消除默认 MIX 编译带来的空转 AIC 派发（.so 含 *_mix_aic 即命中）；`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` 每个入口一行，tiling 不变；Σasc -9.4%、geomean_speedup 0.92→1.18（Maximum round3） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x75.md |
| X76 | 分组相位化调度 | 每核 G 块一组、G×MTE2→G×VEC→G×MTE3，打破同序发射下 per-tile DMA 往返暴露；组深度有尖峰最优（f16 大流式 D=2，D=3 反而退回）|
| X77 | 输出依赖图驱动的 stage/barrier 消除 | L2 | discovered | 调用方只取部分输出时，先画输出依赖闭包再删非必需 stage：intra 判定裁掉跨 chunk 状态递推段 + 折叠 layout-only stage（identity gate 实为 BSND→head-major 转置）+ 全局 SyncAll 换 AIV→AIC flag + AIV0↔AIV1 子块 barrier；门控只复用已有 tiling 字段（禁追加字段，overlay host tiling 库不被 rebuild 刷新）；ChunkKdaFwdIntra 同窗口 1.395x 相对基线 | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x77.md |
| X78 | 逐 token 串行 scan 的标量归约链清零（分组 WholeReduceSum 保序归约 + 归约块 DMA 合并 + 去 per-task 全屏障） | L1 | discovered | 见 knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x78.md | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x78.md |
| X79 | host 准备全融合单 launch（含**布局核/计算核分离**变体） | L2 | discovered | 判定信号=`task_time.csv` 里真内核占比<40% 而 ViewCopy/Transpose/ZerosLike 占比>50%；把 pad/permute/packing 全搬进内核，原始布局 `DataCopyPad` 跨步直读 + UB pitch 取 32B 倍数用 rightPadding 表达，整算子 1 个 device task（608.12us→31.76us，19.2x）。**边界（ChunkKdaBwdIntra 补证）**：当最小操作数粒度远小于行宽时（32B 粒度/4KB 步长）单核直读会摧毁 L2 sector 效率（实测净回归 0.6256x）；此时改用**分离布局核**变体——新增 pack/unpack 两个纯 AIV 布局核、计算核 md5 不变，device task 112→5，935.39→382.07us = 2.448x | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x79.md |
| X80 | 阶段融合的「生产者==消费者」任务重分区（整体删掉一次设备级 SyncAll） | L2 | discovered | 把生产者阶段重划到消费者**完全相同的 task→core 映射**上，使中间量不跨核 ⇒ 该 SyncAll 语义冗余，只留核内 MTE3→MTE2 事件；配 bit-exact 的逐行链整块化（逐元素提升到整 tile、只留顺序累积 `out[r]=out[r-1]+g[r]`，每行 ~11 barrier→~1、64 次小 MTE3→一次 32KB）与 host 精确门控开关。ChunkKdaWithFusedGateFwd：geomean 1233.7→947.3us（**−23.2%**）= 1.3068x，40/40 精度按位等价；同一次改动把 tail 二次项 c 由 0.1530→−0.0094（基线中占总时长 14.9% 的 partial-chunk 惩罚一并消除）。边界：单次设备级 SyncAll 仅 ≈10.5us；旗标不能与融合叠加（FFTS_MAX_FLAG=7 已耗尽）；被删阶段残留 UB 序言会吃掉收益；阶段内「减指令/barrier/cube 次数」在本类算子实测五次全 0 | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x80.md |
| X81 | Kc 单块化（恢复被硬编码 TEMP 覆盖的 host K 维分块常量） | L2 | discovered | 判定信号：host tiling 里「先算出 Kc、紧随一行硬编码覆盖」（注释常带 TEMP/TODO），或实测「每多一个 nKb 块 per-step 约 +2us」。逐 token T 串行 scan 型向量核里，Kc 被钉死会使 `nKb=ceil(Kp/Kc)` 从 1 涨到 3~6 —— 每步把同一份 sdec/q/k/b 按块再从 GM 读一遍、并在 512~768 元素的微 tile 上重发 Cast/Broadcast/ReduceSum，全部落在串行关键路径上。恢复原生 Kc（`min(Kp, max(16,(4608/max(Vp,16))/16*16))`）后 `nKb=1`。**三件配套必做**：①归约 workLocal 随 kvSize 重定尺（硬定尺 8KB 会在恢复 Kc 后溢出到相邻 UB buffer，静默污染 sdec：`dg` max_abs_diff 482→1.5e-5）；②host 控制字段的打包槽位不要迁（`KcPacked=Kc+1000*phase+10000*nChunks` 本就放得下 Kc≤128，迁到与 kernel 成员同名的参数槽会造成形参遮蔽 → 27MB UB 越界）；③UB 预算按新 kvSize 重算 + nChunks 取 `totalTasks % aivCoreNum == 0` 的均衡候选。Gdn2Bwd（反向 adjoint）round_3/parallel_1：vs PyTorch geomean 5.093x→**11.2134x**（相对基线 kernel **2.20x**，case44 25951.8→10684.4us=2.43x），在其上一任 champion（T 段切分 6.9249x，case44 23270.0us）之上再取 1.62x（case44 2.18x）。**精度边界**：Kc 改动必然重排跨 Kb 归约分组 ⇒ bf16 输出 1~2 ULP 偏差、fp32 ~1e-7，逐位一致在数学上不可达（见 failure pattern f22 的 `Kc` 回退对照实验）| knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x81.md |
| X82 | 门控幅度预算驱动的 sub-chunk 收缩（衰减可分离化）+ V 轴并行 | L2 | discovered | 分块 delta-rule 的 sub-chunk 长度 C 是自由参数但非数值自由：参考 C=64 使 |cumsum(gate)| 达 181（exp 上/下溢，参考的 clamp(..,max=0) 在严格下三角是 no-op）；按门控幅度预算收缩 C 到 16（实测最大 |cumsum| 52.6）后 decay[i,j]=exp(G[i])·exp(-G[j]) 可分离且恒有限，块内由 [C,C,K] rank-3 衰减张量的 elementwise 乘+归约 塌缩为 4 个小 GEMM（Aqk/Lm/w/u）。配套 V 作并行轴（task=(b,hv,v-block)，状态切片独立演化 ⇒ 零核间通信、无 SyncAll、state 全程 UB 常驻），V tile 宽取一个 32B 单位。FusedRecurrentKda：asc 329.83→81.16us（同一 45 case 配对，相对基线 4.06x；vs PyTorch 6.639x→26.9788x）。边界：C 收缩增加块数（串行段），T 很长时可能反超；本线 batch 模式崩溃 507035 | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x82.md |
| X83 | chunk-scan 内核的 AIV/AIC 双流解耦软件流水（状态链外提 + 索引偏移重排握手） | L2 | discovered, cv-decoupling, software-pipelining, cross-core-flag, chunk-scan | MIX(Cube+Vector) 的 chunk-scan/SSM 内核每 chunk V/C 交替 rendezvous 导致两单元交替空转；判据 aiv_busy_frac 与 aic_busy_frac 均 <1 且可展开出与状态无关的 cube/向量工作 → 把状态链外提、两条流水按 index 偏移推进（1-deep handshake，超前 cube 段放轮末） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x83.md |
| X84 | 固定延迟受限内核的预计算描述符 + 拆 AscendC 缓冲框架 | L2 | matmul, batch-matmul, fixed-latency-bound, one-task-per-core | host tiling 保证 taskNum ≤ 核数（每核恰好 1 task）⇒ 无摊销，唯一杠杆是缩短每核固定指令流：(1) host 打紧凑 launch 描述符（最终值：偏移/rowsValid/colsValid/全描述符标量），device 只剩「读自己 cacheline + 发描述符」，删掉 2 次整数除法+~15 派生算术+cPadFlag 分支，且搭 per-shape 缓存 ⇒ 稳态 host 成本 0；(2) 拆 TPipe/TBuf（其 ctor 做 ResetPool + 对不用的 matmul 宏无条件 AllocEventID/SetFlag<M_MTE1>，再 InitBuffer×N + TBuf::Get×N）——TBuf 地址是池内相对偏移，可用 `LocalTensor<T>(pos,addr,size)` 在原偏移重建，等价替换；(3) 单任务直线路径，去循环与前导 PipeBarrier<PIPE_ALL>。1_BatchMatmul: 1.2709x→**1.3744x** vs PyTorch（配对 A/B 0.9120/0.9111，噪声底 0.9944，约 15× 噪声底；51/51 精度、49/51 例 ≥1.0x） | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x84.md |
| X85 | 冻结参考 max 的延迟归一化在线 softmax（deferred-rescaling） | L2 | discovered | 判定信号：流式/分块 softmax（FA/GQA/MLA 的 KV 循环）已在线归一化、向量侧 per-tile rescale 是主要开销（向量占用高、cube 与之交替空转）。做法：整行只取**第一个 KV tile 的 rowmax** 作唯一参考 m* 并冻结，后续所有 tile 都用 `exp(S_t − m*)`，运行态 `acc_o` **只加不缩**（V2 从 `Broadcast(alpha)+Mul+Add` 退化为一条裸 `Add`），`reduce_max` 从每 tile 一次降为每 task 一次，`alpha_ring`/`sumexp_i_ring`/`alpha_2d` 三缓冲可删；末尾一次 `acc_o /= Sum`，`exp(−m*)` 在除法里整行约掉 ⇒ 与经典在线式**数学恒等**。前提：S 幅值上界有实测证据且远离 exp 溢出界 88（本例 hd≤256 时 max|S|~O(10)、σ≤2）；除零不可能（m* 项 exp(0)=1 ⇒ Sum≥1）。配套 UB 再规划：逐迭代 `stage_o` 别名到已死的 S/P tile 区、归一化改 `[M,64]` 行复制 + 带 repeat-stride 的 `Div`。8_SelfAttention round_1：50/50 精度 PASS（fp32 max_abs≤5e-8）、45 例无参考伪影 geomean 1.0044x vs PyTorch；同变体修复基线 da≥96 UB 越界与 da>128 host 守卫（35/50→50/50 可运行）。**边界**：无 logits 上界证据时禁用（退回经典 rescale 或加 clamp）；单 tile 序列零收益 | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x85.md |
| X86 | 零填充 staging buffer 的构造方式消除（strided zeros+切片赋值 → F.pad/torch.cat） | L2 | discovered, host-overhead, padding, scalar-path | 判定信号：`--keep-prof` 读 task_time.csv，若 `aclnnInplaceCopy_*ViewCopy*` / `aclnnInplaceZero_*` 合计占该 case 设备时间 >20%（padded shape 尤甚）即命中——`out = torch.zeros(shape); out[slices] = x` 因「目标行比源行宽」退化成带 stride 的写，910B2C 上落字节粒度标量路径（实测 15~23 ns/element ≈ 0.05TB/s，比 HBM 慢一个数量级），比同一次 call 的 `quant_matmul_mm` 还大。改逐维 `torch.nn.functional.pad`（或 `torch.cat`，沿源连续方向遍历）后 pad 段辅助 kernel 降 77~84%（case10 637.6→146.2、case19 833.0→145.5、case22 416.9→272.9、case35 1539.9→253.3us，逐 kernel Task Duration 口径），padded case 1.9~2.7x，未 pad case 中性（case0 占比 0.2%）。**逐位等价**：F.pad 对非连续输入（expand 广播视图）须先 `.contiguous()`。与 X39（消 memset 地板）/X52（消 reshape 物化）同族可叠加。4_QuantMatmul round_1：x0（纯 F.pad）24 例 geomean 0.6292→0.7419（+17.9%），n3（torch.cat+X24 AIC_ONLY/AIV_ONLY）→**0.8396**（+33.5%），两路独立命中同一瓶颈 | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x86.md |
| X87 | Grouped 量化 matmul 的 ND-int32 路由 + 行块批处理 epilogue | L2 | discovered, grouped-matmul, moe, quantization, swiglu, nd-route, row-block-batching | 判定信号：task_time 显示 AIV epilogue 是 AIC matmul 的数倍（本例基线 5–31×）且 epilogue 成本是**指令条数/行**而非算力。两级改造：(1) Cube C 由 NZ 路由 **ND int32**（host SetCType 必须与 kernel 模板 C 一致，混配即 507015），删掉 NZ 碎片重组与逐行 2×n1 个 64B strided DataCopyPad；(2) epilogue 由「逐 token 行」改「RB 行成块」——一次 2D DataCopyPad（blockCount=RB, blockLen=No*4, srcStride=(N-No)*4, dstStride=0，**两半分开取**才可表达），Cast/scale/Silu/Mul/Abs/Div/clamp/存各只跑一次 over RB*No lanes（每行 ~18 条 → ~2.7 条），absmax 折叠改行数无关（chunk-split Max, mask=64, repeatTimes=RB, repStride=No/8）。**注意**：UB→GM 的 srcStride 单位是 32B；absmax 折叠需 32B 对齐；行宽 pad 到 align-64（不是 2 的幂）省 UB；取数用 expert 内行偏移；sub-32B 元素数的 VEC 前缀运算会 507015。**规模边界**：No_ 大（N≥1536）后转 cube-bound（本例 case47 322 GFLOP ≈1.17ms @~268 TOPS），收益趋零。6_GroupedMatmulSwigluQuantV2：round_1/parallel_0（ND 路由+单遍 epilogue+baseM128+解除 M 守卫）0.0233x→0.5639x；round_2/parallel_2（行块批处理）→0.7924x；round_3/parallel_0（cube 侧让框架自导 L1 计划：SetFixSplit(128,512)+SetBufferSpace(512KB,128KB)，A 整条 K 轴钉 L1，per-K 增益随 K 单调升）→**0.8060x/0.8164x，已达 0.8x 目标**（50/50 精度）。同窗口 A/B：r1最优0.57x → r2最优0.78x → r3最优0.8060x | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x87.md |
| X88 | 删 gemm_v0 入口 MTE2→MTE1 全量栅栏（恢复逐拍 pipe 重叠） | L2 | discovered, gemm-v0, pipe-overlap, no-overlap, handshake-removal, cube-mmad, arch-2201 | 判定指纹：aic 三条 pipe 占比**之和 ≈ 1.0**（本例 MTE2 0.517 + MTE1 0.157 + MAC 0.310 = 0.984）——说明搬运与计算逐拍零重叠，多槽 ring/GEMM_STAGES 再深也无效。根因是 `gemm_v0` 入口那条 `Set/Wait<MTE2_MTE1>(L0AB_EVENT)`：L1 的真实消费者是 MTE1，而调用方 slot 事件是 `MTE2_M`，这条栅栏把 MTE1 压到整条 MTE2 队列排空之后。**删掉这两行**即可（L1 读侧由调用方 `WaitFlag<MTE2_M>(EV_READY+slot)` 经 M 管道间接保护；写侧由 `Set/Wait<M_MTE2>(2)` 保护）。自检：`aic_mte2_ratio` 显著上升（0.517→0.887）+ 三 pipe 占比之和 >1.5（→1.783）。**反例**：把 slot 事件统一改成 `MTE2_MTE1` 的教科书写法在 CANN 8.5.1/arch 2201 行为不符预期，每拍 3.077us ≈ 3× 单拍 MTE2，全线慢 1.45~1.59x——收益全部来自删栅栏。与 X73/P1 多槽 ring 互补（本卡管"能重叠"，多槽管"重叠多深"）。10_FlashAttention round_2/parallel_2：case33 `fa_qkv_proj_fp16` 16104→8676us（1.86x），整算子 24703→17883us；全量 50 case 同 run geomean 0.5222→**0.6546**（父→本变体 1.2586x，47/47 交集 case 全变快），与父实现 5/5 case 逐位相同 | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x88.md |
| X89 | fp32 投影 GEMM 的 exact-N 编译期分发 + L0C 预算面板分组 | L2 | discovered, exact-n, l0c-budget, panel-grouping, projection-gemm, mte2-bound, fp32-core, 910b2c | 投影侧被 msprof 证实是 **MTE2/DRAM 流量 bound**（12.9GB/11.25ms = 1.15TB/s ≈ 本机上限）时，一切改动都要围绕**减字节**。四项正交：①`G = min(3, floor(32768/(BLOCK_M*DA)))` 面板分组（一个 task 内 Q/K/V 三累加器并排，A 每 k-step 只载一次被 G 个面板共用）；②exact-N 编译期分发（N = 真实 da，消除列 `[da,128)` 零填充列计算与搬运，da>128 顺带 `n_loops 2->1`）；③BLOCK_M 模板化 + 按 da 分派（BM=256 减半 B 侧重读）；④host 侧权重预处理缓存（本算子 prep 22834→3691us = 6.19x）。**分派表（实测）**：`da<=85 → BM=128+G=3`（K1 1.64~1.82x）；`85<da<=128 → BM=256+G=1`（~1.20x，A 侧重读不可能低于 2）；`da>128 → BM=128+G=1+exact-N`（1.14~1.44x）。`da=128` 档是物理墙（cube_utilization 96.4%）。**坑**：`TBuf::GetWithOffset` 第 2 参数是字节偏移（按元素写会让 G>=2 累加器静默重叠）；host grid 必须与 kernel block_num 同源。8_SelfAttention round_2/parallel_0 + round_3/parallel_2：整链 opref ratio 2.0215→**1.2916**，vs PyTorch 1.1145x→**1.7443x**，10/50 case 反超 ops-ref，50/50 PASS | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x89.md |
| X90 | 归约轴 group-flatten（块对角权重打包）消除 per-group padding | L2 | conv, cv_fusion, matmul, im2col, per-group-padding, block-diagonal | 分组/批维上的 128 对齐 padding 被重复付 G 次时（col/acc 搬运量 >> x+y 字节）：把 group 折进 K（Kflat=Ci·k·k，权重 host 侧打成 (NS,KS) 块对角），KS=RoundUp(Kflat+1,16)、NS=RoundUp(Co,128)；配套四项缺一不可 —— 变长 K panel、bias 折入 ones 行（wp[co,Kflat]=bias + col 第 Kflat 行写 1.0，epilogue 整条 bias 通路删除）、col=at::empty + 定点写有限值（防 fp32 mmad 的 0*NaN=NaN）、条件式 Fixpipe 直写 y（Ws==Wo && Do*Ho*Wo%128==0 && Co%16==0 时 acc 不分配、epilogue 不发射）。8_ConvStandard3d：0.0568x→0.1811x vs PyTorch（3.19x）；case32 col 9.13GB→1.32GB、acc 写 9.13GB→285MB、ZerosLike 6137→56us | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x90.md |
| X91 | 参考舍入同源 | 精度门的同精度兜底不可用时，对齐参考内核的中间量舍入序列（在线 softmax 统计块宽 / 中间量 dtype），而非提高自身精度 |
| X92 | 跨分支机制合并（sibling diff 再合并）把并行兄弟各自已实测的净增量叠进当前最优基座 | L2 | discovered, sibling-merge, cross-branch, verified-diff, barrier-hoisting, host-dispatch-elimination | 判定信号：同一轮多个 `passed` 节点共享**同一个父基座**、改动文件集合几乎不相交（逐文件 diff 各只剩少数 net 变更），而当前最优只吃到其中一份。三步：①用逐文件 `diff -u 最优基座/kernel/<f> 兄弟/kernel/<f>` 分离"净增量"与"基座差异"（**兄弟的基座往往是更早的分叉，整文件拷贝会把基座收益回退**）；②对两边都改过的共享文件逐行对账，**保留最优基座语义、只叠兄弟净增量**（本轮否决了兄弟的 `AIV_BLOCKS=20` 与无条件 `at::zeros`）；③验收 = 全量精度 + **同卡同命令 asc-vs-asc vs 父基座**（父基座的同卡同口径 geomean 直接拿来当分母，**不要重测父内核**浪费一整轮）。8_ConvStandard3d 三连合并：`m1=n1+x0` **-10.42%（50/50 例全变快）** → `m2=m1+w1(host 侧 wp 单次构造)` **-5.65%** → `m1_s1=m1+gemm 栅栏窗口重排` -1.68%；累计 0.34953x→**0.54165x**（-36.27%），是本会话唯一连续三轮都为正的路线。**配套机制**：host 侧每个 aclnn dispatch 的设备 task 时间是**固定 ~10-15us、与元素数无关**（`wp` 构造要 `1+G+(bias?1:0)` 次 ⇒ `11us×(G+2)`/例；小例占 27~86%），逐例扣掉后 per-case geomean **+8.7%** —— 但 device 侧自建填充 kernel 会吃 ADDR_MISALIGN/507035（GM 侧 32B 对齐 ⊕ UB 侧必须 dense 冲突），**必须在 host 侧做** | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x92.md |
| X93 | 复制型上采样的『UB 布局重排 + 整带单条 dense DMA』（H 复制烘焙进 Gather 偏移表 + 工作项上移到输入平面） | L2 | discovered, upsample, nearest, replication, integer-scale, band-dense-dma, gather-offset-table-baked-h-rep, work-item-to-input-plane, sub-32b-descriptor-overhead, 910b2c | 判定信号：**纯复制型重采样**（输出 = 输入沿空间轴的整数倍复制，零算术）且 task_time 显示 asc 吞吐**跨 shape 高度一致**且远低于实测峰值带宽（本例大 case 恒定 ~391 GB/s = 1.3 TB/s 的 30%，跨 case 波动 <1%；参考核同批 800-1175 GB/s）—— 「跨 case 恒定」就是每输出行/每元素一条 DMA 描述符开销 bound 的指纹。三步：①工作项上移到**输入平面**（job=(n,c,m)，它喂养的输出平面是连续区间 od∈[ceil(m*d_out/d_in),ceil((m+1)*d_out/d_in))，D 轴复制退化为纯 MTE3 扇出，MTE2+Vector 有效量除以 d_rep；= OKF X66 在复制型算子上的特化）；②把 **H 轴复制烘焙进 Gather 偏移表**（表长取 h_rep*W_out、尾部重复头部，按 (w_in,w_out,h_rep,elem) host 缓存 ⇒ 一条输入行一次 Gather 直接产出 h_rep 份首尾相接的 W 展开行，H 复制的 MTE3 成本归零）；③**整带一条 blockCount=1 零 stride dense DataCopyPad**（staging 逐元素等于输出行 [q0*h_rep,(q0+nb)*h_rep) 首尾相接 ⇒ UB dense→GM dense，OKF f29 由构造满足；输入侧同理整带一条载入）。大 case 每 lane DMA 条数 ~3 万条 2KB → 几百条 32~120KB。**关键约束**：`DataCopyExtParams` 的 stride 是**块间 gap**，srcStride=0 会让源**前进**而非重复读同一行 ——「同一行重复 N 次」必须先 UB 内密排复制再发 blockCount=N（否则 maxdiff 为输出量级常量 = f29）；Gather count ≤1024；srcBaseAddr 是相对 src LocalTensor 的字节偏移（行选择交给它，表可全行复用）；**删净 k 迭代边界 PipeBarrier 后单跑与 3 连跑都逐位正确，但 kernel 退出时 MTE3 在飞 DMA 会让后续 torch.npu.empty_cache() 永久挂死**（任何单次测试看不见，判据必须是「连跑+empty_cache」）；按 x75 改 AIV_ONLY（40 lane）+ 手工双缓冲 pong-pong 在**已带手工事件编排**的 kernel 上挂死，回退 MIX_AIC_1_2 —— x75 不可照搬到这类 kernel，需独立同窗口 A/B。**前提/回退**：需 h_out%h_in==0 且 w_in*sizeof(T)%32==0，否则自动回退逐行路径。15_upsample_nearest3d round_1/parallel_2（x0, open_exploration，未读策略库）：0.4492x→**1.2520x** vs PyTorch（asc geomean 1275.96→457.81us，-64.1%；逐例 min 0.821 / max 2.626，无例低于目标；50/50 PASS / 50/50 BITEXACT；同窗口 ref 内参 573.19 逐位一致）。同轮同族兄弟 n7（index-scatter workflow 逐行写回）1.1317 —— 差异即本卡第 1、2 步 | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x93.md |
| X94 | 可分离重采样/插值的 Gather 提升（逐输出行 4 次 → 每输入行 2 次）+ 取消列块循环 + 输入读一次 | L2 | discovered, resize, interpolate, bilinear, separable, gather-hoisting, ring-cache, read-once | 判定信号：**可分离线性重采样**（`out = A·X·Bᵀ`，A/B 为稀疏插值矩阵；双线性 resize / remap 的可分离特化）且逐输出点直算退化成「每输出元素 4 次 Gather + 9 条向量算子」、外层还按输出列块循环（`ncb=ceil(WO/ctile)`）导致输入读放大约 `2·ncb` 倍/输出行。关键观察：横向重采样 `Hor[h][j]=w0[j]·X[h][i0[j]]+w1[j]·X[h][i1[j]]` **只依赖（输入行号, 输出列号），与输出行号无关**——原实现却让每个输出行各自 gather 一遍（scale=2 时同一输入行被 4 个输出行各 gather 一次）。改两趟：①横向趟对每个**输入行**只做一次（2 次 Gather，宽 WO）；②纵向趟退化为 `hl0·in_h[h1]+hl1·in_h[h2]` 的纯向量 Mul/Add（**零 Gather**）；③`in_h` 行按**输入行号打 tag 的环形缓存**（2-3 槽）保证每输入行的横向趟恰好一次；④列权/索引表按输出**整宽 WO 一次物化**（`ctile=WO ⇒ ncb≡1`）使**列块循环整体消失**。归因：Gather 元素/输出行 `4·WO→≈1·WO`（scale=2）、向量算子/输出元素 `9→3/scale_h+3`、输入 GM 读/输出行 `2·ncb 行→1/scale_h 行`、每输出行 `PipeBarrier<PIPE_ALL>` 1→0。细化次序（每步单独 A/B）：v1 两趟+tag 滚动窗口 1.34x → v2 去逐行 PIPE_ALL + 顺序流式 staging 环 + 输出双缓冲 + 预取1行 2.19x(+63%) → v3 去 int64 除法（(plane,oh) 增量推进）+ 预取2行/staging 4 环 **2.68x**(+19.6%) → v4 再删 2 道 PIPE_V 栅栏 +0.1%（噪声内，回退）。**坑**：`AscendC::Gather` 的 `srcOffset` 单位是**字节**（OKF f28）；`--quick` 路径的 `aiv_<pipe>_ratio` 分母不一致（同一算子出现 `scalar 0.912/vec 0.047` 与 `vec 0.912/scalar 0.415` 并列）、且 `MIX_AIC_1_2` 派发的空转 AIC 贡献 `aic_mac_ratio 0.491/aic_scalar_ratio 0.36-0.45` 自旋记账 ⇒ 只能定性、优先用 `--aic-metrics=PipeUtilization`；屏障不要一次删光（f21 非确定性挂死）；坐标/权重表的运算顺序必须逐位沿用原码（验收看逐例 MERE 是否与重构前完全相同）；算法级优化对 launch 支配的小 shape 无效（本例唯一 >0.8 的例是 `[1,16,94,256]`）。14_resize_bilinear round_1：`parallel_2`(x0, open_exploration) 与 `parallel_0`(n2, workflow_guided) **两线独立收敛到同一数据流**；in-loop 10 例子集 core 比值 3.4202 → 0.6153/0.6280，**全量 50 例 core 比值 3.0783 → 0.5122**（min 0.2841/max 1.0590，相对基线生成核 **6.01x**），50/50 精度 PASS。与 X29（规则 stride 去 Gather）/X70（数据相关 gather 指令削减）/X93（复制型上采样，本卡权重全 1、整数 scale 的退化特例）同族可叠加 | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x94.md |
| X95 | 描述符/块数 KPI —— 搬运块数最小化优先于字节最小化 | L2 | discovered, block-count-kpi, elementwise, conv, ssm, 910b2c | 每输出/每输出元素的 DMA 描述符数（block count）作为结构 KPI：描述符固定开销 ~70 cycle/块、与 payload 无关，块数削减优先于字节削减（含 T1-T6 六条手法：多块连续合并、整面单条传输、dense-flat strided 读、A 操作数 L1 常驻、列面板宽 vs 块数、行批 dense 写）。已跨三族复现：16_iou_v2 0.2931x→9.5026x、8_ConvStandard3d 0.5461x→0.6496x（-17.41%）、33_Mamba2Fwd；附负对照（块数上升方向净损失 +2.17%）。cross_op_verified | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x95.md |
| X96 | 卷积归约轴单维递延（kd-defer）——im2col 展开维 k²→k + panel 对齐 B 平移读恢复 | L2 | discovered, conv, im2col, kd-defer, mte2-stall, 910b2c | kW==1 型三维滑窗卷积：把 im2col 展开维从 k² 降到 k（递延 kd 维），用 panel 对齐的 B 平移读把递延维恢复，col 写量与搬运块数同步下降。8_ConvStandard3d round_3/parallel_2：0.5461x→0.7695x vs PyTorch（四轮累计；本节点单轮 718.76→642.27us，asc-vs-asc -10.64%），与 X95/X90/X43/P105 同族可叠加 | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x96.md |
| X97 | 逐 token 串行 scan 的『门控/标量壳』清零与向量体遍数收敛（门控链整片外提 + 同管全屏障 N→1 + 内层 Ps×NP 读写遍数压到算法下界） | L2 | discovered, ssm, recurrent, chunk-scan, per-token-shell, gate-chain-hoist, slab-vectorization, pipe-all-collapse, pass-count-floor, 910b2c | 判定信号：逐 token 串行 scan 的**纯向量**核，per-token 墙钟不由向量体吞吐决定 —— (i) `aiv_scalar_ratio 0.38~0.46` 高于 `aiv_vec_ratio 0.27~0.34`；或 (ii) 两个 MTE 管都低（`mte2 0.13~0.21`/`mte3 0.09~0.15`）而 AIV 占 Task Duration 55~60%；或 (iii) vec 头名但逐 token 内层读写 pass 数已等于「递推+读出+归约+原位缩放」的算法下界（本算子 = 4 遍）。**前提区分**：先 grep 是否存在 L 长标量 FADD 链 —— 有则用 X78（分组保序向量归约）；无（归约已全向量化）用本卡。三子杠杆（机制不同、可叠加、逐项单独 A/B）：**(a) 门控/预处理链整片外提**（per-token `GetValue`+softplus/clamp/exp 的 ~15 步标量壳 → token-slab 整片向量化，`nrows=min(64,1024/H)`，片内每 token 只读 2 个 UB 标量，需与旧形态逐位同构）；**(b) 同管全屏障 N→1**（per-token `PipeBarrier<PIPE_ALL>` 3→1，其余改每片一次定向 `V_S`；★禁一刀切删屏障 = f21 挂死）；**(c) 内层 pass 8→4**（删 B/C 两次 Broadcast，改 `src1RepStride=0` 行复用 + `vmla` 一步；Broadcast 共享 tmp 按 `oneRepeateSize=tmpElems/8` 放大 256B→4096B 把 8 轮收成 1 轮；`Brcb`+`src0BlkStride=0` 消灭最后一次行广播 sweep；第三项改乘积结合序需单独证据）。**实测（同窗/同协议 A/B 为锚，绝对数只作区间）**：33_Mamba2Fwd（纯 AIV_ONLY、cube 不可用、本构建 SyncAll 挂死）：(a)+(b) 同窗 A/B **1.0616x**（12 例 9 例变快：c6 −19.8%/c0,c5 −11.3%/c8 −8.4%）；(c) 同会话同协议链路 30.855→25.361us（vec-主导例 c5 −32%/c8 −24%/c6 −22%）。绝对 geomean 46.886 → **25.1~25.8us（1.82~1.87x vs 基线；vs PyTorch 16.9~17.2x）**，置信度中（本机间歇劣化窗口见 f29 补记；哨兵门）。**配套必读**：f40（`reduce_sum_dyn` 的 `sharedTmpBuffer` 实需 ~`Ps*NP*4`，常量 16384 只够 `Ps<=64`，抬高 UB 布局后爆越界）、f26 补记（UB 侧 2D `DataCopyPad` 块步进 = `align32(blockLen)+dstStride*32`；新增缓冲片基址必须 32B 对齐）。 | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x97.md |
| X98 | GEMM K 轴恒等重排消权重 repack（K 序 (kk,ci)->(ci,kk) 使 repack 布局与原生 W 恒等，cube 直读 W，repack kernel+workspace+LRU cache 整体移除） | L2 | discovered, conv, weight-repack-elimination, k-axis-reorder, layout-identity, cube-direct-read, launch-reduction, 910b2c | 判定信号：im2col+GEMM 卷积每 call 权重 repack 成独立发射（aic_mte2 ~0.04 纯固定开销形态），repack 布局 `wk[co][k*ci+kk]` 与原生 `W[co][ci][kk]` 仅 K 轴遍历序不同，且计时 harness 下 repack 缓存命中率 ~0%（每 call 全量 repack，缓存方案死路）。改 GEMM K 轴序 (kk,ci)->(ci,kk) 使两布局逐字节恒等，cube 直读 W（a_stride=ci_g*k_dim，对任意 k 生效），repack launch/wk workspace/LRU 整体移除（extern "C" 保留）；k_st tight 打包同步收缩、bias 下沉 epilogue、im2col 仅改行索引（勿漏 flip 路径 !flip 守卫）。可与 X14 发射合并叠加。6_ConvStandard1d round_3 sv3：1.1290->1.3540 canonical（+19.9%，双窗互证 1.3540/1.3694），50/50 PASS；小 case 每 call 省一次发射+全量 repack 往返 | knowledge/contrib/ascendc/optimizations/evolution_strategies/discovered/disc_x98.md |
