# AscendC Kernel Optimization Strategy Index

Source: 43 production-grade operators + 6 CV fused operator expert implementations (IdeaPool). Select all applicable strategies for your operator, then **Read each referenced detail file** before writing AscendC code.

## 编号说明
P 系列编号中存在间隙（P15, P23, P27, P31, P36, P39），这些编号为预留位，当前无对应策略文件。
P53-P88 为 fork-B 增量补充的高级策略，已纳入本索引。P94-P100 为标量计算向量化替代策略（2026-07 补充）。
所有有效策略 ID 均在下方表格中列出。Agent 在 strategy_combination 中只应使用下方表格中的 ID。

## Layer 定义
- **L0 (Universal)**: 适用于几乎所有算子的通用策略（D1-D5, P1-P12, A1-A8）
- **L1 (Advanced)**: 适用于特定类别算子的高级策略（P14-P100）
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
| D1 | Mixed precision architecture | L0 | datapath | Any op with FP16/BF16 input | strategies/dtype_01_mixed_precision.md |
| D2 | Template kernel type dispatch | L0 | datapath | Compile-time multi-type kernel | strategies/dtype_02_template_kernel.md |
| D3 | TilingKey-driven type dispatch | L0 | datapath | Host-side multi-type selection | strategies/dtype_03_tilingkey_dispatch.md |
| D4 | FP8/INT4 quantization conversion | L0 | quantization | Quantization output operators | strategies/dtype_04_fp8_int4_conversion.md |
| D5 | BF16-specific handling | L0 | datapath | BF16 output with rounding control | strategies/dtype_05_bf16_special_handling.md |

## 性能优化策略 — 通用 (P1-P12) [Layer: L0]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P1 | Double buffering (BUFFER_NUM=2) | L0 | datapath | Any memory-bound kernel | strategies/perf_01_double_buffer.md |
| P2 | Adaptive tiling (Split-N vs Split-D) | L0 | datapath | Variable row/column dimensions | strategies/perf_02_adaptive_tiling.md |
| P3 | Small-D multi-row merging | L0 | datapath | Hidden size ≤ 640 or small D | strategies/perf_03_small_d_optimization.md |
| P4 | Multi-core load balancing | L0 | datapath | Uneven data distributions | strategies/perf_04_load_balance.md |
| P5 | Pipeline sync (PipeBarrier/events) | L0 | datapath | All double-buffered kernels | strategies/perf_05_pipeline_sync.md |
| P7 | 32B alignment + DataCopyPad | L0 | datapath | Non-aligned input shapes | strategies/perf_07_data_alignment.md |
| P8 | UB memory partitioning | L0 | datapath | Kernels with multiple UB tensors | strategies/perf_08_ub_memory_mgmt.md |
| P9 | Deterministic output (workspace) | L0 | datapath | Training ops needing reproducibility | strategies/perf_09_deterministic_output.md |
| P10 | Vectorized data copy | L0 | datapath | CopyIn/CopyOut optimization | strategies/perf_10_vectorized_copy.md |
| P11 | Tail block handling (GatherMask) | L0 | datapath | Pooling/gather with uneven splits | strategies/perf_11_tail_block_handling.md |
| P12 | Broadcast & mask operations | L0 | datapath | Operators with mask inputs, broadcasting | strategies/perf_12_broadcast_mask.md |

## 性能优化策略 — CV融合/高级调优 (P14-P18) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P14 | CV pipeline preload (3-slot task cache) | L1 | cv_fusion, flash_attention | Flash Attention Cube-Vector 流水预发射 | strategies/perf_14_cv_pipeline_preload.md |
| P16 | Cost-driven core partition | L1 | cv_fusion, manual | Alignment-aware 代价模型分核 | strategies/perf_16_cost_driven_core_partition.md |
| P17 | CV parallel block sizing | L1 | cv_fusion | CV 并行基本块设计 | strategies/perf_17_cv_parallel_block_sizing.md |
| P18 | L1 7-buffer resident partition | L1 | cv_fusion, flash_attention | L1 七缓冲常驻分区 | strategies/perf_18_l1_7buf_resident_partition.md |

## 性能优化策略 — 高级数据搬运 (P19-P45) [Layer: L1]

### 队列管理与双缓冲 (P19-P22)

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P19 | Custom ping-pong double buffer | L1 | datapath | 自定义 Ping-Pong 双缓冲管理 | strategies/perf_19_custom_pingpong_double_buffer.md |
| P20 | Triple buffer rotation | L1 | datapath | 三缓冲轮转（搬入/计算/搬出完全重叠） | strategies/perf_20_triple_buffer_rotation.md |
| P21 | Matrix 2×2 buffer policy | L1 | datapath | 2×2 矩阵缓冲策略 | strategies/perf_21_matrix2x2_buffer_policy.md |
| P22 | TQueBind bidirectional reuse | L1 | datapath | TQueBind 双向复用 | strategies/perf_22_tquebind_bidirectional_reuse.md |

### DataCopy 参数与对齐 (P24-P26)

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P24 | ND2NZ format copy | L1 | datapath | ND→NZ 格式转换搬运 | strategies/perf_24_nd2nz_format_copy.md |
| P25 | DataCopy pad alignment | L1 | datapath | DataCopy 填充对齐优化 | strategies/perf_25_datacopy_pad_alignment.md |
| P26 | Stride fused transpose | L1 | datapath | 步长融合转置 | strategies/perf_26_stride_fused_transpose.md |

### 同步与流水控制 (P28-P30)

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P28 | HardEvent producer-consumer | L1 | datapath | HardEvent 生产者-消费者同步 | strategies/perf_28_hardevent_producer_consumer.md |
| P29 | Cross-core AIC-AIV sync | L1 | cv_fusion | 跨核 AIC-AIV 同步 | strategies/perf_29_crosscore_aic_aiv_sync.md |
| P30 | Dense PipeBarrier vector | L1 | datapath | 密集 PipeBarrier 向量同步 | strategies/perf_30_dense_pipebarrier_vector.md |

### 特殊搬运模式 (P32-P33)

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P32 | Scatter block stride | L1 | datapath | 散射块步长搬运 | strategies/perf_32_scatter_block_stride.md |
| P33 | Gather offset table | L1 | datapath | 聚集偏移表搬运 | strategies/perf_33_gather_offset_table.md |

### 缓冲驻留与复用 (P34-P45)

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P34 | Weight load-once reuse | L1 | datapath | 权重一次加载多次复用 | strategies/perf_34_weight_load_once_reuse.md |
| P35 | TBuf resident accumulator | L1 | datapath | TBuf 常驻累加器 | strategies/perf_35_tbuf_resident_accumulator.md |
| P37 | Grad accumulator cross-batch | L1 | datapath | 梯度跨 batch 累加器 | strategies/perf_37_grad_accumulator_cross_batch.md |
| P38 | Softmax state resident | L1 | flash_attention | Softmax 状态常驻 buffer | strategies/perf_38_softmax_state_resident.md |
| P40 | Workspace double buffer | L1 | datapath | Workspace 双缓冲 | strategies/perf_40_workspace_double_buffer.md |
| P41 | Gamma/scale resident | L1 | datapath | Gamma/scale 常驻 buffer | strategies/perf_41_gamma_scale_resident.md |
| P43 | Grad ping-pong deferred scatter | L1 | datapath | 梯度 Ping-Pong 延迟散射 | strategies/perf_43_grad_pingpong_deferred_scatter.md |
| P44 | Transpose buffer resident | L1 | datapath | 转置 buffer 常驻 | strategies/perf_44_transpose_buffer_resident.md |
| P45 | TopK index cache iterator | L1 | datapath | TopK 索引缓存迭代器 | strategies/perf_45_topk_index_cache_iterator.md |
## 性能优化策略 — CV融合算子架构模式 (P46-P52) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P46 | MatmulImpl high-level API | L1 | cv_fusion, matmul | 替代手写 matmul 流水线 | strategies/perf_46_matmulimpl_highlevel_api.md |
| P47 | Diagonal block scheduling | L1 | cv_fusion, matmul | 多核 M×N 分块 L2 cache 优化 | strategies/perf_47_diagonal_block_scheduling.md |
| P48 | Multi-quantization mode dispatch | L1 | quantization, matmul | 多量化模式编译期分发 | strategies/perf_48_multi_quant_dispatch.md |
| P49 | Hardware-accelerated dequantization | L1 | quantization | AscendDequant 硬件反量化 | strategies/perf_49_hw_dequantization.md |
| P50 | SwiGLU fusion pipeline | L1 | moe, cv_fusion | MoE FFN SwiGLU 融合 (narrow scope) | strategies/perf_50_swiglu_fusion.md |
| P51 | Dynamic AIC/AIV core ratio | L1 | cv_fusion, matmul | K 维阈值动态核配比 | strategies/perf_51_dynamic_aic_aiv_ratio.md |
| P52 | L2 Cache Hint optimization | L1 | cv_fusion, matmul | 多核竞争场景 L2 禁用 | strategies/perf_52_l2_cache_hint.md |

## 性能优化策略 — 片上缓存与常驻复用 (P53-P56) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P53 | L1 resident reuse | L1 | cv_fusion, flash_attention | Q 矩阵 L1 常驻跨迭代复用，消除重复搬运 | strategies/perf_53_l1_resident_reuse.md |
| P54 | Launch overhead optimization | L1 | datapath | 减少 kernel launch 开销，合并小 kernel | strategies/perf_54_launch_overhead_optimization.md |
| P55 | Block reduce-max optimization | L1 | datapath | 块级 ReduceMax 优化，二分累加 | strategies/perf_55_block_reduce_max_optimization.md |
| P56 | DataCopy batch output | L1 | datapath | 批量 DataCopy 输出，减少搬运次数 | strategies/perf_56_datacopy_batch_output.md |

## 性能优化策略 — FlashAttention/Decode 专用 (P57-P58) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P57 | FlashDecode G-partition reduction | L1 | flash_attention | FlashDecode G 维分区归约 | strategies/perf_57_flashdecode_g_partition.md |
| P58 | TND load-balanced partition | L1 | flash_attention | T/N/D 维负载均衡分区 | strategies/perf_58_tnd_load_balance_partition.md |

## 性能优化策略 — 数据搬运与格式优化 (P59-P66) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P59 | Output transpose fusion | L1 | datapath | 输出转置融合，减少额外搬运 | strategies/perf_59_output_transpose_fusion.md |
| P60 | KV NZ format optimization | L1 | matmul, flash_attention | KV cache NZ 格式优化 | strategies/perf_60_kv_nz_format_optimization.md |
| P61 | L2 cache optimization | L1 | flash_attention | 关闭双页表开启 L2 Cache，KV 数据缓存 | strategies/perf_61_l2_cache_optimization.md |
| P63 | Iterate async interface | L1 | matmul | 异步迭代接口，隐藏搬运延迟 | strategies/perf_63_iterate_async_interface.md |
| P64 | GM address conflict avoidance | L1 | datapath | GM 地址冲突规避，优化多核访存 | strategies/perf_64_gm_address_conflict_avoidance.md |
| P65 | UB bank conflict avoidance | L1 | datapath | UB bank 冲突规避，优化地址分配 | strategies/perf_65_ub_bank_conflict_avoidance.md |
| P66 | GM 512B alignment bandwidth | L1 | datapath | GM 512B 对齐带宽优化 | strategies/perf_66_gm_512b_alignment.md |

## 性能优化策略 — Vector 计算效率与标量向量化 (P67-P69, P94-P100) [Layer: L1]

P67-P69 为 Vector 计算效率基础策略（Counter 模式、低延迟归约、UB 融合链）。P94-P100 为标量→向量专项替代策略，覆盖归约（P94）、逐元素运算（P95）、排序（P96）、零拷贝切片（P97）、自适应调度（P98）、Broadcast 批量（P99）、GatherMask（P100）七种反模式。

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P67 | Vector counter mode | L1 | datapath | Vector counter 模式，Scalar 控制流优化 | strategies/perf_67_vector_counter_mode.md |
| P68 | Low-latency reduction | L1 | datapath | 低延迟归约指令组合 (BlockReduceSum+WholeReduceSum) | strategies/perf_68_low_latency_reduction.md |
| P69 | UB fused vector chain | L1 | datapath | UB 融合连续 Vector 计算链，中间结果保留 UB | strategies/perf_69_ub_fused_vector_chain.md |
| P94 | Scalar reduction → Vector Reduce API (ReduceSum/ReduceMax/WholeReduceSum) | L1 | datapath | 逐元素标量归约循环（for(i) sum+=x[i]），Reduce/Softmax/Norm 算子 | cards/P94_scalar_reduction_to_vector.md |
| P95 | Scalar element-wise → Adds/Muls vector chain | L1 | datapath | 逐元素标量算术循环（for(i) y[i]=f(x[i])），Elementwise/Norm 算子 | cards/P95_scalar_elementwise_to_vector.md |
| P96 | Hardware sort engine (Sort32/MrgSort) replacement | L2 | datapath | 标量冒泡/选择排序或 TopK 选择，Sort/TopK/NMS 算子 | cards/P96_hardware_sort_replacement.md |
| P97 | Zero-copy sub-tensor slicing + batch vector ops | L1 | datapath | 从 UB tensor 按偏移取子集做批量运算，需避免逐元素 gather | cards/P97_zero_copy_subtensor.md |
| P98 | Hybrid scalar/vector adaptive dispatch (per-tile workload) | L1 | datapath | 小 tile 向量化收益为负（EnQue/DeQue 开销>标量循环），需自适应切换 | cards/P98_hybrid_scalar_vector_dispatch.md |
| P99 | BroadCast + batch vector op replaces row-wise scalar loop | L1 | datapath | 小向量广播到大矩阵做逐元素运算，避免逐行 Muls/Adds dispatch 开销 | cards/P99_broadcast_batch_vector_ops.md |
| P100 | GatherMask replaces element-wise SetValue/GetValue | L1 | datapath | 按固定间隔从连续 tensor 取元素重组（解交织、RoPE 奇偶分拆等） | cards/P100_gathermask_replacement.md |

## 性能优化策略 — Matmul/量化专用 (P70-P72) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P70 | FixPipe buffer inline quantization | L1 | matmul, quantization | FixPipe buffer 内联量化 | strategies/perf_70_fp_buffer_inline_quant.md |
| P71 | Matmul IBShare L1 sharing | L1 | matmul | Matmul IBShare L1 共享，减少重复搬运 | strategies/perf_71_matmul_ibshare_l1.md |
| P72 | Matmul Split-K | L1 | matmul | K 轴切分多核并行，AtomicAdd 累加 | strategies/perf_72_matmul_split_k.md |

## 性能优化策略 — CV 并行与核配比 (P73-P75) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P73 | CV parallel AIC:AIV ratio | L1 | cv_fusion | AIC:AIV 核配比与多 Workspace 流水 | strategies/perf_73_cv_parallel_aic_aiv_ratio.md |
| P74 | Selective L2 cache disable | L1 | datapath | 选择性 L2 cache 禁用，减少多核竞争 | strategies/perf_74_selective_l2_cache_disable.md |
| P75 | Dual AIV workload split | L1 | cv_fusion | 双 AIV M/S1 轴负载拆分 | strategies/perf_75_dual_aiv_workload_split.md |

## 性能优化策略 — Sparse Attention 专用 (P76-P78) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P76 | V_TEMPLATE KV pre-merge workspace | L1 | flash_attention | KV 预合并 Workspace 管理 | strategies/perf_76_v_template_kv_premerge.md |
| P77 | L1 chunk iterator sparse aggregation | L1 | flash_attention | L1 分块迭代器稀疏自适应聚合 | strategies/perf_77_l1_chunk_iterator_sparse_aggregation.md |
| P78 | On-chip cache matmul acceleration | L1 | matmul | 片上缓存加速 Matmul | strategies/perf_78_onchip_cache_matmul_acceleration.md |

## 性能优化策略 — 高级流水与融合 (P79-P84) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P79 | Grad Cube2+Cube3 fusion | L1 | matmul | 梯度 Cube2+Cube3 融合 | strategies/perf_79_grad_cube23_fusion.md |
| P80 | Sink task pipeline injection | L1 | flash_attention | Sink task 流水注入与 S2 block skip | strategies/perf_80_sink_task_pipeline_injection.md |
| P81 | Resident UB/TBuf reuse | L1 | datapath | 小参数/状态常驻 UB/TBuf，跨迭代复用 | strategies/perf_81_resident_ub_tbuf_reuse.md |
| P83 | L1 resident reuse partition | L1 | matmul | L1 常驻复用多缓冲分区 | strategies/perf_83_l1_resident_reuse_partition.md |
| P84 | Vector compute efficiency patterns | L1 | datapath | Counter 模式+低延迟指令+UB 融合链 | strategies/perf_84_vector_compute_efficiency_patterns.md |

## 性能优化策略 — 缓冲区复用与 Matmul 调优 (P85-P88) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P85 | On-chip buffer zone reuse | L1 | datapath | 片上 buffer 分区复用 | strategies/perf_85_on_chip_buffer_zone_reuse.md |
| P86 | Matmul API transport efficiency tuning | L1 | matmul | Matmul 内部数据通路效率调优 | strategies/perf_86_matmul_api_transport_efficiency_tuning.md |
| P87 | Manual mmad pipeline timing control | L1 | matmul, manual | 手动 Mmad 流水时序控制 | strategies/perf_87_manual_mmad_pipeline_timing_control.md |
| P88 | Multiphase compute phase decomposition | L1 | cv_fusion | 多阶段计算拆分，独立 TPipe 资源重分配 | strategies/perf_88_multiphase_compute_phase_decomposition.md |

## 性能优化策略 — UB 共享与 Matmul 微调 (P89-P93) [Layer: L1]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| P89 | UB high-level API temp buffer sharing | L1 | datapath | SoftMax 等高阶 API 与计算阶段 TBuf 分时复用 | strategies/perf_89_ub_api_temp_buffer_sharing.md |
| P90 | Conditional PipeBarrier by matrix size | L1 | matmul, flash_attention | 手动 Matmul L0C 累加模式按 fractal 数动态同步 | strategies/perf_90_conditional_pipebarrier.md |
| P91 | Softmax gradient pre-computation phase | L2 | flash_attention | FA 反向 dY*O ReduceSum 预计算为独立 VecSfmg 阶段 | strategies/perf_91_softmax_grad_precompute.md |
| P92 | Matmul base block compute-to-memory ratio | L1 | matmul, cv_fusion | 优化 baseM/baseN/baseK 最大化 Cube 计算访存比 | strategies/perf_92_matmul_block_compute_ratio.md |
| P93 | Matmul K-axis staggered memory access | L1 | matmul | 多核 Matmul K 轴错峰访问缓解 GM 地址冲突 | strategies/perf_93_matmul_k_stagger_access.md |

## 精度优化策略 (A1-A8) [Layer: L0]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| A1 | FP32 intermediate computation | L0 | datapath | BF16/FP16 with precision requirement | strategies/acc_01_fp32_intermediate.md |
| A2 | Welford numerically stable mean/var | L0 | datapath | LayerNorm, BatchNorm, RMSNorm | strategies/acc_02_welford_algorithm.md |
| A3 | Rounding mode control (CAST_*) | L0 | datapath | Any Cast to lower precision | strategies/acc_03_rounding_mode.md |
| A4 | SetFlag/WaitFlag event sync | L0 | datapath | Data dependencies across pipes | strategies/acc_04_pipeline_barrier.md |
| A5 | Softmax numerical stability | L0 | datapath | Softmax, attention score ops, NaN/Inf risk | strategies/acc_05_softmax_stability.md |
| A6 | High-precision rsqrt (Newton-Raphson) | L0 | datapath | Normalization ops needing rsqrt | strategies/acc_06_high_precision_rsqrt.md |
| A7 | Index & boundary safety | L0 | datapath | Gather/scatter, user-controlled indices | strategies/acc_07_index_boundary.md |
| A8 | Quantization-specific precision | L0 | quantization | Quantization output, dequantization with bias | strategies/acc_08_quant_precision.md |

## 探索发现策略 (Discovered Strategies) [Layer: L2]

| ID | Strategy | Layer | Tags | When to Apply | Detail File |
|----|----------|-------|------|---------------|-------------|
| X1 | Inline Target Capture | L2 | — | Reduce + single-element lookup ops | strategies/disc_X1.md |
| X2 | Big-Tensor-On-Device, Index-Only-Over-Bus | L2 | gather, routing, bus-bound | Host-coordinated gather/scatter where data tensor ≫ index tensor | strategies/disc_X2.md |
| X3 | End-to-End Dtype Passthrough | L2 | elementwise, foreach, multi-dtype, bandwidth-bound | Multi-dtype elementwise where wrapper upcasts fp16/bf16→fp32 and inflates HBM traffic ~5x; sink dtype boundary into kernel | discovered/disc_X3.md |
| X4 | Concat-Fused Strided DMA + Two-Level Skv Reduction | L2 | decode-attention, latency-bound, GQA N_kv=1 | Collapse nope|rope concat into strided K-tile DMA + two-level vmul→vcadd→vcadd + one-pass FA-2; kills per-key DMA/sync latency (decode 0.11-0.27x→0.25-0.79x) | discovered/disc_X4.md |
| X5 | GQA-Decode GEMV→GEMM (M=N_q) | L2 | decode-attention, compute-domain, GQA N_kv=1, cube | For GQA decode/MTP (S∈{1,2}, N_kv=1) take decode M=N_q(128/64) so QK^T=Q[N_q,D_qk]@K^T and PV=P[N_q,S_kv]@V become dense cube GEMM instead of N_q vec GEMVs; stream K once (NC=256), V once (sub-col nd2nz), N-tile PV to L0C; decode 11-33x (geomean 0.57x→3.53x) | discovered/disc_X5.md |

---

## How to Use

1. Identify your operator's characteristics (dtype, shape, op type)
2. Select ALL applicable strategy IDs from the tables above
3. Read each referenced `strategies/XXX.md` detail file
4. Apply the selected patterns in `tiling_pass`, `init_pass`, `process_pass`, `process_nonaligned_pass`
5. **Check compatibility**: Read `strategy-compatibility.md` to verify no mutual exclusions and all dependencies are met

### 按算子类型快速查表 (Quick Reference by Operator Type)

| Operator Type | L0 Strategies | L1 Strategies |
|---------------|---------------|---------------|
| LayerNorm / RMSNorm / GroupNorm | D1, D2, P1, P2, P3, A1, A2, A6 | P19, P55, P68, P69, P81, P84, P94, P95, P98, P99 |
| Quantization ops | D1, D4, D3, P2, P4, A3, A8 | P48, P49, P70 |
| Element-wise (foreach) | D1, D2, P1, P2, A1, A4 | P67, P69, P84, P89, P95, P97, P100 |
| Softmax / Attention | D1, P1, P5, A3, A4, A5 | P14, P18, P38, P55, P68, P81, P94, P100 |
| Pooling / Gather | D1, P1, P2, P11, A7 | P33, P56, P59 |
| Optimizer ops | D1, P1, P2, P9, A1 | P81, P84, P94, P95 |
| Broadcast / Mask ops | D1, P1, P12, A5 | P67, P99 |
| Index / Scatter ops | D1, P1, P7, A7 | P32, P64, P100 |
| Special / Complex ops | D2, D3 | P85, P88, P96 |
| CV Matmul (Cube+Vector) | D1, D3, P1, P4, P7, A3 | P14, P17, P46, P47, P49, P51, P52, P63, P70, P71, P72, P73, P78, P79, P83, P86, P87, P90, P92, P93 |
| CV FFN (MoE) | D1, D3, P1, P4, A1 | P46, P48, P49, P50, P73, P75, P88 |
| Flash Attention | D1, P1, P5, A4, A5 | P14, P16, P17, P18, P29, P38, P53, P57, P58, P60, P61, P76, P77, P80, P90, P91 |
| Data Path (通用数据通路) | D1, D5, P1, P8, P10, P20, P25, P40, P54, P65, P66, P74 | — |

### 按瓶颈类型查表 (Bottleneck-Driven Strategy Selection)

供世界模型 profiling_evidence 参考：

| Bottleneck Type | Primary | Secondary | Anti |
|-----------------|---------|-----------|------|
| mte2_stall | P1, P10 | P2, P7, P19, P18, P56, P59, P66, P92, P93 | P3 |
| mte3_stall | P1 | P8, P19, P40, P56, P59 | — |
| tiling_imbalance | P4, P2 | P11, P47, P51, P58, P72, P73, P75 | P3 |
| scalar_loading | P5, P67 | P2, P54, P90, P97, P99, P100 | — |
| scalar_compute | P5, P67, P94, P95, P98, P99, P100 | P84 | — |
| compute_bound | D1, P46, P94, P95, P96 | A1, P47, P68, P69, P79, P84, P91, P92 | — |
| near_optimal | P98 | — | — |
| no_overlap | P1 | P10, P19, P63 | — |
| partial_overlap | P1, P2 | P8, P18, P26, P53, P83, P90 | — |
| undersize_transfer | P2, P10 | P7, P25, P45, P66 | — |
| icache_miss | — | — | — |
| bus_contention | P8, P65 | P1, P28, P64, P93 | — |
| l2_cache_thrash | P52, P74 | P61, P78 | — |
| ub_memory_pressure | P8, P85 | P81, P88, P89 | P20 |
