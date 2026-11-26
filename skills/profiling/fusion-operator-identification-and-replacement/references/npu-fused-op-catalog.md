# 融合算子清单

本章节整理 op-plugin 仓库（TorchNPU 自定义算子）中已有的**融合算子**（Fused Operator）。融合算子指在一个算子内部完成多个基础计算、或通信与计算流水化融合的算子，通过减少张量在中间过程中的搬入搬出（Host/Device、内存带宽）开销，提升网络整体性能。

- 依据：[torch_npu接口列表](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu_list.md)
- 整理口径：以接口列表中名称或功能说明明确体现“融合”特性的接口为主，并补充部分以多算子合并实现（如 Grouped、Quant+激活、通信+MatMul 等）的接口。纯基础算子（如单个激活、单个归一化、单个量化）不在本清单范围内。
- 说明：标记为“（beta）”的接口为实验性接口，可能存在变动；标记“已废弃”的接口建议不再使用。

> **版本与来源标注**（2026-10-14 核对）
> - **文档快照**：本清单引用的接口文档来自 op-plugin 仓库（https://gitcode.com/Ascend/op-plugin）`master` 分支 commit `b80a2fdf`（2026-10-10），对应 TorchNPU `26.2.0` 发布时点；清单内 77 个接口的文档已逐一核对在该 commit 的本地副本（`references/torch_npu/`）中存在，与下文统计表合计 77 一致。
> - **版本命名**：TorchNPU 产品版本（`26.x`，与 op-plugin 分支同名）与 pip 安装包版本（`2.x`，跟随 PyTorch）是两套编号。例如产品版本 26.1.0 配套安装包 `2.10.0.post4` / PyTorch 2.10.0 / CANN 9.1.0；完整映射以官方[版本配套表](https://gitcode.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)为准。
> - **本地副本**：本 Skill 已内置该快照的完整接口文档副本（`references/torch_npu/`，199 个文档，含接口总表），文件名与在线文档一致；离线分析时优先读取本地副本。
> - **表内链接**：均指向固定 commit（`b80a2fdfce8b32da11ec3647e3946417ee39c463`，短号 `b80a2fdf`）的官方在线文档，与本 Skill 本地副本内容一致；查看最新版本请将 URL 中的完整 commit 号替换为 `master`。
> - **使用注意**：各接口在不同 TorchNPU/CANN 版本与芯片型号上的支持情况不同。评估候选时必须以接口文档"产品支持情况"章节与目标环境核实结果为准，不得仅凭本清单断言可用性。

## 统计

| 分类 | 数量 |
| --- | --- |
| Attention / 注意力融合算子 | 14 |
| 归一化（Norm）融合算子 | 11 |
| 通信与计算融合算子 | 9 |
| MoE 融合算子 | 19 |
| 矩阵乘/量化/激活融合算子 | 14 |
| 损失函数融合算子 | 6 |
| 其他融合算子 | 4 |
| **合计** | **77** |

---

## 一、Attention / 注意力融合算子

将 FlashAttention 的 QK 缩放、Softmax、PV 计算，或 MLA（Multi-head Latent Attention）的 RmsNorm、RoPE、KV Cache 更新等融合为一个算子。

| API 名称 | 融合的计算 | 说明 |
| --- | --- | --- |
| [（beta）torch_npu.npu_fused_attention_score](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/%EF%BC%88beta%EF%BC%89torch_npu-npu_fused_attention_score.md) | matmul、transpose、add、softmax、dropout、batchmatmul、permute | Transformer Attention Score 融合计算逻辑。 |
| [torch_npu.npu_fusion_attention](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fusion_attention.md) | Attention Score 全流程（QK、Softmax、PV 等） | 实现“Transformer Attention Score”的融合计算。 |
| [torch_npu.npu_fusion_attention_v2](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fusion_attention_v2.md) | Attention Score 全流程 | 在融合计算基础上支持 RoPE 扩展、偏置 sink（bias sink）及 GQA。 |
| [torch_npu.npu_fusion_attention_v3](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fusion_attention_v3.md) | Attention Score 全流程 | 在融合计算基础上支持图模式。 |
| [torch_npu.npu_fused_infer_attention_score](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score.md) | FlashAttention（全量 + 增量） | 适配推理场景，同时支持全量计算（PromptFlashAttention）与增量计算（IncreFlashAttention）。 |
| [torch_npu.npu_fused_infer_attention_score_v2](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score_v2.md) | FlashAttention（全量 + 增量）+ MLA 量化 | 在 v1 基础上新增支持 MLA（MultiHead Latent Attention）全量化功能。 |
| [torch_npu.npu_fused_floyd_attention](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_floyd_attention.md) | QK / PV 注意力（批量矩阵乘） | 训练场景下将 seq 维度作为 batch 轴，将注意力计算转换为批量矩阵乘（batchMatmul）。 |
| [torch_npu.npu_advance_step_flashattn](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_advance_step_flashattn.md) | 生成步骤推进（token 位置、seq 长度、slot 映射更新） | 在 NPU 上实现 vLLM 中 advance_step_flashattn 的功能，原地更新 input_tokens、input_positions、seq_lens 和 slot_mapping。 |
| [torch_npu.npu_kv_rmsnorm_rope_cache](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_kv_rmsnorm_rope_cache.md) | RMSNorm + RoPE + KV Cache 更新 | 融合 MLA 结构中 RMSNorm 归一化、RoPE 位置编码与更新 KVCache 的 ScatterUpdate 操作。 |
| [torch_npu.npu_kv_rmsnorm_rope_cache_v2](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_kv_rmsnorm_rope_cache_v2.md) | RMSNorm + RoPE + KV Cache 更新 | 同上，v2 版本。 |
| [torch_npu.npu_apply_rotary_pos_emb](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-apply_rotary_pos_emb.md) | query、key 两路 RoPE 计算 | 将 query 和 key 两路算子融合为单路，在旋转位置编码计算中直接对结果执行原地更新，提升推理性能。 |
| [torch_npu.npu_mla_prolog](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mla_prolog.md) | Matmul + RmsNorm + RoPE + Cache（多路） | MLA 前处理计算，包含下采样、RmsNorm、上采样、RoPE、Cache 写入等多路融合。<br>（接口计划废弃，不建议使用） |
| [torch_npu.npu_mla_prolog_v2](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mla_prolog_v2.md) | Matmul + RmsNorm + RoPE + Cache（多路） | MLA 前处理计算，v2 版本。 |
| [torch_npu.npu_mla_prolog_v3](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mla_prolog_v3.md) | Matmul + RmsNorm + RoPE + Cache（多路） | MLA 前处理计算，v3 版本新增 query_norm、dequant_scale_q_norm 输出及多量化配置支持。 |

## 二、归一化（Norm）融合算子

将残差 Add、RMSNorm/GroupNorm 与量化、激活等算子融合，减少归一化结果在中间过程的搬入搬出，常用于大模型训练与推理。

| API 名称 | 融合的计算 | 说明 |
| --- | --- | --- |
| [torch_npu.npu_add_rms_norm](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_add_rms_norm.md) | Add + RMSNorm | 将 Add 计算与 RMSNorm 归一化融合，常用于大模型残差连接后的归一化。 |
| [torch_npu.npu_add_rms_norm_v2](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_add_rms_norm_v2.md) | Add + RMSNorm | AddRmsNorm 算子，Add 算子与 RmsNorm 算子的融合。 |
| [torch_npu.npu_add_rms_norm_cast](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_add_rms_norm_cast.md) | Add + RMSNorm + Cast | 将 AddRmsNorm 后的 Cast 算子融合起来，减少搬入搬出操作。 |
| [torch_npu.npu_add_rms_norm_quant](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_add_rms_norm_quant.md) | Add + RMSNorm + Quantize | 将 RmsNorm 前的 Add 算子以及 RmsNorm 后的 Quantize 算子融合，减少搬入搬出操作。 |
| [torch_npu.npu_add_rms_norm_dynamic_quant](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_add_rms_norm_dynamic_quant.md) | Add + RMSNorm + DynamicQuant | 将 RmsNorm 前的 Add 算子和归一化输出接的 1 个或 2 个 DynamicQuant 算子融合。 |
| [torch_npu.npu_add_rms_norm_dynamic_mx_quant](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_add_rms_norm_dynamic_mx_quant.md) | Add + RMSNorm + DynamicMxQuant | 将 RmsNorm 前的 Add 算子和归一化输出接的 DynamicMxQuant 算子融合。 |
| [torch_npu.npu_rms_norm_quant](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_rms_norm_quant.md) | RMSNorm + Quantize | 融合 RmsNorm 算子和 Quantize 算子，减少搬入搬出操作。 |
| [torch_npu.npu_rms_norm_quant_v2](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_rms_norm_quant_v2.md) | RMSNorm + Quantize | 融合 RmsNorm 算子和 Quantize 算子，v2 版本新增 rstd 输出。 |
| [torch_npu.npu_rms_norm_dynamic_mx_quant](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_rms_norm_dynamic_mx_quant.md) | RMSNorm + DynamicMxQuant | 先做 RmsNorm 归一化，结果与 beta 相加后再做动态 MX 量化。 |
| [torch_npu.npu_group_norm_swish](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_group_norm_swish.md) | GroupNorm + Swish | 计算组归一化结果 y、均值 mean、标准差倒数 rstd 及 swish 输出。 |
| [torch_npu.npu_group_norm_silu](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_group_norm_silu.md) | GroupNorm + SiLU | 对输入依次执行 GroupNorm 和 SiLU 激活，返回 out、meanOut、rstdOut。 |

## 三、通信与计算融合算子

在 TP（Tensor Parallelism）等切分场景下，将 AllGather / AllReduce / ReduceScatter / AlltoAll 等集合通信与 MatMul 计算融合，算子内部实现通信与计算流水并行，降低通信等待与中间存储开销。

| API 名称 | 融合的计算 | 说明 |
| --- | --- | --- |
| [torch_npu.npu_all_gather_base_mm](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_all_gather_base_mm.md) | AllGather + Matmul | TP 切分场景下 allgather 与 matmul 融合，内部实现通信和计算流水并行。 |
| [torch_npu.npu_all_gather_quant_mm](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_all_gather_quant_mm.md) | AllGather + 量化 Matmul | TP 切分的量化场景下 allgather 与 matmul 融合，输出反量化结果。 |
| [torch_npu.npu_mm_all_reduce_base](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mm_all_reduce_base.md) | Matmul + AllReduce | TP 切分场景下 mm 与 all_reduce 融合，内部实现计算和通信流水并行。 |
| [torch_npu.npu_mm_reduce_scatter_base](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mm_reduce_scatter_base.md) | Matmul + ReduceScatter | TP 切分场景下 matmul 与 reduce_scatter 融合，内部实现计算和通信流水并行。 |
| [torch_npu.npu_quant_mm_reduce_scatter](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_mm_reduce_scatter.md) | 量化 Matmul + ReduceScatter | 融合 MatMul 与 ReduceScatter，对矩阵乘结果按卡数做归约散射通信，支持低精度量化和反量化。 |
| [torch_npu.npu_all_to_all_matmul](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_all_to_all_matmul.md) | AlltoAll + Permute + Matmul | 完成 AlltoAll 通信、Permute（保证通信后地址连续）和 Matmul 计算的融合，先通信后计算。 |
| [torch_npu.npu_all_to_all_quant_matmul](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_all_to_all_quant_matmul.md) | AlltoAll + Permute + 量化 Matmul | 完成量化的 Matmul、Permute 和 AlltoAll 通信的融合，先通信后计算，支持 K-C 动态量化模式和 mx 量化模式。 |
| [torch_npu.npu_matmul_all_to_all](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_matmul_all_to_all.md) | Matmul + Permute + AlltoAll | 完成 Matmul、Permute 和 AlltoAll 通信的融合，先计算后通信。 |
| [torch_npu.npu_quant_matmul_all_to_all](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_matmul_all_to_all.md) | 量化 Matmul + Permute + AlltoAll | 完成量化的 Matmul、Permute 和 AlltoAll 通信的融合，先通信后计算，支持 K-C 量化模式和 mx 量化模式。 |

## 四、MoE 融合算子

混合专家（Mixture of Experts）模型相关的融合算子，包括分组矩阵乘（GroupedMatmul）、专家路由/分发/汇聚（Dispatch/Combine）以及与通信、量化、激活的融合。

| API 名称 | 融合的计算 | 说明 |
| --- | --- | --- |
| [torch_npu.npu_grouped_matmul](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md) | 多个 Matmul 分组批量计算 | 对多个矩阵乘法操作进行分组计算，减少内存访问开销和计算资源浪费。 |
| [torch_npu.npu_grouped_matmul_add](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul_add.md) | 分组 Matmul + Add | 多个矩阵乘法分组计算并与累加矩阵相加，支持 K 轴分组。 |
| [torch_npu.npu_grouped_matmul_add_](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul_add_.md) | GroupedMatmul + InplaceAdd | micro-batch 训练梯度累计场景下，将 GroupedMatmul 与 InplaceAdd 融合。 |
| [torch_npu.npu_grouped_matmul_finalize_routing](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul_finalize_routing.md) | GroupedMatMul + MoeFinalizeRouting | GroupedMatMul 与 MoE Finalize Routing 的融合算子。 |
| [torch_npu.npu_grouped_matmul_swiglu_quant](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul_swiglu_quant.md) | GroupedMatmul + Dequant + SwiGLU + Quant | 融合分组矩阵乘、反量化、SwiGLu 混合激活函数、量化，用于模型性能优化。 |
| [torch_npu.npu_grouped_matmul_swiglu_quant_v2](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul_swiglu_quant_v2.md) | GroupedMatmul + Dequant + SwiGLU + Quant | 同上，v2 版本。 |
| [torch_npu.npu_ffn](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_ffn.md) | Matmul + 激活 + Matmul（可选量化） | FFN 算子提供 FFN 和 MoeFFN 的计算功能（expert_tokens 为空时为 FFN，非空时为 MoeFFN）。 |
| [torch_npu.npu_gmm_alltoallv](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_gmm_alltoallv.md) | GroupedMatmul + AlltoAllv + 共享专家 MatMul 并行 | MoE 网络中先计算后通信，路由专家 GroupedMatMul 与 AlltoAllv 融合，并与共享专家 MatMul 并行融合。 |
| [torch_npu.npu_alltoallv_gmm](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_alltoallv_gmm.md) | AlltoAllv + Permute + GroupedMatmul + 共享专家 MatMul 并行 | MoE 网络中先通信后计算，路由专家 AlltoAllv、Permute、GroupedMatMul 融合，并与共享专家 MatMul 并行融合。 |
| [torch_npu.npu_quant_gmm_alltoallv](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_gmm_alltoallv.md) | 量化 GroupedMatmul + AlltoAllv | 路由专家 GroupedMatmul 和 AlltoAllv 的量化融合，先计算后通信，支持 T-T 和 mx 量化模式。 |
| [torch_npu.npu_alltoallv_quant_gmm](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_alltoallv_quant_gmm.md) | AlltoAllv + 量化 GroupedMatmul | 路由专家 AlltoAllv 和 GroupedMatmul 的量化融合，先通信后计算。 |
| [torch_npu.npu_moe_distribute_dispatch](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_moe_distribute_dispatch.md) | 量化（可选）+ AlltoAllv + AllGatherv | 对 Token 数据先量化（可选），再进行 EP 域 alltoallv 通信，再进行 TP 域 allgatherv 通信（可选）。 |
| [torch_npu.npu_moe_distribute_dispatch_v2](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_moe_distribute_dispatch_v2.md) | 量化（可选）+ AlltoAllv + AllGatherv | 同上，v2 版本新增 comm_alg 参数。 |
| [torch_npu.npu_moe_distribute_combine](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_moe_distribute_combine.md) | ReduceScatterv + AlltoAllv + 整合（乘权重相加） | MoE 并行部署下的 token combine，先 reduce_scatterv，再 alltoallv，最后将接收数据乘权重并相加。 |
| [torch_npu.npu_moe_distribute_combine_v2](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_moe_distribute_combine_v2.md) | ReduceScatterv + AlltoAllv + 整合 | 同上，v2 版本新增特殊专家场景（零专家、拷贝专家、常量专家）支持。 |
| [torch_npu.npu_moe_distribute_combine_add_rms_norm](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_moe_distribute_combine_add_rms_norm.md) | Combine + Add + RMSNorm | 完成 moe_distribute_combine + add + rms_norm 融合，需与 npu_moe_distribute_dispatch 配套使用。 |
| [torch_npu.npu_moe_gating_top_k_softmax](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_moe_gating_top_k_softmax.md) | Softmax + TopK | MoE 门控专家路由计算，先做 Softmax，再 topk 挑选权重最高的前 K 个专家。 |
| [torch_npu.npu_moe_gating_top_k](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_moe_gating_top_k.md) | Sigmoid + 分组排序 + TopK | MoE 计算中对输入做 Sigmoid，分组排序后选取前 k 个专家。 |
| [torch_npu.npu_add_quant_gmm_](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_add_quant_gmm_.md) | 量化 GroupedMatmul + InplaceAdd | micro-batch 训练场景下将量化 GroupedMatmul 与 InplaceAdd 融合，提高网络性能。 |

## 五、矩阵乘 / 量化 / 激活融合算子

将矩阵乘（含分组、转置、批量）与量化、反量化、激活函数（GELU、SwiGLU、Rotate 等）融合，减少中间精度转换与数据搬运。

| API 名称 | 融合的计算 | 说明 |
| --- | --- | --- |
| [（beta）torch_npu.npu_swiglu](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/%EF%BC%88beta%EF%BC%89torch_npu-npu_swiglu.md) | Swish × Gated Linear | 提供 SwiGLU 融合激活函数。 |
| [torch_npu.npu_swiglu_quant](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_swiglu_quant.md) | SwiGLU + Quant | 在 swiglu 激活后添加 quant 操作，支持 int8 / int4 输出、分组量化、动态/静态量化，支持 MoE 与非 MoE 场景。 |
| [torch_npu.npu_dequant_swiglu_quant](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dequant_swiglu_quant.md) | Dequant + SwiGLU + Quant | 对张量做反量化 + swiglu + 量化操作，同时支持分组。 |
| [torch_npu.npu_swiglu_mx_quant](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_swiglu_mx_quant.md) | SwiGLU + DynamicMxQuant | 在 swiglu 激活后添加 DynamicMxQuant 操作，支持 MoE 与非 MoE 场景及多种 swiglu 计算模式。 |
| [torch_npu.npu_gelu_quant](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_gelu_quant.md) | GELU + Quant | 对张量进行 GELU 激活操作后再进行静态/动态量化。 |
| [torch_npu.npu_gelu_mul](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_gelu_mul.md) | GELU + Mul | 尾轴 32B 对齐场景下将 GELU 与 MUL 结合为复合计算以提高算子性能。 |
| [torch_npu.npu_rotate_quant](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_rotate_quant.md) | Rotate + Quant | 融合旋转（Rotate）和量化（Quant）的计算方法，用于模型性能优化。 |
| [torch_npu.npu_quant_matmul_gelu](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_matmul_gelu.md) | 量化 Matmul + GELU | 完成量化的矩阵乘和 GELU 激活函数的融合计算，支持 A8W8 和 A4W4 量化。 |
| [torch_npu.npu_quant_matmul_reduce_sum](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_matmul_reduce_sum.md) | 分组量化 Matmul + ReduceSum | 完成量化的分组矩阵计算，然后将所有组的矩阵计算结果相加后输出。 |
| [torch_npu.npu_kronecker_quant](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_kronecker_quant.md) | Matmul + Matmul + Quant | 对输入依次进行两次矩阵乘法，并将结果量化至 int4 或 float4_e2m1fn_x2 类型。 |
| [torch_npu.npu_weight_quant_batchmatmul](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_weight_quant_batchmatmul.md) | Weight 量化 + BatchMatmul | 实现矩阵乘计算中 weight 输入和输出的量化，支持 pertensor、perchannel、pergroup 多场景量化。 |
| [torch_npu.npu_dual_level_quant_matmul](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dual_level_quant_matmul.md) | QuantMatmul 二级量化 | QuantMatmul 的二级量化功能，减少精度损失。 |
| [torch_npu.npu_transpose_quant_batchmatmul](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_transpose_quant_batchmatmul.md) | Transpose + 量化 Matmul | 完成张量 x1、x2 量化的矩阵乘计算，支持按转置序列（perm_x1 / perm_x2）变更转置。 |
| [torch_npu.npu_transpose_batchmatmul](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_transpose_batchmatmul.md) | Transpose + BatchMatmul | 完成张量 input 与 weight 的矩阵乘计算。 |

## 六、损失函数融合算子

将交叉熵 / 损失计算中的 softmax、log_softmax、nll_loss 及反向梯度融合，降低计算内存，适配词汇表并行等大模型场景。

| API 名称 | 融合的计算 | 说明 |
| --- | --- | --- |
| [torch_npu.npu_cross_entropy_loss](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_cross_entropy_loss.md) | LogSoftmax + NLLLoss（可选 ZLoss） | 将原生 CrossEntropyLoss 中的 log_softmax 和 nll_loss 融合，降低计算时使用的内存。 |
| [torch_npu.npu_fused_linear_online_max_sum](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_linear_online_max_sum.md) | Matmul + 交叉熵前处理 | 词汇表并行场景下融合矩阵乘与交叉熵前处理算子，为通信前部分，需与 npu_fused_cross_entropy_loss_with_max_sum 配合使用。 |
| [torch_npu.npu_fused_cross_entropy_loss_with_max_sum](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_cross_entropy_loss_with_max_sum.md) | Softmax + 交叉熵 Loss | 词汇表并行场景下交叉熵计算模块的 Loss 与 Softmax 计算部分，多卡场景需先对 logits_max、sum_exp_logits 做 all-reduce。 |
| [torch_npu.npu_fused_linear_cross_entropy_loss_with_max_sum_backward](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_linear_cross_entropy_loss_with_max_sum_backward.md) | 交叉熵反向梯度 | 词汇表并行场景下交叉熵损失计算的梯度算子，用于计算叶子节点 input 和 weight 的梯度。 |
| [torch_npu.npu_sparse_lightning_indexer_grad_kl_loss](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_lightning_indexer_grad_kl_loss.md) | Lightning Indexer 反向 + Loss | 实现 npu_lightning_indexer 的反向功能，并融合 Loss 的计算。 |
| [torch_npu.npu_dense_lightning_indexer_grad_kl_loss](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_dense_lightning_indexer_grad_kl_loss.md) | Lightning Indexer 反向 + Loss | 实现 Lightning Indexer 组件 warmup 阶段训练的反向梯度计算，并融合 Loss 的计算。 |

## 七、其他融合算子

| API 名称 | 融合的计算 | 说明 |
| --- | --- | --- |
| [（beta）torch_npu.npu_confusion_transpose](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/%EF%BC%88beta%EF%BC%89torch_npu-npu_confusion_transpose.md) | Reshape + Transpose | 融合 reshape 和 transpose 运算。 |
| [torch_npu.npu_fused_matmul](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_matmul.md) | 矩阵乘 + 通用向量计算 | 矩阵乘与通用向量计算融合，减少数据搬运提升性能。 |
| [torch_npu.npu_fused_causal_conv1d](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_causal_conv1d.md) | 因果 Conv1d + Cache 更新 + 残差连接 | 因果一维卷积使用缓存对各序列头部 padding，卷积完成后更新缓存，并在输出基础上叠加原始输入实现残差连接，支持 APC、MTP 等特性。 |
| [（beta）torch_npu.npu_dropout_with_add_softmax](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/%EF%BC%88beta%EF%BC%89torch_npu-npu_dropout_with_add_softmax.md) | axpy（Add）+ Softmax + Dropout | 实现 axpy_v2、softmax_v2、drop_out_domask_v3 功能。 |

---

## 相关说明

- 本清单中的接口均为 TorchNPU 对外公开接口，具体参数、约束与调用示例请跳转对应接口文档查看（本 Skill 在 `references/torch_npu/` 内置同一快照的本地副本，按 `torch_npu-<接口名>.md` 索引）。
- 接口的 beta 标记、产品支持情况以各接口文档页面为准。
- 若需要查看完整接口列表，可参考 [torch_npu接口列表](https://gitcode.com/Ascend/op-plugin/blob/b80a2fdfce8b32da11ec3647e3946417ee39c463/docs/zh/custom_APIs/torch_npu/torch_npu_list.md)。
