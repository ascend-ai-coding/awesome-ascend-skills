# torch_npu接口列表

本章节包含常用自定义接口，包括创建tensor及计算类操作。

共计 **144** 个接口（正式接口 79 个，Beta 接口 65 个）。

## 正式接口

| API名称 | 说明 |
|---------|------|
| [torch_npu.empty_with_swapped_memory](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-empty_with_swapped_memory.md) | 申请一个device信息为NPU且实际内存在host侧的`Tensor`。 |
| [torch_npu.erase_stream](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-erase_stream.md) | `Tensor`通过record_stream在内存池上添加的已被stream使用的标记后，可以通过该接口移除该标记。 |
| [torch.npu.get_device_limit](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-get_device_limit.md) | 通过该接口，获取指定Device上的Device资源限制。 |
| [torch.npu.get_stream_limit](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-get_stream_limit.md) | 通过该接口，获取指定Stream的Device资源限制。 |
| [torch_npu.matmul_checksum](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-matmul_checksum.md) | 提供基于原生torch.matmul和Tensor.matmul接口的AIcore错误硬件故障检测接口。内部执行矩阵计算结果校验过程，校验误差和实时计算的校验门限进行对比，判断校验误差是否超越门限，若超越则认为发生了AIcore错误。 |
| [torch_npu.npu_advance_step_flashattn](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_advance_step_flashattn.md) | 在NPU上实现vLLM库中advance_step_flashattn的功能，在每个生成步骤中原地更新`input_tokens`，`input_positions`，`seq_lens`和`slot_mapping`。 |
| [torch_npu.npu_all_gather_base_mm](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_all_gather_base_mm.md) | TP切分场景下，融合`allgather`和`matmul`，实现通信和计算流水并行。 |
| [torch_npu.npu_alltoallv_gmm](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_alltoallv_gmm.md) | MoE（Mixture of Experts，混合专家模型）网络中，完成路由专家AlltoAllv、Permute、GroupedMatMul融合并实现与共享专家MatMul并行融合，先通信后计算。 |
| [torch_npu.npu_anti_quant](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_anti_quant.md) | 对张量`x`进行反量化操作，即将整数恢复为浮点数。 |
| [torch_npu.npu_convert_weight_to_int4pack](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_convert_weight_to_int4pack.md) | 将`int32`类型的输入`tensor`打包为`int4`存放，每8个`int4`数据通过一个`int32`数据承载，并进行交叠排放。 |
| [torch_npu.npu_cross_entropy_loss](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_cross_entropy_loss.md) | 计算输入`input`和标签`target`之间的交叉熵损失。此API将原生`CrossEntropyLoss`中的log_softmax和nll_loss融合，降低计算时使用的内存。 |
| [torch_npu.npu_dequant_swiglu_quant](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_dequant_swiglu_quant.md) | 对张量`x`做dequant反量化+swiglu激活+quant量化操作，同时支持分组。 |
| [torch_npu.npu_dynamic_block_quant](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_dynamic_block_quant.md) | 对输入张量，通过给定的`row_block_size`和`col_block_size`将输入划分成多个数据块，以数据块为基本粒度进行量化。在每个块中，先计算出当前块对应的量化参数`scale`，并根据`scale`对输入进行量化。输... |
| [torch_npu.npu_dynamic_quant](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_dynamic_quant.md) | 对输入的张量进行pertoken对称动态量化。 |
| [torch_npu.npu_dynamic_quant_asymmetric](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_dynamic_quant_asymmetric.md) | 对输入的张量进行动态非对称量化。支持pertoken、pertensor和MoE（Mixture of Experts，混合专家模型）场景。 |
| [torch_npu.npu_fast_gelu](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_fast_gelu.md) | 快速高斯误差线性单元激活函数（Fast Gaussian Error Linear Units activation function），对输入的每个元素计算`FastGelu`的前向结果。 |
| [torch_npu.npu_ffn](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_ffn.md) | FFN算子提供MoeFFN和FFN的计算功能。在没有专家分组（`expert_tokens`为空）时是FFN，有专家分组（`expert_tokens`不为空）时是MoeFFN。 |
| [torch_npu.npu_fused_infer_attention_score](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_fused_infer_attention_score.md) | 适配增量和全量推理场景的FlashAttention算子，既可以支持全量计算场景（PromptFlashAttention），也可支持增量计算场景（IncreFlashAttention）。当`query`矩阵的S为1，进入Incre... |
| [torch_npu.npu_fused_infer_attention_score_v2](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_fused_infer_attention_score_v2.md) | 适配增量&全量推理场景的FlashAttention算子，既可以支持全量计算场景（PromptFlashAttention），也可支持增量计算场景（IncreFlashAttention）。当不涉及system prefix、左pad... |
| [torch_npu.npu_fusion_attention](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_fusion_attention.md) | 实现“Transformer Attention Score”的融合计算，实现的计算公式如下： |
| [torch_npu.npu_gather_sparse_index](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_gather_sparse_index.md) | — |
| [torch_npu.npu_gelu](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_gelu.md) | 计算高斯误差线性单元的激活函数。 |
| [torch_npu.npu_gelu_mul](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_gelu_mul.md) | 当输入Tensor的尾轴为32B对齐场景时，使用该API可对输入Tensor进行GELU与MUL结合的复合计算操作以提高算子性能，若尾轴为非32B对齐场景时，建议走小算子拼接逻辑，即按照下述公式分步拼接计算。 |
| [torch_npu.npu_gmm_alltoallv](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_gmm_alltoallv.md) | MoE网络中，完成路由专家GroupedMatMul、AlltoAllv融合并实现与共享专家MatMul并行融合，先计算后通信。 |
| [torch_npu.npu_group_norm_silu](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_group_norm_silu.md) | 计算输入张量`input`按组归一化的结果，包括张量out、均值meanOut、标准差的倒数rstdOut以及silu的输出。 |
| [torch_npu.npu_group_norm_swish](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_group_norm_swish.md) | 计算输入`input`的组归一化结果`y`，均值`mean`，标准差的倒数`rstd`，以及swish的输出。 |
| [torch_npu.npu_group_quant](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_group_quant.md) | 对输入的张量进行分组量化操作。 |
| [torch_npu.npu_grouped_matmul](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_grouped_matmul.md) | `npu_grouped_matmul`是一种对多个矩阵乘法（matmul）操作进行分组计算的高效方法。该API实现了对多个矩阵乘法操作的批量处理，通过将具有相同形状或相似形状的矩阵乘法操作组合在一起，减少内存访问开销和计算资源的浪费... |
| [torch_npu.npu_grouped_matmul_swiglu_quant_v2](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_grouped_matmul_swiglu_quant_v2.md) | `npu_grouped_matmul_swiglu_quant_v2`是一种融合分组矩阵乘法（GroupedMatmul）、SwiGLU混合激活函数、量化（quant）的计算方法。该方法适用于需要对矩阵乘法结果进行SwiGLU激活函... |
| [torch_npu.npu_incre_flash_attention](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_incre_flash_attention.md) | 增量FA实现。 |
| [torch_npu.npu_interleave_rope](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_interleave_rope.md) | 针对单输入`x`进行旋转位置编码。 |
| [torch_npu-npu_kv_quant_sparse_flash_attention](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_kv_quant_sparse_flash_attention.md) | `kv_quant_sparse_flash_attention`在`sparse_flash_attention`的基础上支持了[Per-Token-Head-Tile-128量化]输入。随着大模型上下文长度的增加，Sparse A... |
| [torch_npu.npu_kv_rmsnorm_rope_cache](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_kv_rmsnorm_rope_cache.md) | 融合了MLA（Multi-head Latent Attention）结构中RMSNorm归一化计算与RoPE（Rotary Position Embedding）位置编码以及更新KVCache的ScatterUpdate操作。 |
| [torch_npu-npu_lightning_indexer](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_lightning_indexer.md) | `lightning_indexer`基于一系列操作得到每一个token对应的Top-$k$个位置。 |
| [torch_npu.npu_mla_prolog](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_mla_prolog.md) | 推理场景下，Multi-Head Latent Attention（MLA）前处理的计算。 |
| [torch_npu.npu_mla_prolog_v2](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_mla_prolog_v2.md) | 推理场景下，Multi-Head Latent Attention（MLA）前处理的计算。主要计算过程分为五路； |
| [torch_npu.npu_mm_all_reduce_base](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_mm_all_reduce_base.md) | TP切分场景下，实现mm和all_reduce的融合，融合算子内部实现计算和通信流水并行。 |
| [torch_npu.npu_mm_reduce_scatter_base](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_mm_reduce_scatter_base.md) | TP切分场景下，实现matmul和reduce_scatter的融合，融合算子内部实现计算和通信流水并行。支持perchannel，pertoken量化。 |
| [torch_npu.npu_moe_compute_expert_tokens](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_moe_compute_expert_tokens.md) | MoE（Mixture of Experts，混合专家模型）计算中，通过二分查找的方式查找每个专家处理的最后一行的位置。 |
| [torch_npu.npu_moe_distribute_combine](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_moe_distribute_combine.md) | 先进行reduce_scatterv通信，再进行alltoallv通信，最后将接收的数据整合（乘权重再相加）。需与torch_npu.npu_moe_distribute_dispatch配套使用，相当于按npu_moe_distri... |
| [torch_npu.npu_moe_distribute_combine_add_rms_norm](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_moe_distribute_combine_add_rms_norm.md) | 需与torch_npu.npu_moe_distribute_dispatch_v2配套使用，相当于按`npu_moe_distribute_dispatch_v2`算子收集数据的路径原路返回后对数据进行`add_rms_norm`操作。 |
| [torch_npu.npu_moe_distribute_combine_v2](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_moe_distribute_combine_v2.md) | 需与torch_npu.npu_moe_distribute_dispatch_v2配套使用，相当于按npu_moe_distribute_dispatch_v2算子收集数据的路径原路返回。 |
| [torch_npu.npu_moe_distribute_dispatch](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_moe_distribute_dispatch.md) | 需与torch_npu.npu_moe_distribute_combine配套使用，完成MoE的并行部署下的token dispatch与combine。对token数据先进行quant量化（可选），再进行EP（Expert Par... |
| [torch_npu.npu_moe_distribute_dispatch_v2](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_moe_distribute_dispatch_v2.md) | 需与torch_npu.npu_moe_distribute_combine_v2或torch_npu.npu_moe_distribute_combine_add_rms_norm配套使用，完成MoE的并行部署下的token dis... |
| [torch_npu.npu_moe_finalize_routing](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_moe_finalize_routing.md) | 在MoE计算的最后，合并MoE FFN(Feedforward Neural Network)的输出结果。 |
| [torch_npu.npu_moe_gating_top_k](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_moe_gating_top_k.md) | MoE计算中，对输入x做Sigmoid/SoftMax计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。 |
| [torch_npu.npu_moe_gating_top_k_softmax](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_moe_gating_top_k_softmax.md) | MoE计算中，对输入`x`做Softmax计算，再做topk操作。 |
| [torch_npu.npu_moe_init_routing](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_moe_init_routing.md) | MoE的routing计算，根据torch_npu.npu_moe_gating_top_k_softmax的计算结果做routing处理。 |
| [torch_npu.npu_moe_init_routing_v2](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_moe_init_routing_v2.md) | MoE（Mixture of Experts）的routing计算，根据torch_npu.npu_moe_gating_top_k_softmax的计算结果做routing处理，支持不量化和动态量化模式。 |
| [torch_npu.npu_moe_re_routing](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_moe_re_routing.md) | MoE网络中，进行AlltoAll操作从其他卡上拿到需要算的token后，将token按照专家顺序重新排列。 |
| [torch_npu.npu_moe_update_expert](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_moe_update_expert.md) | 本API支持负载均衡和专家剪枝功能。经过映射后的专家表和mask可传入Moe层进行数据分发和处理。 |
| [torch_npu.npu_prefetch](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_prefetch.md) | 提供网络`weight`预取功能，将需要预取的权重搬到L2 Cache中。尤其在做较大`Tensor`的MatMul计算且需要搬移到L2 Cache的操作时，可通过该接口提前预取权重，适当提高模型性能，具体效果取决于用户采用的并行方式... |
| [torch_npu.npu_prompt_flash_attention](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_prompt_flash_attention.md) | 全量FA实现。 |
| [torch_npu.npu_quant_lightning_indexer](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_quant_lightning_indexer.md) | QuantLightningIndexer是推理场景下，SparseFlashAttention（SFA）前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8，获取最大收益。 |
| [torch_npu.npu_quant_matmul](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_quant_matmul.md) | 完成量化的矩阵乘计算，最小支持输入维度为2维，最大支持输入维度为6维。 |
| [torch_npu.npu_quant_matmul_reduce_sum](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_quant_matmul_reduce_sum.md) | 完成量化的分组矩阵计算，然后所有组的矩阵计算结果相加后输出。 |
| [torch_npu.npu_quant_scatter](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_quant_scatter.md) | 先将`updates`进行量化，然后将`updates`中的值按指定的轴`axis`和索引`indices`更新`input`中的值，并将结果保存到输出tensor，`input`本身的数据不变。 |
| [torch_npu.npu_quant_scatter_](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_quant_scatter_.md) | 先将`updates`进行量化，然后将`updates`中的值按指定的轴`axis`和索引`indices`更新`input`中的值，`input`中的数据被改变。 |
| [torch_npu.npu_quantize](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_quantize.md) | 对输入的张量进行量化处理。 |
| [torch_npu.npu_recurrent_gated_delta_rule](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_recurrent_gated_delta_rule.md) | 该接口实现了变步长Recurrent Gated Delta Rule（RGDR）的计算逻辑，是Transformer线性注意力机制的关键算子之一。通过引入门控机制与递归更新策略，RGDR能够在保持线性时间复杂度的同时，有效捕捉长距离... |
| [torch_npu.npu_rotary_mul](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_rotary_mul.md) | 实现Rotary Position Embedding (RoPE) 旋转位置编码，通过对输入特征进行二维平面旋转注入位置信息。 |
| [torch_npu.npu_scaled_masked_softmax](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_scaled_masked_softmax.md) | 计算输入张量`x`缩放并按照`mask`遮蔽后的`Softmax`结果。 |
| [torch_npu.npu_scatter_nd_update](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_scatter_nd_update.md) | 将`updates`中的值按指定的索引`indices`更新`input`中的值，并将结果保存到输出tensor，`input`本身的数据不变。 |
| [torch_npu.npu_scatter_nd_update_](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_scatter_nd_update_.md) | 将`updates`中的值按指定的索引`indices`更新`input`中的值，并将结果保存到输出tensor，`input`中的数据被改变。 |
| [torch_npu.npu_scatter_pa_kv_cache](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_scatter_pa_kv_cache.md) | 更新KVCache中指定位置的`key`和`value`。 |
| [torch_npu.npu_sim_exponential_](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_sim_exponential_.md) | 根据参数`lambd`生成指数分布随机数，并原地填充至输入张量`input`。 |
| [torch_npu.npu_sparse_flash_attention](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_sparse_flash_attention.md) | sparse_flash_attention（SFA）是针对大序列长度推理场景的高效注意力计算模块，该模块通过“只计算关键部分”大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。 |
| [torch_npu.npu_sparse_lightning_indexer_grad_kl_loss](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_sparse_lightning_indexer_grad_kl_loss.md) | 该接口实现了npu_lightning_indexer的反向功能，并融合了Loss的计算。npu_lightning_indexer用于筛选Attention的`query`与`key`间最高内在联系的Top-k项，存放在`spars... |
| [torch_npu.npu_swiglu_quant](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_swiglu_quant.md) | 在swiglu激活函数后添加quant操作，实现输入`x`的`SwiGluQuant`计算，支持`int8`或`int4`量化输出，支持MoE场景和非MoE场景（`group_index`为空），支持分组量化，支持动态/静态量化。 |
| [torch_npu.npu_top_k_top_p](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_top_k_top_p.md) | 对原始输入`logits`进行`top-k`和`top-p`采样过滤。 |
| [torch_npu.npu_top_k_top_p_sample](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_top_k_top_p_sample.md) | 根据输入词频`logits`、`top_k`/`top_p`采样参数、随机采样权重分布`q`，进行topK-topP-Sample采样计算，输出每个batch的最大词频`logits_select_idx`，以及topK-topP采样... |
| [torch_npu.npu_trans_quant_param](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_trans_quant_param.md) | 完成量化计算参数`scale`数据类型的转换，将`float32`数据按照bit位存储进一个`int64`数据里。 |
| [torch_npu.npu_transpose_batchmatmul](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_transpose_batchmatmul.md) | 完成张量`input`与张量`weight`的矩阵乘计算。仅支持三维的Tensor传入。Tensor支持转置，转置序列根据传入的数列进行变更。`perm_x1`代表张量input的转置序列，`perm_x2`代表张量weight的转置... |
| [torch_npu.npu_weight_quant_batchmatmul](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-npu_weight_quant_batchmatmul.md) | 该接口用于实现矩阵乘计算中`weight`输入和输出的量化操作，支持pertensor、perchannel、pergroup多场景量化。 |
| [torch.npu.reset_stream_limit](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-reset_stream_limit.md) | 调用`torch.npu.set_stream_limit`接口设置指定Stream的Device资源限制后，可调用本接口重置指定Stream的Device资源限制，恢复默认配置，此时可通过`torch.npu.get_stream_... |
| [torch_npu.scatter_update](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-scatter_update.md) | 将tensor updates中的值按指定的轴axis和索引indices更新tensor data中的值，并将结果保存到输出tensor，data本身的数据不变。 |
| [torch_npu.scatter_update_](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-scatter_update_.md) | 将tensor updates中的值按指定的轴axis和索引indices更新tensor data中的值，并将结果保存到输出tensor，data本身的数据被改变。 |
| [torch.npu.set_device_limit](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-set_device_limit.md) | 设置一个进程上指定device，执行算子时所使用的cube和vector核数。 |
| [torch.npu.set_stream_limit](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/torch_npu-set_stream_limit.md) | 设置指定Stream的Device资源限制。 |

## Beta 接口

| API名称 | 说明 |
|---------|------|
| [（beta）torch_npu._npu_dropout](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-_npu_dropout.md) | 不使用种子（seed）进行dropout结果计数。 |
| [（beta）torch_npu.copy_memory_](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-copy_memory_.md) | 从src拷贝元素到self张量，并原地返回self张量。 |
| [（beta）torch_npu.empty_with_format](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-empty_with_format.md) | 返回一个填充未初始化数据的张量。 |
| [（beta）torch_npu.fast_gelu](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-fast_gelu.md) | 快速高斯误差线性单元激活函数（Fast Gaussian Error Linear Units activation function），对输入的每个元素计算FastGelu。支持FakeTensor模式。 |
| [（beta）torch_npu.npu_alloc_float_status](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_alloc_float_status.md) | 申请一个专门用于存储浮点运算状态标志的Tensor。该Tensor用于后续记录计算过程中的溢出状态。 |
| [（beta）torch_npu.npu_anchor_response_flags](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_anchor_response_flags.md) | 在单个特征图中生成锚点的响应标志，即标识哪些锚点需要参与训练或推理。 |
| [（beta）torch_npu.npu_apply_adam](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_apply_adam.md) | 获取adam优化器的计算结果。 |
| [（beta）torch_npu.npu_batch_nms](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_batch_nms.md) | 以批量处理方式对每个类别的检测框进行非极大值抑制（Non-Maximum Suppression，NMS），从而去除冗余检测框，输出保留下来的检测框及其对应的类别和得分。 |
| [（beta）torch_npu.npu_bert_apply_adam](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_bert_apply_adam.md) | 针对bert模型，获取adam优化器的计算结果。 |
| [（beta）torch_npu.npu_bmmV2](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_bmmV2.md) | 将矩阵“a”乘以矩阵“b”，生成“a*b”。支持FakeTensor模式。 |
| [（beta）torch_npu.npu_bounding_box_decode](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_bounding_box_decode.md) | 根据rois和deltas生成标注框。自定义Faster R-CNN算子。 |
| [（beta）torch_npu.npu_bounding_box_encode](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_bounding_box_encode.md) | 计算标注框和ground truth真值框之间的坐标变化。自定义Faster R-CNN算子。 |
| [（beta）torch_npu.npu_broadcast](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_broadcast.md) | 返回self张量的新视图，其单维度扩展，结果连续。张量也可以扩展更多维度，新的维度添加在最前面。 |
| [（beta）torch_npu.npu_ciou](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_ciou.md) | 应用基于NPU的CIoU操作。在DIoU的基础上增加了penalty term，并propose CIoU。 |
| [（beta）torch_npu.npu_clear_float_status](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_clear_float_status.md) | 清除溢出检测相关标志位。 |
| [（beta）torch_npu.npu_confusion_transpose](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_confusion_transpose.md) | 同时执行reshape与transpose运算。 |
| [（beta）torch_npu.npu_conv2d](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_conv2d.md) | 在由多个输入平面组成的输入图像上应用一个2D卷积。 |
| [（beta）torch_npu.npu_conv3d](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_conv3d.md) | 在由多个输入平面组成的输入图像上应用一个3D卷积。 |
| [（beta）torch_npu.npu_conv_transpose2d](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_conv_transpose2d.md) | 在由多个输入平面组成的输入图像上应用一个2D转置卷积算子，有时这个过程也被称为“反卷积”。 |
| [（beta）torch_npu.npu_convolution](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_convolution.md) | 在由多个输入平面组成的输入图像上应用一个2D或3D卷积。 |
| [（beta）torch_npu.npu_convolution_transpose](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_convolution_transpose.md) | 在由多个输入平面组成的输入图像上应用一个2D或3D转置卷积算子，有时这个过程也被称为“反卷积”。 |
| [（beta）torch_npu.npu_deformable_conv2d](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_deformable_conv2d.md) | 使用预期输入计算变形卷积（deformable convolution）的输出。 |
| [（beta）torch_npu.npu_diou](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_diou.md) | 应用基于NPU的DIoU操作。考虑到目标之间距离，以及距离和范围的重叠率，不同目标或边界需趋于稳定。 |
| [（beta）torch_npu.npu_dropout_with_add_softmax](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_dropout_with_add_softmax.md) | 实现axpy_v2、softmax_v2、drop_out_domask_v3功能。即： |
| [（beta）torch_npu.npu_dtype_cast](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_dtype_cast.md) | 执行张量数据类型（dtype）转换。支持FakeTensor模式。 |
| [（beta）torch_npu.npu_format_cast](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_format_cast.md) | 修改`input`的数据格式为目标格式，修改后的新张量不会替换原有张量。 |
| [（beta）torch_npu.npu_format_cast_](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_format_cast_.md) | 原地修改`input`的数据格式为目标格式。 |
| [（beta）torch_npu.npu_fused_attention_score](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_fused_attention_score.md) | 实现“Transformer attention score”的融合计算逻辑，主要将matmul、transpose、add、softmax、dropout、batchmatmul、permute等计算进行了融合。 |
| [（beta）torch_npu.npu_get_float_status](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_get_float_status.md) | 获取溢出检测结果。 |
| [（beta）torch_npu.npu_giou](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_giou.md) | 首先计算两个框的最小封闭面积和IoU，然后计算封闭区域中不属于两个框的封闭面积的比例，最后从IoU中减去这个比例，得到GIoU。 |
| [（beta）torch_npu.npu_grid_assign_positive](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_grid_assign_positive.md) | 执行position-sensitive的候选区域池化梯度计算。 |
| [（beta）torch_npu.npu_gru](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_gru.md) | 计算DynamicGRUV2。 |
| [（beta）torch_npu.npu_indexing](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_indexing.md) | 以begin为起始索引，end为结束索引，strides为步长，对输入张量进行切片。 |
| [（beta）torch_npu.npu_iou](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_iou.md) | 根据ground-truth和预测区域计算交并比（IoU）或前景交叉比（IoF）。 |
| [（beta）torch_npu.npu_layer_norm_eval](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_layer_norm_eval.md) | 对层归一化结果进行计算。与`torch.nn.functional.layer_norm`相同，优化NPU设备实现。 |
| [（beta）torch_npu.npu_linear](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_linear.md) | 将矩阵“a”乘以矩阵“b”，生成“a*b”。 |
| [（beta）torch_npu.npu_lstm](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_lstm.md) | 计算DynamicRNN。 |
| [（beta）torch_npu.npu_max](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_max.md) | 使用dim对最大结果进行计算。类似于torch.max，优化NPU设备实现。 |
| [（beta）torch_npu.npu_min](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_min.md) | 使用dim对最小结果进行计算。类似于`torch.min`，优化NPU设备实现。 |
| [（beta）torch_npu.npu_mish](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_mish.md) | 按元素计算self的双曲正切。 |
| [（beta）torch_npu.npu_mla_prolog_v3](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_mla_prolog_v3.md) | 推理场景下Multi-Head Latent Attention前处理的计算操作。该算子实现四条并行的计算路径： |
| [（beta）torch_npu.npu_multi_head_attention](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_multi_head_attention.md) | 实现Transformer模块中的MultiHeadAttention计算逻辑。 |
| [（beta）torch_npu.npu_nms_rotated](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_nms_rotated.md) | 按分数降序选择旋转标注框的子集。 |
| [（beta）torch_npu.npu_nms_v4](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_nms_v4.md) | 按分数降序选择标注框的子集。 |
| [（beta）torch_npu.npu_nms_with_mask](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_nms_with_mask.md) | 生成值0或1，用于nms算子确定有效位。 |
| [（beta）torch_npu.npu_one_hot](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_one_hot.md) | 返回一个one-hot张量。input中index表示的位置采用on_value值，而其他所有位置采用off_value的值。 |
| [（beta）torch_npu.npu_pad](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_pad.md) | 填充张量。 |
| [（beta）torch_npu.npu_ps_roi_pooling](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_ps_roi_pooling.md) | 执行Position Sensitive ROI Pooling。 |
| [（beta）torch_npu.npu_ptiou](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_ptiou.md) | 根据ground-truth和预测区域计算交并比（IoU）或前景交叉比（IoF）。 |
| [（beta）torch_npu.npu_random_choice_with_mask](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_random_choice_with_mask.md) | 获取非零元素的index，混洗后输出。 |
| [（beta）torch_npu.npu_reshape](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_reshape.md) | reshape张量。仅更改张量shape，其数据不变。 |
| [（beta）torch_npu.npu_rms_norm](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_rms_norm.md) | RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。 |
| [（beta）torch_npu.npu_roi_align](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_roi_align.md) | 从特征图中获取ROI特征矩阵。自定义Faster R-CNN算子。 |
| [（beta）torch_npu.npu_rotated_iou](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_rotated_iou.md) | 计算旋转框的IoU。 |
| [（beta）torch_npu.npu_rotated_overlaps](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_rotated_overlaps.md) | 计算旋转框的重叠面积。 |
| [（beta）torch_npu.npu_sign_bits_pack](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_sign_bits_pack.md) | 将`float`类型的输入打包为`uint8`类型。每8个浮点数打包为一个`uint8`数值，-1.0编码为二进制位0，1.0编码为二进制位1，并按小端序进行打包。 |
| [（beta）torch_npu.npu_sign_bits_unpack](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_sign_bits_unpack.md) | 将`uint8`类型的输入拆包为`float`类型。将`uint8`数值中的8个二进制位解码为8个浮点数，0解码为-1.0，1解码为1.0，并以小端序进行返回。 |
| [（beta）torch_npu.npu_silu](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_silu.md) | 计算self的Swish。Swish是一种激活函数，计算公式为' x * sigmoid(x) '。 |
| [（beta）torch_npu.npu_slice](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_slice.md) | 从张量中提取切片。 |
| [（beta）torch_npu.npu_softmax_cross_entropy_with_logits](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_softmax_cross_entropy_with_logits.md) | 计算softmax的交叉熵损失。 |
| [（beta）torch_npu.npu_sort_v2](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_sort_v2.md) | 沿给定维度，对输入张量元素进行升序排序（不返回索引）。若dim未设置，则选择输入的最后一个维度。如果descending为True，则元素将按值降序排序。 |
| [（beta）torch_npu.npu_swiglu](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_swiglu.md) | Swish门控线性单元激活函数，实现张量`input`的swiglu计算。 |
| [（beta）torch_npu.npu_transpose](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_transpose.md) | 返回原始张量视图，其维度已permute，结果连续。支持FakeTensor模式。 |
| [（beta）torch_npu.npu_yolo_boxes_encode](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-npu_yolo_boxes_encode.md) | 根据YOLO的锚点框（anchor box）和真值框（ground-truth box）生成标注框。自定义mmdetection算子。 |
| [（beta）torch_npu.one_](https://gitcode.com/Ascend/op-plugin/blob/7.3.0/docs/context/（beta）torch_npu-one_.md) | 用1填充self张量。 |
