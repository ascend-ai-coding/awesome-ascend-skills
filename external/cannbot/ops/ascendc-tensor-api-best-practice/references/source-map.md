# 本地 CANN 官方源码知识库

本文档只管理“从哪里找可复用的算子实现与工程参考”。Tensor API 公开 API、ABI、Layout、
同步和数据类型以 asc-devkit 手册与头文件为权威源；asc-devkit `07_tensor_api/` 官方样例是
Tensor API 调用形态的第一参考；本表中的源码仓只提供算法结构、tiling、边界处理和组件架构参考。

## 1. 使用规则

1. 先用 `ensure_docs.py` 确定 asc-devkit 事实源，再用 `ensure_sources.py` 检查本地源码仓。
   两个脚本均在 skill 根目录（包含 `SKILL.md` 的目录）下执行；后者只读，不 clone、不 fetch、
   不修改任何源码仓。仓、URL 和 required_paths 的唯一机器事实源是
   `scripts/sources.yaml`；本文只保留使用规则和能力映射。
2. 判定优先级：asc-devkit 手册/头文件 > Tensor API 官方样例 > 本地源码仓的算法/tiling/边界参考。
3. 引用前先做路线审计：
   - include `tensor_api/tensor.h` 或使用 `asc::te::make_tensor/make_copy/make_mmad`：
     可作为 Tensor API custom 直接参考。
   - include `blaze/` 或 block/kernel 组件头：只作为组件架构、调度和性能模式参考；
     Matmul 类 Blaze 场景路由到 `ascendc-blaze-best-practice`。
   - 传统 AscendC、MicroAPI、AICPU 或框架封装实现：只提取算法结构、tiling、边界处理和测试语义。
4. 不得因参考实现存在就推断 Tensor API 支持；最终以当前 asc-devkit 手册/头文件和最小编译探针为准。
5. 引用具体实现时记录仓、commit、路径和关键结论。
6. 路径中的 build、build-93、二进制和日志产物不作为事实源。

## 2. 源码仓定位

| 仓 | 默认根 | 主要价值 | 使用方式 |
|---|---|---|---|
| ops-tensor | `~/ops-tensor` | GEMM、GroupedMatmul、量化 epilogue、Tensor API/Blaze 组件架构文档 | 组件架构、调度和性能模式参考；具体样例必须先做路线审计 |
| ops-nn | `~/ops-nn` | NN 基础算子：conv、norm、quant、activation、index、hash、sort/select | 算法结构、tiling 和边界处理参考；先看 arch35 kernel |
| ops-transformer | `~/ops-transformer` | GMM、MoE、MHC、attention cache、动态 shape 与多输出融合 | 以复杂融合算法、tiling、多输出合同为主；注意 API 路线差异 |
| cann-samples | `~/cann-samples` | Tensor API/SIMT/CV 通用样例与性能 story | 高优先级；最易读，适合作为从官方样例到实现的起点 |

## 3. 能力族与事实源

| 能力族 | 首选源码仓 | 代表路径 | 复用等级 |
|---|---|---|---|
| Cube 收缩、量化、epilogue | ops-tensor, cann-samples | `ops-tensor/docs/API/gemm/block`、`ops-tensor/docs/API/epilogue/block`、`cann-samples/Samples/2_Performance/matmul_story` | 架构 / 调度 / 性能模式参考 |
| 卷积与空间采样 | ops-nn | `ops-nn/conv/conv2d_v2`、`ops-nn/conv/conv3d_backprop_filter_v2`、`ops-nn/conv/depthwise_conv2d_backprop_input`、`ops-nn/pooling/dilation2_d` | 算法/tiling 参考 |
| 行归约、归一化、动态量化 | ops-nn, cann-samples | `ops-nn/norm/rms_norm_dynamic_quant`、`ops-nn/quant/dynamic_quant`、`ops-nn/quant/swiglu_mx_quant`、`cann-samples/Samples/2_Performance/rms_norm_quant_story` | 算法参考 + story 参考 |
| 索引、scatter、hash | ops-nn, ops-transformer, cann-samples | `ops-nn/hash/embedding_hash_table_lookup_or_insert`、`ops-nn/index/embedding_bag`、`ops-nn/index/scatter_update`、`ops-transformer/attention/scatter_pa_kv_cache`、`cann-samples/Samples/2_Performance/simt_scatter_story` | 算法参考 + SIMT 样例 |
| TopK、排序、去重、compaction | ops-nn, ops-transformer, cann-samples | `ops-nn/index/unique`、`ops-nn/index/top_k_top_p_sample`、`ops-transformer/moe/moe_gating_top_k_softmax`、`cann-samples/Samples/2_Performance/moe_init_routing_story` | 算法/合同参考 |
| 任意 ND 布局变换 | cann-samples, ops-nn | `cann-samples/Samples/0_Introduction/03_simt_api/01_gather/general_gather`、`cann-samples/Samples/1_Features/memory_optimization/transpose_batch_matmul`、`ops-nn/matmul/transpose_batch_mat_mul` | 坐标映射/Layout 参考 |
| AIC/AIV 融合 | cann-samples, ops-tensor | `cann-samples/Samples/1_Features/hardware_features/cv_datapath`、`ops-tensor/docs/API/epilogue` | 数据通路与同步形态参考 |

## 4. L3 算子映射

| 算子 | 直接/近邻参考 | 需要组合的能力 |
|---|---|---|
| grouped_matmul | `ops-transformer/gmm/grouped_matmul`、`ops-tensor/examples/grouped_matmul`、`cann-samples/Samples/2_Performance/grouped_matmul_story` | cumsum 分组、expert weight、输出列表 |
| weight_quant_batch_matmul | `ops-nn/matmul/weight_quant_batch_matmul`、`ops-tensor/examples/weight_quant_batch_matmul` | antiquant scale/offset、bias、batch |
| quant_matmul | `ops-nn/matmul/quant_matmul`、`cann-samples/Samples/2_Performance/matmul_story/matmul_recipes/examples/quant_matmul_int8`、`ops-tensor/examples/quant_matmul_activation_quant` | int8 Mmad、Fixpipe 量化、可选 epilogue |
| conv_2d | `ops-nn/conv/conv2d_v2` | stride/pad/dilation、im2col 或直接窗口 |
| conv_3d_backprop_filter | `ops-nn/conv/conv3d_backprop_filter_v2` | 5D 坐标映射、filter 梯度归约 |
| depthwise_conv_2d | `ops-nn/conv/depthwise_conv2d_backprop_input` | 通道独立、边界 mask |
| dilation_2d | `ops-nn/pooling/dilation2_d` | 窗口 max、dilation/stride/pad/ceil |
| add_rms_norm_dynamic_quant | `ops-nn/norm/rms_norm_dynamic_quant`、`cann-samples/Samples/2_Performance/rms_norm_quant_story` | row sum、RMSNorm、动态量化、多输出 |
| dequant_swiglu_quant | `ops-nn/quant/swiglu_mx_quant` | 反量化、split、SiLU、row amax |
| categorical_embedding_bag | `ops-nn/index/embedding_bag` | gather、sum/mean、padding_idx、bias |
| embedding_hash_table_lookup_or_insert | `ops-nn/hash/embedding_hash_table_lookup_or_insert` | slot hash、first-insert-wins、flags |
| kv_cache_append | `ops-transformer/attention/scatter_pa_kv_cache` | cache 复制、slot scatter |
| mhc_post_grad | `ops-transformer/mhc/mhc_post_backward` | 梯度公式、混合 dtype、多输出 |
| mhc_sinkhorn | `ops-transformer/mhc/mhc_sinkhorn` | row softmax、交替归一化、eps |
| moe_gating_top_k_softmax | `ops-transformer/moe/moe_gating_top_k_softmax` | softmax、top-k、row_idx、finished |
| moe_re_routing | `ops-transformer/moe/moe_re_routing` | prefix sum、token permutation、expert count |
| unique | `ops-nn/index/unique` | 去重、first-occurrence、inverse map |
| scatter_block_update | `ops-nn/index/scatter_update`、`ops-nn/index/scatter_nd_update` | 输入复制、block scatter |
| top_k | `ops-nn/index/top_k_top_p_sample`、`ops-transformer/moe/moe_gating_top_k_softmax` | 任意 dim、tie-break、跨核 merge |
| transpose | `cann-samples/Samples/1_Features/memory_optimization/transpose_batch_matmul` | 任意 perm、内维向量化、真实物化 |
| strided_slice | `cann-samples/Samples/0_Introduction/03_simt_api/01_gather/general_gather` | 8 维 mask 语义、坐标映射 |
| q4_0_dequant_matmul | `ops-tensor/examples/weight_quant_matmul`、`cann-samples/Samples/2_Performance/matmul_story/matmul_recipes/examples/quant_matmul_mxfp4` | GGUF Q4_0 解包、block scale、反量化 |
| engram_gate_fusion | `ops-transformer/experimental/engram`、`ops-transformer/mhc`、`ops-transformer/gmm` | RMSNorm、门控、因果卷积、state |
| nms | 未找到同名实现 | score 排序、IoU、变长输出 |
| roi_align | 未找到同名实现 | ROI bin、双线性采样、pooling |
| aggregate_hidden / grad | 未找到同名实现 | W=3 因果滑窗、mask、全局归约 |

## 5. 检查命令

    cd <skill-root>
    python3 scripts/ensure_sources.py

检查本文引用路径与机器事实源一致：

    python3 scripts/ensure_sources.py --check-map references/source-map.md

第一条命令输出 JSON，记录每个仓的 path、branch、commit、dirty、origin 和 missing_paths。
`status=ready` 表示四个仓均可用；`dirty=true` 时该仓仍可作为参考，但引用时必须记录
commit 与 dirty 状态，并优先复核对应文件。origin 不匹配或 required_paths 缺失的源
不得作为已验证事实引用，只能标记为本地源不可用，继续以 asc-devkit 和官方样例为准。
