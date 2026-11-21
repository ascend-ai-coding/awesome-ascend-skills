# TileLang/PTO Reference 索引

先阅读 [通用实现指南](tilelang_pto_guide.md)、[模板成熟度](template_status.md) 和 [验证门禁](validation.md)。目标仓库已有实现直接引用仓库内真源；TileLang API 通过实际导入路径定位，不在 Skill 中复制第二份。

结构文档只提供提炼后的优化决策，不索引或要求读取对应历史优化实现；从当前目标 baseline 自主实现并验证。

| 算子族 | 文档 | 代码参考 | 关键优化 |
|---|---|---|---|
| Common | 名为 `npu-arch` 的 Skill · [copy](common/datacopy_optimization_design.md) · [tail](common/tail_block_design.md) · [resident](common/ub_resident_design.md) · [task map](common/persistent_task_mapping.md) | TileLang `examples/ascend/` | 硬件事实、合并搬运、padding、常驻、多版本、多轴任务映射 |
| Broadcast | [guide](broadcast/broadcast_design.md) | [broadcast code](broadcast/code/) | 单轴/多轴、resident input、shape specialization |
| Conversion | [guide](conversion/guide.md) | `testing/ascend/layout/test_ascend_l0_transpose.py` | L0 布局测试与搬运 API 参考；通用转置需提供已验证 factory |
| Elementwise / Gather | [vector](elementwise/vector_efficiency_design.md) · [tiling/task](elementwise/tiling_task_vector_search.md) · [pipeline](elementwise/double_buffer_design.md) · [indexed short record](elementwise/indexed_short_record.md) | `examples/ascend/example_simdvf_per_token_cast_to_fp8.py`、`testing/ascend/layout/` | SimdVF、短记录打包、共享字段融合、partial 打包、静态整 tile/JIT、连续搬运、任务展平、指令数据流、2/3 stage |
| MatMul | [guide](matmul/guide.md) | TileLang `examples/ascend/example_gemm.py` | tile 搜索、L1/L0、流水、resident、Stream-K |
| Reduction | [guide](reduce/guide.md) · [rowwise reduce epilogue](reduce/rowwise_reduce_epilogue.md) · [wide output](reduce/wide_output_tiling.md) · [batched short reduction](reduce/batched_short_reduction.md) · [fixed small state](reduce/fixed_small_state.md) | [reduction strategies](reduce/templates/dav310/) | fp32 状态、full-load、行归约与逐元素写回融合、wide-output tile、recompute、online、SIMD 短归约、小状态常驻、split-axis |
| FlashAttention | [guide](flash_attention/guide.md) | TileLang `examples/ascend/flash_attention/example_mha.py` | Q resident、online softmax、Cube/Vector forwarding |
| Scan | [guide](scan/guide.md) | [scan strategies](scan/templates/dav310/) | row-owner、resident carry、lane scan、三阶段 split |
| RoPE | [guide](rope/guide.md) | [rope_vf_common.py](rope/code/rope_vf_common.py) | fp32 pair rotation、layout specialization |
| Scalar | [guide](scalar/guide.md) | 各算子 Python factory | 常量化、地址复用、生命周期、代码体积 |
| SIMT | [guide](simt/optimization-guide.md) | reduction 模板及当前仓库的 SIMT 示例 | threads、fragment、分支特化 |
| Sort/TopK | [guide](sort/radix_sort.md) | `examples/ascend/example_simdvf_topk_gate.py` | padded load、局部选择、分层 merge |
| Conv | [guide](conv/guide.md) | PTO GEMM primitive | tiled im2col/grouped GEMM |
| 通算组合 | [guide](mc2/guide.md) | local GEMM + 正式 distributed runtime | 异步 collective、chunk pipeline |

## 验证要求

- 从目标仓库复制的模板仍须在当前分支重新运行对应测试。
- Scan、RoPE 和转换后的旧模板先做 PTO lowering，再做设备精度与端到端性能测试。
- 先按 [template_status.md](template_status.md) 标注成熟度；`PARTIAL`/`DESIGN_ONLY` 不得列为可直接拷贝模板。
- fp16/bf16 reduction、norm、softmax 和 GEMM 默认使用 fp32 统计或累加状态。
- 尾块、动态 shape、NaN/Inf、极小任务和多阶段 workspace 必须纳入测试矩阵。
