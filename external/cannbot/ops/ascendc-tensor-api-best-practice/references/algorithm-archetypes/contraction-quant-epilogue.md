+# Cube 收缩与量化 epilogue

本 archetype 是路由层，覆盖 grouped_matmul、quant_matmul、weight_quant_batch_matmul、q4_0_dequant_matmul 等 Contraction 类算子。它不是完整实现手册；实施前必须继续读任务合同、DESIGN/PLAN 模板、api-dataflow-layout、precision-model，以及 source-map 中对应参考实现的路线审计结论。

## 路线选择

| 场景 | 执行单元 | 说明 |
|---|---|---|
| 大 M/N/K、规整 GEMM | AIC | 走 GM→L1→L0A/L0B→L0C→Fixpipe/GM |
| Mmad 后接逐元素、scale、激活量化 | __mix__ | L0C→UB 直通，AIV 做 epilogue |
| 纯量化输出、无跨 tile 依赖 | AIC | Fixpipe scalar/tensor 随路量化 |
| 小 M 或 GEMV 类 case | 先做最小 AIC/AIV 探针 | M=1 不一定收益于大 Cube tile |

## 关键设计点

1. 先冻结数学语义：量化输入先反量化还是直接 int8 Mmad；bias 是 pre-scale 还是 post-scale；
   pertoken/offset/浮点 bias 的顺序必须与任务合同一致。
2. 量化 ABI 与逻辑 ABI 分开：Fixpipe quant 参数是 uint64_t 或 L1 quant tensor，不是任务侧 float32/bfloat16。
3. int8×int8→int32 走 mmad；输出 fp16/bf16 可用 L0C 随路量化，超出随路量化表达能力时转 UB/AIV 后处理。
4. GroupedMatmul 用 cumsum group_list 推 expert 行区间；每个 expert 的 weight/bias 偏移必须显式写入 ABI crosswalk。
5. 尾块分 M/N/K 三轴处理；padding 不参与 Golden，输出 writer 必须唯一。
6. L1/L0C/UB 预算在 DESIGN 中列满；多槽 weight 复用必须核对 expert 维度，避免槽位语义互踩。

## 验证顺序

1. 最小对齐 2D case：无 bias、无 optional、per-channel scale。
2. 加 int32 bias /浮点 bias，再加 pertoken、offset、batch。
3. 非对齐 M/N/K、M=1、大 N、大 batch、连续重复 launch。
4. 量化 scale 用有限、非 Inf/NaN、非非规格化数生成；冻结舍入和饱和语义。

## 事实源

- asc-devkit：examples/01_simd_cpp_api/07_tensor_api/{mmad_tensor_api,batch_matmul_tensor_api,matmul_bias_fusion,matmul_mxfp4_tensor_api_high_performance}
- ops-tensor：docs/API/gemm/block、docs/API/epilogue/block、examples/{grouped_matmul,weight_quant_batch_matmul,quant_matmul_activation_quant}
- ops-transformer：gmm/grouped_matmul 及其量化/activation 变体
- cann-samples：Samples/2_Performance/{matmul_story,grouped_matmul_story}、Samples/1_Features/hardware_features/cv_datapath

L3 映射：grouped_matmul、quant_matmul、q4_0_dequant_matmul、weight_quant_batch_matmul。
