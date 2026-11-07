# 局部计算模式

先根据摘要选择候选，再读取卡片正文。C 表示 Cube，V 表示 Vector；流程仅示意主要计算顺序，变体及循环见正文。

| ID | 名称 | tags | flow_pattern |
|---|---|---|---|
| AT-01 | [Online Softmax](AT-01-online-softmax.md) | softmax | V |
| AT-02 | [Standard Softmax](AT-02-standard-softmax.md) | softmax | V |
| AT-03 | [RMSNorm](AT-03-rmsnorm.md) | norm | V |
| AT-04 | [RoPE (Rotary Position Embedding)](AT-04-rope.md) | positional-encoding | V |
| AT-05 | [Per-Token Quantization (FP → INT8)](AT-05-int8-quant.md) | quant | V |
| AT-06 | [Per-Token Dequantization (INT → FP)](AT-06-int8-dequant.md) | dequant | V |
| AT-07 | [SwiGLU Activation](AT-07-swiglu.md) | activation | V |
| AT-08 | [FP8 Quantization (Symmetric Per-Token)](AT-08-fp8-quant.md) | quant | V |
| AT-09 | [Linear Projection (Quantized MatMul)](AT-09-quant-linear.md) | matmul | C, V |
| AT-10 | [RMSNorm + Linear (Fused)](AT-10-norm-linear.md) | norm-linear-fused | V, C |
| AT-11 | [RMSNorm + Linear + Quant (Fused)](AT-11-norm-quant-linear.md) | norm-linear-quant-fused | V, C, V |
| AT-12 | [QK^T MatMul + Scale](AT-12-qk-matmul.md) | matmul | C |
| AT-13 | [P @ V MatMul](AT-13-pv-matmul.md) | matmul | C |
| AT-14 | [Gated Recurrent Update](AT-14-recurrent-update.md) | recurrent-update | V, C, V, C |
| AT-15 | [Expert Gating (TopK Selection)](AT-15-expert-gating.md) | routing | V |
| AT-16 | [Paged Cache Scatter/Update](AT-16-cache-scatter.md) | scatter | V |
| AT-17 | [Block Table Gather (Paged KV)](AT-17-block-gather.md) | gather | V |
| AT-18 | [Iterative Normalize (Sinkhorn 族)](AT-18-iterative-normalize.md) | iterative-normalize | V |
| AT-19 | [Optimizer Update (Adam/RMSProp/SGD 族)](AT-19-optimizer-update.md) | optimizer-step | V |
| AT-20 | [Valid Shape Tail Block (尾块对齐处理)](AT-20-tail-block.md) | shape-adapt | 见正文 |
| AT-21 | [Attention 分阶段子图合图（sg_set_scope）](AT-21-attention-scope.md) | subgraph-boundary | C, V |
| AT-22 | [K-Split MatMul（大 K 投影）](AT-22-k-split-matmul.md) | matmul | C |
| AT-23 | [Unroll-and-Jam 多链交错（Multi-Chain Interleave）](AT-23-multi-chain-interleave.md) | loop-interleave | 见正文 |
