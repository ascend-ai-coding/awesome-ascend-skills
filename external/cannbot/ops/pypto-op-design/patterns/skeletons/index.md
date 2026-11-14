# 整体计算骨架

先根据摘要选择候选，再读取卡片正文。C 表示 Cube，V 表示 Vector；流程仅示意主要计算顺序，变体及循环见正文。

| ID | 名称 | tags | flow_pattern |
|---|---|---|---|
| SK-01 | [Online Flash Attention](SK-01-online-flash-attention.md) | attention | C1, V1, C2 |
| SK-02 | [Single-Pass Attention](SK-02-single-pass-attention.md) | attention | C1, V1, C2 |
| SK-03 | [Linear Projection (Norm→MatMul)](SK-03-linear-projection.md) | projection | V1, C1 |
| SK-04 | [Multi-Stage Fused Prolog](SK-04-multi-stage-prolog.md) | projection | C1, V1, C2, C3, V2, C4 |
| SK-05 | [Fused Pre-Attn (Two-Phase)](SK-05-fused-pre-attention.md) | attention, projection | V1, C1, V2, C2, V3, C3 |
| SK-06 | [FFN / SwiGLU](SK-06-ffn-swiglu.md) | ffn | C1, C2, V1 |
| SK-07 | [MOE (Gate + Select + FFN)](SK-07-moe.md) | moe | V1, C1, V2, C2, V3, C3, V4 |
| SK-08 | [Vector Element-wise (Batch Unroll)](SK-08-vector-batch.md) | vector | V1 |
| SK-09 | [Vector Tiling (Multi-Axis)](SK-09-vector-multi-axis.md) | vector | V |
| SK-10 | [Recurrent State Machine](SK-10-recurrent-state.md) | recurrent | V, C, V |
| SK-11 | [Cache Compressor](SK-11-cache-compressor.md) | cache | C, V |
| SK-12 | [Expert Gating (Pure Vec)](SK-12-expert-gating.md) | moe, vector | V |
| SK-13 | [General Pure Vector](SK-13-general-vector.md) | vector | V |
| SK-14 | [General Multi-MatMul](SK-14-general-matmul.md) | matmul | C, V |
| SK-15 | [General CV Fusion](SK-15-general-cv-fusion.md) | fusion | C, V |
| SK-16 | [Single-Pass Attention Backward (FA Grad)](SK-16-attention-backward.md) | attention, backward | C1, V1, C2, V2 |
