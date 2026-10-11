# PTO FlashAttention 优化索引

## 优化目标

高性能实现 softmax(QK^T·scale)·V，避免显式写出完整 score/probability。

## PTO 实现

按 query block 分配 kernel task；Q 驻留，KV 分块流式。每个 KV block 更新 fp32 m、l 和 O accumulator，最后 O/l 写回。factory 按 D、sequence、causal 与 dtype 选择 BR/BC 和 stage。

FlashAttention 参考 `examples/ascend/flash_attention/example_mha.py`、`examples/ascend/flash_attention/core.py` 及配套 `test_mha.py`。按当前源码核对 QK/PV、online softmax、mask、buffer 布局和流水；示例存在不代表本轮后端已验证。修改 tile、stage、mask 或 forwarding 后运行对应精度用例，mask 必须在 row max 前应用。

## 精度门禁

同时验证数学输出与每行 softmax 归一化；不以较低精度 reference 掩盖误差。

必须与 torch scaled_dot_product_attention reference 比较，覆盖全遮蔽行、causal 对角、sequence/block 边界、极大正负 logits、重复最大值、NaN/Inf 契约及支持的 dtype。online 状态和输出累加保持 fp32。

## 性能门禁

基线、tile、stage、mask specialization、Q resident 分开 A/B。

记录完整 kernel latency、TFLOPS、Q/K/V/O GM bytes、Cube/Vector/MTE 时间、L1/L0/UB 占用和 stage。一次只改变一个优化变量，定向精度测试全通过后才能比较相关完整测试套。
