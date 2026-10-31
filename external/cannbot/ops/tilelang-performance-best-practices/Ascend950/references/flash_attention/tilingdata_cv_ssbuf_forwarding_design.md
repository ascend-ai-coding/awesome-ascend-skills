# Cube-Vector Buffer 转发

## 优化目标

QK 的 fp32 score 要交给 Vector softmax，概率再交回 Cube PV，目标是减少不必要的 GM 与中间复制。

## PTO 实现

使用 L0C→UB 的 T.dual_copy、UB 中 SimdVF softmax、UB→L1 的 T.dual_copy，再执行 T.gemm(P,V)。BR/BC 与 buffer 大小由 Python factory 固化。

FlashAttention 参考 `examples/ascend/flash_attention/example_mha.py`、`examples/ascend/flash_attention/core.py` 及配套 `test_mha.py`。按当前源码核对 QK/PV、online softmax、mask、buffer 布局和流水；示例存在不代表本轮后端已验证。修改 tile、stage、mask 或 forwarding 后运行对应精度用例，mask 必须在 row max 前应用。

## 精度门禁

bf16 probability 转换前在 fp32 中完成 max、exp 与 sum；评估转换误差对最终 O 的影响。

必须与 torch scaled_dot_product_attention reference 比较，覆盖全遮蔽行、causal 对角、sequence/block 边界、极大正负 logits、重复最大值、NaN/Inf 契约及支持的 dtype。online 状态和输出累加保持 fp32。

## 性能门禁

统计 L0C/UB/L1 搬运与同步 gap；增大 buffer 只有在重叠收益超过容量代价时启用。

记录完整 kernel latency、TFLOPS、Q/K/V/O GM bytes、Cube/Vector/MTE 时间、L1/L0/UB 占用和 stage。一次只改变一个优化变量，定向精度测试全通过后才能比较相关完整测试套。
