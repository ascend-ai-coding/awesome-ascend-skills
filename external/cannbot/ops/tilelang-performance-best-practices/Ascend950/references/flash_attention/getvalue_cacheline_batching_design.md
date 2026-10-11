# KV 连续批量搬运

## 优化目标

K/V block 的小片段或不连续访问导致 copy transaction 过多。

## PTO 实现

选择 BC 使 K[BC,D] 和转置 V[D,BC] 形成连续矩形 T.copy；V 使用已验证的 transpose=True 路径。相邻 KV block 由 Pipelined stage 多版本，不编写标量逐元素 load。

FlashAttention 参考 `examples/ascend/flash_attention/example_mha.py`、`examples/ascend/flash_attention/core.py` 及配套 `test_mha.py`。按当前源码核对 QK/PV、online softmax、mask、buffer 布局和流水；示例存在不代表本轮后端已验证。修改 tile、stage、mask 或 forwarding 后运行对应精度用例，mask 必须在 row max 前应用。

## 精度门禁

验证 K/V layout、head stride、GQA/MQA 共享规则和 KV 尾块。

必须与 torch scaled_dot_product_attention reference 比较，覆盖全遮蔽行、causal 对角、sequence/block 边界、极大正负 logits、重复最大值、NaN/Inf 契约及支持的 dtype。online 状态和输出累加保持 fp32。

## 性能门禁

比较 copy 数、burst、MTE2 时间和增大 BC 后的 L1/L0 压力。

记录完整 kernel latency、TFLOPS、Q/K/V/O GM bytes、Cube/Vector/MTE 时间、L1/L0/UB 占用和 stage。一次只改变一个优化变量，定向精度测试全通过后才能比较相关完整测试套。
