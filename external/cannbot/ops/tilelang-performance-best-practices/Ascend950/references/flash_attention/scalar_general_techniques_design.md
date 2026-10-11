# FlashAttention 标量热循环精简

## 优化目标

KV 循环的地址、循环和状态管理占用明显标量周期。

## PTO 实现

缩短局部变量生命周期，复用静态 offset，避免在循环内构造复杂 view；把固定 row/vector 循环用 range/T.Unroll 表达，把动态 sequence 只保留在 block 数和尾块。

FlashAttention 参考 `examples/ascend/flash_attention/example_mha.py`、`examples/ascend/flash_attention/core.py` 及配套 `test_mha.py`。按当前源码核对 QK/PV、online softmax、mask、buffer 布局和流水；示例存在不代表本轮后端已验证。修改 tile、stage、mask 或 forwarding 后运行对应精度用例，mask 必须在 row max 前应用。

## 精度门禁

精简不能删除真实的 mask、尾块或状态更新依赖。

必须与 torch scaled_dot_product_attention reference 比较，覆盖全遮蔽行、causal 对角、sequence/block 边界、极大正负 logits、重复最大值、NaN/Inf 契约及支持的 dtype。online 状态和输出累加保持 fp32。

## 性能门禁

通过生成源码与 timeline 统计标量 gap，只有端到端 latency 改善才保留。

记录完整 kernel latency、TFLOPS、Q/K/V/O GM bytes、Cube/Vector/MTE 时间、L1/L0/UB 占用和 stage。一次只改变一个优化变量，定向精度测试全通过后才能比较相关完整测试套。
