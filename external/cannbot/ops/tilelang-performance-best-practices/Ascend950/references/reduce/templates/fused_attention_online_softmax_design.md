# Attention Online Softmax

## 选型

Softmax 结果立即与 V 相乘，无需输出完整 probability。

## TileLang/PTO 实现

每个 KV block 计算 tile max/sum，使用 m_new=max(m,m_tile)、l_new=l*exp(m-m_new)+l_tile*exp(m_tile-m_new)，并用相同 alpha 重标定 fp32 O accumulator。实现直接复用 examples/ascend/flash_attention/example_mha.py。

基础 reduction 代码见 references/reduce/templates/dav310/kernel_utils.py 和 examples/ascend/example_rmsnorm.py：GM 数据通过 T.copy 进入 UB，在 T.SimtVF 中使用 fp32 fragment 与 T.reduce_max/T.reduce_sum 或 alloc_reducer/finalize_reducer；任务用一维 T.Kernel 分配，单个输出只有一个 owner。

## 精度门禁

mask 在 max 前应用；全遮蔽行按算子契约处理，不能产生未定义除零。

低精度输入的 max、sum、平方和、方差、rsqrt、exp 与 online 状态保持 fp32。尾部 max lane 填负无穷、sum lane 填 0。覆盖 tile±1、极长归约轴、抵消、极值、NaN/Inf 契约以及 forward/backward 的公开范围。

## 性能门禁

比较 full score 与 online 两条路径的 GM bytes、exp 次数和端到端 latency。

PTO 定向精度测试全通过后，比较端到端 latency、有效 GM bytes、Vector/MTE 时间、核利用率、UB 字节与 stage。多 kernel 方案必须计入 workspace 和全部 launch，不能只报告局部 kernel。
