# EuclideanNorm 连续尾块

## 选型

R 连续但最后 tile 非整除。

## TileLang/PTO 实现

T.copy valid 元素到 fp32 tile，剩余 lane 清零，fragment 计算平方并 reduce_sum；carry 常驻 fp32。

基础 reduction 代码见 references/reduce/templates/dav310/kernel_utils.py 和 examples/ascend/example_rmsnorm.py：GM 数据通过 T.copy 进入 UB，在 T.SimtVF 中使用 fp32 fragment 与 T.reduce_max/T.reduce_sum 或 alloc_reducer/finalize_reducer；任务用一维 T.Kernel 分配，单个输出只有一个 owner。

## 精度门禁

验证 R=1、lane±1、tile±1 和大值平方溢出边界。

低精度输入的 max、sum、平方和、方差、rsqrt、exp 与 online 状态保持 fp32。尾部 max lane 填负无穷、sum lane 填 0。覆盖 tile±1、极长归约轴、抵消、极值、NaN/Inf 契约以及 forward/backward 的公开范围。

## 性能门禁

比较 pad copy 与显式 clear 的 latency。

PTO 定向精度测试全通过后，比较端到端 latency、有效 GM bytes、Vector/MTE 时间、核利用率、UB 字节与 stage。多 kernel 方案必须计入 workspace 和全部 launch，不能只报告局部 kernel。
