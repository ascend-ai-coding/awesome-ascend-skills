# Reduction 状态常驻

## 选型

跨输入 tile 只需少量状态，例如 sum、max、平方和、online softmax 的 m/l。

## TileLang/PTO 实现

状态分配为单版本 fp32 UB/fragment，在 tile 循环前初始化，循环内更新，结束后写回。输入输出 tile 可以多版本，状态不得加入 buffer versions。

基础 reduction 代码见 references/reduce/templates/dav310/kernel_utils.py 和 examples/ascend/example_rmsnorm.py：GM 数据通过 T.copy 进入 UB，在 T.SimtVF 中使用 fp32 fragment 与 T.reduce_max/T.reduce_sum 或 alloc_reducer/finalize_reducer；任务用一维 T.Kernel 分配，单个输出只有一个 owner。

## 精度门禁

每个新输出必须重新初始化状态，避免跨 task 污染。

低精度输入的 max、sum、平方和、方差、rsqrt、exp 与 online 状态保持 fp32。尾部 max lane 填负无穷、sum lane 填 0。覆盖 tile±1、极长归约轴、抵消、极值、NaN/Inf 契约以及 forward/backward 的公开范围。

## 性能门禁

比较减少的 GM partial 流量与串行依赖；状态小但依赖长时主要优化 DMA overlap。

PTO 定向精度测试全通过后，比较端到端 latency、有效 GM bytes、Vector/MTE 时间、核利用率、UB 字节与 stage。多 kernel 方案必须计入 workspace 和全部 launch，不能只报告局部 kernel。
