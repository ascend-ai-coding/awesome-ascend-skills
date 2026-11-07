# Softmax AR Full-load

## 选型

整行输入、fp32 工作区和必要 buffer 可同时放入 UB。

## TileLang/PTO 实现

一次 T.copy 整行到 fp32 UB/fragment，reduce_max，计算 exp(x-max)，reduce_sum，再归一化并写回。对齐 padding 参与 reduction 前设置幺元。

基础 reduction 代码见 references/reduce/templates/dav310/kernel_utils.py 和 examples/ascend/example_rmsnorm.py：GM 数据通过 T.copy 进入 UB，在 T.SimtVF 中使用 fp32 fragment 与 T.reduce_max/T.reduce_sum 或 alloc_reducer/finalize_reducer；任务用一维 T.Kernel 分配，单个输出只有一个 owner。

## 精度门禁

稳定形式与 fp32 reference 比较，输出 dtype 只在最终 store 转换。

低精度输入的 max、sum、平方和、方差、rsqrt、exp 与 online 状态保持 fp32。尾部 max lane 填负无穷、sum lane 填 0。覆盖 tile±1、极长归约轴、抵消、极值、NaN/Inf 契约以及 forward/backward 的公开范围。

## 性能门禁

小中等 R 比较 single-stage 与 2-stage 行流水，避免为单行内部强行多版本。

PTO 定向精度测试全通过后，比较端到端 latency、有效 GM bytes、Vector/MTE 时间、核利用率、UB 字节与 stage。多 kernel 方案必须计入 workspace 和全部 launch，不能只报告局部 kernel。
