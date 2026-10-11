# Reduction Kernel 使用流程

## 选型

新增 ReduceSum/Max、LayerNorm、RMSNorm、Softmax 时使用。

## TileLang/PTO 实现

Python factory 接受静态 reduction width、dtype、threads、tile 与 stages，返回 T.prim_func。动态 rows 由 T.dynamic 表达；核数为 min(num_cores, rows)，其中 num_cores 由硬件查询结果显式传入。先复用现有 kernel，再添加 shape specialization。

基础 reduction 代码见 references/reduce/templates/dav310/kernel_utils.py 和 examples/ascend/example_rmsnorm.py：GM 数据通过 T.copy 进入 UB，在 T.SimtVF 中使用 fp32 fragment 与 T.reduce_max/T.reduce_sum 或 alloc_reducer/finalize_reducer；任务用一维 T.Kernel 分配，单个输出只有一个 owner。

## 精度门禁

测试文件与 kernel 同步提交，reference 使用 torch fp32 或更高精度稳定表达式。

低精度输入的 max、sum、平方和、方差、rsqrt、exp 与 online 状态保持 fp32。尾部 max lane 填负无穷、sum lane 填 0。覆盖 tile±1、极长归约轴、抵消、极值、NaN/Inf 契约以及 forward/backward 的公开范围。

## 性能门禁

定向精度测试后运行相关完整测试套，并记录每个 dispatch 分界。

PTO 定向精度测试全通过后，比较端到端 latency、有效 GM bytes、Vector/MTE 时间、核利用率、UB 字节与 stage。多 kernel 方案必须计入 workspace 和全部 launch，不能只报告局部 kernel。
