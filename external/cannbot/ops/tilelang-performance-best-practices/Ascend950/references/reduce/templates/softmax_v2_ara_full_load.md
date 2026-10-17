# ARA Full-load Softmax

## 选型

归约轴位于中间维，outer×inner 产生足够独立输出且对应 slice 可驻留。

## TileLang/PTO 实现

把每个 outer/inner 组合视为一个输出，按 R gather 到 fp32 fragment；若源 stride 可合并为矩形则先 T.copy，否则使用已验证 SIMT 索引。

基础 reduction 代码见 references/reduce/templates/dav310/kernel_utils.py 和 examples/ascend/example_rmsnorm.py：GM 数据通过 T.copy 进入 UB，在 T.SimtVF 中使用 fp32 fragment 与 T.reduce_max/T.reduce_sum 或 alloc_reducer/finalize_reducer；任务用一维 T.Kernel 分配，单个输出只有一个 owner。

## 精度门禁

验证任意 axis 归一化和非连续 stride。

低精度输入的 max、sum、平方和、方差、rsqrt、exp 与 online 状态保持 fp32。尾部 max lane 填负无穷、sum lane 填 0。覆盖 tile±1、极长归约轴、抵消、极值、NaN/Inf 契约以及 forward/backward 的公开范围。

## 性能门禁

比较转置后连续 softmax与直接 strided 访问的总 GM bytes 和 latency。

PTO 定向精度测试全通过后，比较端到端 latency、有效 GM bytes、Vector/MTE 时间、核利用率、UB 字节与 stage。多 kernel 方案必须计入 workspace 和全部 launch，不能只报告局部 kernel。
