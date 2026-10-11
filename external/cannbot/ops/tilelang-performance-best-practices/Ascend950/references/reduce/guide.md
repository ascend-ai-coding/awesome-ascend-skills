# PTO Reduction、Norm 与 Softmax

## 选型

根据归约轴是否能驻留 UB、输出行数是否足以占满向量核、是否需要输出每个元素，选择 full-load、recompute、online 或两阶段 split-axis。

## TileLang/PTO 实现

先实现一输出一核的 fp32 基线。Norm 常驻 weight；softmax 使用稳定 max-subtract。只有行数太少且轴极长时，才用 Kernel A 写 fp32 partial、Kernel B 确定顺序合并。

基础 reduction 代码见 references/reduce/templates/dav310/kernel_utils.py 和 examples/ascend/example_rmsnorm.py：GM 数据通过 T.copy 进入 UB，在 T.SimtVF 中使用 fp32 fragment 与 T.reduce_max/T.reduce_sum 或 alloc_reducer/finalize_reducer；任务用一维 T.Kernel 分配，单个输出只有一个 owner。

短归约能放入一个 SIMD 寄存器时，可用有效 lane mask + `S.vcadd`/`S.vdupv` 替代 SIMT reducer，并把相邻输出批量搬入 UB；必须 A/B 验证运算顺序、spill 和收益。若 partial 只供下游归约，先评估在 producer 内完成 split-axis 合并并特化 `n_splits==1`，同时保留确定性与原子语义。匹配该结构时读取 [批量短归约](batched_short_reduction.md)。

固定小矩阵的归一化可让当前矩阵及反向梯度跨整轮计算常驻 SIMD 寄存器，仅将反向所需快照写 UB；每组先求一次倒数再乘各元素，须验证 local 布局、同步、误差、寄存器压力和展开代码量。匹配该结构时读取 [固定小状态](fixed_small_state.md)。

若每行先生成归约标量，再使用该标量逐元素写回整行，读取 [行归约与逐元素写回](rowwise_reduce_epilogue.md)。

## 精度门禁

Softmax 额外检查每行和、重复最大值、全负无穷和正无穷语义；Norm 的 eps 加在 fp32 统计量上。

低精度输入的 max、sum、平方和、方差、rsqrt、exp 与 online 状态保持 fp32。尾部 max lane 填负无穷、sum lane 填 0。覆盖 tile±1、极长归约轴、抵消、极值、NaN/Inf 契约以及 forward/backward 的公开范围。

## 性能门禁

依次测 tile、threads、resident weight、1/2 stage 和 split-axis；保持正确 fallback。

归约计算较轻、非归约连续轴较宽且小 tile 的调度或 DMA 开销明显时，按 [宽输出轴 Tiling](wide_output_tiling.md) 合并连续输出元素。

PTO 定向精度测试全通过后，比较端到端 latency、有效 GM bytes、Vector/MTE 时间、核利用率、UB 字节与 stage。多 kernel 方案必须计入 workspace 和全部 launch，不能只报告局部 kernel。
