# PTO RoPE 优化索引

## 实现

以 references/rope/code/rope_vf_common.py 为基线。Python factory 固化 layout、dim、dtype 与 position offset；一维 T.Kernel 以 grid-stride 分配 token×head task。输入、sin、cos 通过 T.copy 进入 UB，已通过 PTO 编译的 T.SimtVF/T.Parallel 路径以 fp32 完成成对旋转，最后转换输出 dtype。

half-split 与 interleaved 必须生成不同 kernel，不能在热循环中动态分支。任意 position tensor 需要先验证 PTO 标量/离散索引 lowering；当前可执行基线使用编译期 position offset 的连续位置。该 SimtVF 路径只用于精度回归；生产路径必须实现可 lowering 的 SIMD intrinsic、跨 head 复用或算子融合，并附带硬件、版本、命令和原始性能结果。

## 精度

覆盖 position=0/最大值、offset、奇偶 token/head、dim 边界、两种 layout、原位/非原位、bf16/fp16/fp32。reference 的 sin/cos、乘加和拼接使用 fp32；输出只在最终边界转换。

## 性能

比较每 head 重载 sin/cos 与 token-grouped resident 版本，报告 GM bytes、Vector/MTE 时间、UB 占用和 latency。定向精度测试全通过后运行相关完整测试套。
