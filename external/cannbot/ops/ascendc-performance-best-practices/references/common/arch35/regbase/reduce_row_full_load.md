# `reduce-row-full-load`

## 解决的问题

当归约行可全量放入 UB 时，范例把多行批量载入，用 `RegTensor<float>` 做稳定 max、`exp(x-max)`、sum 和归一化，并把 B16/F32 Cast 合入 load/store。

## 为什么这样写

- 全行驻留后，第一次遍历求 max/exp/sum，第二次归一化可复用 UB 输入，避免跨 tile 合并统计量。
- FP16/BF16 在 load 时转 FP32、在 store 时再转回，归约精度和 Cast 次数都比逐阶段落 UB 更可控。
- `FirstNormCompute` 与 `SecondNormCompute` 对应“先得到整行统计量，再输出”；不能在统计量尚未完成时提前写结果。
- 下游若只消费 `logsumexp`，应复用 `max + log(sum(exp(x-max)))` 统计量直接输出标量，不必先物化完整归一化向量；目标位置读取、权重或其他语义在归约链之外单独保持。

## 怎么使用

- 代码：[`../reduce/templates/dav310/softmax_v2_ar_full_load.template`](../../../reduce/templates/dav310/softmax_v2_ar_full_load.template)
- 集成说明：[`../reduce/templates/softmax_v2_ar_full_load.md`](../../../reduce/templates/softmax_v2_ar_full_load.md)
- 重点函数：`FirstNormCompute`、`SecondNormCompute`、`LoadTensorForDtypeTIn`、`StoreTensorForDtypeTOut`
- 源构建契约：`unverified`；当前无入库的最终编译命令，声明位置和入口必须在目标工程重验

先用真实 dtype 和并行行数计算 UB 预算，确认输入、输出和 FP32 中间量能同时驻留；然后按目标选择“归一化向量输出”或“直接消费 max/sum 统计量”，连同 load/store Cast、两阶段计算和 Tiling 一起迁移。临界 UB 容量和尾行必须单独验证。

## 前置条件

- 2D `[A1, R]`，归约轴为最后一轴。
- 输入行、输出和中间 FP32 空间满足文档的 UB 预算。
- FP16/BF16 输入仍在 FP32 域累积；不能为了少一次 Cast 改变精度顺序。

这是 `integration_skeleton`。使用时需一并复制公共基类、TilingData 和 Host tiling，不能把模板文件原样当成完整工程。
