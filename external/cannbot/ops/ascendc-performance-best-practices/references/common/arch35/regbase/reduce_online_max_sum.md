# `reduce-online-max-sum`

## 解决的问题

归约轴不能一次放入 UB 时，范例沿 R 分 chunk，为每个 A0 列维护 FP32 running max 和 running sum：

```text
m_new = max(m_old, chunk_max)
l_new = l_old * exp(m_old - m_new) + sum(exp(x - m_new))
```

## 为什么这样写

- chunk 的局部 max 改变时，旧 sum 所在指数基准也改变；乘 `exp(m_old-m_new)` 才能与新 chunk 合并。
- running max/sum 保持 FP32，避免长归约轴上反复 B16 舍入放大误差。
- 只保存两个统计量而不保存所有 chunk 的 exp 中间值，使超 UB 归约仍能流式处理。
- 下游若只需要 `logsumexp`，最后使用 `m_final + log(l_final)`，不必回扫并写出完整 Softmax；这也是长轴损失计算避免多余输出的关键。

## 怎么使用

- 代码：[`../reduce/templates/dav310/softmax_v2_ara_online.template`](../../../reduce/templates/dav310/softmax_v2_ara_online.template)
- 集成说明：[`../reduce/templates/softmax_v2_ara_online.md`](../../../reduce/templates/softmax_v2_ara_online.md)
- 重点函数：`CalcOnlineMaxSum`、`CalcOutput`
- 源构建契约：`unverified`；当前无入库的最终编译命令，声明位置和入口必须在目标工程重验

先证明目标归约也满足这组在线更新公式，再迁移 `CalcOnlineMaxSum`；最终按目标选择完整向量输出或直接消费 `m_final/l_final`。同时保留 FP32 统计量、chunk 边界 mask 和历史 sum 重标定。用极值跨 chunk、全负数和尾 chunk 做精度验证。

## 前置条件

- 3D `[A1, R, A0]`，沿 R 归约，`A0 > 1`。
- running max/sum 必须保持 FP32，并保留历史 sum 的 `exp(m_old-m_new)` 修正。
- 该模板输出完整 Softmax，未包含矩阵乘、mask 或其他融合计算。

这是 `integration_skeleton`。把在线公式移植到其他归约时，必须先证明统计量与更新公式满足目标精度语义。
