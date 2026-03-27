# TF To PyTorch Numerical Validation

## 验证顺序

1. 先验证每层权重 shape 和 dtype。
2. 再验证关键中间层输出。
3. 最后验证端到端输出和下游指标。

## 推荐阈值

- 中间层数值优先用 `allclose` 或误差统计来判定。
- 如果是深层网络，允许误差逐步放大，但必须能解释来源。
- 对分类任务，除分数差异外，还要看 top-k 排名是否稳定。

## 常见差异来源

- LayerNorm 或 BatchNorm 默认 epsilon 不一致。
- Dense、Conv1D、LSTM 权重转置或 bias 拆分错误。
- `K.dot`、`einsum`、维度重排的语义理解错误。

## 结论写法建议

- 先说明比对范围。
- 再说明误差大小和可能原因。
- 最后给出是否可以接受，还是必须回到实现继续修正。
