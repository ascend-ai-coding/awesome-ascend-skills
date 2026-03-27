# DeepFRI Weight Conversion Checklist

## 重点映射规则

- Dense/Linear 权重要转置。
- Conv1D 要处理输入布局和 `(K, Cin, Cout)` → `(Cout, Cin, K)` 的转换。
- BatchNorm 必须显式保持 `eps=1e-3`。
- CuDNNLSTM bias 需要拆成 `bias_ih` 与 `bias_hh`。

## 精度对齐顺序

1. 先验证单层权重 shape 是否一致。
2. 再验证中间层输出均值和方差。
3. 最后验证端到端 top-k 和分数差异。

## 最容易出错的点

- TF 默认 epsilon 和 PyTorch 默认值不一致。
- LSTM 或 GraphConv 的输入维度顺序写反。
- 直接引用了不存在的 `references/` 路径而不是当前 skill 的 `scripts/`。

## 推荐验收标准

- CNN 路径的关键输出与基线差异保持在可解释范围内。
- 至少保留一组具体输入和对应预测结果作为回归样例。
