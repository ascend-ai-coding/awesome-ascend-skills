# ProteinBERT Benchmark And Debug Notes

## 推荐验证顺序

1. 先完成预训练权重转换。
2. 再做 embedding 逐层对比。
3. 最后做 benchmark 微调和注意力可视化。

## 调试优先级

- 先比对 embedding 与中间层输出。
- 再比对最终任务指标，如 AUC、Accuracy、Spearman。
- 对于训练差异，优先区分框架差异与迁移错误。

## 常见问题

- LayerNorm epsilon 设置错误。
- 权重顺序与 TF pickle 中的存储顺序不一致。
- 在同一环境同时安装 TensorFlow 与 `torch_npu` 导致冲突。

## 推荐沉淀物

- 一份 embedding 对比结果。
- 一份 benchmark 汇总表。
- 一份最关键层的调试样例输出。
