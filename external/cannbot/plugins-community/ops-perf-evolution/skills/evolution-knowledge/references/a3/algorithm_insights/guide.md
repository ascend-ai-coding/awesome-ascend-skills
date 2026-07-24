# 算法洞见快速参考

## 核心原则

> 指令级优化碰到"平衡墙"时，唯一出路是从算法层面同时减少计算和搬运。

## 按算子族查阅

| 算子族 | 文件 | 关键洞见 |
|--------|------|---------|
| Attention (MHA/GQA/MQA) | `attention_family.md` | 块稀疏跳过、在线 softmax、KV-cache |
| Reduction (Norm/Softmax) | `reduction_ops.md` | 双遍 vs 单遍、Welford、树归约 |
| Elementwise 融合 | `elementwise_fusion.md` | 算子融合、原地计算 |

## 何时读详细文件

- Init 阶段分析算子特性时 → 读匹配的算子族文件
- open_exploration 模式 → 必读匹配族文件寻找灵感
- 停滞反思 → 检查是否存在未尝试的算法方法
