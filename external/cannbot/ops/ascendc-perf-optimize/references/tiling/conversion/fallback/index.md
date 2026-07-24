# Conversion 数据转换类 — 兜底算法

> 当前为占位文档，内容待补充。

## 适用算子

Transpose, Concat, Split

## 待补充内容

- 多核切分策略（Transpose 沿输出维度 / Concat 按输入段 / Split 按输出段）
- 单核切分策略（block 为单位搬移，考虑跨 stride 对齐）
- Buffer 规划
- 分支覆盖

---

> 当前返回：「Conversion Tiling 建模暂未收录。」
