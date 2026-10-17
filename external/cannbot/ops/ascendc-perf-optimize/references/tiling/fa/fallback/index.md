# FA FlashAttention 类 — 兜底算法

> 适用目标：arch35（DAV_3510 / A5）。当前为占位文档，内容待补充。arch22（A2/A3）返回上级路由并读取目标专用 FIA 知识，不使用本占位页兜底。

## 适用算子

FlashAttention、FusedInferAttentionScore、QSFA、SparseFlashAttention、SCFA

## 待补充内容

- 多核切分策略（Batch × Head / Query × KV 等）
- 单核切分策略
- Buffer 规划
- 分支覆盖

---

> 当前返回：「FA Tiling 建模暂未收录。」
