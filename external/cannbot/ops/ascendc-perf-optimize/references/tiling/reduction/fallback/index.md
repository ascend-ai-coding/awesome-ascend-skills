# Reduction 归约类 — 通用 Tiling 算法

> 归约类算子的通用 Tiling 建模：合轴 → 五模板选型 → UB 预算 → 多核切分。
> 适用 ReduceSum、ReduceMax、Softmax、LayerNorm、ArgMax 及同类 Norm 算子。

---

## 1. 适用算子

| 算子类型 | 沿 R 轴的典型计算 | 模板选型注意点 |
|---------|-----------------|--------------|
| ReduceSum / ReduceMax | 单次归约 | 标准 UB 预算；Sum 可选二分累加 |
| Softmax | max → exp → sum → div | 多遍流水，Recompute 需重读原数据 |
| LayerNorm / RMSNorm | mean → var → normalize | 两个关联统计量，分载时用 Welford |
| ArgMax / ArgMin | 归约 + 索引跟踪 | 启用 with_index 增强 |

---

## 2. 文档索引

| 文档 | 内容 |
|------|------|
| [tiling-flow.md](tiling-flow.md) | 五模板决策树、UB 预算公式、可选增强 |
| [tiling-fields.md](tiling-fields.md) | TilingData 字段语义 |
| [script/reduction_tiling.py](script/reduction_tiling.py) | 参考实现（简化版预算） |
| [example/](example/) | 实践案例（Softmax 等） |

---

## 3. 输入与输出

**输入**

- `shape` + `axes`：原始张量形状与归约轴
- `dtype`：FP32 / FP16 / BF16
- `op_type`：sum / max / softmax / norm / argmax 等（影响可选增强）
- 平台参数：`ub_size`（单核 UB 可用字节）、`core_num`（可用核数）

**输出**

- 选定模板（五选一）
- TilingData：形状参数、切分粒度、多核分配、尾块信息
- 可选增强标志：Group Reduce、Welford、二分累加、索引跟踪

---

## 4. 推导流程概览

```
Step 0  合轴 → (A1, R, A0)
Step 1  A0==1 ? AR 族 : ARA 族
Step 2  按 R 与 UB 容量选模板（SmallR / FullLoad / Recompute）
Step 3  计算 UB 切分粒度与多核分配
Step 4  （可选）启用 Group Reduce / Welford / 二分累加 / With-Index
```

详细决策树与公式见 [tiling-flow.md](tiling-flow.md)。

---

## 5. Agent 使用指南

为新的归约类算子（如 LayerNorm、RMSNorm）生成 Tiling 方案时：

1. **读通用算法** — [tiling-flow.md](tiling-flow.md) 中的五模板决策树
2. **分析算子数学** — 列出沿 R 轴需要的中间量（如 max、sum、var）及每步的 buffer 需求
3. **调整 UB 预算** — 在通用公式基础上，按算子的中间 buffer 数量修正分母/预留字节
4. **参考实践案例** — [example/softmax/experience.md](example/softmax/experience.md) 展示了 Softmax 四步流水如何映射到模板选型

---

## 6. 实践案例

| 算子 | 案例文档 | 说明 |
|------|---------|------|
| Softmax | [example/softmax/experience.md](example/softmax/experience.md) | 数学流程 → 模板映射 → 预算修正 |

---

## 7. 贡献新案例

在 `fallback/example/` 下创建算子名目录（如 `layernorm/`），描述该算子如何复用五模板及预算差异。
通用决策逻辑保持在 `fallback/` 根目录，案例目录只记录算子特有的数学映射与参数修正。
