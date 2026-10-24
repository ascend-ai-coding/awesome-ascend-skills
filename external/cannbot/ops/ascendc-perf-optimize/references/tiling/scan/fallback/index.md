# Scan / Cum 前缀扫描类 — 通用 Tiling 算法

> 前缀扫描类算子的通用 Tiling 建模：合轴 (M,R,N) → 模板选型（流式 / oneway·twoway × ub × core）→ UB 预算 → 保序进位多核切分。
> 适用 Cumsum、Cummin、Cummax、Cumprod、LogCumsumExp。

---

## 1. 适用算子

| 算子类型 | 沿 R 轴的 combine | identity | 附加输出 | 选型注意点 |
|---------|-----------------|----------|---------|-----------|
| Cumsum | 加法 | 0 | — | fp32 累加；大 R 二分进位保精度 |
| Cumprod | 乘法 | 1 | — | fp32 累加；溢出/下溢 |
| Cummax | max | -inf | argmax（并列取更晚） | **带索引 → 流式模板**；NaN 传播 |
| Cummin | min | +inf | argmin（并列取更晚） | 同上（本族实践案例） |
| LogCumsumExp | logaddexp | -inf | — | 在线 running max 稳定化 |

> **Scan 与 Reduce 的本质区别**：Reduce 沿 R 求 1 个统计量（可交换，分核合并顺序无关）；Scan 输出每个前缀位置（**有序**，跨核/跨块合并必须**保序传播进位**）。因此 Scan 多核切 R 时的进位不是"部分和相加"，而是"前段累积作为后段初值"，见 [tiling-flow.md §4 保序进位](tiling-flow.md#4-跨核-r-分段的保序进位scan-特有)。

---

## 2. 文档索引

| 文档 | 内容 |
|------|------|
| [tiling-flow.md](tiling-flow.md) | 模板选型决策树、UB 预算、保序进位多核 |
| [tiling-fields.md](tiling-fields.md) | TilingData 字段语义 |
| [script/scan_tiling.py](script/scan_tiling.py) | 参考实现（流式选块 + 跨核借 R 判定） |

---

## 3. 输入与输出

**输入**：`shape` + 扫描轴 `dim`、`dtype`（FP32/FP16/BF16/INT32）、`op_type`（sum/prod/min/max/…，影响 combine 与是否带索引）、平台参数（`ub_size`、`core_num`）。

**输出**：选定模板；TilingData（形状 M/R/N、切分粒度 mBlk/nTileLen 或 borrow/fold、多核分配、尾块）；是否带索引跟踪；是否跨核借 R（保序进位）。

---

## 4. 推导流程概览

```
Step 0  合轴 → (M, R, N)：dim 前维度乘积=M，dim 后维度乘积=N（dim=-1 → N=1；dim=0 → M=1）
Step 1  带索引？(Cummin/Cummax) → 流式模板；否则 Cumsum/Cumprod → oneway/twoway Sklansky
Step 2  按 N 宽窄 + R 与 UB 关系选内层（oneway / twoway 折叠 / 是否 ub chunk）
Step 3  并行度判定：M×N 切块能否用满核？
          够 → 切 M/行（流式/oneway/ub）
          不够 + R 超长 → 跨核借 R（core_partition，保序进位）
Step 4  计算 UB 切分与多核分配（blockDim）
```

详见 [tiling-flow.md](tiling-flow.md)。

---

## 5. Agent 使用指南

为新的 Scan 算子生成 Tiling 方案：

1. **读通用算法** — [tiling-flow.md](tiling-flow.md) 的选型决策树。
2. **确定 combine/identity** — 见 §1 表；带索引算子走流式模板并明确并列语义（torch = last occurrence）。
3. **判并行度** — `M×N` 切块的 work-item 数 vs 核数：够则切行，不够且 R 超长走跨核借 R。
4. **参考模板使用指南** — best-practices 的 [usage_guide.md](../../../../../../ascendc-performance-best-practices/references/scan/templates/usage_guide.md)（含模板集成、选型实测轨迹与已知问题）。

---

## 6. 贡献新案例

在 `fallback/example/` 下创建算子名目录（如 `cumsum/`），记录该算子的 combine 映射、并列语义、UB 预算差异与实测选型。
