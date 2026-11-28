# Reduction / Normalization 族策略选型指南

> 算子族：`reduction` / `normalization` / `softmax`
> 适用算子：`ReduceSum`, `ReduceMax`, `ReduceMean`, `LayerNorm`, `RMSNorm`, `Softmax`, `LogSoftmax`, `GroupNorm` 等
> 关联设计：[knowledge-strategy-architecture-v3.2](../../../../../docs/design/knowledge-strategy-architecture-v3.2.md) §3.2

## 选型决策树

```
归约维度大小？
├── 小维 (≤ 128) → 单核内 reduce + 树状归约 (P68)
├── 中维 (128 - 2048)
│   ├── 沿 inner 轴 → BlockReduceMax/Sum (P55) 单指令归约
│   └── 沿 outer 轴 → 多核切分 + workspace 归约 (P40)
└── 大维 (> 2048) → 两轮 reduce (Welford 算法 A2) + double buffer (P1)

是否数值稳定性敏感？
├── 是 (Softmax / LogSoftmax) → A5 max-trick + A1 FP32 中间计算
└── 否 (ReduceSum / Mean) → 直接 reduce 即可
```

## 族特定策略

> ⚠️ **当前状态**：根据 INDEX.json 筛查，reduction 族当前**没有专属 op_families 标签**。所有 reduction 类策略都打成了 `omni`（通用）或 `softmax/attention`（更窄）。
>
> **运维 TODO**（不在本 Phase 范围）：扫描以下策略，把 `omni` 中明显偏 reduction 类型的（如 P68 低延迟归约）改打 `reduction` 标签：
> - P68 低延迟归约
> - A1 FP32 中间计算
> - A2 Welford 算法（如果有对应策略卡）

## 推荐组合

| 组合 | 适用 | 说明 |
|---|---|---|
| **P1 + P5 + P10** | 标准 RMSNorm / LayerNorm | 双缓冲 + 流水同步 + 向量化拷贝（最基础） |
| **A5 + A1 + P5** | Softmax / LogSoftmax | 数值稳定基础组合（max-trick + FP32 中间 + 同步） |
| **P55 + P67** | inner 轴归约的 attention/norm | BlockReduceMax 单指令替代 + Vector Counter |
| **P68 + Welford** | 大维归约的精度敏感算子 | 树状归约 + 两轮算法（避免单次溢出） |

## 涉及的通用策略（omni）

参见 attention_guide.md 中的 P1/P5/P10 等 L0 通用策略——这些对 reduction 族同样适用。

## Playbook 状态

- ✅ P1, P5, P10 已有 Playbook（reduction 族最常用三件套）
- ❌ A1, A5, P68 待 Phase C 补

---

## Ascend C Reduction Tiling 系统化设计

> 来源：插件本地 [`tiling-design/reduction/patterns.md`](../tiling-design/reduction/patterns.md)、[`algorithms.md`](../tiling-design/reduction/algorithms.md)
>
> 本章节沉淀 reduction 族在 Ascend C 上的 Tiling 设计方法论，供进化优化时直接引用。

### 1. 合轴（A/R 标记）

将 N 维 shape + axes 简化为 A/R 交替序列：

1. **消除冗余维度**：size=1 且内存连续的维度消除。
2. **合并相邻同类型轴**：相邻维度都是 A（或都是 R）时乘积合并。

**标记示例**：

| 原始 shape | axes | 标记 | 合轴后 | 模式 |
|---|---|---|---|---|
| [2,100,4] | [1,2] | A,R,R | [2,400] = (A,R) | AR |
| [2,3,4,5,6,7,8] | [1,2,4] | A,R,R,A,R,A,A | [2,12,5,6,56] = (A,R,A,R,A) | 多轴归约 |

### 2. 场景决策树

```
给定: shape, axes
Step 0: 合轴 → 标记 A/R → 消除冗余维 → 合并相邻同类型轴
Step 1: 单轴还是多轴？
  ├─ 单轴（AR 或 ARA）→ Step 2
  └─ 多轴（ARAR...）→ Shape 三步变换 → 每个 R 按 Step 2 处理
Step 2: A0 决定模式
  ├─ A0 = 1 → AR 模式（尾轴归约）
  │   ├─ UB 可放整行 R → AR-FullLoad
  │   └─ 否则 → AR-ColSplit
  └─ A0 > 1 → ARA 模式（非尾轴归约）
      ├─ UB 可放 R*tileA0Len（32B 对齐）→ ARA-FullLoad
      └─ 否则 → ARA-RowSplit
```

### 3. AR 与 ARA 模式要点

**AR 模式**：每行 R 个元素连续，使用 Level 2 Reduce API。

```cpp
uint32_t alignedCols = ((tileA0Len * sizeof(float) + 31) / 32) * 32 / sizeof(float);
uint32_t srcShape[] = {R, alignedCols};
ReduceMax<float, Pattern::Reduce::RA>(resultLocal, xLocal, tmpLocal, srcShape, true);
```

**ARA 模式**：`[A1, R, A0]`，沿 A1 切分后每次处理 `[R, A0_inner]` 块，整块搬入 UB 后用 `Pattern::Reduce::RA`。

| 分支 | 条件 | API / 说明 |
|---|---|---|
| FullLoad | UB 能容纳全部归约数据 | 一次 CopyIn + Reduce，中间结果复用 |
| ColSplit / RowSplit | UB 放不下全部数据 | 分 chunk 归约，跨 chunk 用 Max/Add 合并 |

### 4. 多轴归约

合轴后为 A/R 交替序列（如 ARARA）时，使用 [multi-axis-transform.md](../tiling-design/reduction/multi-axis-transform.md) 提供的 Shape 三步变换，把问题展开为嵌套循环，每个 R 维单独按 AR/ARA 处理。

### 5. 算法选择

| 条件 | 推荐算法 | 典型算子 |
|---|---|---|
| FullLoad | 直接顺序计算 | reduce_sum/max |
| FullLoad + 两次相关归约 | TwoPass | reduce_var/std, layer_norm |
| 分载 + 两次相关归约 | Welford Online | reduce_var/std |
| 分载 + 单核处理不完 R 且 A 小 | Group Reduce | arg_max, reduce_var |
| Sum 精度敏感 | 二分累加 | reduce_sum, reduce_var |

### 6. rLength vs rLengthAlign 使用对照

| 参数位置 | rLength（有效数据） | rLengthAlign（对齐后） |
|---|---|---|
| DataCopyPad blockLen | ✅ | ❌ |
| Reduce API count | ✅ | ❌ |
| UB 内 rowOffset 计算 | ❌ | ✅ |
| Buffer 大小分配 | ❌ | ✅ |

### 7. 全局归约

`reduce_sum(axes=所有轴)` / `reduce_max(axes=所有轴)` 时，所有元素按核均分，各核独立归约后通过 AtomicAdd 或显式 Merge 两阶段合并。

### 8. 设计检查清单

- [ ] 是否已完成合轴并得到 A/R 交替序列？
- [ ] 是否正确区分 AR（A0=1）与 ARA（A0>1）？
- [ ] 是否按 UB 容量选择 FullLoad / Split？
- [ ] `rLengthAlign` 是否用于 buffer 分配与 offset 计算？
- [ ] 多轴归约是否走 `multi-axis-transform.md` 三步变换？
- [ ] 精度敏感场景是否启用 Welford / 二分累加？
