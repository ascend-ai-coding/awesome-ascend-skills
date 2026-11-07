# Convolution 族 — 兜底算法（FORMULAS）

> Conv2D / DepthwiseConv 官方参考 Tiling 算法。
>
> 按切分模式分为 **Mmode**（M-split）和 **HWmode**（HW-split）两个子算法。

---

## 1. 路由流程

```
给定：算子名 + Shape (N, Ci, Hi, Wi, Co, Kh, Kw) + dtype + attributes

Step 0 — 算子分类：
  ├─ Conv2D → 进入 Step 1
  ├─ DepthwiseConv → 进入 Step 1（groups = Ci = Co）
  └─ 其他 → 返回「暂未收录」

Step 1 — Group 决策：
  ├─ groups=1 → 普通卷积
  ├─ groups>1, opt-group weight 能装入 UB → OPT_GROUP
  └─ groups>1, opt-group 不满足 UB 约束 → ORI_GROUP

Step 2 — 切分模式决策：
  ├─ outputOrder = M → M-split
  └─ outputOrder = HW → HW-split

Step 3 — 算法路由：
  ├─ M-split → Mmode（M 优先 tiling）
  └─ HW-split → HWmode（HW 优先 tiling）
```

---

## 2. 子算法

| 子算法 | 切分模式 | L1 tiling 维度 | L0 tiling 维度 |
|--------|---------|---------------|---------------|
| **Mmode** | M-split | mAL1 × nBL1 × kAL1/kBL1 | mL0 × kL0 × nL0 |
| **HWmode** | HW-split | hoAL1 × woAL1 × nBL1 × kAL1/kBL1 | hoL0 × woL0 × kL0 × nL0 |

---

## 3. 索引

| 文档 | 内容 |
|------|------|
| [tiling-flow.md](tiling-flow.md) | Tiling 生成七步流程 + 代价模型 + TilingKey 编码 |
| [tiling-fields.md](tiling-fields.md) | L1/L0/UB/Scalar 各阶段字段语义 |
| [script/conv_tiling.py](script/conv_tiling.py) | Python 公式化 Tiling 脚本（支持普通卷积 & Depthwise） |
