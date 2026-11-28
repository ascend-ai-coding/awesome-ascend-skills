---
type: "PyPTO Performance Optimization Card"
title: "门控 / 超越函数加权收缩的等价改写（逐元素实现改为 cube GEMM）"
description: "关键路径上的门控加权收缩，经参考点差分分解把逐元素超越函数改写为 cube GEMM，并将递推状态片上常驻，在不改变数学结果的前提下转移工作与依赖。"
status: "stable"
tags: ["pypto-pro", "algorithm-reformulation", "gated-contraction", "gemm", "resident-state"]
item_id: "algo-01"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/gemm.md#thin-row-contractions-compare-arithmetic-and-transport-paths"
      scope: "在保持数学与数值合同的前提下，把可分离的加权收缩作为 VF 与 Cube 路径比较对象。"
      boundary: "目标只提供核心与运输路线的比较原则，不证明本卡权重可分离；指数范围、阶段重排与工作量变化须独立推导。"
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "document:skill/api/numerics.md#4-fixed-references-in-bounded-exponential-normalization"
      scope: "选择指数参考点前推导全输入域的中间量范围、下溢、上溢及最终误差。"
      boundary: "归一化中的公共平移抵消不自动适用于本卡未归一化乘积；两侧指数、乘积、低精度操作数及累加误差均须另证。"
bound_hint: "VEC、MAC、MTE2"
applicability: "关键路径存在门控或超越函数加权的 (查询 × 键) 型收缩，当前以 Vector 逐元素 exp 与乘累加实现，呈 VEC bound 或逐元素依赖链；且逐元素权重可按差值分解为行项与列项之积，并存在参考点使分解后的指数中间量落在可表示范围内"
target_api_gate: "依赖已核验的 exp、逐元素乘、转置与 cube matmul（含批量）及其 dtype 组合；分解后的数值范围与操作数 dtype 须在目标版本核验，其它 SoC 或版本须重新核验"
---

# 技术卡片 algo-01：门控 / 超越函数加权收缩的等价改写（逐元素实现改为 cube GEMM）

本卡归属结构性优化的“计算图等价改写”维度：不改变算子的数学结果，仅改变在哪个计算单元、按何种数据流图计算。其收益通常高于同结构内的调度微调。

## 诊断特征

1. 关键路径存在 (查询 × 键) 型收缩，其权重为逐元素门控或超越函数（如 `exp(Δg)`）。
2. 该收缩当前以 Vector 逐元素 `exp` 与乘累加实现，规模 O(N²·D)，呈 VEC bound 或逐元素依赖链。
3. 逐元素权重可按差值分解为行项与列项之积。
4. profiler 显示各 pipe 利用率均低、无 dominant bound，或当前实现与已验证结构存在数量级差距。

## 不适用条件

1. 逐元素权重无法分解为行项与列项之积。
2. 分解后的指数中间量超出可表示范围，且以参考点或 `clamp` 无法约束。
3. 收缩规模过小，收益不足以抵消 cube 固定开销。
4. 精度门不接受分解引入的额外舍入。

## 原理

设加权收缩 `S[r,c] = Σ_d x[r,d]·y[c,d]·w(r,c,d)`。当权重可分离、即 `w(r,c,d) = f(r,d)·g(c,d)` 时，令
`A[r,d] = x[r,d]·f(r,d)`、`B[c,d] = y[c,d]·g(c,d)`，则 `S = A·Bᵀ`。逐元素权重与超越函数的计算量由
O(N²) 降至 O(N)，O(N²) 的收缩迁移至 cube 的 MAC 单元。

门控场景下 `w(r,c,d) = exp(g[r,d] − g[c,d])`，取逐维参考 `ref[d]` 后：

```
exp(g[r,d] − g[c,d]) = exp(g[r,d] − ref[d]) · exp(ref[d] − g[c,d])
```

即 `f(r,d) = exp(g[r,d] − ref[d])`、`g(c,d) = exp(ref[d] − g[c,d])`。参考点须使每块 `|g − ref|` 落在指数
可表示范围内。消去重复超越函数、合并逐元素链，同属本维度。

## 改法（before / after）

改法分两步：其一，将收缩改写为两侧缩放与一次 cube GEMM；其二，将递推状态片上常驻、输出展平，以
消除 GM 往返。以门控收缩（O(N²)）为例。

`结构示意`（仅表达数据流，缺 tile 与同步上下文，不可直接编译）：

before —— 逐 (行, 列) 逐元素计算权重与收缩，超越函数量 O(N²)，全部落在 Vector；递推状态每步经 GM 往返。

```
for r in range(N):                       # 行
    for c in range(N):                   # 列
        s = 0
        for d in range(D):
            s += x[r,d] * y[c,d] * exp(g[r,d] - g[c,d])   # 逐元素超越函数
        S[r,c] = s
for t in range(steps):                   # 递推: 状态每步经 GM 往返
    st = load_state_gm(t - 1)
    st = st + update(S[t])
    store_state_gm(t, st)
```

after —— 两侧各做一次 O(N·D) 缩放，收缩迁移为一次 cube GEMM；递推状态片上常驻，输出由扁平网格一次算出。

```
ref = <逐维参考点, 取块内累计量>          # 约束 |g - ref| 落在 exp 可表示范围
A = x * exp(g - ref)                     # 行项缩放, O(N·D), VEC
B = y * exp(ref - g)                     # 列项缩放, O(N·D), VEC
S = matmul(A, B^T)                       # O(N²·D) 收缩迁移至 cube, MAC
for t in range(steps):                   # 递推相位: 状态片上常驻, 不经 GM
    state = state + update(S[t])
# 输出相位: 扁平 (chunk, head) 网格一次算出, 铺满核
```

关键约束：

- 逐元素权重须可分解为行项与列项之积（本例经参考点差分分解得到）；否则不适用。
- 参考点取块内累计量，约束每块 `|Δg|` 落在指数可表示范围；超范围时回退逐元素实现，或分块参考并 `clamp`。
- 状态片上常驻与输出展平为配套改造；否则收缩加速后，状态 GM 往返仍位于关键路径（见 `mem-06`、`pipe-06`）。

## 性能与验证

- 正确性：分解引入的舍入为主要风险，逐 case 对照 Golden，须落在 SPEC 精度门内。
- 机制核对：超越函数计数下降，MAC 占比与核占用上升。
- 收益：以 target case 的 device `Task Duration` 为准。

## 技术限制与风险

- **数值范围**：分解后 `exp(ref − g)` 可能上溢，须以参考点约束每块 `|Δg|`；参考点越粗、块越小，收益越低。
- **精度**：一次 `exp` 变为两次 `exp` 与一次乘，累加仍为 FP32，相对误差略增，须对照精度门。放宽精度门属
  独立的合同决策，不得与改写捆绑。
- **阶段划分**：状态片上常驻与输出展平需重排 tile 与同步，跨相位依赖须单独同步；改动面大于局部微调，
  易触发死锁，须预留多轮编译、上板与修复预算。
- **能力核验**：依赖的 `exp`、转置与批量 matmul 的 dtype 组合须按目标版本核验；未核验前仅作结构示意，
  不得作为可运行实现。

## 机制类负结果的处理

机制成立的改写，首次实测为负通常系某一实现变体未达预期，不足以判定机制不成立。应迭代实现变体（dtype、
块宽、参考点、融合点、整合位置）后再结论；仅在可复现“机制在当前工具链或合同下不可用”的最小复现时
方可关闭。放宽精度门或支持范围属合同变更，须独立向用户确认。

## 参考资料

- [结构维度判定](../../workflows/structural-review.md)；
- 配套维度：`../mem/mem-06-resident-state-eliminate-gm-roundtrip.md`、
  `../pipe/pipe-06-flat-2d-work-grid.md`、`../cube/cube-09-bf16-operand-downcast.md`。
