# 优化模式快速参考

## 模式选择决策树

```
第一步：检查标量计算残留（最优先，收益最大）
  内核中有逐元素 for 循环？
    ├─ 是 → 标量归约 (sum/max 循环)？ → scalar_to_vector.md 模式 1 (2-5x)
    ├─ 是 → 标量逐元素算术 (y[i]=f(x[i]))？ → scalar_to_vector.md 模式 2 (3-10x)
    ├─ 是 → 标量排序/选择 (bubble/selection sort)？ → scalar_to_vector.md 模式 3 (5-100x)
    ├─ 是 → 逐元素 gather/scatter？ → scalar_to_vector.md 模式 4 (2-5x)
    ├─ 是 → 逐行标量 Muls/Adds？ → scalar_to_vector.md 模式 6 (BroadCast + 批量, gRows/2x)
    ├─ 是 → 逐元素 SetValue/GetValue？ → scalar_to_vector.md 模式 7 (GatherMask, 5-20x)
    └─ 是 → 其他 → scalar_to_vector.md 模式 5 (混合调度, 避免退化)
  └─ 无标量循环 → 继续常规优化

第二步：常规优化决策
  内核是否内存密集？
    ├─ 是 → 已有双缓冲？
    │       ├─ 否 → double_buffering.md (20-80% 提升)
    │       └─ 是 → tiling_strategies.md (10-50%) + pipeline_overlap.md (5-30%)
    └─ 否（计算密集或平衡）
        ├─ 有因果/掩码逻辑？ → causal_block_skip.md (20-50%)
        ├─ 有跨步访问？ → memory_coalescing.md (10-40%)
        └─ 以上都不是 → 需要算法级优化，转 algorithm_insights/
```

## 模式速查表

| 模式 | 文件 | 适用场景 | 典型提升 |
|------|------|---------|---------|
| 标量计算向量化 | `scalar_to_vector.md` | 所有含标量 for 循环的内核 | 2-11x geomean |
| 双缓冲 | `double_buffering.md` | 所有内存密集型内核 | 20-80% |
| 自适应分块 | `tiling_strategies.md` | 可变形状算子 | 10-50% |
| 因果块跳过 | `causal_block_skip.md` | Attention + causal mask | 20-50% |
| 流水线重叠 | `pipeline_overlap.md` | 已有双缓冲的内核 | 5-30% |
| 内存合并 | `memory_coalescing.md` | 跨步访问模式 | 10-40% |
