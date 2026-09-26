---
id: R2
bottlenecks: []
op_families: [datapath]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: false
has_playbook: false
deprecated: true
deprecated_reason: "Uses fictional RegTensor API. Register allocation is compiler-managed, not a manual kernel optimization strategy."
---

# R2: 寄存器复用

## 核心思想
通过布尔代数简化、指令重排和原地复用减少 RegTensor 使用量，避免超过 32 个限制导致的 spill/fill。

## 代码骨架

// === 改造后（专家模式）===
```cpp
// Baseline: 5 个 RegTensor
RegTensor<float> a, b, c, d, result;
Add(c, a, b, mask);
Mul(d, c, a, mask);
Add(result, d, b, mask);

// Evolved: 3 个 RegTensor
RegTensor<float> a, b, tmp;
Add(tmp, a, b, mask);
Mul(tmp, tmp, a, mask);   // 原地复用 tmp
Add(tmp, tmp, b, mask);   // 继续复用
```

## 关键修改点

1. 预期收益: **: RegTensor 从 5 减少到 3
**Trade-off**: tmp 的依赖链变长，可能影响双发射

## 常见陷阱

⚠️ 收益**: 消除 spill/fill 指令，性能提升 10-30%
⚠️ 风险**: 过度复用可能引入数据依赖，阻碍双发射
⚠️ 复杂度**: 需仔细分析寄存器生命周期
