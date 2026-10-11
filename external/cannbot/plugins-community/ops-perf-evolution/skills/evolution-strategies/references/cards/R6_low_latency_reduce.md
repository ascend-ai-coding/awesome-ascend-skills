---
id: R6
bottlenecks: []
op_families: [datapath]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: false
has_playbook: false
deprecated: true
deprecated_reason: "Content fully duplicated by P68 (Low-Latency Reduction Instruction Combination). Zero references."
---

# R6: 低延迟归约

## 核心思想
选择最优的归约指令组合，减少 Reduce 类操作的延迟。

## 代码骨架

// === 改造后（专家模式）===
```cpp
// Baseline: 单次 WholeReduceSum（高延迟）

// Evolved: 先按 VL 粒度累加，再全局归约
RegTensor<float> accReg;
MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
Duplicate(accReg, 0.0f);

for (uint16_t i = 0; i < repTimes; ++i) {
    MaskReg mask = UpdateMask<float>(count);
    LoadAlign(srcReg, srcAddr + i * oneRepSize);
    Add(accReg, accReg, srcReg, mask);  // VL 粒度累加
}

// 最终归约: VL → 标量
StoreAlign(workAddr, accReg, fullMask);
// 在 VF 外部用高阶 API 完成最终归约
```

## 常见陷阱

⚠️ 收益**: 归约延迟降低 2-3x
⚠️ 风险**: 需要额外 RegTensor 做累加器
⚠️ 复杂度**: 中等
