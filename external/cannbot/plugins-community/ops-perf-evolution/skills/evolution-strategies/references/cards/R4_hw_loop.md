---
id: R4
bottlenecks: []
op_families: [datapath]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: false
has_playbook: false
deprecated: true
deprecated_reason: "Minimal content (uint16_t vs uint32_t loop). Overlaps with P67 Counter Mode. Zero references in lookup tables."
---

# R4: Hardware Loop 规范化

## 核心思想
确保 VF 内的循环满足 Hardware Loop 编码要求，避免退化为 Software Loop。

## 代码骨架

// === 改造后（专家模式）===
```cpp
// ❌ Software Loop: uint32_t 类型
for (uint32_t i = 0; i < bound; i++) { ... }

// ✅ Hardware Loop: uint16_t 类型
for (uint16_t i = 0; i < (uint16_t)bound; i++) { ... }
```

## 常见陷阱

⚠️ 收益**: Hardware Loop 比 Software Loop 快 2-5x（减少分支预测开销）
⚠️ 风险**: 代码可读性略降（for(1) 替代 if）
⚠️ 复杂度**: 低（机械替换）
