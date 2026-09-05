---
id: R5
bottlenecks: []
op_families: [datapath]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: false
has_playbook: false
deprecated: true
deprecated_reason: "Uses fictional UnalignRegForLoad API. Overlaps with P7 (DataCopyPad) and P11 (Tail block). Zero references."
---

# R5: 非对齐访问优化

## 核心思想
使用 UnalignRegForLoad/Store 优化非 32B 对齐地址的数据访问。

## 代码骨架

// === 改造后（专家模式）===
```cpp
// Baseline: 对齐加载 + 手动拼接
// (需要两次 LoadAlign + Shift 操作)

// Evolved: 使用 UnalignReg
UnalignRegForLoad ureg;
LoadUnAlignPre<float>(ureg, srcAddr);  // 预初始化（循环外）

for (uint16_t i = 0; i < repTimes; ++i) {
    LoadUnAlign<float, POST_MODE_UPDATE>(srcReg, ureg, srcAddr, stride);
    // srcAddr 自动更新
    Adds(dstReg, srcReg, scalar, mask);
    StoreAlign(dstAddr + i * repSize, dstReg, mask);
}
```

## 常见陷阱

⚠️ 收益**: 减少非对齐场景的访问次数
⚠️ 风险**: UnalignReg 资源有限（每 VF 最多 4 Load + 4 Store）
⚠️ 复杂度**: 需要 Pre/Post 配对调用
