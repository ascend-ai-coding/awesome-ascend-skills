---
id: P67
bottlenecks: [scalar_compute, scalar_loading]
op_families: [broadcast_mask, elementwise]
complexity: L1
conflicts_with: [P84]
synergizes_with: [P55, P68, P69]
has_preconditions: true
has_playbook: true
---

# P67: Vector Counter 模式 (Vector Counter Mode)

## 核心思想
Normal 模式下 Vector API 需手动计算 repeatTimes、区分主块/尾块、多次设置 mask，涉及大量 Scalar 计算。Counter 模式下只需 SetMaskCount() 后设置 mask 为总元素个数，一次 API 调用完成所有计算，无需主尾块判断，大幅减少指令数量和 Scalar 计算量。

## 代码骨架

// === 改造后（专家模式）===
```cpp
// Normal 模式（反例）：手动计算 repeatTimes 和 mask
uint32_t repeatTimes = dataSize / ONE_REPEAT_SIZE;
uint32_t tailSize = dataSize % ONE_REPEAT_SIZE;
Add(dst, src1, src2, FULL_MASK, repeatTimes, {...});
if (tailSize > 0) {
    SetVectorMask(tailMask);
    Add(dst[offset], src1[offset], src2[offset], tailMask, 1, {...});
}

// Counter 模式（正例）：一次调用完成
SetMaskCount();
SetVectorMask<MaskMode::COUNTER>(ELE_SIZE);  // 总元素个数
Add(dst, src1, src2, MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
ResetMask();
```

## 关键修改点

1. 预期收益: 大幅简化代码，降低 Scalar 和 Vector 计算耗时，减少指令条数

## 常见陷阱

⚠️ 需要硬件支持 Counter 模式
⚠️ 对极小数据量可能无明显收益
