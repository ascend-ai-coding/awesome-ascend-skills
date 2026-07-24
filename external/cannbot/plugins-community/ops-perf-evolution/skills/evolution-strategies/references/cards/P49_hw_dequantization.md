---
id: P49
bottlenecks: [compute_bound]
op_families: [matmul, moe, quantization]
complexity: L1
conflicts_with: [P70]
synergizes_with: [P48]
has_preconditions: true
has_playbook: true
---

# P49: 硬件加速反量化 (Hardware-Accelerated Dequantization)

## 核心思想
用 AscendDequant 硬件融合指令（int32→float32 * scale 一步完成）+ Brcb 广播指令替代软件循环反量化，将 O(M) 次 Vector 操作压缩到 O(1) 次硬件指令。

## 代码骨架

// === 改造前（基线）===
```cpp
Cast(fp32Local, inLocal_, CAST_NONE, tileSize_);  // int32→float32
for (int i = 0; i < subBlockM_; i++) {
    Mul(fp32Local[i*N], fp32Local[i*N], scale[i*N], N);   // per-channel scale
    Muls(fp32Local[i*N], fp32Local[i*N], tokenScale[i], N); // per-token scale
}
// 总计: 1 + M + M = 2M+1 次 Vector 操作
```

// === 改造后（专家模式）===
```cpp
// 单条硬件指令完成反量化
AscendDequant(dequantResult, mmOut, scale, tmpLocal,
              {curVecBaseM, alignBaseN, curVecBaseN});

// Brcb 广播 per-token scale（替代逐行 Muls 循环）
Brcb(scaleLocal, perTokenScaleGm[offset], 1, 1, 0, 0);
Mul(output, dequantResult, scaleLocal, totalSize);
```

## 关键修改点

1. 预期收益: Vector 指令从 O(M) 降到 O(1)，延迟减少 20-40%

## 常见陷阱

⚠️ AscendDequant 需要特定硬件支持（910B+）
⚠️ scale 编码格式（UINT64）需要预处理
⚠️ SetQuantVector 仅支持 Cube 侧融合，部分场景仍需 Vector 后处理

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue" op_kernel/*.cpp op_host/*_tiling.cpp
```
