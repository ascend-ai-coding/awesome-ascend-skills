---
id: P92
bottlenecks: [mte2_stall, compute_bound]
op_families: [matmul, cv_fusion]
complexity: L1
conflicts_with: []
synergizes_with: [P46, P86, P93]
has_preconditions: true
has_playbook: true
---

# P92: Matmul 基本块计算访存比优化 (Matmul Base Block Compute-to-Memory Ratio)

## 核心思想
Matmul 基本块参数 `[baseM, baseN, baseK]` 直接影响 Cube 计算访存比和 MTE2 搬运效率。小基本块（如 64×64）计算密度低、搬出地址可能非 512B 对齐；大基本块（如 128×256×64）计算密度高、搬运总量小。搬运总量公式：`totalCnt = (N/baseN)*M*K + (M/baseM)*K*N`。优化目标是最大化 `计算访存比 = Cube cycle / 搬运数据量`。

## 代码骨架

```cpp
// === 改造前（小基本块，计算访存比低）===
int32_t baseM = 64;   // Cube cycle = 512, 搬运 = 64KB
int32_t baseN = 64;   // 搬出偏移 64*4=256B，非 512B 对齐
tilingApi.SetFixSplit(baseM, baseN, -1);
// 计算访存比：512cycle / 64KB = 低

// === 改造后（大基本块，计算访存比高）===
int32_t baseM = 128;  // Cube cycle = 512, 搬运 = 48KB
int32_t baseN = 256;  // 搬出偏移 256*4=1024B，512B 对齐
tilingApi.SetFixSplit(baseM, baseN, -1);
// baseK 由 tiling 自动计算（此处为 64）
// 计算访存比：512cycle / 48KB = 提升 1.33×
// MTE2 耗时实测：2452us → 808us（3 倍提升）

// 在此基础上进一步使能 MDL 大包搬运
Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> mm;
// MTE2 耗时：808us → 591us，带宽利用率 2491GB/s → 3406GB/s（+36%）
```

## 关键修改点

1. 根据 M/N/K 上限选择大于等于当前最大 shape 的基本块大小
2. 确保 `baseN * sizeof(dtype)` 为 512B 的整数倍，避免带宽损失
3. 使能 `CFG_MDL` 模板配合大基本块，进一步提升带宽利用率
4. 对于小 shape（M/N/K 不足），退化到默认基本块

## 常见陷阱

⚠️ 大基本块需要更多 L1/L0 空间，小 shape 不适用
⚠️ 搬出地址非 512B 对齐会损失 30%+ 带宽
⚠️ `baseN` 为奇数时搬出地址可能不满足对齐要求
⚠️ 与 P93（K 轴错峰访问）配合使用效果更佳
