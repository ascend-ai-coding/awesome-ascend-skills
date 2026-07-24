---
id: P93
bottlenecks: [mte2_stall, bus_contention]
op_families: [matmul]
complexity: L1
conflicts_with: []
synergizes_with: [P46, P86, P92]
has_preconditions: true
has_playbook: true
---

# P93: Matmul K 轴错峰访问内存 (Matmul K-Axis Staggered Memory Access)

## 核心思想
多核执行 Matmul 时，若输入矩阵 A 或 B 的 GM 地址相同，多核同时访问相同 GM 地址会导致地址冲突，MTE2 搬运效率降低。使能 K 轴错峰访问后，不同核从 K 方向不同起始位置开始搬运数据，缓解多核同时访问同一 GM 地址的冲突，提升 MTE2 带宽利用率。

## 代码骨架

```cpp
// === 改造前（多核同地址冲突）===
// 多个核同时从相同 GM 地址搬运 A/B 矩阵
// GM_to_L1 带宽利用率仅 34.4%
Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> mm;

// === 改造后（K 轴错峰访问）===
constexpr MatmulConfig GetMDLKDimReorderConfig() {
    auto CFG = CFG_MDL;
    CFG.enableKdimReorderLoad = true;  // 使能 K 轴错峰
    return CFG;
}
constexpr static MatmulConfig MM_CFG = GetMDLKDimReorderConfig();

Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> matmulObj;
// MTE2 耗时：90us → 69.87us（-22%）
// GM_to_L1 带宽利用率：34.4% → 41.7%（+21%）
// 算子总耗时：98.72us → 85.68us（提升 13.2%）
```

## 关键修改点

1. 使用 `CFG_MDL` 模板（必须，仅 MDL 支持 K 轴错峰）
2. 在 MatmulConfig 中设置 `enableKdimReorderLoad = true`
3. K 轴必须非全载（数据无法一次全部搬入 L1）
4. 多核并行执行 Matmul 时效果最佳

## 常见陷阱

⚠️ 仅支持 MDL 模板，Norm/NBuffer33 等模板不支持
⚠️ K 轴需非全载，全载场景无效果
⚠️ 必须在多核上执行 Matmul，单核场景无意义
⚠️ 与 P64（通用 GM 地址冲突规避）互补：P64 针对 Vector 算子，本策略专门优化 Matmul K 轴
