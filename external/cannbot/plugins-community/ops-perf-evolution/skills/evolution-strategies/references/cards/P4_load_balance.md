---
id: P4
bottlenecks: [tiling_imbalance]
op_families: [matmul, moe, quantization]
complexity: L0
conflicts_with: []
synergizes_with: [P51]
has_preconditions: true
has_playbook: true
---

# P4: 多核负载均衡 (Multi-core Load Balancing)

## 核心思想
将数据均匀分配到所有 NPU 核，避免某些核空闲、某些核过载。根据数据特征选择分核策略。

## 代码骨架
```cpp
// === 改造前（简单均匀分）===
uint32_t rowsPerCore = totalRows / coreNum;  // 行数均匀分
for (uint32_t core = 0; core < coreNum; core++) {
    // 每核处理 rowsPerCore 行
}
// 问题：如果各行计算量不同（如稀疏矩阵），均匀分反而不均衡

// === 改造后（代价感知分核）===
// 方式 A: 按计算代价分（非均匀数据）
uint32_t GetCoreIdByCost(uint32_t rowIdx) {
    // 根据每行的非零元素数/计算量分配核
    return costPrefixSum[rowIdx] * coreNum / totalCost;
}

// 方式 B: 按数据大小分（变长序列）
uint32_t GetRowsPerCore(uint32_t coreId) {
    // 动态计算每核处理的行数，使各核数据量均衡
    return (coreId < remainder) ? (rowsPerCore + 1) : rowsPerCore;
}

// 方式 C: 对角线调度（MoE/矩阵乘场景）
// 将 M×N 网格按对角线分配给不同核，避免热点
uint32_t coreM = blockIdx % coreNumM;  // M 维度分核
uint32_t coreN = blockIdx / coreNumM;  // N 维度分核
```

## 关键修改点
1. **分析数据分布**: 数据是否均匀？是否有热点？
2. **选择分核策略**:
   - 数据均匀 → 简单均匀分
   - 数据不均匀（稀疏/变长）→ 代价感知分
   - 2D 数据（矩阵乘）→ 对角线/M×N 分核
3. **修改分核逻辑**: 通常是 tiling 文件中的 `SfaSplitCore` 或 `GetBlockNum` 相关代码
4. **验证均衡性**: 各核的计算量/数据量差异是否 < 20%

## 常见陷阱
- ❌ **只改 MAX_SPLIT_SIZE 不改分核逻辑** → 分核策略不变，只是阈值变了
- ❌ **忽略数据分布特征** → 对稀疏数据用均匀分，热点核过载
- ❌ **分核后没有重新验证边界** → 尾块处理错误

## 代码搜索关键词
```bash
grep -n "GetBlockNum\|coreNum\|blockIdx\|aiCoreIdx" *.cpp *.h  # 找分核逻辑
grep -n "SplitCore\|SfaSplit\|rowsPerCore\|colsPerCore" *.cpp  # 找分核函数
grep -n "tiling.*core\|core.*tiling" *.cpp *.h                # 找 tiling 中的分核
grep -n "remainder\|mod\|/ coreNum" *.cpp                     # 找均匀分代码
```
