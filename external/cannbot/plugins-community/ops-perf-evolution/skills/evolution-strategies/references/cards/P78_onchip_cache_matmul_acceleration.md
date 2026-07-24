---
id: P78
bottlenecks: [l2_cache_thrash]
op_families: [matmul]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P78: 片上缓存加速 Matmul (On-Chip Cache Matmul Acceleration)

## 核心思想
利用 TSCM（Tensor Shared Cache Memory）或 L1 Carry 路径将 Matmul 的输入矩阵预加载到片上缓存，避免 Matmul 内部的 GM→L1 搬运延迟。TSCM 路径适用于训练场景的中等规模矩阵，L1 Carry 路径适用于推理 decode 场景的小 M 大 N 矩阵。

## 代码骨架

// === 改造前（基线）===
```cpp
// 基线：Matmul 从 GM 读取，内部 MTE2 搬运
mm1.SetAType(TPosition::GM, CubeFormat::ND);
mm1.SetBType(TPosition::GM, CubeFormat::ND);
mm1.IterateAll(qGm, kGm, outGm);  // MTE2 搬运在 Matmul 内部
```

// === 改造后（专家模式）===
```cpp
// TSCM 矩阵缓存
struct FaTscmArray {
    TBuf<TPosition::TSCM> tscmBuf[TSCM_BUF_NUM];  // 2 个 TSCM 槽
    void InitTscmBuffer(TPipe* pipe) {
        pipe->InitBuffer(tscmBuf[Q_VEC1_INDEX], tscmSize);
        pipe->InitBuffer(tscmBuf[K_V_INDEX], tscmSize);
    }
};
// Matmul 使用 TSCM 位置
if (mmPolicyType == MmPolicyType::UNSPLITK) {
    mm1.SetAType(TPosition::TSCM, CubeFormat::NZ);
    mm1.SetBType(TPosition::TSCM, CubeFormat::NZ);
    mm1LoadData(tscmBuf[Q_VEC1_INDEX], qGm);  // GM→TSCM 预加载
}
```

## 关键修改点

1. 预期收益: TSCM 带宽远高于 GM→L1 路径；避免 L1 容量不足时的反复搬运

## 常见陷阱

⚠️ TSCM/L1 容量有限，大 headDim 场景可能放不下
⚠️ 需要额外的 Nd2Nz 预处理步骤
⚠️ 与标准 GM-based Matmul 共存，增加代码复杂度

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|GetBlockNum\|coreNum\|blockIdx\|SplitCore\|InitTscmBuffer" op_kernel/*.cpp op_host/*_tiling.cpp
```
