---
id: P91
bottlenecks: [compute_bound]
op_families: [flash_attention]
complexity: L2
conflicts_with: []
synergizes_with: [P88, P79]
has_preconditions: true
has_playbook: true
---

# P91: Softmax 梯度预计算独立阶段 (Softmax Gradient Pre-Computation Phase)

## 核心思想
在 Flash Attention 反向传播中，将 softmax gradient 的计算（`sum_i = sum(dY_i * O_i, dim=-1)`）提取为独立的 VecSfmg 阶段，在 Main 阶段之前完成。VecSfmg 使用独立的 TPipe 和 buffer 布局，预计算结果存入 sfmgWorkspace，Main 阶段直接读取，避免在 Main 循环内重复计算。

## 代码骨架

```cpp
// === 改造前（Main 循环内重复计算）===
for (int s2 = 0; s2 < s2Loops; s2++) {
    Mul(tmpBuf, dyBuf, oBuf);          // dY * O
    ReduceSum(sfmgBuf, tmpBuf, params); // sum(dY * O)
    // ... 使用 sfmgBuf ...
}
// 每次 S2 迭代重复计算，浪费 Vector 算力

// === 改造后（独立预计算阶段）===
class VectorSoftmaxGrad {
    TPipe pipeSfmg;     // 独立 TPipe，不干扰 Main 阶段
    TBuf<> inputBuf;    // 24K * 2 (ping-pong)
    TBuf<> castBuf;     // 48K * 2 (fp16→fp32)

    void Process() {
        for (int s1 = 0; s1 < s1Loops; s1++) {
            CopyInSfmg(inputBuf, dyGm, oGm, s1);        // 搬入 dY 和 O
            Cast(castBuf, inputBuf, RoundMode::CAST_NONE); // FP16→FP32
            Mul(castBuf, castDyBuf, castOBuf);           // dY * O
            ReduceSum(outputBuf, castBuf, reduceParams);  // ReduceSum along D
            DataCopy(sfmgWorkspaceGm[s1], outputBuf, params); // 写入 workspace
        }
    }
};

// Main 阶段：直接读取预计算结果
void VecMainProcess() {
    DataCopy(sfmgUb, sfmgWorkspaceGm[offset], params);  // 读取 sum(dY*O)
    Sub(dsUb, dyvUb, sfmgUb);  // dS = P * (dY*V^T - sum(dY*O))
}
```

## 关键修改点

1. 新增独立 TPipe（`pipeSfmg`）和 VecSfmg 阶段类
2. Buffer 布局：input 24K×2（ping-pong）+ cast 48K×2（FP16→FP32 扩展）
3. 预计算结果存入 `sfmgWorkspace`（GM workspace），Main 阶段通过 DataCopy 读取
4. VecSfmg 和 Main 之间需要 `SyncAll()` 全局同步

## 常见陷阱

⚠️ 需要额外的 `sfmgWorkspace` 空间
⚠️ `SyncAll()` 引入全核同步开销，小规模场景可能得不偿失
⚠️ 独立 TPipe 的初始化/销毁有固定开销
⚠️ 仅适用于 Flash Attention Backward 场景，泛化到其他算子需验证
