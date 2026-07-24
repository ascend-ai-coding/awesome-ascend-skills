---
id: P94
bottlenecks: [scalar_compute, compute_bound]
op_families: [reduction, normalization, elementwise, softmax, optimizer]
complexity: L1
conflicts_with: []
synergizes_with: [P67, P68, P69, P84]
requires: []
has_preconditions: true
has_playbook: false
---

# P94: 标量归约向量化 (Scalar Reduction → Vector Reduce API)

## 核心思想
检测 kernel 中逐元素的标量归约循环（`for(i) sum += x[i]`、`for(i) if (x[i] > max) max = x[i]`），用 AscendC `ReduceSum`/`ReduceMax`/`WholeReduceSum` Vector API 替代。标量归约在 910B 上每次迭代仅处理 1 个元素，而 Vector 单元可同时处理 8×FP32 或 16×FP16，且 ReduceSum Level 2 API 内部使用硬件树归约达到 O(log n) 延迟。这是生成代码中最常见且修复收益最大的性能问题。

## 代码骨架

```cpp
// === 改造前（基线）：标量归约循环 ===
float* xPtr = reinterpret_cast<float*>(xLocal.GetPhyAddr());
float sum = 0.0f;
float sum_sq = 0.0f;
for (int32_t i = 0; i < count_spatial; ++i) {
    float val = xPtr[i];
    sum += val;
    sum_sq += val * val;
}

// === 改造后：ReduceSum Level 2 API ===
LocalTensor<float> scalarLocal = scalarBuf.Get<float>();
LocalTensor<float> reduceTmp = sumBuf.Get<float>();

// 第一遍：sum
AscendC::ReduceSum<float, true>(scalarLocal, xIn, reduceTmp, count_spatial);
sum_local[g] += scalarLocal.GetValue(0);

// 第二遍：sum_sq（先 in-place 平方，再归约）
AscendC::Mul<float>(xIn, xIn, xIn, count_spatial);  // x² in-place
AscendC::ReduceSum<float, true>(scalarLocal, xIn, reduceTmp, count_spatial);
sum_sq_local[g] += scalarLocal.GetValue(0);
```

## 关键修改点

1. **Buffer 分配**：需要 `scalarBuf`（`TBuf<VECCALC>`，≥ 64 × sizeof(float)，8B 对齐）存放 ReduceSum 标量输出，`sumBuf`（`TBuf<VECCALC>`，≥8KB）作为 ReduceSum Level 2 workspace
2. **批量归约**：同一 batch 内多个 channel 的归约结果可在 scalarLocal 上累加，最后一次性使用
3. **大 n 优化**：当 count > 65536 时，考虑先 `Add` 折半缩小到 256B 以内，再用 `WholeReduceSum` 一次完成（P68 组合方案）
4. **预期收益**：归约维度 ≥ 1024 时可达 3-5x 加速；归约维度 64-1024 时 1.5-3x；< 64 时收益递减

## 适用性检测 (grep)

```bash
# 检测标量归约循环
grep -nE "for\s*\(.*\+\+.*\{\s*(sum|max_val|min_val|result)\s*[\+\-]?=" op_kernel/*.cpp

# 检测 raw pointer 使用（标量计算的前兆）
grep -nE "reinterpret_cast.*GetPhyAddr|\.GetPhyAddr\(\)" op_kernel/*.cpp
```

## 常见陷阱

⚠️ ReduceSum 的 `isTail` 参数：当数据可能不足一个 repeat 时设为 `true`；精确整除时设为 `false` 可获更好性能
⚠️ `scalarBuf` 必须 `TBuf<VECCALC>` 类型且 8B 对齐，否则 `GetValue(0)` 返回错误数据
⚠️ `Mul(xIn, xIn, xIn, n)` in-place 平方会修改原始输入；若后续仍需原始值，需先拷贝
⚠️ ReduceSum 完成后，scalarLocal 上的数据在下一次 ReduceSum 前可能被覆盖；先取值再调度下一次归约

## 代码搜索关键词

```bash
grep -nE "ReduceSum|WholeReduceSum|ReduceMax|scalarBuf|VECCALC" op_kernel/*.cpp op_host/*_tiling.cpp
```

## 来源

- GroupNorm 进化 (11_GroupNorm_evo) P0: ReduceSum 替代 sum/sum_sq 标量循环，geomean 1.84x
- TopK 进化 (TopK_evo_0629): ReduceMax 迭代方案被证无效（握手开销大），转入硬件 Sort 方案
