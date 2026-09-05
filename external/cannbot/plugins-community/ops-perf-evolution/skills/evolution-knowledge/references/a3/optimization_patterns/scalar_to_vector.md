# 标量计算向量化 (Scalar-to-Vector Conversion)

## 核心思想

NPU Vector 单元可同时处理多个元素（910B: 256-bit 向量宽度 = 8×FP32 或 16×FP16），但 TileLang 生成的 AscendC 代码常包含逐元素的标量 `for` 循环，完全绕过了 Vector 单元。标量计算向量化是单次优化中收益最大的模式之一（实测 geomean speedup 2x-11x），应在所有新生成内核上优先检查。

## 标量计算检测清单

在内核代码中搜索以下模式（按严重程度排序）：

```
优先级 1（致命）：逐元素标量 for 循环
  grep -nE "for\s*\(.*i\s*[<>].*\+\+i" op_kernel/*.cpp
  症状：sum += x[i]、y[i] = f(x[i])、if (x[i] > max) max = x[i]

优先级 2（严重）：raw pointer 算术
  grep -nE "reinterpret_cast|GetPhyAddr" op_kernel/*.cpp
  症状：float* ptr = ...GetPhyAddr(); ptr[i] = ...

优先级 3（中等）：标量排序/选择
  grep -nE "bubble|selection.sort|for.*swap" op_kernel/*.cpp
  症状：嵌套 for 循环做比较交换

优先级 4（关注）：逐元素 gather/scatter
  grep -nE "for\s*\(.*\).*\[.*order" op_kernel/*.cpp
  症状：for (i) dst[i] = src[order[i]]
```

## 模式 1：标量归约 → ReduceSum/ReduceMax

### 检测特征
```cpp
// 典型标量归约反模式
float sum = 0.0f;
for (int32_t i = 0; i < count; ++i) {
    sum += xPtr[i];
}
```

### 向量化方案
```cpp
// 方案 A：ReduceSum Level 2 API（推荐，单次归约）
LocalTensor<float> scalarOut = scalarBuf.Get<float>();
AscendC::ReduceSum<float, true>(scalarOut, xTensor, reduceTmp, count);
float sum = scalarOut.GetValue(0);

// 方案 B：WholeReduceSum（适合已对齐的数据，延迟更低）
AscendC::WholeReduceSum<float, true>(scalarOut, xTensor, reduceTmp, count);

// 方案 C：二分累加 + WholeReduceSum（大 count 场景，P68 组合）
// 先 Add 折半缩小到 256B 以内，再 WholeReduceSum 一次完成
```

### 适用条件
- 归约维度元素数 ≥ 64（小于此值，标量循环开销可忽略）
- UB 上有连续的 LocalTensor（非 raw pointer）
- 需要 `scalarBuf`（TBuf<VECCALC>，≥64B，8B 对齐）和 `reduceTmp` workspace

### 实测案例
| 算子 | 原始模式 | 优化后 | 加速比 |
|------|---------|--------|--------|
| GroupNorm | `for(i) sum += x[i]; sum_sq += x[i]*x[i]` | ReduceSum × 2 | 1.84x geomean, float16 1.93x |
| NMS | N/A（不涉及归约） | — | — |
| TopK | ReduceMax 迭代 | 硬件 Sort32（见模式 3） | — |

**来源**: GroupNorm 进化 (11_GroupNorm_evo), Round 1

---

## 模式 2：标量逐元素运算 → Adds/Muls/向量链

### 检测特征
```cpp
// 典型标量逐元素反模式
for (int32_t i = 0; i < count_spatial; ++i) {
    float val = xPtr[i];
    float normalized = (val - mean) * inv_std;
    yPtr[i] = normalized * w_val + b_val;
}
```

### 向量化方案
```cpp
// AscendC Adds/Muls 标量-向量指令链（4 条指令替代 O(4N) 标量循环）
// Adds/Muls 可直接接收标量参数，无需 Duplicate 广播
AscendC::Adds<float>(yLocal, xIn, -mean, count_spatial);       // y = x - mean
AscendC::Muls<float>(yLocal, yLocal, inv_std, count_spatial);  // y = y * inv_std
AscendC::Muls<float>(yLocal, yLocal, w_val, count_spatial);    // y = y * w_val
AscendC::Adds<float>(yLocal, yLocal, b_val, count_spatial);    // y = y + b_val
```

### 可用标量-向量指令速查表

| 指令 | 签名 | 替代的标量模式 |
|------|------|---------------|
| `Adds<T>` | `Adds(dst, src, scalar, len)` | `y[i] = x[i] + c` |
| `Muls<T>` | `Muls(dst, src, scalar, len)` | `y[i] = x[i] * c` |
| `Subs<T>` | `Subs(dst, src, scalar, len)` | `y[i] = x[i] - c` (需 AscendC 版本支持) |
| `Relu<T>` | `Relu(dst, src, len)` | `y[i] = max(0, x[i])` |
| `Abs<T>` | `Abs(dst, src, len)` | `y[i] = |x[i]|` |
| `Exp<T>` | `Exp(dst, src, len)` | `y[i] = exp(x[i])` |
| `Sqrt<T>` | `Sqrt(dst, src, len)` | `y[i] = sqrt(x[i])` |
| `Rsqrt<T>` | `Rsqrt(dst, src, len)` | `y[i] = 1/sqrt(x[i])` |
| `Mul<T>` | `Mul(dst, src0, src1, len)` | `y[i] = a[i] * b[i]` |
| `Add<T>` | `Add(dst, src0, src1, len)` | `y[i] = a[i] + b[i]` |
| `Sub<T>` | `Sub(dst, src0, src1, len)` | `y[i] = a[i] - b[i]` |
| `Div<T>` | `Div(dst, src0, src1, len)` | `y[i] = a[i] / b[i]` |

### 多操作融合
```cpp
// 连续向量链（参考 NMS IoU 的 15 条向量指令链）：
// 中间结果保留在 UB 上直通消费，不写回 GM
AscendC::Duplicate<float>(inter, src, len);      // 广播
AscendC::Max<float>(inter, inter, other, len);   // max
AscendC::SubRelu<float>(inter, src1, src2, len); // max(a-b, 0)
AscendC::Mul<float>(inter, inter0, inter1, len); // 乘法
AscendC::Div<float>(inter, inter, other, len);   // 除法
// 最终写出
```

### 适用条件
- 逐元素操作 ≥ 3 次（少于 3 次时向量化的固定开销可能超过标量）
- 连续的元素数量 ≥ 128（910B 单次向量指令吞吐最优区间）
- 数据类型为 FP32/FP16/BF16（整型需要额外检查）

### 实测案例
| 算子 | 原始模式 | 优化后 | 加速比 |
|------|---------|--------|--------|
| GroupNorm | `for(i) y[i]=(x[i]-mean)*inv_std*w+b` | Adds→Muls→Muls→Adds 链 | 整体 1.84x, bf16 4.31x |
| NMS | `for(i) 逐候选框标量 IoU 计算` | 15 条向量指令链 (Duplicate/Max/Min/SubRelu/Mul/Div/Add/Adds) | 11.25x geomean |
| TopK | running merge 逐元素比较 | in-place 向量 merge | 含在整体 4.08x 中 |

**来源**: GroupNorm (11_GroupNorm_evo) P0, NMS (30_NMS) IoU 优化, TopK (TopK_evo_0629) Round 2

---

## 模式 3：标量排序/选择 → 硬件 Sort32/MrgSort

### 检测特征
```cpp
// 典型标量排序反模式
for (int32_t i = 0; i < M; ++i) {
    for (int32_t j = 0; j < M - 1; ++j) {
        if (scores[j] < scores[j+1]) {
            swap(scores[j], scores[j+1]);
            swap(indices[j], indices[j+1]);
        }
    }
}
// 或 selection-sort topK：
for (int32_t i = 0; i < K; ++i) {
    float maxVal = -INFINITY; int maxIdx = -1;
    for (int32_t j = 0; j < n; ++j) { if (vals[j] > maxVal) ... }
    result[i] = maxVal; vals[maxIdx] = -INFINITY;
}
```

### 向量化方案
```cpp
// 方案 A：Sort32 + MrgSort（TopK 场景，推荐）
constexpr int32_t SORT_SZ = 4096;
AscendC::Sort<float, true>(svScore, svScore, orderU32, tmpF, SORT_SZ / 64);
// Sort<true, T> = 升序排序 + 同步输出索引到 orderU32
// 如需降序：先 Negate → Sort → 再 Negate 恢复

// 方案 B：Sort32(repeat=B) 批量并行（小行场景）
// rowLength ≤ 32 时，多行打包为一次 Sort32(repeat=B) 共享一次 barrier
constexpr int32_t BATCH_ROWS = 32;
for (int32_t r = 0; r < totalRows; r += BATCH_ROWS) {
    int32_t curBatch = min(BATCH_ROWS, totalRows - r);
    // 将 curBatch 行数据打包到连续 UB 区域
    AscendC::Sort<T, true>(packedBuf, packedBuf, orderBuf, tmpBuf, 
                           curBatch * rowLen / 64);
}

// 方案 C：下沉排序（NMS 场景，降序 = Negate + 升序 Sort + Negate）
for (int32_t k = 0; k < M; k++) svScore(k) = -svScore(k);          // Negate
for (int32_t k = M; k < SORT_SZ; k++) svScore(k) = 1e38f;           // Padding
AscendC::Sort<float, true>(svScore, svScore, orderU32, tmpF, SORT_SZ / 64);
for (int32_t k = 0; k < M; k++) svScore(k) = -svScore(k);          // Restore
```

### 适用条件
- 排序/选择元素数 ≥ 32（小于此值标量更优）
- 需要降序排序时，用 Negate → Sort → Negate 模式
- Sort32 需要两个同等大小的临时 buffer（`tmpF` + 输出 buffer）
- MrgSort 需要 `AscendC::MrgSortParams` 参数结构体

### 实测案例
| 算子 | 原始模式 | 优化后 | 加速比 |
|------|---------|--------|--------|
| NMS | O(M²) bubble sort | Sort<float, true> bitonic sort O(M log²M) | 11.25x geomean (含其他优化) |
| TopK | O(n×k) scalar selection sort | Sort32 + MrgSort O(n log n) | 4.08x geomean (含其他优化) |
| TopK 小行 | 逐行 Sort32 | Sort32(repeat=BATCH_ROWS) 批量 | 2.2x-5.9x (小行 case) |

**来源**: NMS (30_NMS) 排序优化, TopK (TopK_evo_0629) Round 1-2

---

## 模式 4：零拷贝子张量切片 + 向量运算

### 检测特征
```cpp
// 典型逐候选框标量取值反模式
for (int32_t j = 0; j < numCandidates; ++j) {
    float cx1 = svX1[j];   // 逐元素标量 load
    float cy1 = svY1[j];
    // ... 标量算术 ...
    float iou = interArea / unionArea;
    if (iou > threshold) { ... }
}
```

### 向量化方案
```cpp
// 零拷贝子张量视图：LocalTensor<T> view = tensor[offset]
// 不触发数据拷贝，直接在原 UB tensor 上创建视图
// 后续向量操作批量处理 BATCH 个候选框

LocalTensor<float> candX1 = svX1[bStart];  // 零拷贝切片
LocalTensor<float> candY1 = svY1[bStart];
LocalTensor<float> candX2 = svX2[bStart];
LocalTensor<float> candY2 = svY2[bStart];

// 批量向量化 IoU 计算
AscendC::Duplicate<float>(w0, curX1, bCnt);           // 广播当前框坐标
AscendC::Max<float>(w2, w0, candX1, bCnt);            // max(xx1, candX1)
AscendC::Min<float>(w0, curX2, candX2, bCnt);         // min(xx2, candX2)
AscendC::SubRelu<float>(w0, w0, w2, bCnt);            // ReLU(x2_inter - x1_inter)
AscendC::Mul<float>(w4, w0, w1, bCnt);                // inter_area
AscendC::Mul<float>(w0, curW, curH, bCnt);            // area = w*h
AscendC::Add<float>(w1, w0, subArea, bCnt);           // union = areaA + areaB
AscendC::Sub<float>(w1, w1, w4, bCnt);                // union -= inter_area
AscendC::Div<float>(w4, w4, w1, bCnt);                // iou = inter / union

// 阈值判断：因 910B Compare API 有 256B 对齐限制，改为标量比较
for (int32_t k = 0; k < bCnt; ++k) {
    if (iouLocal.GetValue(k) > threshold) { ... }
}
```

### 关键要点
- `LocalTensor<T> view = tensor[offset]` 是零拷贝操作，仅创建设图元数据
- 所有向量 API 操作在视图上执行，无数据搬运
- 避免了 gather+pad 的开销（实测 padding 版抵消了向量化收益）
- 阈值判断如果 Compare API 对齐条件不满足，回退到标量比较（小批量 BATCH ≤ 64 时开销可忽略）

### 适用条件
- 需要从大 tensor 中按偏移量取子集做批量运算
- 批量大小 ≥ 32 且 ≤ UB 容量 / (子张量数 × sizeof(T))
- 子张量在 UB 上连续存储

### 实测案例
| 算子 | 原始模式 | 优化后 | 加速比 |
|------|---------|--------|--------|
| NMS IoU | 逐候选框标量 load + 标量算术 | 零拷贝切片 + 15 条向量指令 | 11.25x geomean 的关键贡献因子 |

**来源**: NMS (30_NMS) IoU 向量化, 参考 `ops-cv/objdetect/iou_v2/op_kernel/iou_v2_align_iou.h`

---

## 模式 5：混合标量/向量路径调度

### 核心思想
向量化并非总是最优选择。对于极小 workload，向量指令的 EnQue/DeQue 同步开销可能超过标量循环。根据 workload 大小自动选择路径：

```
spatial_size >= 128  →  向量化路径 (EnQue/DeQue 双缓冲 + Vector API)
spatial_size <  128  →  标量轻量路径 (AllocTensor + raw pointer + FreeTensor, 无队列开销)
```

### 实现框架
```cpp
// 混合调度决策
constexpr int32_t VECTOR_MIN_SPATIAL = 128;

if (count_spatial >= VECTOR_MIN_SPATIAL) {
    ProcessBatchedVectorized(xGm, yGm, weightGm, biasGm, ...);
} else {
    ProcessBatchedScalar(xGm, yGm, weightGm, biasGm, ...);
}

// 标量路径：无 EnQue/DeQue，适合每 tile 数据量小的场景
void ProcessBatchedScalar(...) {
    LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    DataCopyPad(xLocal, xGm + offset, count);      // 直接 DMA
    float* xPtr = reinterpret_cast<float*>(xLocal.GetPhyAddr());
    for (int32_t i = 0; i < count; ++i) {           // 标量循环
        sum += xPtr[i];
        sum_sq += xPtr[i] * xPtr[i];
    }
    inQueueX.FreeTensor(xLocal);
}

// 向量路径：EnQue/DeQue + ReduceSum/Adds/Muls
void ProcessBatchedVectorized(...) {
    LocalTensor<float> xIn = inQueueX.DeQue<float>();
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    
    AscendC::ReduceSum<float, true>(scalarLocal, xIn, reduceTmp, count_spatial);
    AscendC::Adds<float>(yLocal, xIn, -mean, count_spatial);   // 向量链
    AscendC::Muls<float>(yLocal, yLocal, inv_std, count_spatial);
    // ...
    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xIn);
}
```

### 阈值调优建议
- 初始设置 `VECTOR_MIN_SPATIAL = 128`，基于 profiling 数据微调
- 对 2D GroupNorm (N×C)，若 C 很小，向量化 tile 过多 → 降低阈值
- 对 4D 大 spatial，向量化收益大 → 阈值可更低
- 更精确的阈值：基于 `group_elements × tile_size_aligned` 而非仅 `spatial_size`
- 后续优化方向：自动调优阈值（基于 profiling 反馈）

### 实测案例
| 算子 | 阈值 | 效果 |
|------|------|------|
| GroupNorm | VECTOR_MIN_SPATIAL=128 | 大 spatial 受益 (3688x 个别 case)，小 spatial case 保持原速或略优 |
| TopK | rowLength ≤ 32 → Sort32(repeat=B), 32-128 → staged batch, >128 → tiled path | 全方位提升 |

**来源**: GroupNorm (11_GroupNorm_evo) P0 混合架构, TopK (TopK_evo_0629) 3-way dispatch

---

## 模式 6：BroadCast + 批量向量运算替代逐行标量循环

### 检测特征
```cpp
// 典型逐行标量广播反模式：用小向量逐行缩放大矩阵
for (int32_t i = 0; i < gRows; i++) {
    int32_t off = (gStart + i) * UD;
    float rstd = rstdLocal.GetValue(i);        // 标量取值
    Muls(xF[off], xF[off], rstd, UD);          // 逐行 Muls，每行一条 Vec 指令
}
// gRows 条 Vec 指令，每条仅处理 UD 个元素 → 指令开销 × gRows
```

### 向量化方案
```cpp
// BroadCast 将 rstd 从 [gRows, 1] 展开为 [gRows, D]，然后 1 条 Mul 覆盖全部
const uint32_t src_shape[2] = {static_cast<uint32_t>(gRows), 1};
const uint32_t dst_shape[2] = {static_cast<uint32_t>(gRows), UD};
BroadCast<float, 2, 1>(xSq, rstdLocal, dst_shape, src_shape, bcast_tmp);
//  ↑ rstdLocal 从 [gRows, 1] broadcast 到 [gRows, D]，输出到 xSq
PipeBarrier<PIPE_V>();
Mul(xF[gStart * UD], xF[gStart * UD], xSq, gRows * UD);
//  ↑ 1 条 Mul 指令处理 gRows × UD 个元素，替代原来的 gRows 条 Muls
// 无 barrier: BroadCast 写 xSq，Mul 写 xF，不同 buffer
```

### 关键要点
- `BroadCast<T, NDIM, BROADCAST_DIM>`：`NDIM` = 目标维度数，`BROADCAST_DIM` = 被广播的维度索引（0-based）
- 此处 `BroadCast<float, 2, 1>` 表示：2D tensor，第 1 维（dim=1 即列）从 1 广播到 UD
- 需要一个临时 buffer（`bcast_tmp`）存储广播后的结果
- `BroadCast` 和 `Mul` 写入不同 buffer（xSq vs xF），无需额外 barrier
- **性能提升**：gRows 条 Vec 指令 → 2 条（BroadCast + Mul），提升 ≈ gRows/2 倍

### 适用条件
- 小向量（如 `rstd[gRows]`）需要广播到大矩阵（`[gRows, D]`）做逐元素运算
- gRows ≥ 4（太少时分条 Muls 的开销与 BroadCast + Mul 相当）
- 有足够的 UB 空间存放广播后的中间结果（`gRows × UD × sizeof(float)` 字节）

---

## 模式 7：GatherMask 替代逐元素 SetValue/GetValue

### 检测特征
```cpp
// 典型逐元素解交织反模式：用 SetValue/GetValue 逐元素重组数据
for (int i = 0; i < half; ++i) {
    dst.SetValue(i, src.GetValue(base + 2 * i));         // 取偶数位
}
for (int i = 0; i < half; ++i) {
    dst.SetValue(half + i, src.GetValue(base + 2 * i + 1)); // 取奇数位
}
// O(n) 次标量 SetValue/GetValue 调用，每条仅操作 1 个元素
```

### 向量化方案
```cpp
// GatherMask 解交织：deinterleave=1 取偶数位，deinterleave=2 取奇数位
uint64_t rsvdCnt = 0;
LocalTensor<float> ropeSrc = src[base];  // 零拷贝偏移

// deinterleave=1: 从 ropeSrc 的步长 1 位置开始取 → 偶数索引
GatherMask(dst, ropeSrc,
    static_cast<uint8_t>(1), true,              // deinterleave=1, 从 offset=1 开始
    static_cast<uint32_t>(total_elements),      // 取 total_elements 个
    {1, 1, 8, 0}, rsvdCnt);                    // repeat/mask 参数

// deinterleave=2: 从 ropeSrc 的步长 2 位置开始取 → 奇数索引
GatherMask(dst[half], ropeSrc,
    static_cast<uint8_t>(2), true,              // deinterleave=2, 从 offset=2 开始
    static_cast<uint32_t>(total_elements),
    {1, 1, 8, 0}, rsvdCnt);
PipeBarrier<PIPE_V>();
// O(1) 条向量指令替代 O(n) 次标量操作
```

### GatherMask 参数说明
| 参数 | 含义 | 示例值 |
|------|------|--------|
| `deinterleave` | 间隔步长：1=连续, 2=隔1取1, N=隔N-1取1 | `1`(偶数位), `2`(奇数位) |
| `isReverse` | 是否反向序 | `true` |
| `count` | 要收集的元素总数 | `total_elements` |
| `repeatMaskParams` | repeatTimes, dstBlkStride, srcBlkStride, dstRepStride, srcRepStride | `{1, 1, 8, 0}` |

### 适用条件
- 需要从连续 tensor 中按固定间隔取元素（解交织、矩阵转置重排、RoPE 奇偶分拆等）
- 元素数量 ≥ 64（较少时标量 SetValue/GetValue 开销可忽略）
- GatherMask 后需 `PipeBarrier<PIPE_V>()`

---

## 综合检查清单（子 Agent 在代码生成后必检）

每个 AscendC kernel 生成后，必须逐项检查以下标量计算残留：

| # | 检查项 | grep 命令 | 若命中则 |
|---|--------|----------|---------|
| 1 | 标量归约循环 | `grep -nE "for.*\+\+.*\{.*\+=|for.*\+\+.*\{.*sum" *.cpp` | 用 ReduceSum/WholeReduceSum 替代（模式 1） |
| 2 | raw pointer + 循环 | `grep -nE "GetPhyAddr|reinterpret_cast" *.cpp` | 用 LocalTensor + Vector API 替代 |
| 3 | 嵌套比较交换循环 | `grep -nE "for.*for.*swap|bubble|selection.sort" *.cpp` | 用 Sort32/MrgSort 替代（模式 3） |
| 4 | 逐元素 gather scatter | `grep -nE "for.*\[order\[" *.cpp` | 用 Cycle Permutation 或 Sort+Extract 替代 |
| 5 | 逐行标量循环 + Muls | `grep -nE "for.*GetValue.*Muls|for.*GetValue.*Adds" *.cpp` | 用 BroadCast + 批量 Mul/Add 替代（模式 6） |
| 6 | 逐元素 SetValue/GetValue | `grep -nE "for.*SetValue.*GetValue|for.*GetValue.*SetValue" *.cpp` | 用 GatherMask 替代（模式 7） |
| 7 | 除 AscendC API 外的 for 循环 | `grep -n "for\s*(" *.cpp \| grep -v "for.*BlockIdx\|for.*tile\|for.*group\|for.*batch"` | 逐一审查是否可向量化 |

## 性能预期

| 优化类型 | 典型加速比 | 风险等级 |
|---------|-----------|---------|
| 标量归约 → ReduceSum | 2-5x（归约维度） | 低（API 成熟） |
| 标量逐元素 → Adds/Muls 链 | 3-10x（逐元素路径） | 低 |
| 标量排序 → Sort32/MrgSort | 5-100x（n 较大时） | 中（需处理对齐和 buffer） |
| 零拷贝 + 批量向量化 | 2-5x（相对于 gather+pad 版） | 中（需仔细计算 buffer 大小） |
| 混合调度 | 消除退化 case（避免负优化） | 中（需 profiling 验证阈值） |
| BroadCast + 批量向量运算 | gRows/2 倍（逐行→批量） | 低（需额外 UB buffer） |
| GatherMask 替代 SetValue/GetValue | 5-20x（n 较大时） | 低（API 成熟，需 PipeBarrier） |
