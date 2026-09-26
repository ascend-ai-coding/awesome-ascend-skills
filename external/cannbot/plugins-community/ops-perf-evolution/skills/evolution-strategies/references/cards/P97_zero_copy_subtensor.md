---
id: P97
bottlenecks: [scalar_compute, scalar_loading, mte2_stall]
op_families: [elementwise, reduction, special, index_scatter]
complexity: L1
conflicts_with: []
synergizes_with: [P95, P69, P7]
requires: []
has_preconditions: true
has_playbook: false
---

# P97: 零拷贝子张量切片 + 批量向量运算 (Zero-Copy Sub-Tensor Slicing)

## 核心思想
当需要从 UB 大 tensor 中按偏移量取子集做批量运算时（如 NMS 的逐候选框 IoU、TopK 的逐行合并），传统做法是 `for` 循环逐元素标量 load + 算术，或 gather 拷贝到连续 buffer 再向量运算。AscendC 的 `LocalTensor<T> view = tensor[offset]` 是零拷贝子张量切片操作，仅创建视图元数据（偏移量 + 长度），不触发任何数据搬运。后续对 view 的向量操作（`Max`/`Min`/`SubRelu`/`Mul`/`Div`）直接在原 UB tensor 上执行，批量处理 BATCH 个元素而无需拷贝。关键优势：避免了 gather+pad 的搬运开销（实测 padding 版抵消了全部向量化收益）。

## 代码骨架

```cpp
// === 改造前（基线）：逐元素标量 gather ===
for (int32_t j = 0; j < numCandidates; ++j) {
    float cx1 = svX1[j];  // 逐元素标量 load
    float cy1 = svY1[j];
    float cx2 = svX2[j];
    float cy2 = svY2[j];
    // ... 标量算术计算 IoU ...
    float iou = interArea / unionArea;
    if (iou > threshold) { ... }
}

// === 改造后：零拷贝切片 + 批量向量运算 ===
// 子张量视图 — 零拷贝，仅创建视图元数据
LocalTensor<float> candX1 = svX1[bStart];   // 从偏移 bStart 开始
LocalTensor<float> candY1 = svY1[bStart];
LocalTensor<float> candX2 = svX2[bStart];
LocalTensor<float> candY2 = svY2[bStart];

// 批量向量化 IoU 计算（BATCH 个候选框同时处理）
AscendC::Duplicate<float>(w0, curX1, bCnt);          // 广播当前框 x1
AscendC::Max<float>(w2, w0, candX1, bCnt);           // x1_inter = max(xx1, candX1)
AscendC::Min<float>(w0, curX2, candX2, bCnt);        // x2_inter = min(xx2, candX2)
AscendC::SubRelu<float>(w0, w0, w2, bCnt);           // w_inter = ReLU(x2_inter - x1_inter)
AscendC::Max<float>(w3, w0, candX1, bCnt);           // y1_inter = max(yy1, candY1)
AscendC::Min<float>(w1, curY2, candY2, bCnt);        // y2_inter = min(yy2, candY2)
AscendC::SubRelu<float>(w1, w1, w3, bCnt);           // h_inter = ReLU(y2_inter - y1_inter)
AscendC::Mul<float>(w4, w0, w1, bCnt);               // inter_area = w_inter * h_inter
AscendC::Mul<float>(w0, curW, curH, bCnt);           // area = w * h
AscendC::Add<float>(w1, w0, subArea, bCnt);          // union = area + subArea
AscendC::Sub<float>(w1, w1, w4, bCnt);               // union -= inter_area
AscendC::Div<float>(w4, w4, w1, bCnt);               // iou = inter / union

// 阈值判断：若 Compare API 不满足对齐条件，回退标量比较（小批量开销可忽略）
for (int32_t k = 0; k < bCnt; ++k) {
    if (iouLocal.GetValue(k) > threshold) { /* suppress */ }
}
```

## 关键修改点

1. **切片语法**：`LocalTensor<T> view = tensor[offset]` 创建视图，`offset` 为元素偏移（非字节偏移），视图长度 = `tensor.GetLen() - offset`
2. **批量大小选择**：`bCnt = min(IOU_BATCH, remaining)`，`IOU_BATCH` 受限于 UB 容量 / (子张量数 × sizeof(T))
3. **Compare API 回退**：910B `Compare` API 要求 `count × sizeof(float) ≡ 0 mod 256`；不满足时改用标量 `GetValue(k)` 比较（BATCH ≤ 64 时开销可忽略）
4. **与 gather+pad 对比**：零拷贝方案无需额外 buffer，无需搬运时间；gather+pad 版在 NMS 实测中抵消了全部向量化收益
5. **预期收益**：2-5x（相对于标量 gather 方案）；相对于 gather+pad 向量方案可能有 10-30% 额外提升（消除搬运开销）

## 适用性检测 (grep)

```bash
# 检测从大 tensor 逐元素取值的模式
grep -nE "for\s*\(.*j.*\].*=" op_kernel/*.cpp | grep -v "for.*tile\|for.*group\|for.*batch"

# 检测 gather/copy 到连续 buffer 的模式（零拷贝可替代）
grep -nE "for.*gather|for.*collect|for.*copy.*\[.*\+" op_kernel/*.cpp

# 确认 UB tensor 在切片前已完整加载
grep -nE "DataCopy|DeQue" op_kernel/*.cpp
```

## 常见陷阱

⚠️ 子张量视图的生命周期绑定原 tensor；原 tensor 被 FreeTensor 后视图失效
⚠️ 视图偏移量必须在原 tensor 的有效范围内（越界无编译时检查，运行时可能崩溃或静默错误）
⚠️ 多个视图同时活跃时需确保它们不重叠或语义上不冲突（重叠写入会导致 UB）
⚠️ 批量大小过大（超过 UB 容量）导致编译失败或运行时内存错误
⚠️ `tensor[offset]` 创建的视图长度 = `原长度 - offset`，如需限制长度需使用 `tensor[offset].template SubTensor<0, len>()` 或等效 API

## 代码搜索关键词

```bash
grep -nE "LocalTensor.*=.*\[.*\]|SubTensor|GetValue|view.*tensor" op_kernel/*.cpp
```

## 来源

- NMS 进化 (30_NMS): 零拷贝子张量切片替代标量 gather，IoU 计算从全标量→全向量，11.25x geomean 的关键贡献因子
- 参考 `ops-cv/objdetect/iou_v2/op_kernel/iou_v2_align_iou.h` 的生产级实现
