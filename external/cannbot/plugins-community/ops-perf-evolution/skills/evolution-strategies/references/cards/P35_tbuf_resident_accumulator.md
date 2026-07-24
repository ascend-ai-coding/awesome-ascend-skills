---
id: P35
bottlenecks: [partial_overlap, ub_memory_pressure]
op_families: [reduction, normalization, elementwise]
complexity: L1
conflicts_with: []
synergizes_with: [P8]
has_preconditions: true
has_playbook: true
---

# P35: TBuf 常驻中间累加器

## 核心思想
使用 TBuf<VECCALC> 分配的 buffer 不参与队列管理，在 Init 中分配后在整个 Process 生命周期内常驻，用于存储跨迭代的递推状态或梯度累加器。

## 代码骨架

// === 改造后（专家模式）===
```cpp
TBuf<TPosition::VECCALC> y0Buf_;
TBuf<TPosition::VECCALC> y1Buf_;
TBuf<TPosition::VECCALC> y2Buf_;

void ComputeAndUpdate() {
    LocalTensor<float> y0BufLocal = y0Buf_.Get<float>();
    LocalTensor<float> y1BufLocal = y1Buf_.Get<float>();
    Mul(y2BufLocal, x32BufLocal, weight2, curDim_);
    Add(y2BufLocal, y1BufLocal, y2BufLocal, curDim_);
    Mul(y1BufLocal, x32BufLocal, weight1, curDim_);
    Add(y1BufLocal, y0BufLocal, y1BufLocal, curDim_);
    Mul(y0BufLocal, x32BufLocal, weight0, curDim_);
}
```

## 关键修改点

1. 预期收益: 常驻 buffer 避免每次迭代重新分配，支持跨迭代状态传递

## 常见陷阱

⚠️ TBuf 不受队列同步保护，需要程序员自行通过 PipeBarrier 保证一致性
⚠️ 常驻 buffer 在整个 Process 期间占用 UB，减少可用于数据 tile 的空间

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|ComputeAndUpdate" op_kernel/*.cpp op_host/*_tiling.cpp
```
