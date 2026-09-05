# AscendC API Knowledge Index

Common API patterns, pitfalls, and correct usage for AscendC kernel development.

## Files

| File | Content | Priority |
|------|---------|----------|
| `common_pitfalls.md` | Top 15 most frequent coding mistakes in AscendC kernels | Must-read before writing any kernel |

## Quick Reference

### Essential APIs
```
Data Transfer:  DataCopy, DataCopyPad, DataCopyExtParams
Vector Compute: Add, Mul, Adds, Muls, Div, Exp, Log, Gelu, Relu, Cast
Reduction:      ReduceSum, ReduceMax, ReduceMin, WholeReduceSum
Comparison:     Compare, Select, CompareScalar
Sync:           EnQue, DeQue, SetFlag, WaitFlag, PipeBarrier
Memory:         AllocTensor, FreeTensor, TQue, TBuf
Mask:           SetMaskCount, SetMaskNorm, GetMask
```

### Memory Hierarchy
```
Global Memory (GM) ←→ MTE2/MTE3 ←→ Unified Buffer (UB) ←→ VECTOR/SCALAR
  GlobalTensor<T>                    LocalTensor<T>
```

## 按症状快速定位

| 症状 | 对应陷阱# | 关键词 |
|------|----------|--------|
| 输出全 0 或随机值 | #3, #8 | PipeBarrier, EnQue/DeQue |
| 输出部分正确部分乱码 | #1, #7 | 32B 对齐, stride |
| 多 tile 后死锁 | #2, #11 | FreeTensor, EnQue/DeQue 匹配 |
| FP16 精度差 | #4, #5 | Cast, CompareScalar 类型 |
| 最后几个元素错误 | #10 | 尾块处理 |
