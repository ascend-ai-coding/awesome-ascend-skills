---
id: P21
bottlenecks: [ub_memory_pressure]
op_families: [matmul, cv_fusion]
complexity: L1
conflicts_with: [P1, P19, P20]
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P21: 二维矩阵缓冲策略（Matrix2x2BufferPolicy）

## 核心思想
针对 Matmul 切 M 和切 K 两个维度同时需要流水的场景，用 2x2=4 个 buffer 组成二维矩阵，按列优先分配、使用和释放。适用于 MLA headDim=576 的 nope/rope 分离等需要双维度流水的 Cube 算子。

## 代码骨架

// === 改造后（专家模式）===
```cpp
template<BufferType bufferType, SyncType syncType>
class Matrix2x2BufferPolicy {
    Buffer<bufferType, syncType> bufferM0k0_, bufferM0k1_, bufferM1k0_, bufferM1k1_;
    int32_t mExtent_ = 0;
    int32_t aIdx_ = -1, uIdx_ = -1, fIdx_ = -1;
    int32_t amIdx_ = 0, akIdx_ = 0, umIdx_ = 0, ukIdx_ = 0, fmIdx_ = 0, fkIdx_ = 0;
    static constexpr int32_t kSize_ = 2;

    __aicore__ inline Buffer<bufferType, syncType> &GetBuffer(
        int32_t xIdx, int32_t &mIdx, int32_t &kIdx) {
        mIdx = (mIdx + mExtent_ - 1) % mExtent_;
        kIdx = (xIdx / mExtent_) % kSize_;
        return buffers_[mIdx * kSize_ + kIdx];
    }
    __aicore__ inline Buffer<bufferType, syncType> &AllocNext() {
        aIdx_++; return GetBuffer(aIdx_, amIdx_, akIdx_);
    }
    __aicore__ inline Buffer<bufferType, syncType> &ReuseNext() {
        uIdx_++; return GetBuffer(uIdx_, umIdx_, ukIdx_);
    }
    __aicore__ inline Buffer<bufferType, syncType> &FreeNext() {
        fIdx_++; return GetBuffer(fIdx_, fmIdx_, fkIdx_);
    }
};
```

## 关键修改点

1. 预期收益: 支持 M/K 双维度同时流水，最大化 Cube 单元利用率

## 常见陷阱

⚠️ 4 个 buffer 占用大量片上存储；Alloc/Reuse/Free 三套独立索引管理复杂
⚠️ 列优先遍历顺序与常规行优先直觉不同，容易出错

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|SyncAll\|SetFlag\|WaitFlag\|PipeBarrier\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
