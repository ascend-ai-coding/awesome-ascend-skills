---
id: P20
bottlenecks: [mte2_stall]
op_families: [datapath]
complexity: L1
conflicts_with: [P1, P19, P21]
synergizes_with: [P5, P8]
has_preconditions: true
has_playbook: true
---

# P20: 三缓冲轮转（BuffersPolicy3buff）

## 核心思想
在 AIC/AIV 混合核架构下，三缓冲允许搬运、Cube 计算、Vector 计算三个阶段同时进行，形成三级流水。每个阶段持有独立的 buffer 和轮转指针，互不干扰。

## 代码骨架

// === 改造后（专家模式）===
```cpp
template<BufferType bufferType, SyncType syncType>
class BuffersPolicy3buff {
    Buffer<bufferType, syncType> a_, b_, c_;
    uint32_t flag1_ = 0, flag1_vec1_ = 0, flag1_bmm2_ = 0;
public:
    __aicore__ inline Buffer<bufferType, syncType> &Get() {
        if (flag1_ == 0) { flag1_ = 1; return a_; }
        else if (flag1_ == 1) { flag1_ = 2; return b_; }
        else { flag1_ = 0; return c_; }
    }
    __aicore__ inline Buffer<bufferType, syncType> &GetVec() {
        if (flag1_vec1_ == 0) { flag1_vec1_ = 1; return a_; }
        else if (flag1_vec1_ == 1) { flag1_vec1_ = 2; return b_; }
        else { flag1_vec1_ = 0; return c_; }
    }
    __aicore__ inline Buffer<bufferType, syncType> &GetCube() {
        if (flag1_bmm2_ == 0) { flag1_bmm2_ = 1; return a_; }
        else if (flag1_bmm2_ == 1) { flag1_bmm2_ = 2; return b_; }
        else { flag1_bmm2_ = 0; return c_; }
    }
    // 三级流水使用示例：搬运(MTE2) / Cube(M) / Vector(V) 同时进行
    // 每级持有独立 buffer，通过 HardEvent 同步数据就绪
    //
    // WaitFlag<HardEvent::MTE2_MTE1>(eventId);  // 等待搬运完成
    // LoadData(Get());                            // Cube 使用搬运完成的 buffer
    // SetFlag<HardEvent::MTE1_M>(eventId);       // 通知 Cube 数据就绪
    //
    // WaitFlag<HardEvent::MTE1_M>(eventId);      // 等待 Cube 完成
    // Compute(GetCube());                         // Vector 使用 Cube 完成的 buffer
    // SetFlag<HardEvent::M_FIX>(eventId);        // 通知 Vector 数据就绪
// ... (truncated)
```

## 关键修改点

1. 用细粒度同步替代粗粒度 SyncAll
2. 预期收益: 搬运/Cube/Vector 三级流水完全重叠，理论吞吐接近硬件峰值

## 常见陷阱

⚠️ 片上存储占用增加到 3 倍；三套独立轮转指针调试难度大
⚠️ 同步事件数量增多，需要为每个阶段独立管理 event

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|SyncAll\|SetFlag\|WaitFlag\|PipeBarrier\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
