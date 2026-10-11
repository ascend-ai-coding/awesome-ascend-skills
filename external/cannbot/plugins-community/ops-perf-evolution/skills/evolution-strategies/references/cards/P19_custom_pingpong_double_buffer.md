---
id: P19
bottlenecks: [mte2_stall, mte3_stall, no_overlap]
op_families: [normalization]
complexity: L1
conflicts_with: [P1, P20, P21]
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P19: 自定义多级 PingPong 双缓冲

## 核心思想
绕过 TQue 标准队列机制，用手写 ping/pong 轮转实现 L0A/L0B/L1 等多级存储的双缓冲。支持 Q 复用（GetPre）和 KV 复用（GetReused），使搬运和计算完全重叠。对于 Cube 密集型算子，还可在 L1A/L1B、L0A/L0B、L0C 三级存储分别做 PingPong 双缓冲，使 GM→L1→L0→Mmad 三级流水线完全重叠（见 Variant B）。

## 代码骨架

// === 改造后（专家模式）===
```cpp
template<BufferType bufferType, SyncType syncType>
class BuffersPolicyDB {
public:
    __aicore__ inline void Init(BufferManager<bufferType> &bufferManager, uint32_t size){
        ping_ = bufferManager.template AllocBuffer<syncType>(size);
        pong_ = bufferManager.template AllocBuffer<syncType>(size);
        ping_.Init(); pong_.Init();
    }
    __aicore__ inline Buffer<bufferType, syncType> &Get() {
        if (flag1_) { flag1_ = 0; return ping_; } else { flag1_ = 1; return pong_; }
    }
    __aicore__ inline Buffer<bufferType, syncType> &GetPre() {
        if (flag1_) { return pong_; } else { return ping_; }
    }
    __aicore__ inline Buffer<bufferType, syncType> &GetReused() {
        if (flag2_ == 0) { flag2_ = 1; return pong_; } else { flag2_ = 0; return ping_; }
    }
};
```

## 关键修改点

1. 预期收益: 支持多种复用模式（Q 复用、KV 复用），使搬运与 Cube 计算完全重叠

## 常见陷阱

⚠️ 需要手动管理同步事件；buffer 数量翻倍，片上存储占用增加一倍
⚠️ 失去 TQue 的自动流控保护，同步错误会导致静默数据错误
⚠️ Cube 三级 PingPong 需要管理 6 组以上 event ID 数组，编程复杂度极高

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|SyncAll\|SetFlag\|WaitFlag\|PipeBarrier\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
