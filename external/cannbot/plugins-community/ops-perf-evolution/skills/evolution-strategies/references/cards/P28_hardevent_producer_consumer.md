---
id: P28
bottlenecks: [bus_contention]
op_families: [matmul, cv_fusion]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P28: 自定义 HardEvent 生产者-消费者同步

## 核心思想
用编译期确定的 HardEvent 类型（MTE2_MTE1, MTE1_M, M_FIX 等）建立精确的生产者-消费者关系，替代粗粒度 PipeBarrier。每个 Buffer 绑定一对事件 ID。

## 代码骨架

// === 改造后（专家模式）===
```cpp
template<BufferType Type>
struct BufferInfo {
    __aicore__ const static constexpr HardEvent ConsWaitProdStatus() {
        if constexpr (Type == BufferType::L1)  return HardEvent::MTE2_MTE1;
        if constexpr (Type == BufferType::L0A) return HardEvent::MTE1_M;
        if constexpr (Type == BufferType::L0C) return HardEvent::M_FIX;
    }
    static constexpr HardEvent EventP2C = ConsWaitProdStatus();
};
buffer_.Wait<EventP2C>();
buffer_.Set<EventP2C>();

// 更多常见 TPosition 类型映射
template<> struct BufferInfo<TPosition::L0B> {
    static constexpr HardEvent WAIT = HardEvent::MTE2_MTE1;
    static constexpr HardEvent SET  = HardEvent::MTE1_M;
};
template<> struct BufferInfo<TPosition::VECIN> {
    static constexpr HardEvent WAIT = HardEvent::MTE2_V;
    static constexpr HardEvent SET  = HardEvent::V_MTE3;
// ... (truncated)
```

## 关键修改点

1. 预期收益: 编译期确定事件类型，零运行时开销；替代粗粒度 PipeBarrier 减少流水线气泡

## 常见陷阱

⚠️ 事件 ID 资源有限（16 个），需轮转复用；Init/UnInit 必须严格配对
⚠️ 每种 BufferType 需要独立的 HardEvent 特化，buffer 类型多时模板代码膨胀

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|GetBlockNum\|coreNum\|blockIdx\|SplitCore\|ConsWaitProdStatus" op_kernel/*.cpp op_host/*_tiling.cpp
```
