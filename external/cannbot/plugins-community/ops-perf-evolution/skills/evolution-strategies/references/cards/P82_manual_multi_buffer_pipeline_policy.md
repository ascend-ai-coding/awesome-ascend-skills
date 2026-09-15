---
id: P82
bottlenecks: [no_overlap]
op_families: [special]
complexity: L1
conflicts_with: [P1]
synergizes_with: []
has_preconditions: false
has_playbook: false
deprecated: true
deprecated_reason: "Code skeleton is identical copy-paste of P19 (BuffersPolicyDB class). P19 has richer content."
---

# P82: BufferPolicy Class for Multi-Stage Pipeline (手写 BufferPolicy 类替代 TQue 的多阶段流水策略)

## 核心思想
当标准 `TQue` 或单一双缓冲无法表达复杂的 buffer 访问路径时，可以手写 `BufferPolicy` 类，通过显式的 buffer 集合、轮转指针和独立的 alloc/use/free 入口管理片上 buffer 生命周期。该策略的边界是“用策略类替代队列流控”：根据流水阶段数与分块维度，选择 2-buffer PingPong、3-buffer 三级轮转或 2×2 matrix buffer，将当前轮、上一轮复用、Cube/Vector 分阶段消费和多维分块释放等访问关系显式编码。

## 代码骨架

// === 改造前（基线）===
```cpp
TQue<QuePosition::VECIN, 2> inQueue;
LocalTensor<float> cur = inQueue.DeQue<float>();
Compute(cur);
// 无法表达上一轮 buffer 复用或独立复用轮转，只能按单一路径顺序消费
```

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

1. 预期收益: 支持当前轮、上一轮和复用轮转三种访问模式，让搬运与 Cube 计算更充分重叠。

## 常见陷阱

⚠️ 需要显式管理 buffer 生命周期、轮转指针和同步事件，失去 `TQue` 自动流控保护
⚠️ buffer 数量通常增加到 2 倍、3 倍或 4 份，显著压缩片上存储预算
⚠️ 多套访问路径（如 alloc/use/free、Cube/Vector/DMA）和多级事件依赖容易引入静默数据错误，调试复杂度高

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|SyncAll\|PipeBarrier\|ExecuteTask\|PRELOAD\|SyncAll" op_kernel/*.cpp op_host/*_tiling.cpp
```
