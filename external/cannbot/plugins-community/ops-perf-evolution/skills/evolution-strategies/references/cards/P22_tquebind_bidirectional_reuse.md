---
id: P22
bottlenecks: [ub_memory_pressure]
op_families: [elementwise, normalization]
complexity: L1
conflicts_with: []
synergizes_with: [P44]
has_preconditions: true
has_playbook: true
---

# P22: TQueBind 双向复用队列

## 核心思想
对于既需要从 GM 搬入又需要搬出到 GM 的数据（如 conv_states），使用 TQueBind 将 VECIN 和 VECOUT 绑定到同一块物理 buffer，节省约 50% 的 buffer 占用。适用于"读-改-写回"的 in-place 操作场景。

## 代码骨架

// === 改造后（专家模式）===
```cpp
TQueBind<TPosition::VECIN, TPosition::VECOUT, NUM_ONE> xQueue_;
TQueBind<TPosition::VECIN, TPosition::VECOUT, NUM_ONE> convStatesQueue_;

x16Local_ = xQueue_.AllocTensor<DTYPE>();
DataCopyPad(x16Local_, this->xGm[xGmOffset], copyParams, padParams);
SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
xQueue_.EnQue(x16Local_);
x16Local_ = xQueue_.DeQue<DTYPE>();
// 同一个 tensor 直接搬出到 GM
DataCopyPad(this->convStateGm[offset], x16Local_, copyParams);
xQueue_.FreeTensor(x16Local_);
```

## 关键修改点

1. 用细粒度同步替代粗粒度 SyncAll
2. 预期收益: 节省约 50% 的 buffer 占用，适合 UB 资源紧张的场景

## 常见陷阱

⚠️ depth 通常只能为 1，无法做双缓冲流水；搬入搬出之间必须显式同步
⚠️ 不适用于输入输出 shape 不同的场景

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|DataCopy\|CopyIn\|CopyOut\|Fixpipe" op_kernel/*.cpp op_host/*_tiling.cpp
```
