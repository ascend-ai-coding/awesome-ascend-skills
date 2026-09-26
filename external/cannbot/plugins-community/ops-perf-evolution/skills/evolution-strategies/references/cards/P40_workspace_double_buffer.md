---
id: P40
bottlenecks: [mte3_stall]
op_families: [datapath]
complexity: L1
conflicts_with: []
synergizes_with: []
has_preconditions: true
has_playbook: true
---

# P40: Workspace 双缓冲中间结果常驻

## 核心思想
Cube 和 Vector 之间通过 workspace GM 传递中间结果，每个 core 分配 2 份 workspace，通过 loop % 2 索引实现双缓冲，Cube 写入和 Vector 读取完全并行。

## 代码骨架

// === 改造后（专家模式）===
```cpp
mm1ResGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + offset +
    aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T)));
uint64_t srcGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize;
DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);
uint64_t vec2ResGmOffset = ((info.loop - 1) % constInfo.preLoadNum) * constInfo.bmm2ResUbSize;
DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset], vec2ComputeSize);
```

## 关键修改点

1. 预期收益: Cube 写入和 Vector 读取完全并行，消除 GM 中间结果的流水气泡

## 常见陷阱

⚠️ workspace 总量 = 核数 × 2 × 多个 buffer size，HBM 占用较大
⚠️ 首次迭代 Vector 无数据可读需要特殊处理（preLoadNum 控制流水启动延迟）

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|SyncAll\|PipeBarrier\|ExecuteTask\|PRELOAD\|GetBlockNum" op_kernel/*.cpp op_host/*_tiling.cpp
```
