---
id: P50
bottlenecks: [compute_bound, partial_overlap]
op_families: [cv_fusion, moe]
complexity: L1
conflicts_with: []
synergizes_with: [P51, P73]
has_preconditions: true
has_playbook: true
---

# P50: SwiGLU 融合流水线 (SwiGLU Fusion Pipeline)

## 核心思想
用高阶 SwiGLU API + Ping-Pong 双缓冲实现 AIC-AIV 流水线，将 5 步手动 sigmoid+gate 操作压缩为单条 API 调用，减少 80% 指令数和 75% PipeBarrier 同步。

## 代码骨架

// === 改造前（基线）===
```cpp
// Step 1: sigmoid(-x)
Muls(tmpLocal, aLocal, -1.0f, tileSize_);
PipeBarrier<PIPE_V>();
// Step 2: exp(-x)
Exp<float, 0>(bLocal, tmpLocal, tileSize_);
PipeBarrier<PIPE_V>();
// Step 3: 1 + exp(-x)
Adds(bLocal, bLocal, 1.0f, tileSize_);
PipeBarrier<PIPE_V>();
// Step 4: sigmoid = 1 / (1 + exp(-x))
Duplicate(onesLocal, 1.0f, tileSize_);
Div(tmpLocal, onesLocal, bLocal, tileSize_);
PipeBarrier<PIPE_V>();
// Step 5: gate = x * sigmoid(x)
Mul(aLocal, aLocal, tmpLocal, tileSize_);
// ... (truncated)
```

// === 改造后（专家模式）===
```cpp
// 单条 API 调用完成 SwiGLU
SwiGLU<float, false>(workspace, src0, src1, beta, halfTokenLen);
PipeBarrier<PIPE_V>();
// Ping-Pong: 当前 tile 计算时，下一 tile 的 Cube 输出已搬入另一 buffer
```

## 关键修改点

1. 用细粒度同步替代粗粒度 SyncAll
2. 预期收益: 指令数 5→1（80% 减少），PipeBarrier 4→1（75% 减少），省去 onesLocal buffer

## 常见陷阱

⚠️ SwiGLU API 可能不支持所有数据类型（需检查硬件版本）
⚠️ 适用范围较窄（narrow scope），仅限 MoE FFN 融合算子
⚠️ Ping-Pong 双缓冲增加 UB 占用

## 代码搜索关键词

```bash
grep -n "BUFFER_NUM\|InitBuffer\|TQue\|SyncAll\|PipeBarrier\|ExecuteTask\|PRELOAD" op_kernel/*.cpp op_host/*_tiling.cpp
```
