---
id: P80
bottlenecks: [no_overlap, partial_overlap]
op_families: [flash_attention]
complexity: L2
conflicts_with: []
synergizes_with: [P14, P38]
has_preconditions: true
has_playbook: true
---

# P80: Sink Token 流水线注入与 S2 块跳过 (Sink Task Pipeline Injection and S2 Block Skip)

## 核心思想
在 Streaming LLM 的 Attention Sink 场景中，将 sink token（前 N 个 token 的 KV cache）作为独立 task 注入到 Flash Attention 的 3-stage pipeline 中，与普通 sparse task 共享同一套流水线调度。同时，在 sink 区域与有效 token 区域之间的 gap 中，通过 IsSkipCal 判断跳过无效 S2 block 的全部计算。

## 代码骨架

// === 改造前（基线）===
```cpp
// 基线：sink token 和 sparse token 分开处理
// 先处理 sink（独立循环），再处理 sparse（独立循环）
for (int i = 0; i < sinkNum; i++) { ProcessSink(i); }
for (int s2 = 0; s2 < s2End; s2++) { ProcessSparse(s2); }
// 两个循环无法共享流水线
```

// === 改造后（专家模式）===
```cpp
// Sink task 注入流水线
void FlashAttention() {
    for (int s1g = 0; s1g < s1gLoops; s1g++) {
        // 第一个 task: sink token
        if (keySinkNumber > 0) {
            TaskInfo sinkTask;
            sinkTask.isSinkTensor = true;
            sinkTask.s2Start = 0;
            sinkTask.s2End = sinkNumber;
            ExecuteTask(sinkTask);  // 从 keySink/valueSink 读取
        }
        // 后续 tasks: sparse token
        for (int s2 = sinkNumber; s2 < s2End; s2 += sInnerSize) {
            TaskInfo sparseTask;
            sparseTask.isSinkTensor = false;
            ExecuteTask(sparseTask);  // 从 KV cache 读取
        }
    }
}
```

## 关键修改点

1. 预期收益: sink 和 sparse 统一在同一流水线中处理；无需额外计算阶段

## 常见陷阱

⚠️ 每个 S1G block 多一个 sink task，增加总 task 数
⚠️ sink tensor 需要额外的 GM 空间
⚠️ sInnerSize 减小（256）可能降低有效 block 的计算效率

## 代码搜索关键词

```bash
grep -n "SyncAll\|PipeBarrier\|ExecuteTask\|PRELOAD\|tileSize\|ubFactor\|Tiling\|BLOCK_DIM" op_kernel/*.cpp op_host/*_tiling.cpp
```
