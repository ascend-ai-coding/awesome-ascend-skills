---
id: X1
origin: discovered
discovered_round: 2
discovered_from: round_2/parallel_2
base_speedup: 2.58x
---

# Strategy X1: Inline Target Capture

## 核心思路
在 Online Softmax 的前向流式扫描过程中，当处理到包含目标索引的 tile 时，立即捕获目标元素的原始值（如 target logit），
避免在扫描完成后再用一次独立的 DataCopy 回读目标 tile。这将 "reduce + lookup" 两步操作融合为单次遍历，
对于 CrossEntropyLoss 等需要同时做全行归约和单点索引查找的算子，可完全消除一个独立的 pass。

## 适用场景
- **算子类型**: 需要全行归约（max/sum）+ 单点索引查找的算子，如 CrossEntropyLoss、NLLLoss、FocalLoss
- **瓶颈类型**: 内存带宽受限或 DataCopy 次数过多的场景
- **前提条件**: 已采用 Online Softmax 或类似流式归约算法（P13），目标索引在 Process 开始前已知
- **预期收益**: 每行减少 1 次 DataCopy（对于 K 个 tile，从 K+1 降至 K），收益随 K 减小（tile 越大）越显著

## 实现要点
1. 在行处理开始前，预计算目标所在的 tile ID 和 tile 内偏移：
   ```cpp
   uint32_t targetTileId = targetIdx / tileLength;
   uint32_t targetLocalIdx = targetIdx % tileLength;
   ```
2. 在 tile 循环内，Cast 到 fp32 后立即检查是否为目标 tile：
   ```cpp
   if (t == targetTileId) {
       targetLogitRaw = castLocal.GetValue(targetLocalIdx);
   }
   ```
3. 循环结束后直接使用已捕获的 targetLogitRaw，无需额外 DataCopy
4. 注意：targetLocalIdx 的 GetValue 是标量操作，对 Vec 流水线无影响；
   条件判断(`if (t == targetTileId)`)的分支开销极低（每行仅 K 次比较）

## 来源
自动发现于第 2 轮进化，算子 cross_entropy_loss，speedup 2.58x
