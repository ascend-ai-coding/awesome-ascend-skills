# P45 Playbook: 稀疏 Attention 离散块处理

> 本 Playbook 为**强制流程**。采纳 P45 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P45 的核心是**将 topK 选出的离散 block indices 预取到片上 cache 数组，封装为 L1ChunkIterator 迭代器，按 L1 容量自动切分连续段，使调度与离散寻址解耦**。

## Step 1: 定位关键结构

```bash
grep -n "topk|TopK|sparse|Sparse|block|Block|attention|Attention" \
    shared/original/op_kernel/*.cpp > /tmp/p45_locations.txt
grep -n "index|Index|indices|Indices|cache|Cache" \
    shared/original/op_kernel/*.cpp >> /tmp/p45_locations.txt
grep -n "Iterator|iterator|迭代器|chunk|Chunk|range|Range" \
    shared/original/op_kernel/*.cpp >> /tmp/p45_locations.txt
grep -n "GetBlockRange|GetSparseBlockCount|blockIndexCursor" \
    shared/original/op_kernel/*.cpp >> /tmp/p45_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前离散块处理方式**：文件 + 行号
- **索引管理方式**：文件 + 行号
- **L1 切分策略**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 索引管理 | `?` (逐计算) | cache 数组预取 | `op_kernel/*.cpp:L?` |
| 迭代器 | `?` (无) | L1ChunkIterator | `op_kernel/*.cpp:L?` |
| 切分 | `?` (手动) | 按 L1 容量自动 | `op_kernel/*.cpp:L?` |
| 调度 | `?` (耦合) | 与寻址解耦 | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整迭代器（cache + Next + 自动切分）**。
- **形态 β — 仅 Gather-Merge**：不封装迭代器，直接合并连续块。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
int32_t blockIndexCache_[TOPK_CACHE_SIZE] = {0};
__aicore__ inline bool Next() {
    while (blockIndexCursor_ < range_->GetSparseBlockCount()) {
        auto [blockStart, blockEnd] = range_->GetBlockRange(blockIndexCursor_, ...);
        if (currentL1Size_ + (chunkEnd - blockCursor_) >= range_->GetL1Size()) {
            l1ChunkJustCompleted_ = true;
        }
        currentSegment_ = {blockCursor_, chunkEnd, l1ChunkJustCompleted_};
        blockCursor_ = chunkEnd;
        return true;
    }
}
```

### 3C. Variant Notes

- TOPK_CACHE_SIZE=128 限制可缓存数量。
- 每个 chunk 边界需重新计算 GM 偏移，chunk 过小时标量开销占比增大。
- Gather-Merge 需要额外 workspace GM 空间。

## Step 4: 约束复核

- CACHE_SIZE=128 限制
- chunk 过小标量开销
- Gather-Merge workspace 需求

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "blockIndexCache_|TOPK_CACHE" modified_files/op_kernel/*.cpp  # >=1
grep -cE "L1ChunkIterator|ChunkIterator" modified_files/op_kernel/*.cpp  # >=1
grep -cE "GetBlockRange|GetSparseBlockCount" modified_files/op_kernel/*.cpp  # >=1
grep -cE "Next\(\)|blockIndexCursor_" modified_files/op_kernel/*.cpp  # >=1
grep -cE "currentSegment_|l1ChunkJustCompleted_" modified_files/op_kernel/*.cpp  # >=1
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| cache 不足 | 分页或增大 TOPK_CACHE_SIZE |
| chunk 过小 | 合并相邻块 |
| 标量开销高 | 预计算 offset |
| 非稀疏场景 | 不适用 |

---

**完成清单**：
```
[P45 Playbook Completion]
Step 1: done (/tmp/p45_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: CACHE_SIZE=128 限制; chunk 过小标量开销; Gather-Merge workspace 需求: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
