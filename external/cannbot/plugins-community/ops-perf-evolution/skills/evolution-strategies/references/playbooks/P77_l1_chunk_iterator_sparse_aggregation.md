# P77 Playbook: L1 Chunk Iterator 稀疏块自适应聚合

> 本 Playbook 为**强制流程**。采纳 P77 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P77 的核心是**在 Sparse Attention 推理中，将多个不连续的 sparse block 动态聚合为一个 L1 chunk，使每个 chunk 恰好填满一个 L1 KP buffer（64KB），最大化 L1 利用率**。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p77_locations.txt`：

```bash
# 1. Sparse Attention
grep -n "sparse|Sparse|SparseAttention|sparseBlock|block.*sparse|SparseBlock" \
    shared/original/op_kernel/*.cpp > /tmp/p77_locations.txt
# 2. L1 buffer
grep -n "L1|l1Tensor|l1Buffer|LoadData|l1BufferCapacity|l1.*buffer" \
    shared/original/op_kernel/*.cpp >> /tmp/p77_locations.txt
# 3. 当前块处理
grep -n "for.*sparseBlockCount|LoadData.*sparseIdx|sparseIdx" \
    shared/original/op_kernel/*.cpp >> /tmp/p77_locations.txt
# 4. Iterator 模式
grep -n "Iterator|iterator|Advance|chunkStart|chunkEnd" \
    shared/original/op_kernel/*.cpp >> /tmp/p77_locations.txt
# 5. Causal mask
grep -n "causal|Causal|mask|Mask|FastAdvance" \
    shared/original/op_kernel/*.cpp >> /tmp/p77_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **Sparse 位置**：sparse block 定义和使用
- **L1 buffer**：L1 容量、当前 L1 使用
- **当前块处理**：逐个 block 搬运的代码
- **Iterator**：是否有迭代器模式
- **Causal mask**：FastAdvance 相关

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| sparse block | `?` (数量/大小) | 不变 | `?_kernel.cpp:L?` |
| L1 容量 | `?` (64KB/其他) | 64KB | `?_kernel.cpp:L?` |
| 当前搬运 | `?` (逐个/聚合) | 聚合 | `?_kernel.cpp:L?` |
| Iterator | `?` (无/有) | L1ChunkIterator | `?_kernel.cpp:L?` |
| Causal mask | `?` (有/无) | FastAdvance | `?_kernel.cpp:L?` |
| 碎片 | `?` (有/无) | 最小化 | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的 block 大小分布和 L1 容量，判断你的代码属于以下哪种形态：

- **形态 α — L1 Chunk Iterator 动态聚合（最常见）**：用迭代器动态聚合多个 block 到一个 chunk。
- **形态 β — 固定 chunk 大小**：预设 chunk 大小，简单聚合。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta`

### 3B. Canonical Template（形态 α — L1 Chunk Iterator）

```cpp
// === 改造前（逐个 block 搬运，L1 利用率低）===
__aicore__ inline void LoadSparseNaive(LocalTensor<half> kvGm,
                                         uint32_t* sparseIdx,
                                         uint32_t sparseBlockCount,
                                         uint32_t headDim) {
    for (int i = 0; i < sparseBlockCount; i++) {
        // ❌ 每个 block 一次 L1 搬运，利用率低
        LoadData(l1Tensor, kvGm[sparseIdx[i] * headDim], params);
    }
}

// === 改造后（L1ChunkIterator 动态聚合）===
class L1ChunkIterator {
public:
    L1ChunkIterator(uint32_t* sparseIdx, uint32_t sparseBlockCount,
                    uint32_t l1BufferCapacity)
        : sparseIdx_(sparseIdx), sparseBlockCount_(sparseBlockCount),
          l1BufferCapacity_(l1BufferCapacity), chunkEndIdx_(0) {}
    
    int64_t Advance() {
        chunkStartIdx_ = chunkEndIdx_;
        int64_t accumulated = 0;
        
        while (chunkEndIdx_ < sparseBlockCount_) {
            int64_t blockSize = GetBlockSize(chunkEndIdx_);
            if (accumulated + blockSize > l1BufferCapacity_) break;
            accumulated += blockSize;
            chunkEndIdx_++;
        }
        return accumulated;  // 当前 chunk 的总大小
    }
    
    uint32_t GetChunkStart() const { return chunkStartIdx_; }
    uint32_t GetChunkEnd() const { return chunkEndIdx_; }
    
private:
    uint32_t* sparseIdx_;
    uint32_t sparseBlockCount_;
    uint32_t l1BufferCapacity_;
    uint32_t chunkStartIdx_;
    uint32_t chunkEndIdx_;
};

__aicore__ inline void LoadSparseOptimized(LocalTensor<half> kvGm,
                                            uint32_t* sparseIdx,
                                            uint32_t sparseBlockCount,
                                            uint32_t headDim,
                                            uint32_t l1BufferCapacity) {
    L1ChunkIterator iter(sparseIdx, sparseBlockCount, l1BufferCapacity);
    
    while (iter.GetChunkEnd() < sparseBlockCount) {
        int64_t chunkSize = iter.Advance();
        uint32_t start = iter.GetChunkStart();
        uint32_t end = iter.GetChunkEnd();
        
        // 聚合搬运：一次搬运多个 block
        for (uint32_t i = start; i < end; i++) {
            LoadData(l1Tensor + offset, kvGm[sparseIdx[i] * headDim], blockParams);
            offset += GetBlockSize(i);
        }
        
        // 处理当前 chunk
        ComputeChunk(l1Tensor, chunkSize);
    }
}
```

### 3C. Variant Notes（若是形态 β）

- **形态 β（固定 chunk 大小）**：
  预设 chunk 大小，简单聚合：
  ```cpp
  constexpr uint32_t CHUNK_SIZE = 64 * 1024;  // 64KB
  for (uint32_t i = 0; i < sparseBlockCount; i += blocksPerChunk) {
      uint32_t end = min(i + blocksPerChunk, sparseBlockCount);
      AggregateAndLoad(i, end);
  }
  ```

- **与 P57 的边界**：P57 处理 G 分核，P77 处理 L1 聚合。两者可同时存在。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: L1 容量通常为 64KB，不能超过
约束 2: 聚合的 block 必须连续存储在 L1 上
约束 3: sparseBlockSize >= L1 容量时退化为单块迭代
约束 4: Causal mask 裁剪后的无效区域需 FastAdvance 跳过
约束 5: DMA 搬运可能产生碎片，需计算正确偏移
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `sparseBlockCount = ?`, `L1 容量 = ?`
- `chunk 数量 = ?`, `碎片 = ?`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 有 L1ChunkIterator 或 chunk 逻辑
grep -cE "L1ChunkIterator|chunkStart|chunkEnd|Advance" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 有聚合搬运（非逐个 block）
grep -cE "for.*chunk|while.*chunk|Aggregate" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: 有 L1 容量检查
grep -cE "l1BufferCapacity|64.*1024|65536" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: 无逐个 block 搬运（循环内独立 LoadData）
grep -cE "for.*sparseBlockCount.*LoadData" modified_files/op_kernel/*.cpp
# 期望: == 0（或 note 中说明 "回退路径"）

# 检查 5: 有 sparse 引用
grep -cE "sparse|Sparse|sparseBlock" modified_files/op_kernel/*.cpp
# 期望: >= 1
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：L1ChunkIterator 未定义 | 确认类定义在 kernel 文件内或头文件中 |
| 运行时：L1 溢出 | 检查 accumulated + blockSize <= l1BufferCapacity |
| 运行时：聚合后数据错乱 | 确认 block 在 L1 上的偏移计算正确 |
| sparseBlockSize >= L1 | 退化为单块迭代，Advance 只取一个 block |
| Causal mask 未处理 | 添加 FastAdvance 跳过无效区域 |
| DMA 碎片 | 不连续 block 的搬运可能产生碎片。检查地址连续性 |
| 迭代器状态错误 | 确认 chunkStartIdx_ 和 chunkEndIdx_ 的更新顺序 |
| 性能不如预期 | 检查聚合后 chunk 是否接近 L1 容量。太小则收益低 |
| 与 P57 冲突 | P57 和 P77 都修改迭代逻辑。不冲突，P77 在 L1 层面 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P77 Playbook Completion]
Step 1: done (/tmp/p77_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta, canonical/variant applied
Step 4: constraints: l1Capacity=? sparseBlockCount=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
