# P58 Playbook: TND 负载均衡分核

> 本 Playbook 为**强制流程**。采纳 P58 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P58 的核心是**在变长序列场景下，用线段抽象和贪心算法将序列按长度均分到各核，避免传统 BS 切分导致的负载不均衡**。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p58_locations.txt`：

```bash
# 1. 变长序列
grep -n "actualKVS|seqLen|sequence|variable.*len|dynamic.*len|actualLen" \
    shared/original/op_kernel/*.cpp > /tmp/p58_locations.txt
# 2. 分核逻辑
grep -n "batchSize|coreNum|blockIdx|SplitCore|GetBlockNum|partition" \
    shared/original/op_kernel/*.cpp >> /tmp/p58_locations.txt
# 3. 线段抽象
grep -n "segment|线段|totalS2Length|avgS2Length|recordSplitInfo" \
    shared/original/op_kernel/*.cpp >> /tmp/p58_locations.txt
# 4. 当前 BS 切分
grep -n "assignToCore|for.*batchSize|bIdx.*core" \
    shared/original/op_kernel/*.cpp >> /tmp/p58_locations.txt
# 5. FlashDecode 归约
grep -n "FlashDecode|flashdecode|lseMax|lseSum|accumOut" \
    shared/original/op_kernel/*.cpp >> /tmp/p58_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **变长序列位置**：actualKVS, seqLen 等参数位置
- **分核逻辑**：当前按 BS 切分的代码位置
- **线段抽象**：segment, totalS2Length 等
- **当前 BS 切分**：assignToCore 等
- **FlashDecode 归约**：lseMax, lseSum, accumOut

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 序列类型 | `?` (变长/定长) | 不变 | `?_tiling.cpp:L?` |
| 当前切分 | `?` (BS/其他) | 线段均分 | `?_tiling.cpp:L?` |
| 基本单位 | `?` (无/512) | 512 (KVS) | `?_tiling.cpp:L?` |
| 跨核切分 | `?` (无/有) | 有 + 记录 | `?_tiling.cpp:L?` |
| 归约信息 | `?` (无/有) | 有 (lseMax/lseSum) | `?_kernel.cpp:L?` |
| 负载均衡指标 | `?` (无/有) | avgS2Length | `?_tiling.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的序列变长程度和归约需求，判断你的代码属于以下哪种形态：

- **形态 α — 线段均分（最常见）**：将变长序列抽象为线段，按长度均分到各核，无跨核切分。
- **形态 β — 线段均分 + 跨核归约**：线段可能跨核切分，需记录切分信息并用 FlashDecode 归约。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta`

### 3B. Canonical Template（形态 α — 线段均分）

```cpp
// === 改造前（按 BS 切分，负载不均）===
__aicore__ inline void PartitionNaive(uint32_t batchSize, uint32_t coreNum,
                                        uint32_t* actualKVS) {
    for (uint32_t b = 0; b < batchSize; b++) {
        assignToCore(b, coreIdx);  // 短序列和长序列可能分到同一核
    }
}

// === 改造后（线段均分，负载均衡）===
__aicore__ inline void PartitionOptimized(uint32_t batchSize, uint32_t coreNum,
                                            uint32_t* actualKVS) {
    constexpr uint32_t KVS_UNIT = 512;
    
    // Step 1: 计算线段总长度（以 KVS_UNIT 为单位）
    uint32_t totalS2Length = 0;
    for (uint32_t b = 0; b < batchSize; b++) {
        totalS2Length += (actualKVS[b] + KVS_UNIT - 1) / KVS_UNIT;  // 向上取整
    }
    
    // Step 2: 平均每个核的线段数
    uint32_t avgS2Length = (totalS2Length + coreNum - 1) / coreNum;
    
    // Step 3: 贪心分配线段到各核
    uint32_t currentCore = 0;
    uint32_t currentLen = 0;
    for (uint32_t b = 0; b < batchSize; b++) {
        uint32_t segmentLen = (actualKVS[b] + KVS_UNIT - 1) / KVS_UNIT;
        
        if (currentLen + segmentLen > avgS2Length && currentLen > 0) {
            currentCore++;
            currentLen = 0;
        }
        
        assignSegmentToCore(b, currentCore);
        currentLen += segmentLen;
    }
}
```

### 3C. Variant Notes（若是形态 β）

- **形态 β（跨核切分 + FlashDecode 归约）**：
  当线段跨核切分时，需记录信息并归约：
  ```cpp
  // 记录跨核切分信息
  for (each segment) {
      if (segment crosses core boundary) {
          recordSplitInfo(b, s1Outer, kvSplitNum);
          // workspace 存储 lseMax/lseSum/accumOut
          workspace[b].lseMax = ...;
          workspace[b].lseSum = ...;
          workspace[b].accumOut = ...;
      }
  }
  
  // FlashDecode 归约
  SyncAll();
  FlashDecodeReduce(workspace, finalOutput);
  ```

- **与 P57 的协同**：P57（G 分核）增加 G 轴并行度，P58 均衡序列负载。两者可同时存在。
- **与 P67 的边界**：P67 处理多 shape 支持，P58 处理特定变长 shape 的负载均衡。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: KVS_UNIT 必须是 512（经验最优值）
约束 2: avgS2Length 向上取整，确保所有线段都被分配
约束 3: 跨核切分时 workspace 必须足够存储 lseMax/lseSum/accumOut
约束 4: 贪心分配时 currentLen 为 0 时可跳过边界检查（避免空核）
约束 5: 定长序列场景无需线段抽象，直接回退 BS 切分
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `batchSize = ?`, `coreNum = ?`, `totalS2Length = ?`
- `avgS2Length = ?`, `KVS_UNIT = ?`
- `跨核切分数 = ?`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 有 totalS2Length 计算
grep -cE "totalS2Length|total.*Length" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 有 avgS2Length 计算
grep -cE "avgS2Length|avg.*Length" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: 有 KVS_UNIT 或基本单位
grep -cE "KVS_UNIT|512|kvs_unit" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: 有跨核切分记录（形态 beta）或无跨核切分（形态 alpha）
grep -cE "recordSplitInfo|cross.*core|lseMax|lseSum" modified_files/op_kernel/*.cpp
# 期望: >= 1（形态 beta）或 == 0（形态 alpha，需 note 说明）

# 检查 5: 无纯 BS 切分（循环内直接 assignToCore(b, coreIdx)）
grep -cE "assignToCore.*b.*coreIdx|for.*batchSize.*assignToCore" modified_files/op_kernel/*.cpp
# 期望: == 0（或 note 中说明 "回退路径"）
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：actualKVS 未定义 | 确认变长序列长度参数已传入 tiling 函数 |
| 运行时：负载仍不均衡 | 检查 KVS_UNIT 是否为 512。过大或过小都会影响均衡度 |
| 运行时：线段遗漏 | 检查 avgS2Length 的向上取整。`(totalS2Length + coreNum - 1) / coreNum` |
| 跨核切分后结果错误 | 确认 workspace 已正确存储 lseMax/lseSum/accumOut，且 FlashDecode 归约顺序正确 |
| 空核问题 | 贪心分配时若 currentLen = 0 不检查边界，避免空核 |
| 定长序列误触发 | 添加判断：若所有 actualKVS 相同，回退到 BS 切分 |
| workspace 溢出 | 跨核切分的 workspace 大小 = batchSize * sizeof(splitInfo)。检查是否足够 |
| 与 P57 冲突 | P57 和 P58 都修改分核。P58 先执行（均衡负载），P57 后执行（G 分核） |
| 性能下降 | 线段抽象增加标量计算。若序列很短（<512）， overhead 可能大于收益 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P58 Playbook Completion]
Step 1: done (/tmp/p58_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta, canonical/variant applied
Step 4: constraints: totalS2Length=? avgS2Length=? KVS_UNIT=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
