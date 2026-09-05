# P57 Playbook: FlashDecode G 分核归约

> 本 Playbook 为**强制流程**。采纳 P57 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P57 的核心是**在 FlashDecode 场景下，当 B 和 N2 很小而 G 很大时，对 G 轴进行分核，实现多核并行归约，充分利用硬件资源**。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p57_locations.txt`：

```bash
# 1. FlashDecode 场景
grep -n "FlashDecode|flashdecode|Flash.*Decode|decode|Decode" \
    shared/original/op_kernel/*.cpp > /tmp/p57_locations.txt
# 2. 分核参数
grep -n "batchSize|kvHeadNum|gSize|GSize|coreNum|blockIdx|GetBlockNum" \
    shared/original/op_kernel/*.cpp >> /tmp/p57_locations.txt
# 3. 当前分核逻辑
grep -n "usedCores|coreNumPerGD|gSizeSub|gOuter|gSizeStart" \
    shared/original/op_kernel/*.cpp >> /tmp/p57_locations.txt
# 4. Tiling 文件
grep -n "gSizeMin|gSizeTail|SplitG|FlashDecodeComputeSplitG" \
    shared/original/op_kernel/*.cpp >> /tmp/p57_locations.txt
# 5. 核数修正
grep -n "coreNumPerGDFix|tmpBlockIdx|return" \
    shared/original/op_kernel/*.cpp >> /tmp/p57_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **FlashDecode 位置**：FlashDecode 相关代码位置
- **分核参数**：batchSize, kvHeadNum, gSize, coreNum
- **当前分核逻辑**：按 BN2 分核 / 其他方式
- **Tiling 文件**：gSizeMin, gSizeSub 等参数
- **核数修正**：coreNumPerGD, coreNumPerGDFix

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| B * N2 | `?` | 不变 | `?_tiling.cpp:L?` |
| G 大小 | `?` | 不变 | `?_tiling.cpp:L?` |
| 当前分核 | `?` (BN2/其他) | G 分核 | `?_tiling.cpp:L?` |
| G 最小切分 | `?` (无/有) | 16 | `?_tiling.cpp:L?` |
| 核数修正 | `?` (无/有) | 有 | `?_tiling.cpp:L?` |
| 超额核处理 | `?` (无/有) | return | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的 G 大小和核利用率，判断你的代码属于以下哪种形态：

- **形态 α — G 分核 + 核数修正（最常见）**：G 轴切分后，修正实际使用的核数，避免超额分配。
- **形态 β — G 分核 + 归约合并**：G 切分后各核结果需归约合并，处理跨核边界。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta`

### 3B. Canonical Template（形态 α — G 分核 + 核数修正）

```cpp
// === 改造前（仅按 BN2 分核，G 大时利用率低）===
__aicore__ inline void TilingNaive(uint32_t batchSize, uint32_t kvHeadNum,
                                     uint32_t gSize, uint32_t coreNum) {
    uint32_t usedCores = batchSize * kvHeadNum;  // 可能远小于总核数
    // ❌ 当 G 很大时，大量核闲置
}

// === 改造后（G 分核，充分利用核数）===
__aicore__ inline void TilingOptimized(uint32_t batchSize, uint32_t kvHeadNum,
                                         uint32_t gSize, uint32_t coreNum) {
    if (batchSize * kvHeadNum <= 24) {
        // Step 1: 计算每个 G 分片可分配的核数
        uint32_t coreNumPerGD = coreNum / (batchSize * kvHeadNum);
        
        // Step 2: G 最小切分大小（经验值 16，避免过小导致劣化）
        uint32_t gSizeMin = 16;
        uint32_t maxCoreNumPerGD = (gSize + gSizeMin - 1) / gSizeMin;
        coreNumPerGD = (coreNumPerGD > maxCoreNumPerGD) ? maxCoreNumPerGD : coreNumPerGD;
        
        // Step 3: 计算 G 子块大小
        uint32_t gSizeSub = (gSize + coreNumPerGD - 1) / coreNumPerGD;
        
        // Step 4: 修正实际核数
        uint32_t coreNumPerGDFix = (gSize + gSizeSub - 1) / gSizeSub;
        coreNumPerGD = (coreNumPerGD > coreNumPerGDFix) ? coreNumPerGDFix : coreNumPerGD;
        
        // Step 5: 超额核直接返回
        uint32_t totalUsedCores = batchSize * kvHeadNum * coreNumPerGD;
        if (tmpBlockIdx >= totalUsedCores) {
            return;  // 超出分配核数，直接返回
        }
        
        // Step 6: 计算当前核的 G 范围
        uint32_t gSizeTail = gSize - (coreNumPerGD - 1) * gSizeSub;
        uint32_t gOuter = coreNumPerGD;
        uint32_t gSizeStart = 0;
        
        FlashDecodeComputeSplitG();
        return;
    }
    // BN2 较大时，回退到原有分核逻辑
}
```

### 3C. Variant Notes（若是形态 β）

- **形态 β（G 分核 + 归约合并）**：
  当 G 切分后各核结果需要归约时：
  ```cpp
  // 各核计算部分 G 的结果
  LocalTensor<float> partialResult = ComputePartialG(...);
  
  // 跨核归约（需 workspace 存储中间结果）
  if (needCrossCoreReduce) {
      DataCopy(workspaceGm[tmpBlockIdx], partialResult, ...);
      SyncAll();
      // 主核归约所有部分结果
      if (isMainCore) {
          Reduce(finalResult, workspaceGm, totalUsedCores);
      }
  }
  ```

- **与 P58 的协同**：P58（TND 负载均衡）处理变长序列的跨核负载均衡，P57 处理 G 轴的并行分核。两者可同时存在：P58 均衡序列长度，P57 增加 G 轴并行度。
- **与 P67 的边界**：P67（Multi-Shape Migration）处理多 shape 支持，P57 处理特定 shape（B*N2 小，G 大）的分核优化。两者可同时存在。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: gSizeMin 必须 >= 16，过小（<=4）会导致性能劣化
约束 2: coreNumPerGD 不能超过 maxCoreNumPerGD，否则 gSizeSub < gSizeMin
约束 3: 超额核必须直接 return，不能参与计算
约束 4: gSizeTail 计算必须正确：gSize - (coreNumPerGD - 1) * gSizeSub
约束 5: 仅当 batchSize * kvHeadNum <= 24 时才启用 G 分核，否则回退原有逻辑
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `B * N2 = ?`, `G = ?`, `总核数 = ?`
- `coreNumPerGD = ?`, `gSizeSub = ?`, `gSizeMin = ?`
- `总使用核数 = ?`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 有 G 分核逻辑
grep -cE "coreNumPerGD|gSizeSub|gOuter|gSizeStart" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 有 G 最小切分限制
grep -cE "gSizeMin|g_size_min|16" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: 有核数修正
grep -cE "coreNumPerGDFix|maxCoreNumPerGD" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: 有超额核处理
grep -cE "tmpBlockIdx.*>=.*return|return.*blockIdx" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: 有 FlashDecode 调用
grep -cE "FlashDecode|flashdecode|FlashDecodeComputeSplitG" modified_files/op_kernel/*.cpp
# 期望: >= 1
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：gSizeMin 未定义 | 确认常量定义 `constexpr uint32_t gSizeMin = 16;` |
| 运行时：核利用率仍低 | 检查 `batchSize * kvHeadNum <= 24` 的条件是否命中。若 BN2 更大，G 分核不触发 |
| 运行时：G 切分后结果错误 | 检查 gSizeTail 计算。最后一个分片的 gSizeTail 可能不同于 gSizeSub |
| gSizeSub < gSizeMin | 检查 maxCoreNumPerGD 计算。coreNumPerGD 不能超过 maxCoreNumPerGD |
| 超额核未 return | 确认 `if (tmpBlockIdx >= totalUsedCores) { return; }` 存在且条件正确 |
| 核数修正后仍超额 | coreNumPerGDFix = (gSize + gSizeSub - 1) / gSizeSub。修正后 coreNumPerGD 必须 <= coreNumPerGDFix |
| 与 P58 冲突 | P58 也修改分核逻辑。两者同时存在时，G 分核应在 TND 分核之后执行 |
| 性能劣化 | gSizeMin = 16 是经验值。若 G < 16，不应触发 G 分核 |
| 归约结果不一致 | 形态 β 的跨核归约需确保 workspace 大小足够存储所有部分结果 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P57 Playbook Completion]
Step 1: done (/tmp/p57_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta, canonical/variant applied
Step 4: constraints: B*N2=? G=? coreNumPerGD=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
