# P11 Playbook: Tail Block Handling (尾块与边界处理实操流程)

> 本 Playbook 为**强制流程**。采纳 P11 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步，禁止"看起来改了"就声明完成。
>
> P11 的核心是**当数据总量不能被 tileSize 或 blockDim 整除时，为最后一块（尾块）和最后一个核（尾核）提供安全且正确的处理路径**。它不改变主循环的高效路径，只补充边界保护。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p11_locations.txt`：

```bash
# 1. 主循环（for/while，计算或搬运）
grep -n "for\s*(.*;.*;.*)\|while\s*(" shared/original/op_kernel/*.cpp > /tmp/p11_locations.txt
# 2. tileSize / blockSize / elemsPerCore 计算
grep -n "tileSize\|tile_size\|blockSize\|elemsPerCore\|dataCount\|totalElems" \
    shared/original/op_kernel/*.cpp shared/original/op_host/*_tiling.cpp >> /tmp/p11_locations.txt
# 3. blockIdx / GetBlockNum / 核数相关
grep -n "blockIdx\|GetBlockNum\|BLOCK_DIM\|usedCoreNum\|coreNum" \
    shared/original/op_kernel/*.cpp shared/original/op_host/*.cpp >> /tmp/p11_locations.txt
# 4. 已有 tail / remain / remainder 处理（可能已有部分实现）
grep -n "tail\|remain\|remainder\|lastCore\|endIdx\|boundary" \
    shared/original/op_kernel/*.cpp shared/original/op_host/*.cpp >> /tmp/p11_locations.txt
# 5. DataCopy / CopyIn / CopyOut 调用（尾块通常需要特殊搬运量）
grep -n "DataCopy\|CopyIn\|CopyOut" shared/original/op_kernel/*.cpp >> /tmp/p11_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **主循环位置**：文件 + 行范围
- **tileSize 定义**：当前值 + 计算位置
- **总数据量**：`totalElems` / `dataCount` / `shape` 的表达式
- **核数分配**：`blockDim` / `elemsPerCore` 的计算
- **已有 tail 处理**：是否已有尾块/尾核分支（文件 + 行号）
- **DataCopy 调用**：所有搬运点的文件 + 行号

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| tileSize | `?` | 不变（主循环保持） | `?_tiling.cpp:L?` |
| totalElems | `?` | 不变 | `?_tiling.cpp:L?` |
| 主循环次数 | `totalElems / tileSize` | `mainLoops = totalElems / tileSize` | `?_kernel.cpp:L?` |
| 尾块大小 | 无 / 越界 | `tailElems = totalElems % tileSize` | `?_kernel.cpp:L?` |
| 尾核数据量 | 无 / 越界 | `tailCoreElems = totalElems - normCoreElems * (blockDim - 1)` | `?_kernel.cpp:L?` |
| 尾块搬运 API | `DataCopy(..., tileSize)` | `DataCopyPad` 或带 `count=tailElems` | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的循环和分块结构，判断你的代码属于以下哪种形态：

- **形态 α — 单维度尾块**（最常见）：只有一个 spatial 维度需要切 tile，尾块是该维度的剩余数据
- **形态 β — 多维度尾块**：两个维度同时切分（如 M×N），每个维度都有尾块，形成 2×2=4 种 block 类型（正常+正常、正常+尾、尾+正常、尾+尾）
- **形态 γ — 跨核尾核**：数据量不能被 `blockDim` 整除，最后一个核处理的数据量少于其他核

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 α — 最常见）

```cpp
// === 改造前（无尾块保护，可能越界）===
for (uint32_t i = 0; i < loops; i++) {
    DataCopy(ubLocal, gmTensor[i * tileSize], tileSize);
    Compute(ubLocal, tileSize);
    DataCopy(gmOut[i * tileSize], ubLocal, tileSize);
}

// === 改造后（主循环 + 尾块分支）===
uint32_t mainLoops   = totalElems / tileSize;
uint32_t tailElems   = totalElems % tileSize;

// 主循环：固定 tileSize，可展开/向量化
for (uint32_t i = 0; i < mainLoops; i++) {
    DataCopy(ubLocal, gmTensor[i * tileSize], tileSize);
    Compute(ubLocal, tileSize);
    DataCopy(gmOut[i * tileSize], ubLocal, tileSize);
}

// 尾块：特殊处理（仅当 tailElems > 0）
if (tailElems > 0) {
    uint32_t tailOffset = mainLoops * tileSize;
    // 方案 A：DataCopyPad（推荐，自动 padding 到 tileSize）
    DataCopyPadParams padParams;
    padParams.isPad = true;
    padParams.leftPadding  = 0;
    padParams.rightPadding = tileSize - tailElems;
    padParams.paddingValue = 0;
    DataCopyPad(ubLocal, gmTensor[tailOffset], tailElems, padParams);
    Compute(ubLocal, tailElems);   // 注意：只算 tailElems，不能算 tileSize
    DataCopy(gmOut[tailOffset], ubLocal, tailElems);  // 只写出实际数据
}
```

### 3C. Variant Notes（若是形态 β 或 γ）

- **形态 β（多维度尾块）**：Host 侧计算 4 种 block 大小组合（normal×normal、normal×tail、tail×normal、tail×tail），Kernel 内按 `blockIdx` 或 `TilingKey` 分支选择。每种分支的 `Compute` 传入不同的实际数据量。
- **形态 γ（跨核尾核）**：Host 侧分配数据时，前 `blockDim - 1` 个核各处理 `elemsPerCore`，最后一个核处理 `totalElems - elemsPerCore * (blockDim - 1)`。Kernel 内通过 `blockIdx` 判断自己是尾核还是正常核。
- **如果尾块很小（tailElems < 16）**：DataCopyPad 的 padding 开销占比大。可考虑将尾块合并到前一个正常 tile 中（前一个 tile 实际处理 `tileSize + tailElems`），减少一次 DMA 调用。但需确保不越界。
- **与 P7（32B 对齐）协同**：尾块的数据量可能不是 32B 对齐。`DataCopyPad` 可自动对齐；若不用 Pad，需手动将 `tailElems` 向上对齐到 32B 倍数，但 Compute 时只处理原始 `tailElems`。
- **与 P1（双缓冲）协同**：尾块循环通常只有 1 次迭代，双缓冲无收益。尾块分支保持单缓冲，主循环保持双缓冲。

## Step 4: 约束复核（防崩溃）

**公式**：
```
约束 1: mainLoops × tileSize + tailElems == totalElems
约束 2: tailElems < tileSize（尾块必须小于正常 tile）
约束 3: tailElems ≥ 0（不能为负）
约束 4: 尾块 GM 偏移 = mainLoops × tileSize ≤ totalElems（不越界）
约束 5: DataCopy(尾块) 的 count = tailElems（不能写成 tileSize）
```

**跨核尾核额外约束**：
```
约束 6: (blockDim - 1) × normCoreElems + tailCoreElems == totalElems
约束 7: tailCoreElems ≤ normCoreElems（尾核数据量不超过正常核）
约束 8: tailCoreElems ≥ 0
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**（每种约束的实际值 + 是否通过）。

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。

```bash
# 检查 1: 存在尾块大小计算（取模或减法）
grep -cE "%\s*tileSize|totalElems\s*-\s*mainLoops|tailElems|remainder" \
    modified_files/op_kernel/*.cpp modified_files/op_host/*.cpp
# 期望: >= 1

# 检查 2: 存在尾块分支（if tail / if remainder / if lastCore）
grep -cE "if\s*\(.*tail|if\s*\(.*remain|if\s*\(.*lastCore|if\s*\(.*endIdx" \
    modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: 尾块搬运量不等于 tileSize（必须小于 tileSize）
# 查找 DataCopy 调用中是否有变量参与 count（而非硬编码 tileSize）
grep -E "DataCopy\s*\([^,]+,\s*[^,]+,\s*[a-zA-Z_]" modified_files/op_kernel/*.cpp | \
    grep -vE "tileSize| TILE_SIZE" | wc -l
# 期望: >= 1（至少有一处 DataCopy 用变量做 count，对应尾块）

# 检查 4: 无越界写（GM 输出不超过 totalElems）
# 确保尾块写出量等于 tailElems，不是 tileSize
grep -E "DataCopy\s*\(.*tailOffset.*tileSize\)|DataCopyPad\(.*tailOffset.*tileSize\)" \
    modified_files/op_kernel/*.cpp | wc -l
# 期望: == 0（如果匹配到，说明尾块可能按 tileSize 写出，会越界）

# 检查 5: 主循环和尾块的总覆盖等于 totalElems
# 通过代码审查确认：mainLoops * tileSize + tailElems == totalElems
# （此条为逻辑检查，agent 需在 implementation_note 中手写验证）
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 运行时：尾块数据错乱 / 越界 | 检查 `DataCopy(尾块)` 的 count 是否为 `tailElems` 而非 `tileSize`；检查 `tailOffset` 是否为 `mainLoops * tileSize` |
| 运行时：精度对不上（尾块结果与 baseline 差异大）| `DataCopyPad` 的 `paddingValue` 必须是 0（identity element）；若 padding 值非 0 且 Compute 包含累加，padding 数据会污染结果 |
| 性能与 baseline 持平 | 若 shape 本来就是 tileSize 的整数倍 → P11 无意义。检查 `totalElems % tileSize == 0` |
| 编译失败：DataCopyPad 模板参数错误 | `DataCopyPad` 需要 `DataCopyPadParams` 结构体；检查是否遗漏了参数初始化 |
| 尾块太小（< 8 elements）导致 DMA 效率极低 | 将尾块合并到前一个正常 tile（前一个 tile 处理 `tileSize + tailElems`），减少一次 DMA。需确保前一个 tile 不越界 |
| 多维度尾块漏了一种组合 | 形态 β 有 4 种 block 类型。检查是否遗漏了 `tail×tail` 组合（最容易漏） |
| 跨核尾核负载不均衡 | 尾核数据量远小于正常核 → 尾核成为瓶颈。考虑调整 `blockDim` 使分配更均匀（如用 P4 负载均衡） |
| 尾块分支与双缓冲冲突 | 尾块通常只有 1 次迭代，不需要双缓冲。确保尾块分支不引用 `CopyIn(i+1)` 等双缓冲模式 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P11 Playbook Completion]
Step 1: done (/tmp/p11_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: constraints calc: mainLoops=total/aligned tailElems=total%aligned TILE_ELEMS_ALIGNED≥TILE_ELEMS: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
