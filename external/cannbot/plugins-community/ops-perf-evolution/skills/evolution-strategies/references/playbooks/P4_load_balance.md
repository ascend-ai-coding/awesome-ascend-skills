# P4 Playbook: 多核负载均衡 (Multi-core Load Balancing)

> 本 Playbook 为**强制流程**。采纳 P4 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P4 的核心是**在 Host 侧 tiling 阶段重新设计数据到核的映射方式**，使各核的计算量/数据量差异最小化。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p4_locations.txt`：

```bash
# 1. 分核参数与调用点
grep -n "BLOCK_DIM\|GetBlockNum\|blockIdx\|coreNum\|aiCoreIdx" \
    shared/original/op_host/*_tiling.cpp shared/original/op_kernel/*.cpp > /tmp/p4_locations.txt
# 2. 简单均匀分核逻辑（重点改造对象）
grep -n "/\s*BLOCK_DIM\|/\s*coreNum\|rowsPerCore\|colsPerCore\|SplitCore\|SfaSplit" \
    shared/original/op_host/*_tiling.cpp >> /tmp/p4_locations.txt
# 3. 数据总量与维度定义
grep -n "totalRows\|totalElems\|GetShapeSize\|batchSize\|seqLen\|M\|N\|K" \
    shared/original/op_host/*_tiling.cpp shared/original/op_kernel/*.cpp >> /tmp/p4_locations.txt
# 4. 已有负载均衡或代价相关逻辑
grep -n "remainder\|costPrefixSum\|workload\|balance\|不均衡" \
    shared/original/op_host/*_tiling.cpp shared/original/op_kernel/*.cpp >> /tmp/p4_locations.txt
# 5. 核内循环边界（判断是否存在尾块）
grep -n "for.*blockIdx\|startRow\|endRow\|myRows\|myCols" \
    shared/original/op_kernel/*.cpp >> /tmp/p4_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **分核参数位置**：BLOCK_DIM / GetBlockNum / blockIdx 使用的文件 + 行号
- **当前分核策略**：是简单除法均匀分、还是有某种 Split 函数
- **数据总量**：各输入 tensor 的 shape、总元素数变量名
- **已有均衡逻辑**：是否已有 remainder 或 cost-aware 处理
- **核内边界**：Kernel 内循环是否依赖 blockIdx 计算起止

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 总分核数 | `?` (BLOCK_DIM / GetBlockNum) | 不变 | `?_tiling.cpp:L?` |
| 当前分核策略 | `?` (均匀除法 / Split函数 / 无) | `alpha/beta/gamma` 见 3A | `?_tiling.cpp:L?` |
| 数据分布 | `?` (均匀 / 稀疏 / 变长 / 2D) | 识别真实分布 | 分析结论 |
| 每核数据量 | `?` (固定 / 有remainder) | 差异 < 20% | `?_tiling.cpp:L?` |
| 核内边界变量 | `?` (startRow/endRow / 无) | 明确定义起止 | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的数据分布和分核方式，判断你的代码属于以下哪种形态：

- **形态 α — 均匀数据但总量不整除核数**：数据本身均匀（dense tensor），但 `totalRows % coreNum != 0`，导致余数行全落在最后一个核（或某些核拿到 0 行）。
- **形态 β — 数据不均匀（稀疏 / 变长 / MoE gating）**：各行/各块计算量差异大，均匀分会导致热点核过载。
- **形态 γ — 2D 数据（Matmul / MoE）**：数据天然是 M×N 网格，单维度分核会导致某些维度上核心利用率不足。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 α — 均匀数据余数均衡，最常见）

```cpp
// === 改造前（简单除法，余数全部压给最后一个核）===
uint32_t rowsPerCore = totalRows / coreNum;
uint32_t startRow = blockIdx * rowsPerCore;
uint32_t endRow = startRow + rowsPerCore;  // 最后一个核可能多干，也可能有核空闲

// === 改造后（remainder 均匀散到前 remainder 个核）===
uint32_t rowsPerCore = totalRows / coreNum;
uint32_t remainder = totalRows % coreNum;
uint32_t myRows = (blockIdx < remainder) ? (rowsPerCore + 1) : rowsPerCore;
uint32_t startRow = blockIdx * rowsPerCore + ((blockIdx < remainder) ? blockIdx : remainder);
uint32_t endRow = startRow + myRows;
```

**核内使用**：
```cpp
for (uint32_t row = startRow; row < endRow; row++) {
    // 处理该行
}
```

### 3C. Variant Notes（若是形态 β 或 γ）

- **形态 β（不均匀数据 / 代价感知）**：
  1. 在 Host 侧预计算每行/每块的代价（如非零元素数、序列长度、MoE gate 权重）。
  2. 构建前缀和 `costPrefixSum[rows+1]`。
  3. 每个核按代价比例取连续段：
  ```cpp
  uint32_t totalCost = costPrefixSum[totalRows];
  uint32_t myCostStart = blockIdx * totalCost / coreNum;
  uint32_t myCostEnd   = (blockIdx + 1) * totalCost / coreNum;
  // 通过二分查 prefixSum 找到对应的 startRow / endRow
  uint32_t startRow = lower_bound(costPrefixSum, myCostStart);
  uint32_t endRow   = lower_bound(costPrefixSum, myCostEnd);
  uint32_t myRows = endRow - startRow;
  ```
  ⚠️ 前缀和数组若很大，Host 侧二分即可；不要传到 Device。

- **形态 γ（2D M×N 网格分核）**：
  将 `coreNum` 拆分为 `coreNumM × coreNumN`（尽量接近平方根）：
  ```cpp
  uint32_t coreNumM = coreNum;  // 或根据 M/N 比例调整
  uint32_t coreNumN = 1;
  while (coreNumM > coreNumN * 2 && coreNumM % 2 == 0) {
      coreNumM /= 2; coreNumN *= 2;  // 尽量让 M/N 分配与数据比例匹配
  }
  uint32_t coreM = blockIdx % coreNumM;
  uint32_t coreN = blockIdx / coreNumM;
  uint32_t mPerCore = M / coreNumM;
  uint32_t mRem = M % coreNumM;
  uint32_t myM = (coreM < mRem) ? (mPerCore + 1) : mPerCore;
  uint32_t mStart = coreM * mPerCore + ((coreM < mRem) ? coreM : mRem);
  // 同理 nStart / myN
  ```
  若 M 或 N 小于 coreNum，2D 分核会导致部分核无工作，此时应退化为形态 α 的一维分核。

- **与 P11 的边界**：形态 α 解决的是**核间**余数分配（每核处理的行数），P11 解决的是**核内** tile 余数（每个 tile 内的尾块）。两者通常一起使用：P4 保证核间均衡，P11 保证核内尾块正确。

- **与 P51 的协同**：形态 β 的代价感知分核若仍无法均衡（如极端稀疏），可配合 P51 动态调度，将过载核的部分 work 推送到轻载核。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: startRow < endRow ≤ totalRows（每个核至少 0 行，最多 totalRows 行）
约束 2: 各核的 [startRow, endRow) 区间不重叠、无遗漏、覆盖 [0, totalRows)
约束 3: 对 2D 分核：coreNumM * coreNumN == coreNum，否则有核空闲
约束 4: 数据量 < coreNum 时，应让多余核 early-exit（if myRows == 0 return），不能访问空区间
约束 5: 修改分核边界后，必须重新验证 P11 尾块逻辑（start/end 变了，tile 内循环边界可能也需调整）
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `totalRows = ?`, `coreNum = ?`, `rowsPerCore = ?`, `remainder = ?`
- 核 0 的 `startRow / endRow / myRows = ?`
- 核 `coreNum-1` 的 `startRow / endRow / myRows = ?`
- 最大差异率 = `(max(myRows) - min(myRows)) / avg(myRows)`，必须 < 20%
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 存在 remainder / cost-aware / 2D 分核中的至少一种均衡逻辑
grep -cE "remainder|costPrefixSum|coreM|coreN|workloadBalance|loadBalance" \
    modified_files/op_host/*_tiling.cpp modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 无裸除法分核（blockIdx * (total / coreNum) 是反模式）
grep -cE "blockIdx\s*\*\s*.*\/\s*(coreNum|BLOCK_DIM|GetBlockNum)" \
    modified_files/op_kernel/*.cpp modified_files/op_host/*_tiling.cpp
# 期望: == 0

# 检查 3: 有明确的每核起止或数据量变量（不是直接用 blockIdx * fixedSize）
grep -cE "myRows|myCols|myCount|startRow|endRow|mStart|nStart|perCore" \
    modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: 有边界保护（防止越界或空核访问数据）
grep -cE "startRow.*total|endRow.*total|if.*myRows.*==.*0|if.*blockIdx.*>=.*coreNum" \
    modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: 分核参数通过 TilingData 或 constexpr 传入 Kernel，不是 hardcode
grep -cE "TilingData.*coreNum|coreNum.*Tiling|BLOCK_DIM" \
    modified_files/op_host/*_tiling.cpp modified_files/op_kernel/*.cpp
# 期望: >= 1
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：startRow/endRow 未定义 | 确保 TilingData 结构体新增了 startRow/endRow 字段，且 Kernel 侧通过 `tilingData.startRow` 读取 |
| 运行时：某些核处理 0 行但仍在执行 Compute | 在 Kernel 入口加 `if (myRows == 0) return;` 或跳过循环 |
| 运行时：最后一个核明显慢于其他核 | 检查 remainder 是否全压在最后一个核。应散到前 `remainder` 个核 |
| 精度对不上（形态 β） | `lower_bound` 在 Host 侧实现时，确保前缀和是闭区间 `[0, totalCost]`，且二分查找的是 cost 值而非行号 |
| 2D 分核后部分核无工作 | 若 M < coreNumM 或 N < coreNumN，应退化为一维分核或让多余核 early-exit |
| 修改分核后 P11 尾块失效 | 重新验证 `startRow + mainLoops * tileSize + tailElems == endRow` |
| 数据量小于核数时性能差 | 此时多核并行本身收益为负，应在 Host 侧判断 `if (totalRows < coreNum * threshold) BLOCK_DIM = 1` |
| costPrefixSum 占用过大 Host 内存 | 若行数极大（>1M），用采样近似：每 1024 行聚合成一个 bucket，再对 bucket 分核 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P4 Playbook Completion]
Step 1: done (/tmp/p4_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: constraints: totalRows=? coreNum=? max_diff_rate=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
