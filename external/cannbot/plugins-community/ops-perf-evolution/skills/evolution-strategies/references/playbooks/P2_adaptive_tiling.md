# P2 Playbook: Adaptive Tiling 实操流程

> 本 Playbook 为**强制流程**。采纳 P2 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。
> 关联设计：[knowledge-strategy-architecture-v3.2](../../../../../../docs/design/knowledge-strategy-architecture-v3.2.md)

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `implementation_note.txt` 的 "Playbook Step 1" 段落：

```bash
# 当前 tiling 计算位置
grep -n "Tiling\b\|TilingFunc\|FillTilingData\|tileSize\|ubFactor" op_host/*.cpp op_host/*_tiling.cpp > /tmp/p2_locations.txt

# 当前核数/分块
grep -n "BLOCK_DIM\|GetBlockNum\|blockDim\|usedCoreNum\|aivNum\|aicNum" op_host/*.cpp op_kernel/*.cpp >> /tmp/p2_locations.txt

# Kernel 内 UB 容量约束
grep -n "InitBuffer\|UB_SIZE\|maxFormerNum\|alignC\|tileLen" op_kernel/*.cpp op_kernel/*.h >> /tmp/p2_locations.txt
```

**交付物**（写入 implementation_note.txt "Playbook Step 1"）：
- 当前 tileSize 计算公式（host 侧）：文件 + 行号
- 当前 blockDim 计算公式（host 侧）：文件 + 行号
- Kernel 内的 UB 占用上限约束（可能是 InitBuffer 大小）：文件 + 行号
- 算子的关键 shape 维度（如 M, K, N, S, D 等）

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。需在 `implementation_note.txt` "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值/逻辑 | 目标值/逻辑 | 修改位置 |
|---|---|---|---|
| tileSize 计算 | 硬编码或单一公式 | 根据 UB - reserved - small_blocks 动态算 | `?_tiling.cpp:L?` |
| blockDim 选择 | 固定值（如 40） | 按 totalCnt / minBlockSize 自适应 | `?_tiling.cpp:L?` |
| TilingKey 分发 | 无 / 单 key | 多 key（按 dtype + shape 模式） | `?_tiling.cpp:L?` |
| Kernel 模板特化 | 单实例 | 多分支（如 SPLIT_C / SPLIT_W / MULTI_W）| `?_kernel.cpp` |

**如果你填不出任何一格** → Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位结果，判断算子属于以下哪种形态：

- **形态 α — 单维分块**（最常见）：算子只在一个 spatial 维度（如 N 或 M）上分块
- **形态 β — 多维分块**：在 (M, N) 等两个维度上同时切分（多发生在 matmul / FA）
- **形态 γ — 维度可选**：根据 shape 大小自动选择 SPLIT_C / SPLIT_W 等不同维度分块

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 α 自适应 tile）

```cpp
// === 改造前（固定 tile）===
constexpr uint32_t TILE_SIZE = 1024;
uint32_t blockDim = 40;
uint32_t elemsPerCore = (totalElems + blockDim - 1) / blockDim;

// === 改造后（自适应）===
constexpr uint32_t UB_SIZE       = 192 * 1024;     // 各芯片不同，请用 hw_params
constexpr uint32_t RESERVED_UB   = 20 * 1024;      // 栈 + 临时
constexpr uint32_t MIN_TILE_BYTES = 256;           // 太小则少切，避免 launch 开销

uint32_t availableUb  = UB_SIZE - RESERVED_UB;
uint32_t bytesPerElem = sizeof(T) * QUEUE_COUNT;   // QUEUE_COUNT 由算子决定
uint32_t maxTileElems = availableUb / bytesPerElem;
uint32_t tileSize     = std::min(maxTileElems, totalElems);

// blockDim 自适应：让 tail block 数据量也至少有 MIN_TILE_BYTES
uint32_t maxBlocks    = (totalElems * sizeof(T) + MIN_TILE_BYTES - 1) / MIN_TILE_BYTES;
uint32_t blockDim     = std::min(static_cast<uint32_t>(aivNum), maxBlocks);
uint32_t elemsPerCore = (totalElems + blockDim - 1) / blockDim;
```

### 3C. Variant Notes

- **形态 β（多维分块）**：`tileSize_M = sqrt(maxTileElems / aspect_ratio)`，`tileSize_N` 同理；避免一个维度全 UB
- **形态 γ（多模式）**：用 TilingKey 编码 (dtype, shape_type, mode)；kernel 内 `if (TilingKey & MODE_MASK)` 分支
- **UB 紧张**：QUEUE_COUNT 是 buffer 数 × 2（双缓冲）+ scratch；如果含 P1 双缓冲，bytesPerElem 翻倍

## Step 4: 约束复核（防崩溃）

**公式**：

```
新 UB 占用 = tileSize × bytesPerElem
约束 1: 新 UB 占用 ≤ availableUb
约束 2: blockDim ≤ aivNum（防超核数）
约束 3: elemsPerCore × blockDim ≥ totalElems（确保覆盖）
约束 4: 若启用 P1 双缓冲，tileSize ≥ 2（双缓冲意义）
```

在 `implementation_note.txt "Playbook Step 4"` 中报告具体数值（实际值 + 是否通过）。

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。

```bash
# 检查 1: tileSize 已改为运行时计算（不再是 constexpr hardcode）
grep -E "tileSize\s*=\s*[a-zA-Z_]" modified_files/op_host/*.cpp
# 期望: >= 1（必须有变量参与计算，不能纯字面量）

# 检查 2: blockDim 自适应（含 std::min 或类似公式）
grep -cE "blockDim\s*=\s*[a-z_]+::min|blockDim\s*=\s*\w+/\w+|std::min.*aiv" \
    modified_files/op_host/*_tiling.cpp
# 期望: >= 1

# 检查 3: UB 容量使用了 hw_params (UB_SIZE 或 GetUBSize)
grep -cE "UB_SIZE|GetUBSize|GetCoreNum|aivNum" modified_files/op_host/*.cpp
# 期望: >= 1

# 检查 4: kernel 入口接收 TilingData (不能 hardcode 在 kernel)
grep -cE "GetTilingData|tilingData\.\w+" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: tail block 有特殊处理（若有 elemsPerCore 切分）
grep -cE "tail|remain|lastCore|elemsRemain" modified_files/op_kernel/*.cpp
# 期望: >= 1（无 tail 处理可能导致越界）
```

## Step 6: Known Pitfalls

| 现象 | 原因 | 修复 |
|---|---|---|
| 编译通过但 UB 越界（崩溃）| availableUb 计算未减去栈空间 | RESERVED_UB ≥ 20KB（栈 + 临时 buffer）|
| 性能比 baseline 还慢 | tileSize 太小，launch 开销占比大 | 加 MIN_TILE_BYTES 下限（典型 256 字节）|
| 部分 shape 精度对不上 | tail block 边界未对齐 | tile 元素数应按 dtype 32B 倍数对齐 |
| blockDim 超过实际可用核 | 用了 hw_params.aicNum 而非 aivNum | 按算子类型区分 AIC/AIV 核 |
| 自适应后多 shape 测试中 1 个慢 5%+ | 该 shape 落进次优分支 | 加 TilingKey 多模式（形态 γ），针对该 shape 写专用分支 |

---

**完成清单**：
```
[P2 Playbook Completion]
Step 1: done (/tmp/p2_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: UB calc: tileSize×bytesPerElem ≤ availableUb; blockDim ≤ aivNum; coverage check passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
