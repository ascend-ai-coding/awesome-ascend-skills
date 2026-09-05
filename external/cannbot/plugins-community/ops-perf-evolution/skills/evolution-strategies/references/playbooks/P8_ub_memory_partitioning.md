# P8 Playbook: UB Memory Partitioning (UB 内存分区管理)

> 本 Playbook 为**强制流程**。采纳 P8 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步，禁止"看起来改了"就声明完成。
>
> P8 的核心是**在有限 UB 容量下，为多个 tensor 合理分配空间**。它不改变数学逻辑，只调整 buffer 大小、数量和分配顺序，消除 UB overflow 并释放空间给更大 tile 或双缓冲。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p8_locations.txt`：

```bash
# 1. 所有 InitBuffer 调用（UB 分配点）
grep -n "InitBuffer" shared/original/op_kernel/*.cpp > /tmp/p8_locations.txt
# 2. 所有 LocalTensor / Get 定义（UB 使用点）
grep -n "LocalTensor\|Get<" shared/original/op_kernel/*.cpp >> /tmp/p8_locations.txt
# 3. tileSize / blockSize / ub_factor 计算（决定单 buffer 大小）
grep -n "tileSize\|tile_size\|ubFactor\|ub_factor\|blockSize\|BLOCK_SIZE" shared/original/op_kernel/*.cpp shared/original/op_host/*_tiling.cpp >> /tmp/p8_locations.txt
# 4. BUFFER_NUM 定义（影响总 UB 倍数）
grep -n "BUFFER_NUM\|bufferNum" shared/original/op_kernel/*.h >> /tmp/p8_locations.txt
# 5. 当前 UB 总估算（粗略统计 InitBuffer 的 size 表达式）
grep -n "InitBuffer.*," shared/original/op_kernel/*.cpp >> /tmp/p8_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **InitBuffer 列表**：每个调用的文件 + 行号 + 分配的 buffer 名
- **LocalTensor 列表**：所有在 kernel 中使用的 UB tensor 名称
- **tileSize / BUFFER_NUM**：当前值 + 定义位置
- **UB 分配公式**：每个 InitBuffer 的 size 表达式（如 `tileSize * sizeof(T)`）

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| BUFFER_NUM | `?` | `?`（若 P1 双缓冲则 2） | `?.h:L?` |
| tileSize | `?` | `?`（按 UB 容量调整） | `?_tiling.cpp:L?` |
| InitBuffer 数量 | `?` | `?`（减少/合并/复用） | `?_kernel.cpp:L?` |
| 每个 buffer 大小 | `? bytes` | `? bytes` | — |
| 总 UB 占用 | `? bytes` | `? bytes`（≤ UB_TOTAL × 0.8） | — |
| UB 分区方案 | 独立分配 | 见 Step 3A 形态 | — |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的 InitBuffer 和 UB 占用情况，判断你的代码属于以下哪种形态：

- **形态 α — UB 充足**：当前总 UB 占用 ≤ UB_TOTAL × 0.5。优化方向是**对齐 + 顺序调整**（使 buffer 边界对齐到 32B，减少 bank conflict），无需缩减 tileSize。
- **形态 β — UB 紧张**：当前总 UB 占用 > UB_TOTAL × 0.5 但 ≤ 0.8。优化方向是**缩减 tileSize** 或**合并生命周期不重叠的 buffer**（同一块 InitBuffer 分 zone 复用，见 P85 on-chip buffer zone reuse）。
- **形态 γ — UB 溢出**：当前总 UB 占用 > UB_TOTAL × 0.8 或编译报 UB overflow。必须**改分区方案**：减小 tileSize、减少 BUFFER_NUM、或将大中间结果搬到 GM workspace。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 β — 最常见）

```cpp
// === 改造前（独立分配，UB 紧张）===
constexpr int32_t BUFFER_NUM = 1;
uint32_t tileSize = 4096;   // 假设 dtype = fp16, 则 4096 * 2 = 8KB per buffer

pipe.InitBuffer(inQueue,  BUFFER_NUM, tileSize * sizeof(T));   // 8KB
pipe.InitBuffer(outQueue, BUFFER_NUM, tileSize * sizeof(T));   // 8KB
pipe.InitBuffer(tmpQueue, BUFFER_NUM, tileSize * sizeof(T));   // 8KB
pipe.InitBuffer(maskQueue,BUFFER_NUM, tileSize * sizeof(T));   // 8KB
// 总 UB = 32KB（示例），若 BUFFER_NUM=2 则 64KB

// === 改造后（缩减 tileSize + 对齐到 32B）===
constexpr int32_t BUFFER_NUM = 2;   // 若配合 P1 双缓冲
// tileSize 减半，使总 UB 回到安全区
uint32_t tileSize = 2048;           // 2048 * 2 = 4KB per buffer

// 每个 InitBuffer 的大小按 32B 对齐
uint32_t bufSize = (tileSize * sizeof(T) + 31) & ~31;  // 向上对齐到 32B

pipe.InitBuffer(inQueue,  BUFFER_NUM, bufSize);   // 4KB × 2 = 8KB
pipe.InitBuffer(outQueue, BUFFER_NUM, bufSize);   // 4KB × 2 = 8KB
// tmp 和 mask 生命周期不重叠 → 合并为同一块 zone 复用（见 P85）
pipe.InitBuffer(sharedQueue, BUFFER_NUM, bufSize);  // 4KB × 2 = 8KB
// 总 UB = 24KB（含 20% 安全余量）
```

### 3C. Variant Notes（若是形态 α 或 γ）

- **形态 α（UB 充足）**：不改 tileSize，只做 **32B 对齐**。将 `InitBuffer(xQueue, N, size)` 改为 `InitBuffer(xQueue, N, (size + 31) & ~31)`。重点消除 UB bank conflict（P65）。
- **形态 γ（UB 溢出）**：
  - **方案 1**：tileSize 减半（如 4096 → 2048），总 UB 降到 1/2。
  - **方案 2**：BUFFER_NUM 从 2 降到 1（放弃双缓冲），总 UB 减半。权衡：性能下降，需确认是否为 memory_bound。
  - **方案 3**：将大中间结果从 UB 搬到 GM workspace（`workspaceGm`），用 `DataCopy` 往返搬运。适用于中间结果 > 50% UB 的场景。
  - **方案 4**：生命周期不重叠的 tensor 共享同一块 buffer 的不同 offset（zone reuse，见 P85）。
- **如果已应用 P1 双缓冲**：BUFFER_NUM=2 已固定，只能从 tileSize 或 buffer 合并下手。若仍溢出，放弃 P1 换回单缓冲。

## Step 4: UB 容量复核（防崩溃）

**公式**：
```
单 buffer 大小  = ((tileSize × sizeof(dtype)) + 31) & ~31   // 32B 对齐
总 UB 占用      = Σ (单 buffer 大小 × BUFFER_NUM_i)        // 所有 InitBuffer 求和
安全上限        = UB_TOTAL × 0.8                            // 留 20% 给栈 + 编译器临时变量
```

- **910B (A3)**：UB_TOTAL ≈ 256 KB（每核）
- **950 (A5)**：UB_TOTAL ≈ 512 KB（每核）

**约束**：`总 UB 占用 ≤ 安全上限`

若不满足，回到 Step 3C 选择降级方案。**在 implementation_note.txt "Playbook Step 4" 中报告具体计算**（每个 buffer 大小、总和、上限、是否通过）。

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。

```bash
# 检查 1: InitBuffer 调用的 size 参数包含 sizeof(T) 或显式对齐
# （确保 buffer 大小计算考虑了 dtype，不是硬编码常数）
grep -En "InitBuffer\s*\([^,]+,\s*[^,]+,\s*[^)]*(sizeof|<<|>>|align|ALIGN)" modified_files/op_kernel/*.cpp | wc -l
# 期望: >= 1

# 检查 2: tileSize / tile_size / ub_factor 在 tiling 文件中被定义
# （确保 UB 分区决策来自 tiling 参数，而非 kernel 内硬编码）
grep -c "tileSize\|tile_size\|ubFactor\|ub_factor" modified_files/op_host/*_tiling.cpp
# 期望: >= 1

# 检查 3: BUFFER_NUM 定义存在（单缓冲=1 或 双缓冲=2）
grep -c "BUFFER_NUM\s*=\s*[12]\|bufferNum\s*=\s*[12]" modified_files/op_kernel/*.h modified_files/op_host/*.h
# 期望: >= 1

# 检查 4: 至少有两个 InitBuffer 调用（UB 分区管理才有意义；单 buffer 无需 P8）
grep -c "InitBuffer" modified_files/op_kernel/*.cpp
# 期望: >= 2

# 检查 5: 无裸数字硬编码的 InitBuffer size（如 InitBuffer(q, 1, 8192) 是反模式）
grep -En "InitBuffer\s*\([^,]+,\s*[^,]+,\s*\d+\s*\)" modified_files/op_kernel/*.cpp | wc -l
# 期望: == 0
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：UB overflow | 回 Step 4 复算；按形态 γ 选择方案 1-4（缩减 tileSize / 降 BUFFER_NUM / 搬 workspace / zone 复用） |
| 编译失败：UB bank conflict | buffer size 未 32B 对齐 → 改为 `((size + 31) & ~31)` 或 `AlignCeil(size, 32)` |
| 运行时：数据错乱 | 检查 zone reuse 的 offset 计算是否重叠。两个 tensor 的 `[offset, offset + size)` 区间不能相交 |
| 运行时：精度退化 | P8 不改计算逻辑；若精度变了，检查是否误改了 dtype 大小（sizeof(T) vs sizeof(float)） |
| 性能提升 < 5% | 若瓶颈不是 ub_memory_pressure（ profiling 中 UB 利用率已低），P8 帮不上。换瓶颈对应的策略（如 compute_bound → P46/P84） |
| tileSize 过小导致 DMA 效率低 | tileSize 减半后若 < 128B，DataCopy 粒度太小 → 改用方案 4（zone reuse）而非继续缩减 tileSize |
| double buffer + UB 分区冲突 | P1 的 BUFFER_NUM=2 已使 UB 翻倍；若此时再分区，总和易溢出。优先保证 P1，用 zone reuse 替代独立 buffer |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P8 Playbook Completion]
Step 1: done (/tmp/p8_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: UB calc: 总占用=sum(queue_buffer_sizes) ≤ UB_TOTAL×0.8: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
