# P18 Playbook: L1 七缓冲常驻分区 (L1 7-Buffer Resident Partitioning)

> 本 Playbook 为**强制流程**。采纳 P18 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P18 的核心是**将 L1 (512KB) 静态分区为 QP 常驻区（4 块）和 KV 旋转区（3 块）**：QP 在 N 方向首次迭代加载后常驻复用，KV 三路轮转实现 GM→L1 预加载、L1→L0 消费、下一轮预加载的三阶段全重叠。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p18_locations.txt`：

```bash
# 1. Cube/Matmul 相关代码
 grep -n "Mmad\|L0A\|L0B\|L0C\|Cube\|A1Buf\|B1Buf\|C1Buf" \
    shared/original/op_kernel/*.cpp > /tmp/p18_locations.txt
# 2. Attention 模式与 Q/P/K/V 数据流
 grep -n "Attention\|attention\|Query\|Key\|Value\|QK\|softmax" \
    shared/original/op_kernel/*.cpp shared/original/op_host/*_tiling.cpp >> /tmp/p18_locations.txt
# 3. L1 buffer 分配现状
 grep -n "InitBuffer.*L1\|L1Buf\|L1Tensor\|buf.*L1\|L1_SIZE\|L1_SPLIT" \
    shared/original/op_kernel/*.cpp >> /tmp/p18_locations.txt
# 4. N 方向循环及 Q 加载模式
 grep -n "for.*nL1\|for.*nStart\|CopyQ\|copyQuery\|QGm\|queryGm" \
    shared/original/op_kernel/*.cpp >> /tmp/p18_locations.txt
# 5. 当前同步机制
 grep -n "SetFlag\|WaitFlag\|HardEvent\|SyncAll\|PipeBarrier" \
    shared/original/op_kernel/*.cpp >> /tmp/p18_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **Cube 流水线**：Mmad / L0A / L0B / L0C 使用位置
- **Attention 结构**：Q/P/K/V 的 GM 地址、加载函数名
- **L1 现状**：当前 L1 buffer 数量、大小、是否已分区
- **N 循环**：nL1 / nStart 循环结构、Q 加载是否在循环内
- **同步现状**：当前事件 ID 分配、SetFlag/WaitFlag 使用情况

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| L1 总容量 | `?` (通常 512KB) | 不变 | 芯片常量 |
| QP 块数 | `?` (当前 =1/2) | 4 块（2×2 布局） | `?_kernel.cpp:L?` |
| KV 块数 | `?` (当前 =1/2) | 3 块（三路旋转） | `?_kernel.cpp:L?` |
| 块大小 | `? bytes` | `L1_TOTAL / 7 ≈ 72KB` | `?_kernel.cpp:L?` |
| headDim | `?` | 需适配 block 大小 | `?_tiling.cpp:L?` |
| 事件 ID | `?` (当前分配) | 7 个独立 ID | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的 L1 空间和 headDim，判断你的代码属于以下哪种形态：

- **形态 α — QP 常驻（4 块）+ KV 双缓冲（2 块）**：L1 空间不充裕或 KV 数据量小，仅 QP 实施常驻，KV 用普通双缓冲。
- **形态 β — 完整 7 缓冲分区（QP 4 块 + KV 3 块）**：L1 空间充裕，实施完整方案。FIA MLA 的标准配置。
- **形态 γ — 适配非标准 headDim**：headDim ≠ 576（如 128/256/1024），需要重新计算 blockSize 和分区比例。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 β — 完整 7 缓冲分区）

```cpp
// === 改造前（QP 双缓冲，每次 N 迭代都重新加载 Q）===
static constexpr uint32_t L1_QP_SIZE = 128 * 1024;
pipe->InitBuffer(qpBufL1, L1_QP_SIZE * 2);  // 256KB，仅 2 块

for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) {
    // 每次循环都搬运 Q（冗余）
    CopyQGmToL1(qpLocal, queryGm, ...);
    // ...
}

// === 改造后（L1 七缓冲分区）===
// 1. 常量定义
static constexpr uint32_t L1_TOTAL = 512 * 1024;           // 512KB
static constexpr uint32_t L1_BLOCK_SIZE = 72 * 1024;       // ~72KB 每块
static constexpr uint32_t QP_BLOCKS = 4;                   // QP 区 4 块
static constexpr uint32_t KV_BLOCKS = 3;                   // KV 区 3 块
static constexpr uint32_t BLOCK_ELEMS = L1_BLOCK_SIZE / sizeof(Q_T);

// 2. L1 buffer 初始化
pipe->InitBuffer(bufQPL1, L1_BLOCK_SIZE * QP_BLOCKS);  // 288KB
pipe->InitBuffer(bufKVL1, L1_BLOCK_SIZE * KV_BLOCKS);  // 216KB
// 剩余：512 - 288 - 216 = 8KB 对齐/保险余量

// 3. QP 2×2 索引映射
__aicore__ inline uint32_t GetQPBlockIdx(uint32_t mIdx, uint32_t k1Idx) {
    // block [0,1] 对应 M 偶数；block [2,3] 对应 M 奇数
    uint32_t base = (mIdx % 2) * 2;
    return base + k1Idx;
}

// 4. KV 三路旋转索引
__aicore__ inline uint32_t GetKVBlockIdx(uint32_t loopIdx) {
    return loopIdx % KV_BLOCKS;
}

// 5. N 方向循环：QP 常驻 + KV 旋转
for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) {
    for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) {
        // --- QP 区：常驻复用 ---
        uint32_t qpIdx = GetQPBlockIdx(mIdx, kL1);
        LocalTensor<Q_T> qpTensor = bufQPL1.Get<Q_T>()[qpIdx * BLOCK_ELEMS];
        
        if (nL1 == 0) {
            // 首次 N 迭代：从 GM 加载 Q/P 到 L1
            WaitFlag<HardEvent::MTE2_MTE1>(qpEventIds[qpIdx]);
            CopyQGmToL1(qpTensor, queryGm, gmCoord);
            SetFlag<HardEvent::MTE2_MTE1>(qpEventIds[qpIdx]);
            WaitFlag<HardEvent::MTE2_MTE1>(qpEventIds[qpIdx]);
        }
        // nL1 > 0：直接复用 L1 中已有数据，零搬运
        
        // --- KV 区：三路旋转 ---
        uint32_t kvIdx = GetKVBlockIdx(nL1);
        LocalTensor<KV_T> kvTensor = bufKVL1.Get<KV_T>()[kvIdx * BLOCK_ELEMS];
        
        // Stage A: 消费当前 KV（L1→L0→Cube）
        WaitFlag<HardEvent::MTE2_MTE1>(kvEventIds[kvIdx]);
        CopyL1ToL0(kvTensor, l0aTensor, ...);
        Mmad(l0cTensor, l0aTensor, l0bTensor, ...);
        
        // Stage B: 预加载下一轮 KV（GM→L1，与当前 Cube 计算重叠）
        if (nL1 < nL1Loops - 1) {
            uint32_t nextKvIdx = GetKVBlockIdx(nL1 + 1);
            // 确保下一轮 buffer 已空闲（上轮消费完成）
            WaitFlag<HardEvent::MTE1_M>(kvFreeEventIds[nextKvIdx]);
            SetFlag<HardEvent::MTE2_MTE1>(kvEventIds[nextKvIdx]);
            CopyKVGmToL1(bufKVL1.Get<KV_T>()[nextKvIdx * BLOCK_ELEMS], kvGm, nextCoord);
        }
    }
}
```

### 3C. Variant Notes（若是形态 α 或 γ）

- **形态 α（QP 常驻 + KV 双缓冲）**：
  当 KV 数据量较小或 L1 空间不足时，仅实施 QP 常驻：
  ```cpp
  static constexpr uint32_t QP_BLOCKS = 4;
  static constexpr uint32_t KV_BLOCKS = 2;  // 双缓冲替代三路旋转
  pipe->InitBuffer(bufQPL1, L1_BLOCK_SIZE * QP_BLOCKS);
  pipe->InitBuffer(bufKVL1, L1_BLOCK_SIZE * KV_BLOCKS);
  // KV 双缓冲无预加载槽位，流水重叠率低于形态 β
  ```
  形态 α 适合 headDim 较小或 batch size 较大的场景，此时 KV 占比低。

- **形态 γ（非标准 headDim 适配）**：
  headDim 决定 `L1_BLOCK_SIZE`。标准值 `headDim=576` → `blockSize≈72KB`。
  若 headDim 不同：
  ```cpp
  // 重新计算 blockSize
  static constexpr uint32_t M_SPLIT = 128;
  static constexpr uint32_t K_SPLIT = headDim / 2;  // 分两半处理
  static constexpr uint32_t L1_BLOCK_SIZE = M_SPLIT * K_SPLIT * sizeof(Q_T);
  // 然后计算可分配的 QP/KV 块数
  static constexpr uint32_t QP_BLOCKS = 4;
  static constexpr uint32_t KV_BLOCKS = (L1_TOTAL - QP_BLOCKS * L1_BLOCK_SIZE) / L1_BLOCK_SIZE;
  // KV_BLOCKS 可能 = 2 或 3，取决于 headDim
  ```
  若 `KV_BLOCKS < 2`，则形态 γ 退化为仅 QP 常驻（形态 α）。

- **与 P14 的协同**：P14 是 Flash Attention 的宏观 tiling 策略，P18 是其中的 L1 微观 buffer 管理。两者通常一起使用：P14 决定 tile 大小，P18 决定 L1 内部分区。

- **与 P38 的边界**：P38 管理 UB 内的 softmax 状态，P18 管理 L1 内的 Q/K/V 数据。两者独立但互补：P18 减少 GM→L1 搬运，P38 减少循环内 UB 分配 overhead。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: QP_BLOCKS * blockSize + KV_BLOCKS * blockSize ≤ L1_TOTAL (512KB)
约束 2: blockSize 必须满足 L0A/L0B 对齐要求（通常为 32B 或 64B）
约束 3: 7 个 buffer 需要 7 个独立事件 ID，加上 Cube/Vector 事件后总数 ≤ 16
约束 4: QP 常驻仅在 Q 数据跨 N 迭代不变时有效。若 Q 被 online 更新（如某些变体 Attention），不能常驻
约束 5: KV 预加载不能覆盖尚未消费的 buffer。三路旋转保证 nL1, nL1-1, nL1-2 三阶段不重叠
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `L1_TOTAL = ?`, `blockSize = ?`, `QP_BLOCKS = ?`, `KV_BLOCKS = ?`
- `QP 占用 = ?`, `KV 占用 = ?`, `剩余 = ?`
- `使用的事件 ID = [?, ?, ?, ?, ?, ?, ?]`，总数 = ?
- `Q 是否跨 N 迭代不变：yes/no`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: L1 分区 buffer 已正确定义
 grep -cE "InitBuffer.*L1_BLOCK_SIZE|bufQPL1|bufKVL1" modified_files/op_kernel/*.cpp
# 期望: >= 2

# 检查 2: 有 QP 2×2 索引映射或 KV 旋转索引
 grep -cE "GetQPBlockIdx|GetKVBlockIdx|loop.*%.*KV_BLOCKS|mIdx.*%.*2" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: Q 加载仅在 nL1 == 0 时发生
 grep -cE "if.*nL1.*==.*0.*CopyQ|if.*nStart.*==.*0.*CopyQ" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: 无冗余 Q 加载（不在 nL1 > 0 的路径内）
 grep -cE "for.*nL1.*\{[^}]*CopyQGmToL1" modified_files/op_kernel/*.cpp
# 期望: == 0（CopyQ 必须在 if nL1==0 分支内）

# 检查 5: KV 预加载与消费有正确事件同步
 grep -cE "WaitFlag.*kvEventIds|SetFlag.*kvEventIds" modified_files/op_kernel/*.cpp
# 期望: >= 2
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：L1 buffer 大小超过芯片限制 | 确认 `L1_TOTAL` 对应当前芯片（a3: 512KB, a5: 可能不同）。超限时减少 `QP_BLOCKS` 或 `KV_BLOCKS` |
| 运行时：L1 数据覆盖 | 检查 KV 旋转索引计算。三路旋转要求 `nL1`, `nL1-1`, `nL1-2` 的 buffer 互不相同。确认 `loopIdx % 3` 不会冲突 |
| 运行时：QP 数据在 N 迭代中被意外修改 | QP 常驻要求 Q 只读。若 kernel 内有代码修改了 QP buffer，会破坏常驻语义。检查所有写 QP 的操作 |
| 事件 ID 超过 16 | 7 个 L1 buffer + Cube/Vector 事件可能超 16。合并部分事件（如 QP 4 块共用 2 个事件 ID，按 M 奇偶分组） |
| headDim 非 576 时 blockSize 不对齐 | `blockSize` 必须是 L0A/B 对齐粒度的整数倍。不对齐时向上取整到对齐边界，重新计算分区 |
| KV 预加载与消费未同步导致数据竞争 | 确保 `WaitFlag` 在消费前、`SetFlag` 在预加载后。漏同步会导致读到旧数据 |
| nL1Loops < 3 时三路旋转退化 | 若 `nL1Loops == 1`，KV 只需 1 块；若 `== 2`，只需 2 块。编译期判断避免浪费 |
| L1 剩余空间为负 | 严格复核约束 1。若 `QP_BLOCKS * blockSize + KV_BLOCKS * blockSize > L1_TOTAL`，必须减小块数或 blockSize |
| Q_rope 和 Q_nope 的存储顺序 | 某些 MLA 实现中 Q 分 nope 和 rope 两部分。确认 `CopyQGmToL1` 的 dst offset 正确，两部分不重叠 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P18 Playbook Completion]
Step 1: done (/tmp/p18_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: constraints: L1_TOTAL=? blockSize=? QP=? KV=? events=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
