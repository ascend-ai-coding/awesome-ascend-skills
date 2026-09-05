# P22 Playbook: TQueBind 双向复用队列 (Bidirectional Queue Reuse)

> 本 Playbook 为**强制流程**。采纳 P22 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P22 的核心是**将同 shape 的 VECIN + VECOUT 两个物理 buffer 合并为一块 TQueBind 双向队列**，节省约 50% UB 空间，适用于读-改-写的 in-place 场景。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p22_locations.txt`：

```bash
# 1. 当前分离的 VECIN / VECOUT 队列
grep -n "TQue.*VECIN\|TQue.*VECOUT\|TBuf.*VECIN\|TBuf.*VECOUT" \
    shared/original/op_kernel/*.cpp > /tmp/p22_locations.txt
# 2. 读-改-写模式（CopyIn → Compute → CopyOut 同数据）
grep -n "DataCopy.*Gm\|CopyIn\|CopyOut\|EnQue\|DeQue\|FreeTensor" \
    shared/original/op_kernel/*.cpp >> /tmp/p22_locations.txt
# 3. UB buffer 分配与压力点
grep -n "InitBuffer\|BUFFER_NUM\|TBuf\|TQue\|ubFactor\|tileSize" \
    shared/original/op_kernel/*.cpp >> /tmp/p22_locations.txt
# 4. 同步机制
grep -n "SyncAll\|SetWaitFlag\|HardEvent\|PipeBarrier" \
    shared/original/op_kernel/*.cpp >> /tmp/p22_locations.txt
# 5. 已有的 TQueBind
grep -n "TQueBind" \
    shared/original/op_kernel/*.cpp >> /tmp/p22_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **队列位置**：所有 VECIN / VECOUT 队列声明位置
- **读写配对**：哪些数据先 CopyIn、Compute、再 CopyOut（同一 tensor）
- **UB 现状**：buffer 数量、总大小、瓶颈 buffer
- **同步现状**：当前使用 SyncAll 还是细粒度事件同步
- **已有 TQueBind**：是否已使用

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 读队列 | `?` (TQue/TBuf VECIN) | `TQueBind VECIN+VECOUT` | `?_kernel.cpp:L?` |
| 写队列 | `?` (TQue/TBuf VECOUT) | 复用同一 TQueBind | `?_kernel.cpp:L?` |
| Buffer 大小 | `? bytes × 2` | `? bytes × 1`（节省 50%） | `?_kernel.cpp:L?` |
| 同步方式 | `?` (SyncAll / 无) | `SetWaitFlag<HardEvent::MTE2_V>` | `?_kernel.cpp:L?` |
| 数据模式 | `?` (读-改-写 / 纯写) | 确认 in-place | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的数据流和同步需求，判断你的代码属于以下哪种形态：

- **形态 α — 纯替换（读-改-写，无额外同步需求）**：数据从 GM 读入 UB，Compute 修改后，直接写回同一位置 GM。无其他核/队列依赖。
- **形态 β — 替换 + 显式事件同步（标准模式）**：读入后可能有 Vector 计算，必须在写回前等待 Vector 完成。使用 `SetWaitFlag<HardEvent::MTE2_V>`。
- **形态 γ — TQueBind + 与其他 buffer 的 ping-pong（深度 1 下的极限并行）**：虽然 TQueBind depth=1 无法双缓冲，但可通过与其他独立 buffer（如 weight resident buffer）形成伪并行。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 β — 显式事件同步，最常见）

```cpp
// === 改造前（分离 VECIN + VECOUT，2 倍 buffer）===
TQue<TPosition::VECIN, BUFFER_NUM> inQueue;
TQue<TPosition::VECOUT, BUFFER_NUM> outQueue;
// pipe.InitBuffer(inQueue, tileSize * sizeof(T));
// pipe.InitBuffer(outQueue, tileSize * sizeof(T));

__aicore__ inline void Compute(...) {
    // --- VECIN 阶段 ---
    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopy(inLocal, inGm, count);
    inQueue.EnQue(inLocal);
    inLocal = inQueue.DeQue<T>();
    
    // Compute (in-place modification)
    Mul(inLocal, inLocal, scale, count);
    
    // --- VECOUT 阶段 ---
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
    DataCopy(outLocal, inLocal, count);  // 额外拷贝！
    outQueue.EnQue(outLocal);
    outLocal = outQueue.DeQue<T>();
    DataCopy(outGm, outLocal, count);
    outQueue.FreeTensor(outLocal);
    inQueue.FreeTensor(inLocal);
}

// === 改造后（TQueBind，1 倍 buffer + 原地修改）===
TQueBind<TPosition::VECIN, TPosition::VECOUT, NUM_ONE> bindQueue;
// pipe.InitBuffer(bindQueue, tileSize * sizeof(T));  // 只占一份 UB

__aicore__ inline void Compute(...) {
    // Alloc 从 VECIN 侧取 buffer
    LocalTensor<T> local = bindQueue.AllocTensor<T>();
    
    // 1. 搬入
    DataCopy(local, inGm, count);
    
    // 2. 显式同步：确保搬入完成后再做 Compute
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    
    // 3. EnQue → DeQue 标记状态转换（VECIN → VECOUT）
    bindQueue.EnQue(local);
    local = bindQueue.DeQue<T>();
    
    // 4. Compute（原地修改，无额外 buffer）
    Mul(local, local, scale, count);
    
    // 5. 搬出到 GM
    DataCopy(outGm, local, count);
    
    // 6. Free 归还 buffer
    bindQueue.FreeTensor(local);
}
```

### 3C. Variant Notes（若是形态 α 或 γ）

- **形态 α（无显式同步需求）**：
  若 Compute 为纯标量/常数操作（如 `Add(local, local, 1.0f, count)`），且 DataCopy 后无 Vector 依赖，可省略 `SetWaitFlag`：
  ```cpp
  DataCopy(local, inGm, count);
  bindQueue.EnQue(local);
  local = bindQueue.DeQue<T>();
  // 直接 Compute，无需 SetWaitFlag
  Add(local, local, bias, count);
  DataCopy(outGm, local, count);
  bindQueue.FreeTensor(local);
  ```
  ⚠️ 形态 α 极少见，AscendC 的 DataCopy (MTE2) 和 Vector 之间通常需要同步。默认按形态 β 处理，省略同步需有 profiling 证据证明安全。

- **形态 γ（与其他 buffer 形成伪并行）**：
  TQueBind depth=1 意味着同一块 buffer 不能同时用于读和写。但如果算子有**多个独立数据流**（如 `conv_state` + `x`），可让其中一路用 TQueBind，另一路用普通 TQue 双缓冲：
  ```cpp
  TQueBind<TPosition::VECIN, TPosition::VECOUT, NUM_ONE> convQueue;  // 节省 UB
  TQue<TPosition::VECIN, 2> xQueue;   // 正常双缓冲，不节省但可流水
  
  // Pipeline: 当 convQueue 在 Compute 时，xQueue 可预取下一 tile
  // 总 UB = convQueue(1x) + xQueue(2x) < 改造前 convQueue(2x) + xQueue(2x)
  ```
  形态 γ 的优化收益取决于两路数据的大小比例。若 convQueue 占 UB 的 80%，则节省 50%  convQueue 收益显著；若只占 10%，收益有限。

- **与 P1 的边界**：P1（双缓冲）要求 `BUFFER_NUM ≥ 2` 实现 Copy/Compute 流水。TQueBind 的 `NUM_ONE` 意味着**同一块 buffer 无法做双缓冲流水**。两者冲突：
  - 若该数据路径是性能瓶颈（带宽 bound），需要 P1 双缓冲流水 → **不要**用 P22 TQueBind。
  - 若该数据路径不是瓶颈，但 UB 空间不足导致其他路径无法优化 → 用 P22 节省空间。

- **与 P8 的协同**：P22 节省的 UB 空间可释放给 P8 的 buffer 分区策略。例如在 UB 紧张时，将次要路径改为 TQueBind，腾出空间给主路径扩大 tileSize。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: 输入 shape == 输出 shape（元素个数相同），TQueBind 不涉及数据 reshape
约束 2: TQueBind depth = NUM_ONE（固定为 1，无法双缓冲）
约束 3: EnQue 后必须 DeQue 才能再次使用同一 tensor，不能跳过
约束 4: SetWaitFlag<HardEvent::MTE2_V> 必须在 DataCopy(in) 之后、Compute 之前，确保数据就绪
约束 5: FreeTensor 必须在 DataCopy(out) 之后，不能提前释放
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `原 VECIN size = ? bytes, 原 VECOUT size = ? bytes, 合计 = ? bytes`
- `TQueBind size = ? bytes, 节省 = ? bytes / ?%`
- `SetWaitFlag 位置：在 DataCopy(in) 之后第 ? 行`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 已引入 TQueBind 声明
grep -cE "TQueBind.*VECIN.*VECOUT|TQueBind.*VECOUT.*VECIN" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 无分离的 VECIN + VECOUT TQue/TBuf 对被改造的数据流
grep -cE "TQue.*VECIN.*[a-zA-Z_]+[^B];\n.*TQue.*VECOUT.*[a-zA-Z_]+[^B];" modified_files/op_kernel/*.cpp
# 期望: == 0（被替换的队列已删除）

# 检查 3: 存在 EnQue / DeQue / FreeTensor 的完整配对
grep -cE "EnQue|DeQue|FreeTensor" modified_files/op_kernel/*.cpp
# 期望: >= 3（至少各出现一次）

# 检查 4: 有显式同步（SetWaitFlag 或 SyncAll）在 DataCopy(in) 和 Compute 之间
grep -cE "SetWaitFlag|SyncAll" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: DataCopy(out) 使用的是 DeQue 后的 tensor，不是新 Alloc 的
grep -cE "DataCopy.*outGm.*local|DataCopy.*Gm.*DeQue" modified_files/op_kernel/*.cpp
# 期望: >= 1
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：TQueBind 模板参数错误 | `TQueBind<TPosition::VECIN, TPosition::VECOUT, NUM_ONE>` 顺序不能交换。`NUM_ONE` 是 depth，必须为 1 |
| 编译失败：AllocTensor 类型不匹配 | `bindQueue.AllocTensor<T>()` 和 `bindQueue.DeQue<T>()` 的类型 `T` 必须一致 |
| 运行时：数据未写完就开始 Compute | 漏加 `SetWaitFlag<HardEvent::MTE2_V>`。DataCopy (MTE2) 和 Vector 计算是异步的，必须显式等待 |
| 运行时：搬出数据是旧值 | `EnQue` → `DeQue` 必须在 Compute 之前完成。若先 Compute 再 EnQue，DeQue 拿到的是未修改的副本 |
| 运行时：UB 越界 | TQueBind 的 InitBuffer 大小必须 ≥ 最大 tile 的数据量。若存在 padding，按 padding 后大小计算 |
| 尝试用 TQueBind 做双缓冲 | TQueBind depth 固定为 1，无法做双缓冲流水。若需要流水，用 P1 的标准 TQue + BUFFER_NUM=2 |
| 输入输出 shape 不同 | TQueBind 要求同一块物理内存进出。若 shape 不同（如 broadcast、transpose），不能用 P22 |
| FreeTensor 在 DataCopy(out) 之前 | FreeTensor 必须在所有使用结束后调用。提前释放会导致 DataCopy 写非法内存 |
| 多个 TQueBind 之间无 SyncAll | 若存在多个 TQueBind 处理不同数据，它们之间可能需要 SyncAll 保证全局顺序 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P22 Playbook Completion]
Step 1: done (/tmp/p22_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: constraints: old_size=? new_size=? saved=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
