# P28 Playbook: 自定义 HardEvent 生产者-消费者同步

> 本 Playbook 为**强制流程**。采纳 P28 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P28 的核心是**用编译期确定的 HardEvent（`SetFlag`/`WaitFlag`）替代粗粒度的 `PipeBarrier`/`SyncAll`**，在不同引擎（MTE2/Vector/MTE3/MTE1/Matrix）之间建立精确的生产者-消费者关系，减少流水线气泡。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p28_locations.txt`：

```bash
# 1. 当前粗粒度同步点
grep -n "SyncAll\|PipeBarrier\|WaitFlag\|SetFlag" \
    shared/original/op_kernel/*.cpp > /tmp/p28_locations.txt
# 2. 流水线阶段（CopyIn / Compute / CopyOut / L0A/L0B/L0C 操作）
grep -n "CopyIn\|CopyOut\|DataCopy\|Compute\|EnQue\|DeQue\|L0A\|L0B\|L0C\|L1" \
    shared/original/op_kernel/*.cpp >> /tmp/p28_locations.txt
# 3. Buffer 类型与队列
grep -n "TQue\|TBuf\|VECIN\|VECOUT\|A1Buf\|B1Buf\|C1Buf\|CO1Buf\|CO2Buf" \
    shared/original/op_kernel/*.cpp >> /tmp/p28_locations.txt
# 4. 已有的 HardEvent 或自定义同步
grep -n "HardEvent\|ConsWaitProdStatus\|MTE2_V\|V_MTE3\|MTE1_M\|M_MTE1\|M_FIX" \
    shared/original/op_kernel/*.cpp >> /tmp/p28_locations.txt
# 5. 循环内的同步（inner loop sync 是重点优化对象）
grep -n "for.*PipeBarrier\|for.*SyncAll\|while.*PipeBarrier" \
    shared/original/op_kernel/*.cpp >> /tmp/p28_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **同步点位置**：所有 `PipeBarrier` / `SyncAll` 的行号
- **流水线阶段**：CopyIn → Compute → CopyOut 的代码段划分
- **Buffer 类型**：使用的 TQue/TBuf 及其绑定的 TPosition
- **已有事件**：是否已有 HardEvent 或细粒度同步
- **循环内同步**：`PipeBarrier` 是否在循环内部（这些是最优先替换的）

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 粗粒度同步 | `?` (PipeBarrier / SyncAll) | `SetFlag/WaitFlag <HardEvent>` | `?_kernel.cpp:L?` |
| 流水线阶段 | `?` (CopyIn/Compute/CopyOut) | 精确事件配对 | `?_kernel.cpp:L?` |
| Buffer 类型 | `?` (VECIN/VECOUT/L0A/L0B/L0C) | 映射到对应 HardEvent | `?_kernel.cpp:L?` |
| 事件 ID 分配 | `?` (无) | 编译期常量 0-15 | `?_kernel.cpp:L?` |
| 循环内同步 | `?` (有 / 无) | 优先替换循环内的 | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的流水线结构和 buffer 类型，判断你的代码属于以下哪种形态：

- **形态 α — VECIN→Compute→VECOUT 基础流水**：仅涉及 Vector 引擎（MTE2 搬入 → Vector 计算 → MTE3 搬出）。最常见。
- **形态 β — 含 Matmul 的多引擎流水（L0A/L0B/L0C 参与）**：涉及 MTE1（填 L0A/B）、Matrix 引擎、MTE2/MTE3。事件类型更复杂。
- **形态 γ — 大量 buffer 需事件 ID 轮转**：算子使用 >8 个独立 buffer，16 个事件 ID 需要精心分配和复用。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 α — Vector 基础流水，最常见）

```cpp
// === 改造前（粗粒度 PipeBarrier，所有引擎全部等待）===
__aicore__ inline void PipelineLoop(uint32_t loopCount) {
    for (uint32_t i = 0; i < loopCount; i++) {
        CopyIn(i);
        PipeBarrier<PIPE_ALL>();  // 所有人等所有人，气泡大
        Compute(i);
        PipeBarrier<PIPE_ALL>();  // 又等
        CopyOut(i);
        PipeBarrier<PIPE_ALL>();  // 再等
    }
}

// === 改造后（SetFlag/WaitFlag 精确配对，仅相关引擎等待）===
// 事件 ID 分配（编译期常量，0-15）
static constexpr int EV_COPY_IN  = 0;  // MTE2 搬入完成 → Vector 可计算
static constexpr int EV_COMPUTE  = 1;  // Vector 计算完成 → MTE3 可搬出
static constexpr int EV_COPY_OUT = 2;  // MTE3 搬出完成 → 下一轮 MTE2 可搬入

__aicore__ inline void PipelineLoop(uint32_t loopCount) {
    for (uint32_t i = 0; i < loopCount; i++) {
        // Stage 1: CopyIn (MTE2)
        // 等待上一轮 CopyOut 完成（避免覆盖上一轮输出 buffer）
        if (i > 0) {
            WaitFlag<HardEvent::MTE3_V>(EV_COPY_OUT);
        }
        CopyIn(i);
        SetFlag<HardEvent::MTE2_V>(EV_COPY_IN);  // Signal: 搬入完成
        
        // Stage 2: Compute (Vector)
        WaitFlag<HardEvent::MTE2_V>(EV_COPY_IN);  // Wait: 搬入完成
        Compute(i);
        SetFlag<HardEvent::V_MTE3>(EV_COMPUTE);    // Signal: 计算完成
        
        // Stage 3: CopyOut (MTE3)
        WaitFlag<HardEvent::V_MTE3>(EV_COMPUTE);   // Wait: 计算完成
        CopyOut(i);
        SetFlag<HardEvent::MTE3_V>(EV_COPY_OUT);   // Signal: 搬出完成
    }
    // 循环结束后等待最后一轮 CopyOut
    WaitFlag<HardEvent::MTE3_V>(EV_COPY_OUT);
}
```

### 3C. Variant Notes（若是形态 β 或 γ）

- **形态 β（含 Matmul 的多引擎流水）**：
  当算子包含 Matmul（L0A/L0B/L0C），事件映射更复杂：
  ```cpp
  // 常见 HardEvent 映射表（按 AscendC 标准定义）
  // MTE2 → Vector:     HardEvent::MTE2_V    (UB/GM 搬入 → Vector 计算)
  // Vector → MTE3:     HardEvent::V_MTE3    (Vector 计算 → UB/GM 搬出)
  // MTE1 → Matrix:     HardEvent::MTE1_M    (L0A/B 填完 → Matmul 开始)
  // Matrix → MTE1:     HardEvent::M_MTE1    (Matmul 结束 → L0C 读出)
  // MTE2 → MTE1:       HardEvent::MTE2_MTE1 (GM → L0A/B 直接搬运)
  // Matrix → Fixpipe:  HardEvent::M_FIX     (L0C → Fixpipe 量化/搬出)
  
  // 示例：L0A/B 填 + Matmul + L0C 读 的同步
  static constexpr int EV_MTE1_A = 3;   // L0A fill done
  static constexpr int EV_MTE1_B = 4;   // L0B fill done
  static constexpr int EV_MM_DONE = 5;  // Matmul done
  
  // Fill L0A
  DataCopy(aLocal, aGm, count);
  SetFlag<HardEvent::MTE2_MTE1>(EV_MTE1_A);
  
  // Fill L0B
  DataCopy(bLocal, bGm, count);
  SetFlag<HardEvent::MTE2_MTE1>(EV_MTE1_B);
  
  // Matmul waits for both L0A and L0B
  WaitFlag<HardEvent::MTE2_MTE1>(EV_MTE1_A);
  WaitFlag<HardEvent::MTE2_MTE1>(EV_MTE1_B);
  Mmad(cLocal, aLocal, bLocal, ...);
  SetFlag<HardEvent::M_MTE1>(EV_MM_DONE);
  
  // CopyOut waits for Matmul
  WaitFlag<HardEvent::M_MTE1>(EV_MM_DONE);
  DataCopy(cGm, cLocal, count);
  ```
  ⚠️ 不同 CANN 版本的 HardEvent 命名可能有差异，以当前版本 `ascendc_ops.h` 为准。

- **形态 γ（大量 buffer，事件 ID 轮转）**：
  若算子使用 >8 个独立 buffer，16 个事件 ID 不够一对一分配。此时按**流水线阶段**而非 buffer 分配事件：
  ```cpp
  // 不是每个 buffer 一个事件，而是每个“阶段转换”一个事件
  // 例如：CopyIn(0) → Compute(0) → CopyOut(0) → CopyIn(1) → ...
  // 只需要 3 个事件：EV_IN_READY, EV_COMPUTE_DONE, EV_OUT_DONE
  // 同一事件 ID 在不同轮次复用，因为 pipeline 保证了时序不重叠
  ```
  若 16 个事件仍不够，说明算子过于复杂，应考虑拆分 kernel 或用 P1 双缓冲简化时序。

- **与 P1 的边界**：P1 双缓冲用 `BUFFER_NUM=2` 让 CopyIn(i+1) 与 Compute(i) 重叠。P28 的 HardEvent 是**在这个双缓冲基础上**进一步减少同步开销。两者叠加：P1 提供并行度，P28 提供精确同步。

- **与 A4 的边界**：A4（SetFlag/WaitFlag 事件同步）使用**软件事件**（`EVENT_ID_X`），P28 使用**硬件事件**（`HardEvent::MTE2_V` 等）。软件事件更灵活但 overhead 稍大；硬件事件零开销但类型固定。若算子已有 A4，评估是否可升级为 P28 的 HardEvent。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: 事件 ID 总数 ≤ 16（0-15）。超出必须复用或拆分 kernel
约束 2: 每个 SetFlag 必须有且仅有一个对应的 WaitFlag，且 WaitFlag 必须在 SetFlag 之后执行
约束 3: 同一事件 ID 不能同时被两组不同的 SetFlag/WaitFlag 使用（会导致死锁或信号混乱）
约束 4: HardEvent 类型必须匹配实际引擎对。例如 MTE2→Vector 不能用 MTE1_M
约束 5: 循环最后一轮结束后，必须 Wait 所有未完成的 SetFlag，否则可能提前退出导致数据未写完
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `使用的事件 ID = [?, ?, ?]`，总数 = ?
- `SetFlag/WaitFlag 配对检查：`
  - `EV_COPY_IN: SetFlag @ L? → WaitFlag @ L?`
  - `EV_COMPUTE: SetFlag @ L? → WaitFlag @ L?`
  - `EV_COPY_OUT: SetFlag @ L? → WaitFlag @ L?`
- `最后一轮后 Wait 了所有未完成事件：yes/no`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 已引入 SetFlag/WaitFlag 且指定了 HardEvent
grep -cE "SetFlag\s*<\s*HardEvent|WaitFlag\s*<\s*HardEvent" modified_files/op_kernel/*.cpp
# 期望: >= 2（至少一对 Set/Wait）

# 检查 2: 循环内的 PipeBarrier 已被替换
grep -cE "for.*PipeBarrier|while.*PipeBarrier|for.*SyncAll" modified_files/op_kernel/*.cpp
# 期望: == 0（或仅剩循环外必要的全局同步）

# 检查 3: 事件 ID 是编译期常量（不是运行时变量）
grep -cE "static\s+constexpr\s+int\s+EV_|constexpr\s+int\s+EV_|#define\s+EV_" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: SetFlag 和 WaitFlag 数量匹配（同一事件不能只有 Set 没有 Wait）
# 人工统计：SetFlag 次数 ?= WaitFlag 次数（允许边界条件分支导致差 1，但需说明）
grep -cE "SetFlag" modified_files/op_kernel/*.cpp
# 记录次数
grep -cE "WaitFlag" modified_files/op_kernel/*.cpp
# 记录次数，两者应相等或在 note 中解释差异

# 检查 5: 无无效 HardEvent 类型（以当前 CANN 版本为准）
grep -cE "HardEvent::MTE2_V|HardEvent::V_MTE3|HardEvent::MTE1_M|HardEvent::M_MTE1|HardEvent::MTE2_MTE1|HardEvent::M_FIX|HardEvent::MTE3_V" modified_files/op_kernel/*.cpp
# 期望: >= 1
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：HardEvent 类型不存在 | 不同 CANN 版本 HardEvent 枚举名不同。以 `ascendc_ops.h` 中定义的为准。常见：MTE2_V, V_MTE3, MTE1_M, M_MTE1, MTE2_MTE1, M_FIX |
| 运行时：死锁（hang） | 检查是否有 WaitFlag 在 SetFlag 之前执行，或同一事件 ID 被两组独立同步复用。用 Ascend 工具链的 `msprof` 死锁检测 |
| 运行时：数据竞争 / 精度对不上 | SetFlag/WaitFlag 只保证**执行顺序**，不保证**内存可见性**。若多核访问共享 GM，仍需 `SyncAll` 或 `AtomicAdd` |
| 事件 ID 超过 16 | 按流水线阶段分配事件（形态 γ），不要按 buffer 分配。若仍超 16，拆分 kernel |
| 循环外遗漏最后一轮 Wait | 循环结束后必须 Wait 所有未完成事件。常见遗漏：`EV_COPY_OUT` 在最后一轮 Set 后没有 Wait |
| PipeBarrier 全部删除后某些路径异常 | 某些边界条件（如第一轮的 CopyIn 不需要 Wait）需要特殊处理。不要简单全局替换 |
| HardEvent 类型与引擎不匹配 | 例如 Vector 计算后设了 `MTE2_V`（应该是 `V_MTE3`）。对照映射表仔细核对 |
| 双缓冲场景事件 ID 冲突 | 双缓冲有两套 buffer，但事件 ID 只有一套。事件按“阶段”分配（CopyIn done → Compute 可开始），不是按 buffer 分配 |
| SetFlag/WaitFlag 嵌套在 if 分支内 | 若 SetFlag 在 if 内而 WaitFlag 在 if 外，条件不满足时 Wait 永远等不到 Signal。确保 Set/Wait 成对出现或条件一致 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P28 Playbook Completion]
Step 1: done (/tmp/p28_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: constraints: event_ids=[?] set_wait_pairs=? final_wait=yes/no passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
