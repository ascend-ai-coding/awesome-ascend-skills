# A4 Playbook: SetFlag/WaitFlag 事件同步保证精度

> 本 Playbook 为**强制流程**。采纳 A4 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> A4 的核心是**在精度敏感的多步序列（如 Cast→Compute→Cast）的关键节点插入细粒度同步（PipeBarrier<PIPE_V> 或 SetFlag/WaitFlag）**，确保前一步操作完全完成后才执行下一步，防止数据竞争和乱序执行导致的精度问题。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/a4_locations.txt`：

```bash
# 1. Cast 操作与精度转换
grep -n "Cast\|CAST_NONE\|CAST_RINT\|RoundMode\|float32\|bf16\|bfloat16" \
    shared/original/op_kernel/*.cpp > /tmp/a4_locations.txt
# 2. 多步 Compute 序列
grep -n "Add\s*(\|Mul\s*(\|Sub\s*(\|Div\s*(\|Exp\s*(\|Sqrt\s*(" \
    shared/original/op_kernel/*.cpp >> /tmp/a4_locations.txt
# 3. 当前同步机制
grep -n "SyncAll\|PipeBarrier\|SetFlag\|WaitFlag" \
    shared/original/op_kernel/*.cpp >> /tmp/a4_locations.txt
# 4. 精度敏感的 buffer（FP32 中间 buffer）
grep -n "float.*Local\|float.*Buf\|fp32\|calcType" \
    shared/original/op_kernel/*.cpp >> /tmp/a4_locations.txt
# 5. 已有的精度同步
grep -n "PipeBarrier.*PIPE_V\|Cast.*PipeBarrier\|precision.*sync" \
    shared/original/op_kernel/*.cpp >> /tmp/a4_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **Cast 位置**：所有 `Cast` 调用位置、RoundMode 类型
- **Compute 序列**：Cast→op→Cast 的完整序列
- **当前同步**：SyncAll / PipeBarrier 使用位置
- **精度 buffer**：FP32 中间 buffer 的分配和使用
- **已有同步**：是否已有 Cast 后的同步

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 精度链 | `?` (Cast→Compute→Cast) | 不变 | `?_kernel.cpp:L?` |
| 当前同步 | `?` (无 / SyncAll) | PipeBarrier<PIPE_V> | `?_kernel.cpp:L?` |
| Cast 次数 | `?` | `?` | `?_kernel.cpp:L?` |
| 插入位置 | `?` (全部 / 部分) | `alpha/beta/gamma` 见 3A | `?_kernel.cpp:L?` |
| 精度类型 | `?` (BF16/FP16/INT8) | 确认转换方向 | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的精度序列和引擎类型，判断你的代码属于以下哪种形态：

- **形态 α — PipeBarrier<PIPE_V>（纯 Vector 精度链，最常见）**：Cast→Compute→Cast 都在 Vector 引擎内，用 `PipeBarrier<PIPE_V>` 同步。
- **形态 β — SetFlag/WaitFlag（跨引擎精度链）**：Cast 在 MTE2/MTE3，Compute 在 Vector，需要跨引擎事件同步。
- **形态 γ — 多级精度链（多个 Cast-Compute 阶段串联）**：链很长（如 Cast→Add→Cast→Mul→Cast），需要分段同步。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 α — PipeBarrier<PIPE_V>，最常见）

```cpp
// === 改造前（无同步，数据竞争风险）===
__aicore__ inline void Bf16Compute(...) {
    // Upcast: BF16 → FP32
    Cast(fp32Buf, bf16In, RoundMode::CAST_NONE, count);
    // ❌ 风险：Add 可能读到 Cast 未完成的值
    Add(fp32Buf, fp32Buf, bias, count);
    // ❌ 风险：Cast down 可能读到 Add 未完成的值
    Cast(bf16Out, fp32Buf, RoundMode::CAST_RINT, count);
}

// === 改造后（PipeBarrier 确保每步完成）===
__aicore__ inline void Bf16Compute(...) {
    // Step 1: Upcast BF16 → FP32
    Cast(fp32Buf, bf16In, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();  // 确保 Cast 结果写入 UB
    
    // Step 2: Compute in FP32
    Add(fp32Buf, fp32Buf, bias, count);
    PipeBarrier<PIPE_V>();  // 确保 Compute 结果写入 UB
    
    // Step 3: Downcast FP32 → BF16
    Cast(bf16Out, fp32Buf, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();  // 确保 Downcast 完成（若后续有读）
}
```

### 3C. Variant Notes（若是形态 β 或 γ）

- **形态 β（跨引擎精度链）**：
  当 Cast 由 MTE2 执行、Compute 由 Vector 执行时，用 `SetFlag/WaitFlag`：
  ```cpp
  // MTE2: Cast BF16→FP32
  Cast(fp32Buf, bf16In, RoundMode::CAST_NONE, count);
  SetFlag<HardEvent::MTE2_V>(EVENT_CAST_DONE);
  
  // Vector: Wait 后 Compute
  WaitFlag<HardEvent::MTE2_V>(EVENT_CAST_DONE);
  Add(fp32Buf, fp32Buf, bias, count);
  SetFlag<HardEvent::V_MTE3>(EVENT_COMPUTE_DONE);
  
  // MTE3: Wait 后 Cast down
  WaitFlag<HardEvent::V_MTE3>(EVENT_COMPUTE_DONE);
  Cast(bf16Out, fp32Buf, RoundMode::CAST_RINT, count);
  ```

- **形态 γ（多级精度链）**：
  多个 Cast-Compute 阶段串联：
  ```cpp
  for (uint32_t i = 0; i < stages; i++) {
      Cast(fp32Buf, stageIn[i], RoundMode::CAST_NONE, count);
      PipeBarrier<PIPE_V>();
      Compute(fp32Buf, ...);
      PipeBarrier<PIPE_V>();
      Cast(stageOut[i], fp32Buf, RoundMode::CAST_RINT, count);
      PipeBarrier<PIPE_V>();
  }
  ```

- **与 P30 的边界**：P30 是递推链（y0→y1→y2）的 barrier，A4 是精度链（Cast→Compute→Cast）的 barrier。两者可同时存在：P30 保证递归正确性，A4 保证精度转换正确性。

- **与 D1 的协同**：D1（Mixed Precision）定义了 Cast 链，A4 确保 Cast 链的同步。两者必须同时使用：D1 做精度转换，A4 做同步保证。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: PipeBarrier<PIPE_V> 仅适用于 Vector 引擎。跨引擎用 SetFlag/WaitFlag 或 SyncAll
约束 2: 每条 PipeBarrier 增加 1-2 周期。形态 γ 的 N 阶段链 = N × 1~2 周期
约束 3: 仅在 true dependency 处插入（dst 是下一步的 src）。独立操作间不需要
约束 4: RoundMode 必须正确：upcast 用 CAST_NONE，downcast 用 CAST_RINT
约束 5: SyncAll 不可完全替代 PipeBarrier（SyncAll 更粗粒度、更慢，仅在多核/跨引擎时用）
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `Cast 次数 = ?`, `Compute 次数 = ?`
- `PipeBarrier 数量 = ?`, `理论 overhead = ? 周期`
- `RoundMode: upcast = ?, downcast = ?`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: Cast 后有 PipeBarrier 或事件同步
grep -cE "Cast\s*\([^)]*\)[^;]*;\s*PipeBarrier|Cast\s*\([^)]*\)[^;]*;\s*SetFlag" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: Compute 后有 PipeBarrier 或事件同步
grep -cE "Add\s*\([^)]*\)[^;]*;\s*PipeBarrier|Mul\s*\([^)]*\)[^;]*;\s*PipeBarrier" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: 无 SyncAll 在精度链内部
grep -cE "Cast\s*\([^)]*\)[^}]*SyncAll|Add\s*\([^)]*\)[^}]*SyncAll" modified_files/op_kernel/*.cpp
# 期望: == 0

# 检查 4: RoundMode 正确（upcast CAST_NONE，downcast CAST_RINT）
grep -cE "CAST_NONE|CAST_RINT" modified_files/op_kernel/*.cpp
# 期望: >= 2（至少一上一下）

# 检查 5: Cast 的 dst/src 类型匹配（BF16↔FP32）
grep -cE "bf16.*float|float.*bf16|half.*float|float.*half" modified_files/op_kernel/*.cpp
# 期望: >= 1
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：PIPE_V 不存在 | 确认 CANN 版本。旧版本用 `PipeBarrier<PIPE_ALL>` 或 `__sync()` |
| 运行时：性能下降明显 | 检查是否在不需同步的位置插了 barrier。独立操作间删除 barrier |
| 运行时：精度仍不对 | 检查 RoundMode：`CAST_NONE` 用于 upcast，`CAST_RINT` 用于 downcast。不要反过来 |
| 运行时：BF16 输出全 0 | 确认 `Cast(bf16Out, fp32Buf, ...)` 的 src 是已完成 Compute 的 fp32Buf，不是原始输入 |
| 跨引擎场景误用 PIPE_V | `PIPE_V` 只同步 Vector。MTE2→Vector 用 `SetFlag<HardEvent::MTE2_V>` |
| 多级链遗漏中间 barrier | 每段 Cast→Compute 和 Compute→Cast 之间都需要 barrier。不要只加首尾 |
| 与 P30 的 barrier 重复 | 若算子同时有递归链和精度链，barrier 可能重复。检查每个 barrier 是否都有独立作用 |
| 过度优化删除必要 barrier | 调试阶段先保留所有 barrier，profiling 确认安全后再选择性删除 |
| Cast 后 buffer 被后续循环覆盖 | 若 Cast 结果要跨循环使用，需额外 SyncAll 或事件保证全局可见性 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[A4 Playbook Completion]
Step 1: done (/tmp/a4_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: constraints: cast_count=? barrier_count=? overhead=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
