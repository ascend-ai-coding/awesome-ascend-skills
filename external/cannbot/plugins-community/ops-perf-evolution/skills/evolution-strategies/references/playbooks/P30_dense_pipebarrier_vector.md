# P30 Playbook: PipeBarrier 密集插入的纯 Vector 同步

> 本 Playbook 为**强制流程**。采纳 P30 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P30 的核心是**在纯 Vector 递推链（y0→y1→y2）的每条指令后插入 `PipeBarrier<PIPE_V>`**，确保前一条 Vector 指令结果已写入 UB 后才被下一条读取，避免乱序执行导致的数据竞争。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p30_locations.txt`：

```bash
# 1. 递推依赖链（y0→y1→y2 或类似模式）
grep -n "y0.*y1\|y1.*y2\|prev.*next\|recurs\|depend\|chain" \
    shared/original/op_kernel/*.cpp > /tmp/p30_locations.txt
# 2. Vector 指令序列
grep -n "Add\s*(\|Mul\s*(\|Sub\s*(\|Div\s*(\|Exp\s*(\|Log\s*(\|Sqrt\s*(" \
    shared/original/op_kernel/*.cpp >> /tmp/p30_locations.txt
# 3. 当前同步机制
grep -n "SyncAll\|PipeBarrier\|SetFlag\|WaitFlag" \
    shared/original/op_kernel/*.cpp >> /tmp/p30_locations.txt
# 4. Cube/Matmul 指令（排除条件）
grep -n "Mmad\|Cube\|L0A\|L0B\|L0C\|A1Buf\|B1Buf\|C1Buf" \
    shared/original/op_kernel/*.cpp >> /tmp/p30_locations.txt
# 5. 已有的 PipeBarrier<PIPE_V>
grep -n "PipeBarrier.*PIPE_V\|PipeBarrier<PIPE_V>" \
    shared/original/op_kernel/*.cpp >> /tmp/p30_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **递推链**：y0→y1→y2 或类似依赖链的位置
- **Vector 序列**：连续的 Vector 指令列表（Add/Mul/Sub/Div/...）
- **当前同步**：SyncAll / PipeBarrier 使用位置
- **Cube 指令**：是否存在 Cube/Matmul（P30 不适用）
- **已有 PIPE_V**：是否已使用

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 递推链长度 | `?` (y0→y1→...→yN) | 不变 | `?_kernel.cpp:L?` |
| 当前同步 | `?` (无 / SyncAll / 其他) | `PipeBarrier<PIPE_V>` | `?_kernel.cpp:L?` |
| Vector 指令数 | `?` | `? + barrier` | `?_kernel.cpp:L?` |
| 插入位置 | `?` (全部 / 部分) | `alpha/beta/gamma` 见 3A | `?_kernel.cpp:L?` |
| 依赖模式 | `?` (WAR / RAW / WAW) | 确认 hazard 类型 | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的依赖模式和 hazard 类型，判断你的代码属于以下哪种形态：

- **形态 α — 密集插入（每条 Vector 指令后都加 barrier）**：递推链极短（≤4 条指令）且每条都依赖上一条结果，安全优先。
- **形态 β — 选择性插入（仅在 true dependency 边界加 barrier）**：部分指令间无依赖，只在 RAW/WAR hazard 处插入。最常见。
- **形态 γ — 交替 buffer 消除 barrier（用双 buffer 替代同步）**：通过交替使用 dst/src buffer，避免 WAR hazard，完全不需要 barrier。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 β — 选择性插入，最常见）

```cpp
// === 改造前（无同步，乱序执行风险）===
__aicore__ inline void RecursiveCompute(...) {
    // 递推链：y1 = f(x, w1, y0), y2 = f(x, w2, y1), ...
    Mul(y1Local, xLocal, weight1, count);
    Add(y1Local, y0Local, y1Local, count);  // ❌ 风险：Add 可能读到 Mul 未完成的结果
    
    Mul(y2Local, xLocal, weight2, count);
    Add(y2Local, y1Local, y2Local, count);  // ❌ 风险：Add 可能读到 Mul 未完成的结果
    
    Mul(y3Local, xLocal, weight3, count);
    Add(y3Local, y2Local, y3Local, count);  // ❌ 同上
}

// === 改造后（在 true dependency 边界加 PipeBarrier<PIPE_V>）===
__aicore__ inline void RecursiveCompute(...) {
    // Step 1: y1 = Mul(x, w1) + y0
    Mul(y1Local, xLocal, weight1, count);
    PipeBarrier<PIPE_V>();  // 确保 Mul 结果写入 UB 后再 Add
    Add(y1Local, y0Local, y1Local, count);
    
    // Step 2: y2 = Mul(x, w2) + y1
    Mul(y2Local, xLocal, weight2, count);
    PipeBarrier<PIPE_V>();  // 确保 Mul 结果写入 UB
    Add(y2Local, y1Local, y2Local, count);
    
    // Step 3: y3 = Mul(x, w3) + y2
    Mul(y3Local, xLocal, weight3, count);
    PipeBarrier<PIPE_V>();
    Add(y3Local, y2Local, y3Local, count);
}
```

**Hazard 分析**：
- `Mul(y1, x, w1)` 后 `Add(y1, y0, y1)`：**WAR hazard**（y1 既是 Mul 的 dst 又是 Add 的 src）
- 必须加 `PipeBarrier<PIPE_V>()` 保证写入完成后才能读取

### 3C. Variant Notes（若是形态 α 或 γ）

- **形态 α（密集插入，每条指令后都加）**：
  当递推链极短且对正确性要求极高时，每条 Vector 指令后都加 barrier：
  ```cpp
  Mul(y1, x, w1, count);
  PipeBarrier<PIPE_V>();
  Add(y1, y0, y1, count);
  PipeBarrier<PIPE_V>();
  Mul(y2, x, w2, count);
  PipeBarrier<PIPE_V>();
  Add(y2, y1, y2, count);
  ```
  形态 α 开销最大（N 条指令 ≈ N 个 barrier），仅在调试或极端场景使用。生产环境推荐形态 β。

- **形态 γ（交替 buffer 消除 barrier）**：
  通过分配独立 src/dst buffer，避免 WAR hazard：
  ```cpp
  // 使用交替 buffer：y1_tmp 存 Mul 结果，Add 写到 y1_out
  Mul(y1Tmp, xLocal, weight1, count);   // dst = y1Tmp
  Add(y1Out, y0Local, y1Tmp, count);    // src = y1Tmp, dst = y1Out（无 WAR）
  
  Mul(y2Tmp, xLocal, weight2, count);   // dst = y2Tmp
  Add(y2Out, y1Out, y2Tmp, count);      // src = y1Out, dst = y2Out（无 WAR）
  ```
  形态 γ 无 barrier 开销，但增加 2× buffer 占用。若 UB 充裕，形态 γ 最优；若 UB 紧张，形态 β 更平衡。

- **与 P28 的边界**：P28 用 `SetFlag/WaitFlag<HardEvent>` 做跨引擎同步，P30 用 `PipeBarrier<PIPE_V>` 做 Vector 引擎内部同步。两者不冲突：P30 解决 Vector 内指令乱序，P28 解决 Vector↔MTE2/Cube 跨引擎时序。

- **与 A4 的协同**：A4（SetFlag/WaitFlag）用于软件事件同步，P30 用于硬件流水线 barrier。若算子同时有跨核/跨引擎同步和 Vector 内依赖链，两者可叠加。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: PipeBarrier<PIPE_V> 仅适用于纯 Vector 流水线。含 Cube/Matmul 时用 HardEvent 或 SyncAll
约束 2: 每条 barrier 增加 1-2 周期开销。密集插入时（形态 α）总开销 = N × 1~2 周期
约束 3: 形态 β 的 barrier 必须放在 true dependency 边界（WAR/RAW），不能放在独立指令之间
约束 4: 形态 γ 的交替 buffer 必须确保 src ≠ dst，否则仍有 WAR hazard
约束 5: PipeBarrier 不能替代跨核 SyncAll。多核场景仍需 SyncAll 保证全局可见性
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `Vector 指令总数 = ?`
- `形态 β 的 barrier 数量 = ?`
- `形态 α 的 barrier 数量 = ?`
- `形态 γ 的额外 buffer = ? bytes`
- `选择的形态 = ?，理论 overhead = ? 周期`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 已引入 PipeBarrier<PIPE_V>
grep -cE "PipeBarrier\s*<\s*PIPE_V\s*>" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 无 Cube/Matmul 指令（纯 Vector）
grep -cE "Mmad|Cube|L0A|L0B|L0C" modified_files/op_kernel/*.cpp
# 期望: == 0

# 检查 3: PipeBarrier<PIPE_V> 在 Vector 指令后（不是无关联的位置）
grep -cE "Add\s*\([^)]*\)[^;]*;\s*PipeBarrier|Mul\s*\([^)]*\)[^;]*;\s*PipeBarrier" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: 无 SyncAll 在递推链内部（已被 PipeBarrier 替代）
grep -cE "SyncAll\s*\(\s*\)" modified_files/op_kernel/*.cpp
# 期望: == 0（或仅在多核入口处保留）

# 检查 5: 递推链的 WAR hazard 已覆盖（每个 dst=src 的指令对后都有 barrier）
# 人工检查：对每个 Mul(dst,...) + Add(dst, src, dst, ...) 对，确认中间有 PipeBarrier
grep -cE "PipeBarrier" modified_files/op_kernel/*.cpp
# 记录次数，应与 WAR hazard 对数匹配
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：PIPE_V 不存在 | 确认 CANN 版本支持 `PipeBarrier<PIPE_V>`。旧版本可能用 `PipeBarrier<PIPE_ALL>` 或 `__sync()` |
| 运行时：性能下降明显 | 形态 α 的密集 barrier 开销大。退化为形态 β（仅 true dependency 处加）或形态 γ（交替 buffer） |
| 运行时：结果仍不对 | 检查是否有遗漏的 WAR hazard。常见遗漏：`Add(y1, y0, y1)` 前有 `Mul(y1, x, w)` 但漏了 barrier |
| 误在 Cube/Vector 混合代码中用 PIPE_V | `PipeBarrier<PIPE_V>` 只同步 Vector 引擎，不保证 Cube 完成。混合代码用 `SyncAll` 或 P28 HardEvent |
| 形态 γ 的交替 buffer 仍出现 WAR | 确认 src ≠ dst。若 `Add(dst, src0, src1)` 中 `dst == src1`，仍有 WAR，需 barrier |
| 多核场景漏 SyncAll | `PipeBarrier<PIPE_V>` 是核内同步。多核写共享 GM 时，仍需 `SyncAll` 或 `AtomicAdd` |
| 过度使用形态 α | 形态 α 仅在调试时使用。生产环境默认形态 β。除非 profiling 证明形态 β 仍有竞争 |
| barrier 放在独立指令之间 | 两条无依赖的 Vector 指令之间不需要 barrier。例如 `Add(a, b, c)` 和 `Mul(d, e, f)` 独立，中间加 barrier 纯浪费 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P30 Playbook Completion]
Step 1: done (/tmp/p30_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: constraints: vector_count=? barrier_count=? overhead=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
