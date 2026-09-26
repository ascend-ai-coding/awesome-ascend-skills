# P5 Playbook: 流水线同步实操流程

> **强制流程**。P5 是把粗粒度 SyncAll / 隐式同步升级为**细粒度 HardEvent 同步**，从而消除流水空泡。误用会导致死锁，**必须**按本 Playbook 逐步执行。

## Step 1: 定位同步关键点

执行下面的 grep，把结果写入 `/tmp/p5_locations.txt`：

```bash
# 1.1 当前同步情况
grep -n "SyncAll\|PipeBarrier" shared/original/op_kernel/*.cpp > /tmp/p5_locations.txt

# 1.2 跨 pipe 数据依赖点（MTE→V, V→MTE, V→S, S→V, MTE→S）
grep -nE "Cast\b|Add\b|Mul\b|ReduceSum\b|CopyIn\b|CopyOut\b|DataCopy\b|SetValue\b|GetValue\b" shared/original/op_kernel/*.cpp >> /tmp/p5_locations.txt

# 1.3 现有 HardEvent 使用（若 Preconditions 已确保为 0，这里验证）
grep -n "HardEvent::\|FetchEventID" shared/original/op_kernel/*.cpp >> /tmp/p5_locations.txt
```

**交付物**（`implementation_note.txt` "Playbook Step 1"）：
- **现有 SyncAll / PipeBarrier 位置**：列出每个粗粒度同步点
- **跨 pipe 依赖链**：识别 Cast (MTE→V) / ReduceSum (V→S) / CopyOut (V→MTE) 等跨 pipe 数据流

## Step 2: 同步升级计划表（强制）

**不填完此表不得进入 Step 3**：

| 依赖点 | 生产者 pipe | 消费者 pipe | 升级方式 | HardEvent 类型 | 预计 event_id |
|---|---|---|---|---|---|
| `CopyIn → Cast` | MTE2 | V | SetFlag/WaitFlag | `HardEvent::MTE2_V` | 0 |
| `Cast → ReduceSum` | V | V | PipeBarrier | `<PIPE_V>` | — |
| `Compute → CopyOut` | V | MTE3 | SetFlag/WaitFlag | `HardEvent::V_MTE3` | 1 |
| ... | ... | ... | ... | ... | ... |

**规则**：
- **同 pipe 内依赖** → 用 `PipeBarrier<PIPE_V>()` 或 `PipeBarrier<PIPE_MTE2>()`
- **跨 pipe 依赖** → 用 `SetFlag<HardEvent::X_Y>(event_id); WaitFlag<HardEvent::X_Y>(event_id);`
- **跨核依赖** → 用 `CrossCoreSetFlag / CrossCoreWaitFlag`（与 P14 协作）

## Step 3: 同步升级实施

### 3A. 形态识别

读 Step 2 的表，判断你的算子属于以下哪种主导形态：

- **形态 α — MTE + V 两 pipe**（Softmax / LayerNorm 类）：以 MTE2 ↔ V ↔ MTE3 为主
- **形态 β — MTE + V + S 三 pipe**（有 ReduceSum + Scalar 后处理）：V → S → V 循环
- **形态 γ — 跨核**（Cube + Vector 融合）：需要 CrossCoreSetFlag，**请优先考虑 P14 CV 预发射**

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template（形态 α）

```cpp
// === 改造前（粗粒度：依赖 TQue 隐式同步 + 可能 SyncAll）===
CopyIn(i);
Cast(xFp32, xLocal, RoundMode::CAST_NONE, numCol);
Add(xFp32, xFp32, yLocal, numCol);
ReduceSum(result, xFp32, temp, numCol);
CopyOut(i);

// === 改造后（细粒度 HardEvent + PipeBarrier）===
CopyIn(i);
event_t e_mte2v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
SetFlag<HardEvent::MTE2_V>(e_mte2v);
WaitFlag<HardEvent::MTE2_V>(e_mte2v);

Cast(xFp32, xLocal, RoundMode::CAST_NONE, numCol);
PipeBarrier<PIPE_V>();
Add(xFp32, xFp32, yLocal, numCol);
PipeBarrier<PIPE_V>();
ReduceSum(result, xFp32, temp, numCol);

event_t e_vmte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
SetFlag<HardEvent::V_MTE3>(e_vmte3);
WaitFlag<HardEvent::V_MTE3>(e_vmte3);
CopyOut(i);
```

### 3C. Variant Notes

- **形态 β（有 Scalar 读 Vector 结果）**：在 `ReduceSum → Scalar` 之间插入 `SetFlag<HardEvent::V_S> / WaitFlag<HardEvent::V_S>`。Scalar 分支计算完还可能要 `SetFlag<HardEvent::S_V> / WaitFlag<HardEvent::S_V>` 让 Vector 能用上结果。
- **形态 γ（跨核）**：跳过本 Playbook，改用 P14 CV 预发射 Playbook。
- **已有部分同步**：保留原有 PipeBarrier，只**替换**粗粒度 SyncAll 为 HardEvent。禁止到处加 PipeBarrier 造成过度同步（会抵消收益）。

## Step 4: Event ID 分配复核

**规则**：
- 同一 event_id 可以在**不同循环迭代**中复用（同 SetFlag/WaitFlag 配对已完成）
- 同一迭代内**不同 HardEvent 类型**可以共用 event_id（类型不同不会冲突）
- **并发活跃的 event_id 不超过 8 个**（硬件限制）

在 `implementation_note.txt` "Playbook Step 4" 报告：
```
event_id 分配：
  e_mte2v = 0 (HardEvent::MTE2_V)
  e_vmte3 = 1 (HardEvent::V_MTE3)
并发活跃数: 2 ≤ 8 ✓
```

## Step 5: 编码后自检（5 条 grep，全部必须过）
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
# 检查 1: SetFlag 出现次数（每个 SetFlag 必须有对应 WaitFlag）
grep -c "SetFlag<HardEvent::" modified_files/op_kernel/*.cpp
# 期望：>= 1

# 检查 1b: WaitFlag 必须与 SetFlag 配对
grep -c "WaitFlag<HardEvent::" modified_files/op_kernel/*.cpp
# 期望：等于检查 1 的数量（agent 必须对比两个数字是否相等，不等则 Step 3C 有漏配对）

# 检查 2: event_id 已通过 FetchEventID 分配（不能硬编码）
grep -c "FetchEventID(HardEvent::" modified_files/op_kernel/*.cpp
# 期望：>= 1

# 检查 3: 粗粒度 SyncAll() 已清理完毕（或极少保留）
grep -c "SyncAll()" modified_files/op_kernel/*.cpp
# 期望：<= 1（P5 的核心目标就是消除 SyncAll。保留 ≤1 个通常是 kernel 启动时的初始同步，可接受）

# 检查 4: 形态 α 必有 MTE2_V 或 V_MTE3 类型事件
grep -En "HardEvent::MTE2_V|HardEvent::V_MTE3" modified_files/op_kernel/*.cpp | wc -l
# 期望: >= 1（形态 α / β 必须）

# 检查 5: 无裸 PipeBarrier<PIPE_ALL>（过度同步，应换细粒度）
grep -c "PipeBarrier<PIPE_ALL>" modified_files/op_kernel/*.cpp
# 期望: == 0
```

**在 implementation_note.txt "Playbook Step 5" 列出每条实际输出**。任一失败 → 回 Step 3 重做。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 算子死锁 / 超时不返回 | SetFlag 和 WaitFlag 的 event_id 或 HardEvent 类型不匹配；检查 Step 4 event_id 分配 |
| 精度错（NaN / 随机数） | WaitFlag 放得太早，消费者在生产者未完成时就读了 UB；确保 WaitFlag 在消费指令**前** |
| 性能比基线还差 | 过度同步（每个 Vector 指令后都加 PipeBarrier）；形态 α 每条 ReduceSum / Cast 之间不需要同步，只在跨 pipe 边界加 |
| Compilation: `event_t not declared` | 漏了 `static_cast<event_t>(GetTPipePtr()->FetchEventID(...))` 这一步 |
| 仿真报 "event overflow" | 并发 event_id 超过 8；合并循环内同 HardEvent 类型的 event |

---

**完成后在 `implementation_note.txt` 末尾贴**：
```
[P5 Playbook Completion]
Step 1: done
Step 2: sync upgrade plan filled (N pairs)
Step 3: form = alpha/beta/gamma
Step 4: event_ids allocated, concurrent = X ≤ 8
Step 5: all 5 grep checks passed
Step 6: no pitfalls / {列出触发的}
```
