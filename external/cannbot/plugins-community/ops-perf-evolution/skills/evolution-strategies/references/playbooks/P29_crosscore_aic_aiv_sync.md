# P29 Playbook: AIC-AIV CrossCore Sync

> 本 Playbook 为**强制流程**。采纳 P29 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P29 的核心是**用细粒度 CrossCoreSetFlag/CrossCoreWaitFlag 替代粗粒度 SyncAll，实现 AIC 与 AIV 间的精确数据交接**。

## Step 1: 定位关键结构

```bash
grep -n "SyncAll|PipeBarrier|SetFlag|WaitFlag" \
    shared/original/op_kernel/*.cpp > /tmp/p29_locations.txt
grep -n "CrossCoreSetFlag|CrossCoreWaitFlag|CROSS_CORE" \
    shared/original/op_kernel/*.cpp >> /tmp/p29_locations.txt
grep -n "ASCEND_IS_AIC|ASCEND_IS_AIV|AIC|AIV" \
    shared/original/op_kernel/*.cpp >> /tmp/p29_locations.txt
grep -n "Cube.*Vector|Vector.*Cube|CV|MIX" \
    shared/original/op_kernel/*.cpp >> /tmp/p29_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前同步原语（SyncAll/SetFlag/CrossCore**：文件 + 行号
- **AIC/AIV 分支逻辑**：文件 + 行号
- **CV 融合确认**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 同步原语 | `?` (SyncAll) | CrossCoreSet/WaitFlag | `op_kernel/*.cpp:L?` |
| 粒度 | `?` (粗) | 细粒度（按 buffer） | `op_kernel/*.cpp:L?` |
| AIC→AIV | `?` (无) | 双 AIV 分别同步 | `op_kernel/*.cpp:L?` |
| 事件 ID | `?` (无) | 独立 ID + OFFSET | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整 CrossCore 替换（所有 SyncAll → CrossCore）**。
- **形态 β — 保留部分 SyncAll**：仅热路径替换。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
if constexpr (bufferType == BufferType::L1) {
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id0_);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id0_ + AIV0_AIV1_OFFSET);
    } else {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id1_);
    }
}
```

### 3C. Variant Notes

- 一个 AIC 对应两个 AIV，需两次 SetFlag/WaitFlag。
- 跨核事件 ID 空间有限，需合理分配。
- 批量数据交接优于逐 tile 交接。

## Step 4: 约束复核

- CrossCore 延迟高于核内同步
- 需保证 AIC/AIV 分支正确性
- 事件 ID 冲突会导致死锁

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "CrossCoreSetFlag|CrossCoreWaitFlag" modified_files/op_kernel/*.cpp  # >=1
grep -cE "CROSS_CORE_SYNC_MODE" modified_files/op_kernel/*.cpp  # >=1
grep -cE "AIV0_AIV1_OFFSET|id0_|id1_" modified_files/op_kernel/*.cpp  # >=1
grep -cE "ASCEND_IS_AIC|ASCEND_IS_AIV" modified_files/op_kernel/*.cpp  # >=1
grep -cE "SyncAll" modified_files/op_kernel/*.cpp  # ==0（或注释说明保留场景）
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 事件 ID 冲突 | 独立分配，检查 OFFSET |
| AIC 漏同步一个 AIV | 确认两次 WaitFlag |
| 跨核延迟高 | 批量交接，减少同步次数 |
| 死锁 | 检查 Set/Wait 配对 |

---

**完成清单**：
```
[P29 Playbook Completion]
Step 1: done (/tmp/p29_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: CrossCore 延迟高于核内同步; 需保证 AIC/AIV 分支正确性; 事件 ID 冲突会导致死锁: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
