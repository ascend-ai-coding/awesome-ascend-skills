# P43 Playbook: 反向梯度 4路PingPong 常驻与 Scatter-Add 延迟写回

> 本 Playbook 为**强制流程**。采纳 P43 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P43 的核心是**dK/dV 梯度累积在 workspace 的 4 路 PingPong buffer 中轮转，延迟到整个 S1 处理完毕后再统一 scatter-add，将 scatter-add 从 O(S2) 次降为 O(S1) 次**。

## Step 1: 定位关键结构

```bash
grep -n "grad|Grad|dK|dV|反向|backward" \
    shared/original/op_kernel/*.cpp > /tmp/p43_locations.txt
grep -n "ScatterAdd|scatter.*add|AtomicAdd" \
    shared/original/op_kernel/*.cpp >> /tmp/p43_locations.txt
grep -n "PingPong|pingpong|PPP|%.*4|selectdK" \
    shared/original/op_kernel/*.cpp >> /tmp/p43_locations.txt
grep -n "CrossCoreWaitFlag|changeS1|SCATTER_SYNC" \
    shared/original/op_kernel/*.cpp >> /tmp/p43_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前 scatter-add 频率**：文件 + 行号
- **workspace buffer 数**：文件 + 行号
- **跨核同步方式**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| buffer 数 | `?` (1) | 4 路 PingPong | `op_kernel/*.cpp:L?` |
| scatter-add | `?` (O(S2)) | O(S1) | `op_kernel/*.cpp:L?` |
| 索引 | `?` (无) | selectdKPPPidx % 4 | `op_kernel/*.cpp:L?` |
| 同步 | `?` (无) | CrossCoreWaitFlag | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整 4 路 PingPong + 延迟 ScatterAdd（含跨核同步）**。
- **形态 β — 仅延迟 ScatterAdd**：不做 4 路 buffer。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
selectedKGmOffset = selectdKPPPidx * selectedKWspOffset;
mmPingPongIdx = 1 - mmPingPongIdx;
selectdKPPPidx = (selectdKPPPidx + 1) % 4;
if (scatterRunInfo.changeS1) {
    CrossCoreWaitFlag<2, PIPE_MTE2>(SCATTER_SYNC_FLAG);
    ScatterAddByS1(vecOp, actual_seq_qlen, actual_seq_kvlen);
}
```

### 3C. Variant Notes

- 4 路 PingPong workspace 占用 = 4 × buffer_size × 核数，HBM 开销显著。
- 跨核同步逻辑复杂，changeS1 条件判断和 ScatterAddByS1 触发时机需要精确控制。

## Step 4: 约束复核

- HBM 占用 = 4 × buffer × core
- 跨核同步复杂度
- changeS1 触发时机

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "selectdKPPPidx|PPPidx" modified_files/op_kernel/*.cpp  # >=1
grep -cE "%.*4|PingPong" modified_files/op_kernel/*.cpp  # >=1
grep -cE "ScatterAddByS1|ScatterAdd" modified_files/op_kernel/*.cpp  # >=1
grep -cE "CrossCoreWaitFlag|SCATTER_SYNC" modified_files/op_kernel/*.cpp  # >=1
grep -cE "changeS1" modified_files/op_kernel/*.cpp  # >=1
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| HBM 不足 | 减 buffer 或取消 |
| 同步死锁 | 检查 CrossCore 配对 |
| changeS1 错 | 验证触发条件 |
| 非反向场景 | 不适用 |

---

**完成清单**：
```
[P43 Playbook Completion]
Step 1: done (/tmp/p43_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: HBM 占用 = 4 × buffer × core; 跨核同步复杂度; changeS1 触发时机: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
