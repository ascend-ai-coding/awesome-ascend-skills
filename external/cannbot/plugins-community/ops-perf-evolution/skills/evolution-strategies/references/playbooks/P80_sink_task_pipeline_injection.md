# P80 Playbook: Sink Token 流水线注入与 S2 块跳过

> 本 Playbook 为**强制流程**。采纳 P80 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P80 的核心是**将 sink token 作为独立 task 注入到 Flash Attention 的 3-stage pipeline 中，与普通 sparse task 共享流水线调度，并跳过无效 S2 block**。

## Step 1: 定位关键结构

```bash
grep -n "sink|Sink|streaming|Streaming LLM" \
    shared/original/op_kernel/*.cpp > /tmp/p80_locations.txt
grep -n "FlashAttention|flash.*attention|FA|IFA" \
    shared/original/op_kernel/*.cpp >> /tmp/p80_locations.txt
grep -n "ExecuteTask|TaskInfo|isSinkTensor" \
    shared/original/op_kernel/*.cpp >> /tmp/p80_locations.txt
grep -n "ProcessSink|ProcessSparse|s2Start|s2End" \
    shared/original/op_kernel/*.cpp >> /tmp/p80_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前 sink 处理方式（独立循环/注入**：文件 + 行号
- **FA pipeline 结构**：文件 + 行号
- **task 调度方式**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| sink 处理 | `?` (独立循环) | 注入流水线 | `op_kernel/*.cpp:L?` |
| task 类型 | `?` (单一) | sink + sparse | `op_kernel/*.cpp:L?` |
| S2 跳过 | `?` (无) | IsSkipCal | `op_kernel/*.cpp:L?` |
| sInnerSize | `?` (大) | 256 | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整注入（sink task + sparse tasks + S2 跳过）**。
- **形态 β — 仅注入**：不做 S2 跳过。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
void FlashAttention() {
    for (int s1g = 0; s1g < s1gLoops; s1g++) {
        // 第一个 task: sink token
        if (keySinkNumber > 0) {
            TaskInfo sinkTask;
            sinkTask.isSinkTensor = true;
            sinkTask.s2Start = 0;
            sinkTask.s2End = sinkNumber;
            ExecuteTask(sinkTask);  // 从 keySink/valueSink 读取
        }
        // 后续 tasks: sparse token
        for (int s2 = sinkNumber; s2 < s2End; s2 += sInnerSize) {
            TaskInfo sparseTask;
            sparseTask.isSinkTensor = false;
            ExecuteTask(sparseTask);  // 从 KV cache 读取
        }
    }
}
```

### 3C. Variant Notes

- 每个 S1G block 多一个 sink task，增加总 task 数。
- sink tensor 需要额外的 GM 空间。
- sInnerSize 减小（256）可能降低有效 block 的计算效率。

## Step 4: 约束复核

- 总 task 数增加
- sink GM 空间
- sInnerSize 效率

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "isSinkTensor|sinkTask" modified_files/op_kernel/*.cpp  # >=1
grep -cE "ExecuteTask|TaskInfo" modified_files/op_kernel/*.cpp  # >=1
grep -cE "keySinkNumber|sinkNumber" modified_files/op_kernel/*.cpp  # >=1
grep -cE "s1gLoops|s1g" modified_files/op_kernel/*.cpp  # >=1
grep -cE "ProcessSink|ProcessSparse" modified_files/op_kernel/*.cpp  # ==0（统一为 ExecuteTask）
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| task 数过多 | 评估调度开销 |
| sink GM 不足 | 分配额外空间 |
| sInner 效率低 | 权衡 block 大小 |
| 非 Streaming LLM | 不适用 |

---

**完成清单**：
```
[P80 Playbook Completion]
Step 1: done (/tmp/p80_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 总 task 数增加; sink GM 空间; sInnerSize 效率: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
