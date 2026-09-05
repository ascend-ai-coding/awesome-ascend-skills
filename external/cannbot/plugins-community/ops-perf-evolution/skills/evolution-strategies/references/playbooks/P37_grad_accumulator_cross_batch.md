# P37 Playbook: 梯度累加器跨 batch 常驻

> 本 Playbook 为**强制流程**。采纳 P37 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P37 的核心是**gradWeight 使用 TBuf 分配常驻 UB，在整个 B×loopS 循环中持续累加，最后一次性 ReduceSum 并搬出**。

## Step 1: 定位关键结构

```bash
grep -n "grad|Grad|梯度|dW|weight_grad" \
    shared/original/op_kernel/*.cpp > /tmp/p37_locations.txt
grep -n "batch|Batch|loopS|B×S|idxB|idxS" \
    shared/original/op_kernel/*.cpp >> /tmp/p37_locations.txt
grep -n "TBuf|VECCALC|AllocTensor|FreeTensor" \
    shared/original/op_kernel/*.cpp >> /tmp/p37_locations.txt
grep -n "Duplicate|Sum|ReduceSum|CopyOut" \
    shared/original/op_kernel/*.cpp >> /tmp/p37_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前梯度累加方式（每 batch 搬出/常驻**：文件 + 行号
- **batch/seq 循环结构**：文件 + 行号
- **TBuf 使用**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 累加位置 | `?` (GM) | UB 常驻 | `op_kernel/*.cpp:L?` |
| 搬出次数 | `?` (每 batch) | 最后 1 次 | `op_kernel/*.cpp:L?` |
| 初始化 | `?` (无) | Duplicate 0 | `op_kernel/*.cpp:L?` |
| 最后处理 | `?` (无) | SumAndCopyOut | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整跨 batch 常驻（TBuf + Duplicate + 循环累加 + SumAndCopyOut）**。
- **形态 β — 仅延迟搬出**：不改为 TBuf，只是减少搬出次数。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
TBuf<QuePosition::VECCALC> gradWeightBuf;
void Process() {
    LocalTensor<float> gradWeight = gradWeightBuf.Get<float>();
    Duplicate(gradWeight, 0.0f, baseS * calNum * 3);
    PipeBarrier<PIPE_V>();
    for (int64_t idxB = 0; idxB < batchSize; idxB++) {
        for (int64_t idxS = 0; idxS < loopS; idxS++) {
            GradWeightConv(sTileLen);
        }
    }
    SumAndCopyOutGradWeight();
}
```

### 3C. Variant Notes

- gradWeightBuf 是 UB 中最大的常驻 buffer 之一，baseS 的 tiling 直接受此约束。
- 最终 ReduceSum 搬出，batch 数很大时累加精度可能需要 FP32 中间结果。

## Step 4: 约束复核

- UB 空间约束 tiling
- 大 batch 累加精度
- FP32 中间结果需求

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "gradWeightBuf|gradWeight" modified_files/op_kernel/*.cpp  # >=1
grep -cE "Duplicate.*0\.0f|Duplicate.*0," modified_files/op_kernel/*.cpp  # >=1
grep -cE "GradWeightConv|grad.*累加" modified_files/op_kernel/*.cpp  # >=1
grep -cE "SumAndCopyOut|CopyOutGrad" modified_files/op_kernel/*.cpp  # >=1
grep -cE "for.*idxB.*DataCopy|每.*batch.*搬出" modified_files/op_kernel/*.cpp  # ==0
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| UB 不足 | 减 baseS 或 tile |
| 精度溢出 | 用 FP32 累加 |
| 未初始化 | Duplicate 0 |
| 非梯度场景 | 不适用 |

---

**完成清单**：
```
[P37 Playbook Completion]
Step 1: done (/tmp/p37_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: UB 空间约束 tiling; 大 batch 累加精度; FP32 中间结果需求: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
