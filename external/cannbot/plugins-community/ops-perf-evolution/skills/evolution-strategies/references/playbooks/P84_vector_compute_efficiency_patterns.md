# P84 Playbook: Vector Compute Efficiency Patterns

> 本 Playbook 为**强制流程**。采纳 P84 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P84 的核心是**通过 Counter 模式减少 Scalar 控制开销、用低延迟指令组合替代高延迟归约、把连续 Vector 链中间结果保留在 UB 内直通消费**。

## Step 1: 定位关键结构

```bash
grep -n "Vector|vector|VEC|Add|Mul|Exp|计算" \
    shared/original/op_kernel/*.cpp > /tmp/p84_locations.txt
grep -n "repeat|mask|Mask|CalcMask|CalcRepeat" \
    shared/original/op_kernel/*.cpp >> /tmp/p84_locations.txt
grep -n "reduce|Reduce|Sum|Max|Min|归约" \
    shared/original/op_kernel/*.cpp >> /tmp/p84_locations.txt
grep -n "SetMaskCount|Counter|COUNTER|ResetMask" \
    shared/original/op_kernel/*.cpp >> /tmp/p84_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前 Vector 计算模式（Normal/Counter**：文件 + 行号
- **repeat/mask 计算方式**：文件 + 行号
- **归约路径**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 控制模式 | `?` (Normal) | Counter | `op_kernel/*.cpp:L?` |
| repeat | `?` (手动计算) | 一次性覆盖 | `op_kernel/*.cpp:L?` |
| 归约 | `?` (高延迟 API) | 低延迟组合 | `op_kernel/*.cpp:L?` |
| 中间结果 | `?` (写回 GM) | UB 内直通 | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整优化（Counter + 低延迟归约 + UB 直通）**。
- **形态 β — 仅 Counter 模式**：不做归约和链优化。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// Normal 模式（反例）：手动计算 repeatTimes 和 tail mask
uint32_t repeatTimes = dataSize / ONE_REPEAT_SIZE;
uint32_t tailSize = dataSize % ONE_REPEAT_SIZE;
Add(dst, src1, src2, FULL_MASK, repeatTimes, {...});
if (tailSize > 0) {
    SetVectorMask(tailMask);
    Add(dst[offset], src1[offset], src2[offset], tailMask, 1, {...});
}

// Counter 模式（正例）：按总元素个数一次性覆盖
SetMaskCount();
SetVectorMask<MaskMode::COUNTER>(ELE_SIZE);
Add(dst, src1, src2, MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
ResetMask();
```

### 3C. Variant Notes

- Counter 模式依赖硬件和 API 支持，使用后必须正确恢复 mask 模式。
- 低延迟归约组合会增加临时 buffer、PipeBarrier 和 shape-specific 分支。
- UB 融合连续 Vector 链会延长中间 tensor 生命周期，链条过长可能挤占双缓冲空间。

## Step 4: 约束复核

- Counter 模式硬件/API 支持
- mask 恢复责任
- UB 空间与链长度权衡

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "SetMaskCount|MaskMode::COUNTER" modified_files/op_kernel/*.cpp  # >=1
grep -cE "ResetMask|恢复.*mask" modified_files/op_kernel/*.cpp  # >=1
grep -cE "MASK_PLACEHOLDER" modified_files/op_kernel/*.cpp  # >=1
grep -cE "repeatTimes|tailSize|CalcMask" modified_files/op_kernel/*.cpp  # ==0（或退化分支）
grep -cE "Add.*dst.*src1.*src2" modified_files/op_kernel/*.cpp  # >=1
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| mask 未恢复 | 确保 ResetMask |
| 归约组合复杂 | 封装为函数 |
| UB 链过长 | 分段或写回 GM |
| 硬件不支持 | 回退 Normal |

---

**完成清单**：
```
[P84 Playbook Completion]
Step 1: done (/tmp/p84_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: Counter 模式硬件/API 支持; mask 恢复责任; UB 空间与链长度权衡: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
