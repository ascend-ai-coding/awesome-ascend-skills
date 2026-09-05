# P41 Playbook: gamma/scale/offset 常驻 UB

> 本 Playbook 为**强制流程**。采纳 P41 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P41 的核心是**gamma/scale/offset 等小张量整个 kernel 生命周期内只搬入一次，常驻 UB 供所有 ubLoop 迭代复用，利用 ReinterpretCast 原地 Cast**。

## Step 1: 定位关键结构

```bash
grep -n "gamma|Gamma|scale|Scale|offset|Offset|RmsNorm|LayerNorm" \
    shared/original/op_kernel/*.cpp > /tmp/p41_locations.txt
grep -n "DataCopy|CopyIn|CopyOut|Cast|ReinterpretCast" \
    shared/original/op_kernel/*.cpp >> /tmp/p41_locations.txt
grep -n "for.*loop|loop|迭代|ubLoop|Process" \
    shared/original/op_kernel/*.cpp >> /tmp/p41_locations.txt
grep -n "AllocTensor|FreeTensor|inQueueGamma" \
    shared/original/op_kernel/*.cpp >> /tmp/p41_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **gamma/scale 搬运位置**：文件 + 行号
- **循环结构**：文件 + 行号
- **当前是否每轮重复搬运**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| gamma 搬运 | `?` (每轮) | 开头一次 | `op_kernel/*.cpp:L?` |
| Cast | `?` (每轮) | 开头一次 | `op_kernel/*.cpp:L?` |
| 复用 | `?` (无) | 所有 ubLoop | `op_kernel/*.cpp:L?` |
| 技巧 | `?` (无) | ReinterpretCast | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整常驻（Load + ReinterpretCast + Cast + 全循环复用）**。
- **形态 β — 仅 Load 常驻**：不做 Cast，保持原格式。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
LocalTensor<float> gammaLocalFp32 = inQueueGamma.AllocTensor<float>();
LocalTensor<KV_DTYPE> gammaLocal = gammaLocalFp32.ReinterpretCast<KV_DTYPE>()[RMS_NORM_LENGTH];
DataCopyPad(gammaLocal, gammaGm, copyParams, padParams);
Cast(gammaLocalFp32, gammaLocal, RoundMode::CAST_NONE, RMS_NORM_LENGTH);
// gammaLocalFp32 在所有 ubLoop 中复用
```

### 3C. Variant Notes

- 常驻 buffer 占用约 2KB UB 空间。
- ReinterpretCast 原地 Cast 技巧要求 float buffer 后半段空间足够存放 half 数据。

## Step 4: 约束复核

- UB 2KB 占用
- ReinterpretCast 空间要求
- 仅适用于小张量

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "gammaLocalFp32|gammaLocal" modified_files/op_kernel/*.cpp  # >=1
grep -cE "ReinterpretCast|reinterpret_cast" modified_files/op_kernel/*.cpp  # >=1
grep -cE "Cast.*gamma|gamma.*Cast" modified_files/op_kernel/*.cpp  # >=1
grep -cE "DataCopyPad.*gamma|gamma.*DataCopy" modified_files/op_kernel/*.cpp  # >=1
grep -cE "loop.*gamma|每.*轮.*gamma" modified_files/op_kernel/*.cpp  # ==0
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| UB 不足 | 减 tile |
| ReinterpretCast 越界 | 确认后半段空间 |
| 大 gamma | 评估常驻代价 |
| gamma 变化 | 不适用 |

---

**完成清单**：
```
[P41 Playbook Completion]
Step 1: done (/tmp/p41_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: UB 2KB 占用; ReinterpretCast 空间要求; 仅适用于小张量: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
