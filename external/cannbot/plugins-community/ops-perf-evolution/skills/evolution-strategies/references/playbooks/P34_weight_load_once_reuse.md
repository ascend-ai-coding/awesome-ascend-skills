# P34 Playbook: Weight 常驻与预计算复用

> 本 Playbook 为**强制流程**。采纳 P34 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P34 的核心是**将 weight/scale 等小尺寸跨 loop 不变数据在 Process 开头搬入 UB 并 cast 到 FP32，整个 B×S 循环中反复使用**。

## Step 1: 定位关键结构

```bash
grep -n "weight|Weight|scale|Scale|gamma|Gamma" \
    shared/original/op_kernel/*.cpp > /tmp/p34_locations.txt
grep -n "DataCopy|CopyIn|CopyOut|Cast|ReinterpretCast" \
    shared/original/op_kernel/*.cpp >> /tmp/p34_locations.txt
grep -n "for.*loop|loop|迭代|Process|ubLoop" \
    shared/original/op_kernel/*.cpp >> /tmp/p34_locations.txt
grep -n "AllocTensor|FreeTensor|inQueueW|outQueue" \
    shared/original/op_kernel/*.cpp >> /tmp/p34_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **weight 搬运位置**：文件 + 行号
- **循环结构**：文件 + 行号
- **当前是否每轮重复搬运**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| weight 搬运 | `?` (每轮) | Process 开头一次 | `op_kernel/*.cpp:L?` |
| Cast | `?` (每轮) | 开头一次 | `op_kernel/*.cpp:L?` |
| 常驻 | `?` (无) | 整个 B×S 循环 | `op_kernel/*.cpp:L?` |
| Broadcast | `?` (无) | 预计算常驻（可选） | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整常驻（Load + Cast + Broadcast 预计算）**。
- **形态 β — 仅 Load+Cast**：不做 Broadcast。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
__aicore__ inline void Process() {
    LocalTensor<float> weightFp32 = this->inQueueW.template AllocTensor<float>();
    DataCopyPad(weightLocal, weightGm, copyParams, padParams);
    Cast(localW0FP32, localW0, RoundMode::CAST_NONE, this->alignBaseH);
    Cast(localW1FP32, localW1, RoundMode::CAST_NONE, this->alignBaseH);
    for (int64_t bIdx = 0; bIdx < this->baseB; ++bIdx) {
        for (int64_t sIdx = 0; sIdx < this->baseS; ++sIdx) {
            Compute<DTYPE>(xLocalFp32, weightFp32, y0Fp32, y1Fp32, y2Fp32, ...);
        }
    }
    this->inQueueW.FreeTensor(weightFp32);
}
```

### 3C. Variant Notes

- weight buffer 常驻减少可用于数据 tile 的空间。
- 需额外 Cast（half→float），增加初始化延迟。
- Broadcast 预计算需额外 UB 空间（原始+Cast+Broadcast 三份）。

## Step 4: 约束复核

- UB 空间被 weight 占用
- Cast 初始化延迟
- Broadcast 仅适用于单维度

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "weightGm|weightLocal|weightFp32" modified_files/op_kernel/*.cpp  # >=1
grep -cE "Cast.*weight|weight.*Cast" modified_files/op_kernel/*.cpp  # >=1
grep -cE "DataCopyPad.*weight|weight.*DataCopy" modified_files/op_kernel/*.cpp  # >=1
grep -cE "for.*bIdx.*sIdx|Process.*loop" modified_files/op_kernel/*.cpp  # >=1
grep -cE "loop.*weight|每.*轮.*weight" modified_files/op_kernel/*.cpp  # ==0
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| UB 不足 | 减 tile 或取消 Broadcast |
| Cast 延迟大 | 容忍或预加载 |
| Broadcast 多维度 | 仅单维度适用 |
| weight 变化 | 不适用 |

---

**完成清单**：
```
[P34 Playbook Completion]
Step 1: done (/tmp/p34_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: UB 空间被 weight 占用; Cast 初始化延迟; Broadcast 仅适用于单维度: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
