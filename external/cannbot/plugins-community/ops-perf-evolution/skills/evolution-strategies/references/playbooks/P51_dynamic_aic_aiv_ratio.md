# P51 Playbook: 动态 AIC/AIV 核配比

> 本 Playbook 为**强制流程**。采纳 P51 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P51 的核心是**根据 K 维大小动态选择 AIC:AIV 比例（1:1 vs 1:2），在 Cube 密集和 Vector 密集场景间自动切换**。

## Step 1: 定位关键结构

```bash
grep -n "KERNEL_TYPE_MIX|MIX|mix|AIC|AIV" \
    shared/original/op_kernel/*.cpp > /tmp/p51_locations.txt
grep -n "k|K|DOUBLE_VECTOR|threshold" \
    shared/original/op_kernel/*.cpp >> /tmp/p51_locations.txt
grep -n "kernelType|SetKernelType|aicAivRatio" \
    shared/original/op_kernel/*.cpp >> /tmp/p51_locations.txt
grep -n "TILING_KEY_IS|SetTilingKeyMode" \
    shared/original/op_kernel/*.cpp >> /tmp/p51_locations.txt
grep -n "GetBlockNum|coreNum|blockIdx" \
    shared/original/op_kernel/*.cpp >> /tmp/p51_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前 kernel type**：文件 + 行号
- **K 维大小**：文件 + 行号
- **分核逻辑**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 当前比例 | `?` (固定) | 动态 | `?_tiling.cpp:L?` |
| K 阈值 | `?` (无) | 2048 | `?_tiling.cpp:L?` |
| kernel type | `?` (MIX_AIC_1_2) | 条件切换 | `?_kernel.cpp:L?` |
| TilingKey | `?` (无) | 0/1 分发 | `?_tiling.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — K 阈值切换（最常见）**：K < 2048 用 1:2，K >= 2048 用 1:1。
- **形态 β — 多阈值分段**：更精细的分段策略。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
bool IsAivAicRatioTwoRequired() {
    return k < DOUBLE_VECTOR_THRESHOLD_K_UPPER;  // 2048
}

if (IsAivAicRatioTwoRequired()) {
    kernelType = KERNEL_TYPE_MIX_AIC_1_2;  // 小 K
} else {
    kernelType = KERNEL_TYPE_MIX_AIC_1_1;  // 大 K
}

if (TILING_KEY_IS(0)) { /* 1:2 */ }
if (TILING_KEY_IS(1)) { /* 1:1 */ }
```

### 3C. Variant Notes

- 与 P73 协同：P73 扩展为 1:2 + 4 workspace，P51 仅调比例。

## Step 4: 约束复核

- 阈值 2048 需按算子调优
- 动态切换影响系统调度
- Host 侧 tiling 逻辑增加

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "KERNEL_TYPE_MIX_AIC_1_2|KERNEL_TYPE_MIX_AIC_1_1" modified_files/op_kernel/*.cpp  # >=1
grep -cE "DOUBLE_VECTOR|threshold.*K" modified_files/op_kernel/*.cpp  # >=1
grep -cE "IsAivAicRatio|aicAivRatio" modified_files/op_kernel/*.cpp  # >=1
grep -cE "TILING_KEY_IS.*0|TILING_KEY_IS.*1" modified_files/op_kernel/*.cpp  # >=1
grep -cE "kernelType.*=.*MIX" modified_files/op_kernel/*.cpp  # ==0（或条件赋值）
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 阈值不准 | 按算子 profiling 调优 |
| 系统调度影响 | 测试多算子共存场景 |
| 1:1 时 Vector 仍瓶颈 | 检查 K 是否足够大 |

---

**完成清单**：
```
[P51 Playbook Completion]
Step 1: done (/tmp/p51_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 阈值 2048 需按算子调优; 动态切换影响系统调度; Host 侧 tiling 逻辑增加: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
