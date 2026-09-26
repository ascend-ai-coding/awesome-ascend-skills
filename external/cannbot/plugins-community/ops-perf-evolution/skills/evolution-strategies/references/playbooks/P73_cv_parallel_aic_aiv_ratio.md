# P73 Playbook: CV 并行 AIC:AIV 比例与多 Workspace 流水

> 本 Playbook 为**强制流程**。采纳 P73 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P73 的核心是**将 AIC:AIV 比例从 1:1 调整为 1:2，扩展 workspace 到 4 份消除 Cube-Vector 流水气泡，Vector 侧开启 double buffer**。

## Step 1: 定位关键结构

```bash
grep -n "Cube.*Vector|Vector.*Cube|CV|MIX|GroupedMatmul" \
    shared/original/op_kernel/*.cpp > /tmp/p73_locations.txt
grep -n "Vector|vector|postprocess|activation|后处理" \
    shared/original/op_kernel/*.cpp >> /tmp/p73_locations.txt
grep -n "aicAivRatio|workspace|double.*buffer|pingpong" \
    shared/original/op_kernel/*.cpp >> /tmp/p73_locations.txt
grep -n "GetSubBlockIdx|subBlockIdx|aivIdx|aivNum" \
    shared/original/op_kernel/*.cpp >> /tmp/p73_locations.txt
grep -n "SetMatmulConfig|MatmulConfigParams" \
    shared/original/op_kernel/*.cpp >> /tmp/p73_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **Cube+Vector 融合确认**：文件 + 行号
- **Vector 工作量**：文件 + 行号
- **当前 workspace 数**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| AIC:AIV | `?` (1:1) | 1:2 | `?_tiling.cpp:L?` |
| workspace | `?` (2) | 4 | `?_tiling.cpp:L?` |
| Vector DB | `?` (无) | double buffer | `?_kernel.cpp:L?` |
| 分工作 | `?` (无) | 按 aivIdx | `?_kernel.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 1:2 + 4 workspace + double buffer（完整版）**。
- **形态 β — 仅调比例**：不扩 workspace。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// Tiling 侧
optiling::MatmulConfigParams matmulConfigParams;
matmulConfigParams.aicAivRatio = {1, 2};  // AIC:AIV = 1:2
cubeTiling.SetMatmulConfigParams(matmulConfigParams);

// Kernel 侧
uint32_t aivIdx = GetSubBlockIdx();  // 0 或 1
uint32_t aivNum = GetSubBlockNum();  // 2
uint32_t vecStart = aivIdx * totalVecWork / aivNum;
uint32_t vecEnd = (aivIdx + 1) * totalVecWork / aivNum;
```

### 3C. Variant Notes

- 与 P51 协同：P51 调比例，P73 扩展 workspace + double buffer。
- 4 workspace 需 4 倍 GM 空间。

## Step 4: 约束复核

- 1:2 减少 AIC 核数，Cube 吞吐降低
- 4 workspace 需足够 GM 空间
- 仅 Cube+Vector 融合算子

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "aicAivRatio.*1.*2|1,\s*2" modified_files/op_kernel/*.cpp  # >=1
grep -cE "GetSubBlockIdx|subBlockIdx" modified_files/op_kernel/*.cpp  # >=1
grep -cE "workspace.*4|4.*workspace" modified_files/op_kernel/*.cpp  # >=1
grep -cE "double.*buffer|doubleBuffer" modified_files/op_kernel/*.cpp  # >=1
grep -cE "SetMatmulConfig|MatmulConfigParams" modified_files/op_kernel/*.cpp  # >=1
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| Cube 吞吐下降 | 确认 Vector 是瓶颈 |
| workspace 不足 | 分配 4 倍 GM |
| 非融合算子 | 不适用 |
| AIV 负载不均 | 按工作量均分 |

---

**完成清单**：
```
[P73 Playbook Completion]
Step 1: done (/tmp/p73_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 1:2 减少 AIC 核数，Cube 吞吐降低; 4 workspace 需足够 GM 空间; 仅 Cube+Vector 融合算子: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
