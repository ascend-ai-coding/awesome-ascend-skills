# P49 Playbook: 硬件加速反量化

> 本 Playbook 为**强制流程**。采纳 P49 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P49 的核心是**用 AscendDequant + Brcb 硬件指令替代软件循环反量化，将 O(M) 次 Vector 操作压缩到 O(1) 次**。

## Step 1: 定位关键结构

```bash
grep -n "Cast.*int32|Mul.*scale|Muls.*token|for.*subBlockM" \
    shared/original/op_kernel/*.cpp > /tmp/p49_locations.txt
grep -n "dequant|Dequant|scale|tokenScale" \
    shared/original/op_kernel/*.cpp >> /tmp/p49_locations.txt
grep -n "AscendDequant|Brcb|SetQuantVector" \
    shared/original/op_kernel/*.cpp >> /tmp/p49_locations.txt
grep -n "fp32Local|inLocal|tileSize" \
    shared/original/op_kernel/*.cpp >> /tmp/p49_locations.txt
grep -n "per.*channel|per.*token" \
    shared/original/op_kernel/*.cpp >> /tmp/p49_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **反量化位置**：文件 + 行号
- **当前循环结构**：文件 + 行号
- **scale 类型**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 反量化方式 | `?` (软件循环) | 硬件指令 | `?_kernel.cpp:L?` |
| scale 类型 | `?` (per-channel/per-token) | 不变 | `?_kernel.cpp:L?` |
| 循环次数 | `?` (O(M)) | O(1) | `?_kernel.cpp:L?` |
| 硬件支持 | `?` (910B+) | 确认 | `?_kernel.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — AscendDequant + Brcb（最常见）**：单条硬件指令完成。
- **形态 β — 仅 Brcb**：场景不支持 AscendDequant 时用 Brcb 广播替代循环。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// 改造前（软件循环，2M+1 次 Vector）
Cast(fp32Local, inLocal_, CAST_NONE, tileSize_);
for (int i = 0; i < subBlockM_; i++) {
    Mul(fp32Local[i*N], fp32Local[i*N], scale[i*N], N);
    Muls(fp32Local[i*N], fp32Local[i*N], tokenScale[i], N);
}

// 改造后（硬件指令，O(1)）
AscendDequant(dequantResult, mmOut, scale, tmpLocal,
              {curVecBaseM, alignBaseN, curVecBaseN});
Brcb(scaleLocal, perTokenScaleGm[offset], 1, 1, 0, 0);
Mul(output, dequantResult, scaleLocal, totalSize);
```

### 3C. Variant Notes

- 形态 β：若 AscendDequant 不可用，仅用 Brcb 替代 Muls 循环。
- 与 P48 协同：P48 分发量化路径，P49 优化反量化指令。

## Step 4: 约束复核

- 需 910B+ 硬件
- scale 编码格式 UINT64 需预处理
- SetQuantVector 仅 Cube 侧融合

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "AscendDequant|Brcb" modified_files/op_kernel/*.cpp  # >=1
grep -cE "for.*subBlockM|for.*scale" modified_files/op_kernel/*.cpp  # ==0
grep -cE "Cast.*int32.*fp32" modified_files/op_kernel/*.cpp  # ==0
grep -cE "scale|tokenScale" modified_files/op_kernel/*.cpp  # >=1
grep -cE "perTokenScale|perChannel" modified_files/op_kernel/*.cpp  # >=1
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 硬件不支持 | 回退软件循环 |
| scale 格式错误 | 预处理为 UINT64 |
| 仅 Cube 融合 | Vector 后处理保留 |

---

**完成清单**：
```
[P49 Playbook Completion]
Step 1: done (/tmp/p49_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 需 910B+ 硬件; scale 编码格式 UINT64 需预处理; SetQuantVector 仅 Cube 侧融合: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
