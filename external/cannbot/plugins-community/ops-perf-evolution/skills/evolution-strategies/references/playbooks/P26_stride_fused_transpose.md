# P26 Playbook: Stride 搬运模式（转置与列提取）

> 本 Playbook 为**强制流程**。采纳 P26 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P26 的核心是**通过 srcStride=0（连读）+ dstStride=跳步（跳写）实现零拷贝转置，避免额外 Transpose 算子**。

## Step 1: 定位关键结构

```bash
grep -n "DataCopy|CopyIn|CopyOut|Fixpipe" \
    shared/original/op_kernel/*.cpp > /tmp/p26_locations.txt
grep -n "Transpose|transpose|转置|Permute|permute" \
    shared/original/op_kernel/*.cpp >> /tmp/p26_locations.txt
grep -n "stride|Stride|srcStride|dstStride" \
    shared/original/op_kernel/*.cpp >> /tmp/p26_locations.txt
grep -n "DataCopyExtParams|DataCopyParams|blockCount|blockLen" \
    shared/original/op_kernel/*.cpp >> /tmp/p26_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前转置方式**：文件 + 行号
- **DataCopy stride 使用**：文件 + 行号
- **布局转换需求**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 转置方式 | `?` (独立算子) | DataCopy stride 融合 | `op_kernel/*.cpp:L?` |
| srcStride | `?` (默认) | 0（连读） | `op_kernel/*.cpp:L?` |
| dstStride | `?` (默认) | 跳写 | `op_kernel/*.cpp:L?` |
| 拷贝次数 | `?` (2次) | 1次 | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 零拷贝转置（连读+跳写）**。
- **形态 β — 列提取（blockCount+srcStride 跳过无关列）**。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
DataCopyExtParams dataCopyParams;
dataCopyParams.blockCount = gCountOneS1;
dataCopyParams.blockLen = headDim * sizeof(OUT_T);
dataCopyParams.srcStride = 0;                                     // 连读
dataCopyParams.dstStride = (tSize - 1) * headDim * sizeof(OUT_T); // 跳写
DataCopyPad(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);
```

### 3C. Variant Notes

- DataCopyParams stride 为 uint16_t 最大 65535，超限需 DataCopyExtParams。
- 仅适用于单维度转置，多维度需组合多次 stride 或 Gather。
- 列提取时 srcStride 和 blockLen 必须满足 32B 对齐。

## Step 4: 约束复核

- stride 上限 65535（uint16_t）
- 单维度限制
- 32B 对齐要求

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "srcStride|dstStride|DataCopyExtParams" modified_files/op_kernel/*.cpp  # >=1
grep -cE "blockCount|blockLen" modified_files/op_kernel/*.cpp  # >=1
grep -cE "srcStride.*=.*0|连读" modified_files/op_kernel/*.cpp  # >=1
grep -cE "Transpose|permute" modified_files/op_kernel/*.cpp  # ==0（或注释）
grep -cE "DataCopyPad.*attentionOutGm|DataCopy.*stride" modified_files/op_kernel/*.cpp  # >=1
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| stride 超 65535 | 换 DataCopyExtParams |
| 多维度转置 | 组合多次或 Gather |
| 未 32B 对齐 | 调整 blockLen |
| 效率低于连续 | 评估收益 |

---

**完成清单**：
```
[P26 Playbook Completion]
Step 1: done (/tmp/p26_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: stride 上限 65535（uint16_t）; 单维度限制; 32B 对齐要求: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
