# P59 Playbook: 输出 Transpose 融合

> 本 Playbook 为**强制流程**。采纳 P59 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P59 的核心是**通过 DataCopy stride 参数实现输出格式转换（Transpose），避免额外转置算子调用**。

## Step 1: 定位关键结构

```bash
grep -n "DataCopy|CopyOut|Fixpipe" \
    shared/original/op_kernel/*.cpp > /tmp/p59_locations.txt
grep -n "Transpose|transpose|转置|Permute|permute" \
    shared/original/op_kernel/*.cpp >> /tmp/p59_locations.txt
grep -n "stride|Stride|srcStride|dstStride" \
    shared/original/op_kernel/*.cpp >> /tmp/p59_locations.txt
grep -n "BNSD|NBSD|BSND|BSH|TND|NTD|layout" \
    shared/original/op_kernel/*.cpp >> /tmp/p59_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前输出布局**：文件 + 行号
- **Transpose 调用位置**：文件 + 行号
- **DataCopy stride 使用**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 转置方式 | `?` (独立算子) | DataCopy stride 融合 | `op_kernel/*.cpp:L?` |
| stride 参数 | `?` (无/0) | 计算并设置 | `op_kernel/*.cpp:L?` |
| 布局 | `?` (BNSD) | NBSD 等 | `op_kernel/*.cpp:L?` |
| 写回次数 | `?` (2次) | 1次 | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整 stride 转置（BNSD→NBSD，含 offset 计算）**。
- **形态 β — 简单 Permute**：仅交换两个轴。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// BNSD → NBSD
DataCopyParams dataCopyParams;
dataCopyParams.blockCount = gCount;
dataCopyParams.blockLen = s1Size * headDim * sizeof(OUT_T) / 32U;
dataCopyParams.srcStride = 0;  // 连读
dataCopyParams.dstStride = (batchSize * qSeqSize - s1Size) * headDim * sizeof(OUT_T) / 32U;  // 跳写

uint64_t attenOutOffset = n2Idx * gSize * batchSize * qSeqSize * headDim +
                          gStartIdx * batchSize * qSeqSize * headDim +
                          bIdx * qSeqSize * headDim;
DataCopy(attentionOutGm[attenOutOffset], attenOutUb, dataCopyParams);
```

### 3C. Variant Notes

- DataCopy stride 效率低于连续搬运。
- 需精确计算各轴偏移量。
- 头块和尾块需单独处理。
- 支持 BNSD→NBSD、BSND→NBSD、BSH→NBSD、TND→NTD。

## Step 4: 约束复核

- stride 模式效率损失
- 偏移量计算复杂度
- 头尾块特殊处理

## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "srcStride|dstStride|DataCopyParams" modified_files/op_kernel/*.cpp  # >=1
grep -cE "blockLen.*headDim|blockCount.*gCount" modified_files/op_kernel/*.cpp  # >=1
grep -cE "attenOutOffset|offset.*n2Idx" modified_files/op_kernel/*.cpp  # >=1
grep -cE "Transpose|permute" modified_files/op_kernel/*.cpp  # ==0（消除独立算子）
grep -cE "DataCopy.*attentionOutGm|DataCopy.*output" modified_files/op_kernel/*.cpp  # >=1
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| stride 效率低 | 评估是否优于独立 Transpose |
| 偏移计算错误 | 逐轴验证 |
| 头尾块遗漏 | 单独处理 |
| 布局不匹配 | 确认输入输出格式 |

---

**完成清单**：
```
[P59 Playbook Completion]
Step 1: done (/tmp/p59_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: stride 模式效率损失; 偏移量计算复杂度; 头尾块特殊处理: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
