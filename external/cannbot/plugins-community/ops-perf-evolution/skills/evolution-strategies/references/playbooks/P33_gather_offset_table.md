# P33 Playbook: Gather 偏移表驱动提取

> 本 Playbook 为**强制流程**。采纳 P33 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P33 的核心是**预计算字节偏移表常驻 UB，使用 Gather 指令一次性提取多个子张量，避免逐元素寻址**。

## Step 1: 定位关键结构

```bash
grep -n "Gather|gather|提取|子张量" \
    shared/original/op_kernel/*.cpp > /tmp/p33_locations.txt
grep -n "offset|Offset|偏移|table|Table" \
    shared/original/op_kernel/*.cpp >> /tmp/p33_locations.txt
grep -n "UB|ub|LocalTensor|TBuf|AllocTensor" \
    shared/original/op_kernel/*.cpp >> /tmp/p33_locations.txt
grep -n "SetValue|GetValue|preOffset|postOffset" \
    shared/original/op_kernel/*.cpp >> /tmp/p33_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前提取方式（逐元素/Gather**：文件 + 行号
- **偏移表存在性**：文件 + 行号
- **UB 容量**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 提取方式 | `?` (逐元素) | Gather 批量 | `op_kernel/*.cpp:L?` |
| 偏移表 | `?` (无) | 预计算常驻 UB | `op_kernel/*.cpp:L?` |
| 计算次数 | `?` (每次) | Init 一次 | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整偏移表 + Gather（预计算 + 批量提取）**。
- **形态 β — 仅预计算**：不改为 Gather，仅缓存 offset。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// Init 阶段预计算偏移表（只执行一次）
for (uint32_t i = 0; i < V1_BASE_T; i++) {
    for (uint32_t j = 0; j < N_; j++)
        preOffsetBuf_.SetValue(offset1++, curOffset * sizeof(P));
    curOffset += N_;
    for (uint32_t j = 0; j < N_; j++)
        postOffsetBuf_.SetValue(offset2++, curOffset * sizeof(P));
    curOffset += nSquare;
}

// 批量提取
Gather(hPreBuff_, matmulRes_, preOffsetBuf_, 0, lenT * N_);
Gather(hPostBuff_, matmulRes_, postOffsetBuf_, 0, lenT * N_);
```

### 3C. Variant Notes

- 偏移表占用 UB 空间（元素数 × 4 字节）。
- offset 必须是字节偏移，Gather 要求源数据在 UB 中连续。
- 子张量数量多时占用显著。

## Step 4: 约束复核

- 偏移表 UB 占用
- Gather 源数据连续性要求
- 子张量数量限制

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "Gather" modified_files/op_kernel/*.cpp  # >=1
grep -cE "preOffsetBuf_|postOffsetBuf_" modified_files/op_kernel/*.cpp  # >=1
grep -cE "SetValue.*offset|offset.*SetValue" modified_files/op_kernel/*.cpp  # >=1
grep -cE "curOffset.*sizeof|sizeof\(P\)" modified_files/op_kernel/*.cpp  # >=1
grep -cE "for.*i.*j.*GetValue|逐.*元素.*提取" modified_files/op_kernel/*.cpp  # ==0
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| UB 不足 | 减少子张量数或分页 |
| 源不连续 | 先 Copy 到连续 UB |
| offset 错误 | 字节对齐校验 |
| 少量子张量 | 逐元素可能更优 |

---

**完成清单**：
```
[P33 Playbook Completion]
Step 1: done (/tmp/p33_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 偏移表 UB 占用; Gather 源数据连续性要求; 子张量数量限制: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
