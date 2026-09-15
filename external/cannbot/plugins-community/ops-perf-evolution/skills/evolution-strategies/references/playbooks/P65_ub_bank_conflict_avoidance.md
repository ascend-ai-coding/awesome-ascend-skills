# P65 Playbook: UB Bank 冲突规避

> 本 Playbook 为**强制流程**。采纳 P65 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P65 的核心是**通过优化 Vector 计算的 blk_stride 参数或地址分配，规避 UB bank group 冲突，使单 Repeat 从 8 拍降至 1-2 拍**。

## Step 1: 定位关键结构

```bash
grep -n "Vector|vector|VEC|Add|Mul|计算" \
    shared/original/op_kernel/*.cpp > /tmp/p65_locations.txt
grep -n "UB|ub|LocalTensor|TBuf|InitBuffer" \
    shared/original/op_kernel/*.cpp >> /tmp/p65_locations.txt
grep -n "blk_stride|block_stride|stride|mask|repeat" \
    shared/original/op_kernel/*.cpp >> /tmp/p65_locations.txt
grep -n "bank|Bank|冲突|conflict" \
    shared/original/op_kernel/*.cpp >> /tmp/p65_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前 Vector 计算模式**：文件 + 行号
- **blk_stride 值**：文件 + 行号
- **bank 冲突热点**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 访问模式 | `?` (跳读连续写) | 连续读跳写 | `op_kernel/*.cpp:L?` |
| blk_stride | `?` (16) | 1 | `op_kernel/*.cpp:L?` |
| bank 冲突 | `?` (有) | 无 | `op_kernel/*.cpp:L?` |
| padding | `?` (无) | 256B | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完全规避（连续读跳写 + padding）**。
- **形态 β — 仅调整 blk_stride**：不增加 padding。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// 反例：跳读连续写，blk_stride=16 导致 8 个 DataBlock 在同一 bank group
Adds(dst, src, scalar, MASK_PLACEHOLDER, 1, {1, 16, 1, 16});

// 正例：连续读跳写，读操作连续不冲突
Adds(dst, src, scalar, MASK_PLACEHOLDER, 1, {16, 1, 16, 1});
```

### 3C. Variant Notes

- UB（192KB）划分为 48 个 bank（16 个 bank group，每组 3 个 bank），每 bank 4KB。
- 同一 bank group 的并发读写会导致冲突。
- 地址优化方案需多申请 UB 空间（如 256 字节 padding）。

## Step 4: 约束复核

- padding 增加 UB 占用
- 计算逻辑调整复杂度
- 需理解 UB bank 结构

## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "{16, 1, 16, 1}|连续读" modified_files/op_kernel/*.cpp  # >=1
grep -cE "{1, 16, 1, 16}|跳读" modified_files/op_kernel/*.cpp  # ==0（或注释）
grep -cE "padding|PAD|256" modified_files/op_kernel/*.cpp  # >=0
grep -cE "Adds|Mul|Vector" modified_files/op_kernel/*.cpp  # >=1
grep -cE "bank|Bank|conflict" modified_files/op_kernel/*.cpp  # >=1
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| padding 不足 | 增加 256B |
| 逻辑复杂 | 封装为宏 |
| 其他 Vector API | 统一检查 blk_stride |
| 非 Vector 瓶颈 | 不适用 |

---

**完成清单**：
```
[P65 Playbook Completion]
Step 1: done (/tmp/p65_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: padding 增加 UB 占用; 计算逻辑调整复杂度; 需理解 UB bank 结构: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
