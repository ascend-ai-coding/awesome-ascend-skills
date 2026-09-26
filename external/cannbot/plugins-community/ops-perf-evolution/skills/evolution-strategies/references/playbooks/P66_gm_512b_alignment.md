# P66 Playbook: GM 地址 512B 对齐优化带宽

> 本 Playbook 为**强制流程**。采纳 P66 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P66 的核心是**确保 GM 访问地址 512B 对齐，发挥最大带宽效率（32B 对齐最差仅为 512B 对齐的 70%）**。

## Step 1: 定位关键结构

```bash
grep -n "DataCopy|CopyIn|CopyOut|Fixpipe|GM|gm" \
    shared/original/op_kernel/*.cpp > /tmp/p66_locations.txt
grep -n "Align|对齐|offset|Offset|stride|Stride" \
    shared/original/op_kernel/*.cpp >> /tmp/p66_locations.txt
grep -n "512|256|32B|align.*512|AlignUp" \
    shared/original/op_kernel/*.cpp >> /tmp/p66_locations.txt
grep -n "workspace|Tiling|kernel.*入参|入参" \
    shared/original/op_kernel/*.cpp >> /tmp/p66_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前对齐粒度（32B/256B/512B**：文件 + 行号
- **GM 访问模式**：文件 + 行号
- **偏移量计算**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 对齐粒度 | `?` (32B/256B) | 512B | `op_kernel/*.cpp:L?` |
| 偏移计算 | `?` (直接) | AlignUp(rawOffset, 512) | `op_kernel/*.cpp:L?` |
| 带宽效率 | `?` (~70%) | ~100% | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整 512B 对齐（所有 GM 访问偏移 AlignUp）**。
- **形态 β — 仅关键路径**：仅对热点 DataCopy 做 512B 对齐。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// 确保偏移量保持 512B 对齐
uint32_t offset = AlignUp(rawOffset, 512);
DataCopy(ubTensor, gmTensor[offset], dataSize);

// 实测带宽对比（GM→UB）：
// 512B 对齐: 100% 带宽效率
// 256B 对齐: ~90% 带宽效率
// 32B 对齐:  ~70% 带宽效率（最差情况）
```

### 3C. Variant Notes

- 可能需要调整数据布局以保持 512B 对齐。
- 与 P7（32B 对齐）互补：P7 关注最低对齐要求，P66 关注最优带宽对齐。

## Step 4: 约束复核

- 数据布局调整代价
- 仅影响 GM 访问
- workspace/Tiling 入参已保证 512B 对齐

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "AlignUp|align.*512|512B" modified_files/op_kernel/*.cpp  # >=1
grep -cE "offset.*Align|Align.*offset" modified_files/op_kernel/*.cpp  # >=1
grep -cE "DataCopy.*gmTensor\[" modified_files/op_kernel/*.cpp  # >=1
grep -cE "32B|256B" modified_files/op_kernel/*.cpp  # ==0（或注释说明）
grep -cE "rawOffset|未对齐" modified_files/op_kernel/*.cpp  # ==0
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 布局调整代价大 | 仅改热点路径 |
| padding 浪费 | 评估总收益 |
| 已 512B 对齐 | 无需修改 |
| 非 GM 访问 | 不适用 |

---

**完成清单**：
```
[P66 Playbook Completion]
Step 1: done (/tmp/p66_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 数据布局调整代价; 仅影响 GM 访问; workspace/Tiling 入参已保证 512B 对齐: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
