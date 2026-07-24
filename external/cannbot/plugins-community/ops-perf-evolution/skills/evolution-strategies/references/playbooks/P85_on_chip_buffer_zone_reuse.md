# P85 Playbook: On-Chip Buffer Zone Reuse

> 本 Playbook 为**强制流程**。采纳 P85 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P85 的核心是**将多个串行计算阶段的临时 buffer 映射到同一块 UB TBuf 或 GM workspace 的不同逻辑区域，从 sum(stage buffers) 压缩到 max(concurrent buffers)**。

## Step 1: 定位关键结构

```bash
grep -n "InitBuffer|TBuf|workspace|wsLocal|zone" \
    shared/original/op_kernel/*.cpp > /tmp/p85_locations.txt
grep -n "stage|Stage|阶段|phase|Phase|串行|serial" \
    shared/original/op_kernel/*.cpp >> /tmp/p85_locations.txt
grep -n "RmsNorm|RoPE|SoftMax|后处理|postprocess" \
    shared/original/op_kernel/*.cpp >> /tmp/p85_locations.txt
grep -n "offset|Offset|alias|Alias" \
    shared/original/op_kernel/*.cpp >> /tmp/p85_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前 buffer 分配方式（独立/共享**：文件 + 行号
- **串行阶段数**：文件 + 行号
- **各阶段 buffer 大小**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 分配 | `?` (独立) | 分区共享 | `op_kernel/*.cpp:L?` |
| 空间 | `?` (sum) | max(concurrent) | `op_kernel/*.cpp:L?` |
| 管理 | `?` (独立) | 手工 offset / alias | `op_kernel/*.cpp:L?` |
| 阶段边界 | `?` (模糊) | 严格串行 | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整 zone reuse（wsLocal 分区 + 严格串行边界）**。
- **形态 β — 仅两阶段复用**：简化版，只合并两个阶段的 buffer。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// 改造前：独立分配
pipe.InitBuffer(rmsBuf0, 1, zoneSize);
pipe.InitBuffer(rmsBuf1, 1, zoneSize);
pipe.InitBuffer(rmsBuf2, 1, zoneSize);
pipe.InitBuffer(ropeCosBuf, 1, zoneSize);
pipe.InitBuffer(ropeSinBuf, 1, zoneSize);

// 改造后：分区复用
int64_t xLocalFp32Offset = 0;                     // zone0: RmsNorm fp32
int64_t xSquareLocalOffset = rows * headSize;    // zone1: RmsNorm square / RoPE cos
int64_t xSumLocalOffset = rows * headSize * 2;   // zone2: RmsNorm sum / RoPE sin
LocalTensor<float> xLocalFp32 = wsLocal[xLocalFp32Offset];
LocalTensor<float> xSquareLocal = wsLocal[xSquareLocalOffset];
LocalTensor<float> xSumLocal = wsLocal[xSumLocalOffset];
```

### 3C. Variant Notes

- 各阶段必须具备严格的串行边界，错误判断生命周期会导致后续阶段覆盖仍在使用的数据。
- 共享区通常依赖手工 offset、MAX 大小估算或 alias 规划，维护成本高于独立分配。
- 当后续优化把串行阶段改成重叠流水时，原有 zone reuse 方案可能失效。

## Step 4: 约束复核

- 严格串行边界要求
- 手工 offset 维护成本
- 流水化改造时可能失效

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "wsLocal|workspace.*offset|zone.*Offset" modified_files/op_kernel/*.cpp  # >=1
grep -cE "xLocalFp32Offset|xSquareLocalOffset|xSumLocalOffset" modified_files/op_kernel/*.cpp  # >=1
grep -cE "LocalTensor.*wsLocal|wsLocal\[" modified_files/op_kernel/*.cpp  # >=1
grep -cE "InitBuffer.*rmsBuf|InitBuffer.*rope" modified_files/op_kernel/*.cpp  # ==0（或注释）
grep -cE "rows.*headSize|zoneSize" modified_files/op_kernel/*.cpp  # >=1
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 生命周期误判 | 严格阶段边界验证 |
| offset 冲突 | 统一规划表 |
| 流水化失效 | 重新评估 zone 分配 |
| 维护成本高 | 封装 zone 管理器 |

---

**完成清单**：
```
[P85 Playbook Completion]
Step 1: done (/tmp/p85_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: 严格串行边界要求; 手工 offset 维护成本; 流水化改造时可能失效: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
