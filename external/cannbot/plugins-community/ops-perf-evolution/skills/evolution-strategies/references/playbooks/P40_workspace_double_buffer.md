# P40 Playbook: Workspace 双缓冲中间结果常驻

> 本 Playbook 为**强制流程**。采纳 P40 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P40 的核心是**Cube 和 Vector 通过 workspace GM 传递中间结果，每个 core 分配 2 份 workspace，通过 loop % 2 索引实现双缓冲，Cube 写入和 Vector 读取完全并行**。

## Step 1: 定位关键结构

```bash
grep -n "workspace|Workspace|ws|中间结果|intermediate" \
    shared/original/op_kernel/*.cpp > /tmp/p40_locations.txt
grep -n "Cube.*Vector|Vector.*Cube|CV|MIX" \
    shared/original/op_kernel/*.cpp >> /tmp/p40_locations.txt
grep -n "loop.*%.*2|dbWorkspace|preLoadNum|pingpong" \
    shared/original/op_kernel/*.cpp >> /tmp/p40_locations.txt
grep -n "SetGlobalBuffer|mmResGm|vec2ResGm" \
    shared/original/op_kernel/*.cpp >> /tmp/p40_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前 workspace 数量**：文件 + 行号
- **CV 融合确认**：文件 + 行号
- **中间结果传递方式**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| workspace 数 | `?` (1) | 2 | `op_kernel/*.cpp:L?` |
| 索引方式 | `?` (固定) | loop % 2 | `op_kernel/*.cpp:L?` |
| 并行度 | `?` (串行) | Cube 写 + Vector 读并行 | `op_kernel/*.cpp:L?` |
| 启动延迟 | `?` (无) | preLoadNum 控制 | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整双缓冲（loop % 2 + preLoadNum + 并行读写）**。
- **形态 β — 仅扩 workspace**：不做双缓冲索引。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
mm1ResGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + offset +
    aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T)));
uint64_t srcGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize;
DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);
uint64_t vec2ResGmOffset = ((info.loop - 1) % constInfo.preLoadNum) * constInfo.bmm2ResUbSize;
DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset], vec2ComputeSize);
```

### 3C. Variant Notes

- workspace 总量 = 核数 × 2 × 多个 buffer size，HBM 占用较大。
- 首次迭代 Vector 无数据可读，需 preLoadNum 控制流水启动延迟。

## Step 4: 约束复核

- HBM 占用 = 核数 × 2 × buffer_size
- 首次迭代启动延迟
- 仅 CV 融合算子

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "loop.*%.*2|%.*preLoadNum" modified_files/op_kernel/*.cpp  # >=1
grep -cE "dbWorkspaceRatio|workspace.*offset" modified_files/op_kernel/*.cpp  # >=1
grep -cE "mmResGm|vec2ResGm|bmm2Res" modified_files/op_kernel/*.cpp  # >=1
grep -cE "SetGlobalBuffer|workspace" modified_files/op_kernel/*.cpp  # >=1
grep -cE "loop.*workspace|单.*workspace" modified_files/op_kernel/*.cpp  # ==0
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| HBM 不足 | 减 buffer size 或取消 |
| 首次迭代空读 | preLoadNum 延迟启动 |
| 索引越界 | 确认 % 2 正确 |
| 非 CV 融合 | 不适用 |

---

**完成清单**：
```
[P40 Playbook Completion]
Step 1: done (/tmp/p40_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: HBM 占用 = 核数 × 2 × buffer_size; 首次迭代启动延迟; 仅 CV 融合算子: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
