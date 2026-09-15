# P88 Playbook: Multi-Phase Compute Phase Decomposition

> 本 Playbook 为**强制流程**。采纳 P88 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P88 的核心是**将算子内多个串行阶段拆分，在阶段边界通过 SyncAll/pipe->Reset 重新分配 UB/L1/Workspace 资源，从 sum(stage buffers) 压缩到 max(concurrent buffers)**。

## Step 1: 定位关键结构

```bash
grep -n "InitBuffer|TBuf|TPipe|pipe|workspace" \
    shared/original/op_kernel/*.cpp > /tmp/p88_locations.txt
grep -n "stage|Stage|阶段|phase|Phase|preprocess|postprocess" \
    shared/original/op_kernel/*.cpp >> /tmp/p88_locations.txt
grep -n "SyncAll|PipeBarrier|Reset|pipe-.*Reset" \
    shared/original/op_kernel/*.cpp >> /tmp/p88_locations.txt
grep -n "MainLoop|Process|Compute" \
    shared/original/op_kernel/*.cpp >> /tmp/p88_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前阶段结构**：文件 + 行号
- **buffer 分配方式**：文件 + 行号
- **是否存在资源竞争**：文件 + 行号
## Step 2: 改造计划表

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 阶段 | `?` (单一) | 多阶段拆分 | `op_kernel/*.cpp:L?` |
| 资源 | `?` (全局分配) | 阶段独立分配 | `op_kernel/*.cpp:L?` |
| 边界 | `?` (无) | SyncAll + pipe->Reset | `op_kernel/*.cpp:L?` |
| 空间 | `?` (sum) | max(concurrent) | `op_kernel/*.cpp:L?` |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 完整阶段拆分（独立 TPipe + SyncAll + Reset + 各阶段最大 buffer）**。
- **形态 β — 仅 Reset**：不做独立 TPipe，只复用 pipe->Reset。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`
### 3B. Canonical Template

```cpp
// 改造前：单一阶段
void Process() {
    InitAllBuffers(pipe);
    MainLoop();
    PostProcess();
}

// 改造后：多阶段拆分
void Process() {
    InitBuffersPhase1(pipe);
    MainLoop();

    SyncAll();
    pipe->Reset();

    vector2Service.InitBuffers(pipe);
    vector2Service.Process();
}
```

### 3C. Variant Notes

- 阶段边界的 SyncAll、独立 TPipe 初始化/销毁或 pipe->Reset 会引入额外固定开销。
- 多阶段之间通常无法像单一深流水那样完全重叠。
- 若阶段拆分粒度过细，workspace 读写、控制流和初始化成本会变成新瓶颈。

## Step 4: 约束复核

- SyncAll/Reset 额外开销
- 阶段切换无法重叠
- 粒度过细的成本

**在 `implementation_note.txt` "Playbook Step 4" 中报告具体数值**（实际数值 + 是否通过）。
## Step 5: 编码后自检
**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。


```bash
grep -cE "SyncAll.*Reset|pipe-.*Reset|pipe->Reset" modified_files/op_kernel/*.cpp  # >=1
grep -cE "InitBuffersPhase|vector2Service" modified_files/op_kernel/*.cpp  # >=1
grep -cE "MainLoop|PostProcess" modified_files/op_kernel/*.cpp  # >=1
grep -cE "stage|phase|阶段" modified_files/op_kernel/*.cpp  # >=1
grep -cE "InitAllBuffers|单一.*分配" modified_files/op_kernel/*.cpp  # ==0（或注释）
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 开销抵消收益 | 评估阶段数 |
| 重叠损失 | 限制阶段数 |
| 粒度过细 | 合并相邻阶段 |
| 资源竞争误判 | 严格生命周期分析 |

---

**完成清单**：
```
[P88 Playbook Completion]
Step 1: done (/tmp/p88_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: SyncAll/Reset 额外开销; 阶段切换无法重叠; 粒度过细的成本: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
