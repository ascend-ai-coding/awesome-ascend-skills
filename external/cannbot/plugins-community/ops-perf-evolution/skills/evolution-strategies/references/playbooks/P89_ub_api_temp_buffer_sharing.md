# P89 Playbook: UB 高阶 API 临时 Buffer 共享复用

> 本 Playbook 为**强制流程**。采纳 P89 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P89 的核心是**识别算子中使用高阶 API（如 SoftMax）产生的临时 Buffer 需求，将其与其他计算阶段的临时 Buffer 共享同一块 TBuf<VECCALC> 空间**。目标是减少 UB 总占用和搬运次数。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p89_locations.txt`：

```bash
# 1. 高阶 API 调用（SoftMax、LayerNorm、Activation 等可能有临时 buffer 的 API）
grep -n "SoftMax\|softmax\|Softmax\|LayerNorm\|BatchNorm\|Activation\|HighLevelAPI" shared/original/op_kernel/*.cpp > /tmp/p89_locations.txt
# 2. 所有 TBuf 分配（临时 buffer 空间）
grep -n "TBuf\|InitBuffer.*VECCALC\|InitBuffer.*TBuf" shared/original/op_kernel/*.cpp >> /tmp/p89_locations.txt
# 3. 普通计算阶段的 buffer 分配（可能可共享的）
grep -n "InitBuffer" shared/original/op_kernel/*.cpp >> /tmp/p89_locations.txt
# 4. PipeBarrier / 阶段边界（用于判断生命周期分界点）
grep -n "PipeBarrier\|SyncAll\|SetFlag\|WaitFlag" shared/original/op_kernel/*.cpp >> /tmp/p89_locations.txt
```

**交付物**（记录到 `implementation_note.txt` "Playbook Step 1"）：
- **高阶 API 列表**：所有需要临时 buffer 的 API 调用位置
- **TBuf 分配列表**：每个 InitBuffer 的名称、大小、用途
- **阶段边界**：PipeBarrier/SyncAll 位置（生命周期分界点）

## Step 2: 改造计划表（强制填写）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| API 临时 buffer 大小 | `?` bytes | — | `?_kernel.cpp:L?` |
| 计算阶段临时 buffer 大小 | `?` bytes | — | `?_kernel.cpp:L?` |
| 共享 buffer 大小 | — | `max(apiBufSize, calcBufSize)` | `?_kernel.cpp:L?` |
| 当前总搬运次数 | `?` | `? * 0.5`（预期减半） | — |
| 阶段间同步方式 | `?` | 不变（确保生命周期隔离） | — |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — 单 API + 单计算阶段**：一个 SoftMax + 一个 Add/Mul 等。共享方案最直接，取 `max(softmaxTempSize, calcTempSize)`。
- **形态 β — 多 API 或多计算阶段**：多个高阶 API 或多次计算。需绘制生命周期时间线，确认任意两个共享 buffer 的阶段时间不重叠。
- **形态 γ — 无法共享**：各阶段临时 buffer 生命周期重叠（流水并行）。放弃 P89，检查 P85（on-chip buffer zone reuse）的跨阶段 workspace 复用方案。

### 3B. Canonical Template（形态 α）

```cpp
// === 改造前 ===
pipe.InitBuffer(softmaxBuf, 1, softmaxTempSize);  // SoftMax 临时空间
pipe.InitBuffer(addBuf, 1, addTempSize);           // Add 临时空间

Softmax(softmaxBuf, input, softmaxParams);
PipeBarrier<PIPE_V>();
Add(addBuf, a, b);

// === 改造后 ===
uint32_t sharedSize = std::max(softmaxTempSize, addTempSize);
pipe.InitBuffer(sharedBuf, 1, sharedSize);

Softmax(sharedBuf, input, softmaxParams);   // 阶段 1: SoftMax 使用
PipeBarrier<PIPE_V>();                       // 生命周期边界
Add(sharedBuf, a, b);                        // 阶段 2: Add 复用
```

## Step 4: 生命周期复核

确认两个阶段之间**有 PipeBarrier 或 SyncAll 保证串行**。检查 sharedBuf 不会被两个阶段同时访问。在 `implementation_note.txt` "Playbook Step 4" 中画出时间线：

```
[SoftMax 使用 sharedBuf] ── PipeBarrier ── [Add 使用 sharedBuf]
         ↑ 独占                          ↑ 独占
```

## Step 5: 编码后自检

```bash
# 检查 1: 共享 buffer 只被 InitBuffer 一次（不再独立分配两个 buffer）
grep -c "InitBuffer.*sharedBuf\|InitBuffer.*shared" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: std::max 或条件判断取最大 buffer 大小
grep -c "std::max\|MAX(" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: 高阶 API 调用存在
grep -c "SoftMax\|softmax\|LayerNorm" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: 阶段之间有同步边界
grep -c "PipeBarrier\|SyncAll" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: 原来的独立 buffer 已被移除
grep -c "softmaxBuf\|addBuf\|tempBuf" modified_files/op_kernel/*.cpp
# 期望: == 0（旧的独立 buffer 名不应再出现）
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 编译失败：UB overflow | sharedSize 计算错误，检查 `std::max` 参数是否正确 |
| 运行时数据错乱 | 两个阶段之间缺少 PipeBarrier 或 SyncAll，sharedBuf 被同时读写 |
| 性能无提升 | 搬运次数未减少，检查是否还有中间变量复用了 sharedBuf |
| 精度退化 | TBuf dtype 不兼容（API 需要 fp32 但 sharedBuf 分配为 fp16），确保 dtype 一致 |
| API 报临时空间不足 | API 文档中的临时空间需求 > sharedSize，需查 API 文档确认实际需求 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P89 Playbook Completion]
Step 1: done (/tmp/p89_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: lifecycle verified, no overlap
Step 5: all 5 grep checks passed
Step 6: no pitfalls triggered / {列出触发的}
```
