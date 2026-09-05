# P69 Playbook: UB 融合连续 Vector 计算

> 本 Playbook 为**强制流程**。采纳 P69 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P69 的核心是**将多次连续的 Vector 计算（如 Exp→Abs→Mul 等）的中间结果留在 UB 上，不搬出 GM，仅在首次搬入和最终搬出时访问 GM，将 2n 次 GM 搬运减少到 2 次**。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p69_locations.txt`：

```bash
# 1. Vector 计算链
grep -n "Add\s*(\|Mul\s*(\|Sub\s*(\|Div\s*(\|Exp\s*(\|Abs\s*(\|Sqrt\s*(\|Max\s*(" \
    shared/original/op_kernel/*.cpp > /tmp/p69_locations.txt
# 2. GM 搬运
grep -n "DataCopy\|CopyIn\|CopyOut\|Fixpipe\|fixpipe\|GM.*UB\|UB.*GM" \
    shared/original/op_kernel/*.cpp >> /tmp/p69_locations.txt
# 3. 队列操作
grep -n "inQueue\|outQueue\|DeQue\|EnQue\|AllocTensor\|FreeTensor" \
    shared/original/op_kernel/*.cpp >> /tmp/p69_locations.txt
# 4. 当前 Vector 调用模式
grep -n "Compute_\|Process\|Exp.*Abs\|Abs.*Exp\|Add.*Mul\|Mul.*Add" \
    shared/original/op_kernel/*.cpp >> /tmp/p69_locations.txt
# 5. UB buffer 分配
grep -n "VECCALC\|VecCalc\|vecCalc\|UB.*中间\|中间.*UB\|tmpLocal\|tmpBuf" \
    shared/original/op_kernel/*.cpp >> /tmp/p69_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **Vector 计算链**：连续的 Vector 操作序列（如 Exp→Abs→Mul）
- **GM 搬运位置**：CopyIn/CopyOut/DataCopy 的调用位置
- **队列操作**：inQueue/outQueue/DeQue/EnQue 的使用模式
- **当前调用模式**：每次 Vector 后是否 CopyOut 再 CopyIn
- **UB buffer**：是否有 VECCALC 或临时 UB buffer

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| Vector 链长度 | `?` (2/3/4+ 次) | 不变 | `?_kernel.cpp:L?` |
| 当前 GM 搬运 | `?` (2n 次/轮) | 2 次 | `?_kernel.cpp:L?` |
| 中间 buffer | `?` (无/GM/UB) | UB (VECCALC) | `?_kernel.cpp:L?` |
| 队列模式 | `?` (单队列/多队列) | `alpha/beta` 见 3A | `?_kernel.cpp:L?` |
| 数据类型 | `?` (FP32/FP16/BF16) | 不变 | `?_kernel.cpp:L?` |
| buffer 大小 | `?` (元素数) | 不变 | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的队列模式和链长度，判断你的代码属于以下哪种形态：

- **形态 α — 单队列 UB 融合（最常见）**：使用单个 inQueue + outQueue，中间结果通过 VECCALC buffer 在 UB 上传递，最终一次性 EnQue。
- **形态 β — 多队列级联**：使用多个 inQueue/outQueue 做流水线，前一阶段的 outQueue 是下一阶段的 inQueue，无 GM 往返。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta`

### 3B. Canonical Template（形态 α — 单队列 UB 融合）

```cpp
// === 改造前（每次 Vector 后搬出 GM，2n 次搬运）===
__aicore__ inline void ProcessNaive() {
    // 第 1 次 Vector: Exp
    CopyIn();           // GM → UB
    Compute_Exp();      // UB 计算
    CopyOut();          // UB → GM ❌
    
    // 第 2 次 Vector: Abs
    CopyIn1();          // GM → UB ❌
    Compute_Abs();      // UB 计算
    CopyOut1();         // UB → GM ❌
}

// === 改造后（UB 融合，仅 2 次 GM 搬运）===
__aicore__ inline void ProcessOptimized() {
    // Step 1: 搬入
    LocalTensor<float> src0Local = inQueueSrc0.DeQue<float>();
    LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
    
    // Step 2: Vector 链在 UB 上连续计算
    Exp(dstLocal, src0Local, 1024);   // VECIN → VECCALC
    Abs(dstLocal, dstLocal, 1024);    // VECCALC → VECCALC（UB 内传递）
    // 若有更多 Vector 操作，继续链式调用
    // Mul(dstLocal, dstLocal, scalar, 1024);
    // Sqrt(dstLocal, dstLocal, 1024);
    
    // Step 3: 一次性搬出
    outQueueDst.EnQue<float>(dstLocal);
    inQueueSrc0.FreeTensor(src0Local);
}
```

### 3C. Variant Notes（若是形态 β）

- **形态 β（多队列级联）**：
  当链很长且需要流水线并行时：
  ```cpp
  // 队列 A → 队列 B → 队列 C，无 GM 往返
  LocalTensor<float> bufA = queueA.DeQue<float>();
  LocalTensor<float> bufB = queueB.AllocTensor<float>();
  Exp(bufB, bufA, count);
  queueB.EnQue<float>(bufB);
  queueA.FreeTensor(bufA);
  
  LocalTensor<float> bufC = queueC.AllocTensor<float>();
  Abs(bufC, bufB, count);
  queueC.EnQue<float>(bufC);
  queueB.FreeTensor(bufB);
  ```

- **与 P68 的协同**：P68（低延迟归约）优化 Reduce 指令，P69 优化 Vector 计算链的 GM 搬运。两者可同时存在：P68 优化归约，P69 优化 Vector 链。
- **与 P81 的边界**：P81（常驻 Buffer）处理参数常驻，P69 处理中间结果在 UB 内传递。两者可同时存在：P81 常驻参数，P69 融合 Vector 链。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: Vector 链的中间结果必须在同一 UB buffer 上（in-place 或独立 VECCALC buffer）
约束 2: 链长度受 UB 容量限制。若链过长导致 buffer 不足，需拆分为多个子链
约束 3: 每个 Vector 操作的输入输出类型必须匹配（FP32→FP32，不能混精度）
约束 4: DeQue/AllocTensor 必须在 EnQue/FreeTensor 之前，顺序不能错
约束 5: 最终 EnQue 的数据必须是完成全部 Vector 操作后的最终结果
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `Vector 链长度 = ?`, `GM 搬运次数 = ?`（改造前/后）
- `UB buffer 大小 = ? bytes`
- `数据类型 = ?`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 有连续的 Vector 调用（dst 是下一个的 src）
grep -cE "Exp\s*\([^,]+,\s*[^,]+,\s*[^)]+\)[^;]*;\s*Abs\s*\(|Add\s*\([^,]+,\s*[^,]+,\s*[^)]+\)[^;]*;\s*Mul\s*\(" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 中间结果不搬出 GM（无 CopyOut/Fixpipe 在 Vector 链中间）
grep -cE "Exp\s*\([^)]*\)[^;]*;\s*CopyOut|Abs\s*\([^)]*\)[^;]*;\s*CopyOut|DataCopy.*GM.*Exp" modified_files/op_kernel/*.cpp
# 期望: == 0（或 note 中说明 "最终输出步骤"）

# 检查 3: 有 DeQue/AllocTensor 和 EnQue/FreeTensor
grep -cE "DeQue|AllocTensor|EnQue|FreeTensor" modified_files/op_kernel/*.cpp
# 期望: >= 2

# 检查 4: 无循环内的 GM 往返（CopyOut + CopyIn 紧邻）
grep -cE "CopyOut[^;]*;\s*CopyIn|Fixpipe[^;]*;\s*DataCopy" modified_files/op_kernel/*.cpp
# 期望: == 0（或 note 中说明 "跨 stage 必要搬运"）

# 检查 5: 有 inQueue/outQueue 或队列操作
grep -cE "inQueue|outQueue|VECCALC|VecCalc" modified_files/op_kernel/*.cpp
# 期望: >= 1
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：dst 和 src 不能是同一张量 | 检查 Vector 操作是否支持 in-place。若不支持，需分配独立的中间 buffer |
| 运行时：UB 溢出 | Vector 链过长导致 buffer 不足。检查链长度，若 >4 步考虑拆分子链 |
| 运行时：数据未写完就被读取 | 缺少 PipeBarrier。在关键节点插入 `PipeBarrier<PIPE_V>()` 保证数据一致性 |
| 性能不如预期 | 确认中间结果确实在 UB 上传递，没有隐式搬出。检查 DataCopy 调用位置 |
| EnQue 前忘记 FreeTensor | 必须 `inQueueSrc0.FreeTensor(src0Local)` 在 EnQue 后，否则队列泄漏 |
| 多精度混用导致错误 | Vector 链中不能混用 FP32 和 FP16。若需要精度转换，在链的起点或终点做 Cast |
| 队列死锁 | DeQue 和 EnQue 必须配对。检查是否所有 AllocTensor 都有对应的 FreeTensor |
| 与 P81 的常驻 buffer 冲突 | P81 常驻参数占用 UB 空间，P69 的 Vector 链需要足够 UB。计算总占用，避免溢出 |
| CopyOut 在循环内 | 检查是否所有非最终输出的 CopyOut 都已移除。Vector 链中间不应有 GM 写 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P69 Playbook Completion]
Step 1: done (/tmp/p69_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta, canonical/variant applied
Step 4: constraints: chain_length=? gm_copies=? buffer_size=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
