# P38 Playbook: Softmax 状态 Buffer 跨 S2 循环常驻

> 本 Playbook 为**强制流程**。采纳 P38 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P38 的核心是**将 online softmax 的三个状态 buffer（softmaxMax / softmaxSum / softmaxExp）从循环内动态分配改为 Init 阶段一次性常驻分配**，通过 `loop % preLoadNum` 索引实现跨循环复用，避免每次 S2 循环的 InitBuffer overhead。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p38_locations.txt`：

```bash
# 1. Online softmax / Flash Attention 相关代码
grep -n "SoftmaxFlash|online.*softmax|softmaxMax|softmaxSum|softmaxExp|FlashAttention" \
    shared/original/op_kernel/*.cpp > /tmp/p38_locations.txt
# 2. S2 循环或外层循环
grep -n "for.*s2|for.*seq|for.*tile.*outer|loop.*s2|loopCount" \
    shared/original/op_kernel/*.cpp >> /tmp/p38_locations.txt
# 3. 当前循环内的 buffer 分配（重点改造对象）
grep -n "InitBuffer.*softmax|InitBuffer.*max|InitBuffer.*sum|InitBuffer.*exp" \
    shared/original/op_kernel/*.cpp >> /tmp/p38_locations.txt
# 4. UB buffer 现状
grep -n "InitBuffer|TBuf|TQue|BUFFER_NUM|ubFactor|tileSize" \
    shared/original/op_kernel/*.cpp >> /tmp/p38_locations.txt
# 5. 已有的 preLoadNum 或 resident 模式
grep -n "preLoadNum|resident|stateBuff|loop.*%" \
    shared/original/op_kernel/*.cpp >> /tmp/p38_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **Softmax 位置**：`SoftmaxFlash` / `online_softmax` 调用位置
- **循环结构**：S2 循环或外层循环的起止、迭代次数变量
- **Buffer 分配**：当前 `InitBuffer` 是否在循环内部（重点改造）
- **UB 现状**：已有 buffer 数量、总占用、剩余空间估算
- **已有模式**：是否已有 `preLoadNum` 或 `%` 索引模式

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 状态变量 | `?` (max/sum/exp 变量名) | 常驻 buffer | `?_kernel.cpp:L?` |
| 循环内 InitBuffer | `?` (有 / 无) | 无（移至 Init） | `?_kernel.cpp:L?` |
| preLoadNum | `?` (当前 =1 或无) | 1 或 2 | `?_kernel.cpp:L?` |
| 状态 buffer 大小 | `? bytes × 3` | `SOFTMAX_TMP × 3 × preLoadNum` | `?_kernel.cpp:L?` |
| UB 剩余空间 | `? bytes` | ≥ 状态 buffer 总和 | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的循环结构和 UB 空间，判断你的代码属于以下哪种形态：

- **形态 α — 单缓冲常驻（preLoadNum=1，无流水重叠）**：UB 空间紧张，只能容纳一套状态 buffer。状态在循环间原地复用，无重叠。
- **形态 β — 双缓冲常驻（preLoadNum=2，支持 Copy/Compute 重叠）**：UB 空间充裕，分配两套状态 buffer。`loop % 2` 索引让下一轮 Compute 与本轮 CopyOut 重叠。最常见。
- **形态 γ — GM spill 常驻（超长序列，UB 放不下）**：序列极长导致 `s2Loops` 很大，但状态 buffer 本身不大，形态 β 通常已足够。形态 γ 仅在状态量本身极大（如 grouped attention）时考虑，将状态 spill 到 GM。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 β — 双缓冲常驻，最常见）

```cpp
// === 改造前（每次 S2 循环都 InitBuffer，严重 overhead）===
__aicore__ inline void S2Loop(uint32_t s2Loops) {
    for (uint32_t s2 = 0; s2 < s2Loops; s2++) {
        // ❌ 反模式：循环内动态分配
        TBuf<QuePosition::VECCALC> maxBuf;
        TBuf<QuePosition::VECCALC> sumBuf;
        TBuf<QuePosition::VECCALC> expBuf;
        pipe.InitBuffer(maxBuf, SOFTMAX_TMP_BUFFER_SIZE);
        pipe.InitBuffer(sumBuf, SOFTMAX_TMP_BUFFER_SIZE);
        pipe.InitBuffer(expBuf, SOFTMAX_TMP_BUFFER_SIZE);
        
        LocalTensor<float> maxLocal = maxBuf.Get<float>();
        LocalTensor<float> sumLocal = sumBuf.Get<float>();
        LocalTensor<float> expLocal = expBuf.Get<float>();
        
        // Compute matmul result for this tile
        LocalTensor<float> mmRes = mmBuf.Get<float>();
        Mmad(mmRes, ...);
        
        // Online softmax update
        SoftmaxFlashV2(mmRes, sumLocal, maxLocal, mmRes, expLocal,
                       inSumTensor, inMaxTensor, ...);
    }
}

// === 改造后（Init 阶段一次性分配，loop % 2 索引）===
// 在 Kernel Init 中一次性分配（不在循环内）
static constexpr uint32_t PRELOAD_NUM = 2;
static constexpr uint32_t STATE_BUF_SIZE = SOFTMAX_TMP_BUFFER_SIZE * PRELOAD_NUM;

TBuf<QuePosition::VECCALC> softmaxMaxBuf;
TBuf<QuePosition::VECCALC> softmaxSumBuf;
TBuf<QuePosition::VECCALC> softmaxExpBuf;

__aicore__ inline void Init() {
    pipe.InitBuffer(softmaxMaxBuf, STATE_BUF_SIZE);
    pipe.InitBuffer(softmaxSumBuf, STATE_BUF_SIZE);
    pipe.InitBuffer(softmaxExpBuf, STATE_BUF_SIZE);
    
    // 可选：初始化第一套状态（max = -inf, sum = 0）
    // InitSoftmaxState(softmaxMaxBuf, softmaxSumBuf, ...);
}

__aicore__ inline void S2Loop(uint32_t s2Loops) {
    for (uint32_t s2 = 0; s2 < s2Loops; s2++) {
        uint32_t bufIdx = s2 % PRELOAD_NUM;
        uint32_t offset = bufIdx * (SOFTMAX_TMP_BUFFER_SIZE / sizeof(float));
        
        // 通过 offset 索引到对应双缓冲槽位
        LocalTensor<float> maxLocal = softmaxMaxBuf.Get<float>()[offset];
        LocalTensor<float> sumLocal = softmaxSumBuf.Get<float>()[offset];
        LocalTensor<float> expLocal = softmaxExpBuf.Get<float>()[offset];
        
        // Compute matmul result
        LocalTensor<float> mmRes = mmBuf.Get<float>();
        Mmad(mmRes, ...);
        
        // Online softmax 更新常驻状态
        SoftmaxFlashV2(mmRes, sumLocal, maxLocal, mmRes, expLocal,
                       inSumTensor, inMaxTensor, ...);
    }
}
```

### 3C. Variant Notes（若是形态 α 或 γ）

- **形态 α（单缓冲常驻，preLoadNum=1）**：
  当 UB 空间紧张（如同时有 L0A/L0B/L0C buffer），无法承受 `×2` 开销：
  ```cpp
  static constexpr uint32_t PRELOAD_NUM = 1;
  // 无 % 索引，直接 Get<>() 不加 offset
  LocalTensor<float> maxLocal = softmaxMaxBuf.Get<float>();
  ```
  形态 α 无流水重叠收益，仅节省 `InitBuffer` 调用开销。若 UB 连 `×1` 的 3 个状态 buffer 都放不下，考虑 P8 UB 分区或 P12 mask 优化腾空间。

- **形态 γ（GM spill，极少见）**：
  仅在 grouped attention / multi-head 场景下，状态量 `numHeads × seqLen × sizeof(float)` 超过 UB 容量时才需要：
  ```cpp
  // 将状态 spill 到 GM workspace
  DataCopy(softmaxMaxGm + headIdx * seqLen, maxLocal, seqLen);
  // 下一轮从 GM reload
  DataCopy(maxLocal, softmaxMaxGm + headIdx * seqLen, seqLen);
  ```
  形态 γ 引入额外 GM 访存，通常性能不如形态 β。仅在 UB 绝对不足时考虑。

- **与 P14 的协同**：P14 是 Flash Attention 的 tiling 策略，P38 是其中的 softmax 状态优化。两者通常一起出现在 FA kernel 中。若算子已有 P14 的 tiling 结构，P38 的改造只需在已有 tiling 框架内添加常驻 buffer。

- **与 P1 的边界**：P1 双缓冲用于 `CopyIn/Compute/CopyOut` 的 input/output tensor，P38 双缓冲用于 `softmaxMax/Sum/Exp` 状态。两者独立：
  - P1 的 BUFFER_NUM 控制 input/output 流水深度
  - P38 的 PRELOAD_NUM 控制 softmax 状态流水深度
  - 若两者都用 `preLoadNum=2`，总 UB = P1_buffers + P38_buffers，需复核容量。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: 3 × SOFTMAX_TMP_BUFFER_SIZE × preLoadNum ≤ UB 可用空间（需扣除其他 buffer 后剩余）
约束 2: preLoadNum 必须是编译期常量（1 或 2），不建议 >2（UB 开销过大）
约束 3: offset 必须按元素对齐：`bufIdx × (SOFTMAX_TMP_BUFFER_SIZE / sizeof(T))`
约束 4: 第一轮循环的状态必须正确初始化（max = -inf, sum = 0, exp = 0）
约束 5: 若 preLoadNum=2，最后循环结束后需 Wait 所有未完成操作（CopyOut 可能滞后）
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `SOFTMAX_TMP_BUFFER_SIZE = ? bytes`
- `preLoadNum = ?`
- `状态总占用 = 3 × ? × ? = ? bytes`
- `UB 其他 buffer = ? bytes`
- `UB 剩余 = ? bytes`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: InitBuffer 在 Init/构造函数中，不在 S2 循环内
grep -cE "for.*s2.*\{[^}]*InitBuffer|for.*loop.*\{[^}]*InitBuffer" modified_files/op_kernel/*.cpp
# 期望: == 0

# 检查 2: 有 preLoadNum 或 % 索引模式
grep -cE "preLoadNum|loop.*%|s2.*%|bufIdx" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: softmaxMaxBuf / softmaxSumBuf / softmaxExpBuf 声明在循环外
grep -cE "softmaxMaxBuf|softmaxSumBuf|softmaxExpBuf|maxBuf|sumBuf|expBuf" modified_files/op_kernel/*.cpp
# 期望: >= 3

# 检查 4: 有 offset 计算（不是直接用 Get<>()）
grep -cE "offset.*SOFTMAX_TMP|Get.*\[.*offset\]|\[.*offset.*\]" modified_files/op_kernel/*.cpp
# 期望: >= 1（形态 alpha 除外，preLoadNum=1 时无 offset，需在 note 中说明）

# 检查 5: SoftmaxFlashV2 或等效调用使用状态 local tensor
grep -cE "SoftmaxFlash|online.*softmax|softmaxMax.*Local|softmaxSum.*Local" modified_files/op_kernel/*.cpp
# 期望: >= 1
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：TBuf 类型不匹配 | `softmaxMaxBuf.Get<float>()[offset]` 的 `offset` 单位是元素个数，不是字节。确认 `offset = bufIdx × (SOFTMAX_TMP_BUFFER_SIZE / sizeof(float))` |
| 运行时：softmax 结果越界 | `SOFTMAX_TMP_BUFFER_SIZE` 必须 ≥ 实际需要。online softmax 的状态量通常是 `tileSize` 或 `seqLen`，不是固定 2KB |
| 运行时：精度对不上 | 第一轮的初始状态必须正确。`max` 初始化为 `-FLT_MAX` 或 `-1e30`，`sum` 初始化为 `0`。不要遗漏初始化 |
| UB 越界 | 复核约束 1。若 `3 × SOFTMAX_TMP × 2` 超过 UB 剩余空间，退化为形态 α（preLoadNum=1）或考虑 P8 UB 分区 |
| preLoadNum=2 但无流水重叠 | 双缓冲需要 `CopyOut(s2-1)` 与 `Compute(s2)` 并行。确认 `PipeBarrier` 或 `SetFlag/WaitFlag` 允许这种重叠。若无重叠，preLoadNum=2 纯属浪费空间 |
| 循环内残留旧的 InitBuffer | 改造后循环内必须无任何 `InitBuffer`。旧代码可能隐藏在条件分支中，需全局搜索 `InitBuffer` |
| offset 计算用错 sizeof | `sizeof(float)` 与 `sizeof(half)` 不同。若计算类型是 FP16，offset 需按 `sizeof(half)` 计算 |
| 最后一轮状态未正确归并 | 若 preLoadNum=2，最后一轮和倒数第二轮的状态可能都需要参与最终输出。确认所有循环的状态已正确累积到输出 |
| 与 P1 双缓冲的 buffer 冲突 | P1 的 input/output 双缓冲和 P38 的状态双缓冲共用 UB。确保总占用不超过容量。建议用 `static_assert` 检查 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P38 Playbook Completion]
Step 1: done (/tmp/p38_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: constraints: state_size=? preLoadNum=? total_ub=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
