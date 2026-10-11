# P81 Playbook: UB/TBuf 常驻 Buffer 复用

> 本 Playbook 为**强制流程**。采纳 P81 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P81 的核心是**将小尺寸参数/状态一次性搬入 UB/TBuf，在循环外完成初始化（DataCopy/Cast），循环内直接复用，消除每轮迭代的 GM 搬运和类型转换开销**。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p81_locations.txt`：

```bash
# 1. 循环结构
grep -n "for\s*(\|while\s*(\|loop\|Loop\|iteration\|Iteration" \
    shared/original/op_kernel/*.cpp > /tmp/p81_locations.txt
# 2. 小尺寸参数
grep -n "weight\|bias\|scale\|gamma\|offset\|param\|state\|accumulator\|grad" \
    shared/original/op_kernel/*.cpp >> /tmp/p81_locations.txt
# 3. DataCopy 和 Cast
grep -n "DataCopy\|DataCopyPad\|Cast\|cast" \
    shared/original/op_kernel/*.cpp >> /tmp/p81_locations.txt
# 4. UB/TBuf 分配
grep -n "AllocTensor\|FreeTensor\|VECCALC\|TBuf\|UB\|inQueue\|outQueue" \
    shared/original/op_kernel/*.cpp >> /tmp/p81_locations.txt
# 5. 当前搬运模式
grep -n "CopyIn\|CopyOut\|Process\|Compute" \
    shared/original/op_kernel/*.cpp >> /tmp/p81_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **循环结构**：for/while 的嵌套层级、循环变量
- **小尺寸参数**：weight/bias/scale/gamma 等的类型和尺寸
- **DataCopy/Cast 位置**：每次迭代是否重复搬运
- **UB/TBuf 分配**：AllocTensor/FreeTensor 的位置（循环内/外）
- **当前搬运模式**：每次迭代都 CopyIn/CopyOut 还是只在循环外一次

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 循环层级 | `?` (1/2/3层) | 不变 | `?_kernel.cpp:L?` |
| 常驻数据 | `?` (weight/bias/scale/其他) | `alpha/beta` 见 3A | `?_kernel.cpp:L?` |
| 当前搬运 | `?` (每轮迭代/循环外一次) | 循环外一次 | `?_kernel.cpp:L?` |
| UB/TBuf | `?` (UB/TBuf) | 不变 | `?_kernel.cpp:L?` |
| 数据尺寸 | `?` (字节数) | 不变 | `?_kernel.cpp:L?` |
| 复用次数 | `?` (循环次数) | 不变 | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的常驻数据类型和 buffer 位置，判断你的代码属于以下哪种形态：

- **形态 α — UB 常驻（最常见）**：小参数通过 `AllocTensor` 分配到 UB，循环外 DataCopy+Cast 一次，循环内直接使用。
- **形态 β — TBuf 常驻**：参数常驻 TBuf（不受队列同步保护），需要手动 PipeBarrier 保证一致性。适合生命周期跨多个 Compute 阶段的场景。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta`

### 3B. Canonical Template（形态 α — UB 常驻）

```cpp
// === 改造前（每轮迭代重复搬运）===
__aicore__ inline void ProcessNaive() {
    for (int64_t bIdx = 0; bIdx < baseB; ++bIdx) {
        for (int64_t sIdx = 0; sIdx < baseS; ++sIdx) {
            // ❌ 每轮迭代都搬运 + Cast
            DataCopyPad(weightLocal, weightGm, copyParams, padParams);
            Cast(weightFp32, weightLocal, RoundMode::CAST_NONE, alignBaseH);
            Compute(xLocalFp32, weightFp32, ...);
        }
    }
}

// === 改造后（循环外一次初始化，循环内复用）===
__aicore__ inline void ProcessOptimized() {
    // Step 1: 循环外 — 分配 UB buffer 并搬运参数
    LocalTensor<float> weightFp32 = this->inQueueW.template AllocTensor<float>();
    DataCopyPad(weightLocal, weightGm, copyParams, padParams);
    Cast(weightFp32, weightLocal, RoundMode::CAST_NONE, alignBaseH);
    
    // Step 2: 循环内 — 直接复用 weightFp32，无搬运开销
    for (int64_t bIdx = 0; bIdx < baseB; ++bIdx) {
        for (int64_t sIdx = 0; sIdx < baseS; ++sIdx) {
            Compute(xLocalFp32, weightFp32, y0Fp32, y1Fp32, y2Fp32);
        }
    }
    
    // Step 3: 循环后 — 释放 buffer
    this->inQueueW.FreeTensor(weightFp32);
}
```

### 3C. Variant Notes（若是形态 β）

- **形态 β（TBuf 常驻）**：
  当参数需要跨多个 Compute 阶段或不受队列保护时：
  ```cpp
  // TBuf 常驻：手动管理生命周期
  LocalTensor<float> weightTBuf = tBufPool.Get<float>(weightSize);
  DataCopyPad(weightTBuf, weightGm, copyParams, padParams);
  Cast(weightTBuf, weightTBuf, RoundMode::CAST_NONE, alignBaseH);
  PipeBarrier<PIPE_V>();  // 确保 Cast 完成
  
  for (...) {
      Compute(xLocalFp32, weightTBuf, ...);
      PipeBarrier<PIPE_V>();  // 确保 Compute 完成后下一轮可用
  }
  ```
  形态 β 需额外注意 PipeBarrier 保证一致性。

- **与 P35 的边界**：P35（TBuf 常驻中间累加器）处理中间结果的 TBuf 常驻，P81 处理参数/状态的常驻。两者可同时存在：P35 常驻中间结果，P81 常驻输入参数。
- **与 P40 的边界**：P40（Workspace 双缓冲）处理 workspace 的常驻复用，P81 处理 UB/TBuf 的常驻复用。两者可同时存在。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: 常驻数据尺寸必须可控（通常 < UB 总容量的 20%），不能挤占主数据 tile 空间
约束 2: 循环外 AllocTensor + 循环内使用 + 循环后 FreeTensor 的顺序不能错
约束 3: TBuf 常驻需手动 PipeBarrier，不受队列同步保护
约束 4: 常驻数据的生命周期必须覆盖整个循环，不能在循环内被覆盖
约束 5: 复用次数必须足够多（通常 >= 4 次迭代才有收益），否则初始化开销得不偿失
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `常驻数据尺寸 = ? bytes`, `UB 总容量 = ? bytes`
- `复用次数 = ?`, `预期节省搬运次数 = ?`
- `Buffer 占用比例 = ?%`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: AllocTensor 在循环外
grep -cE "AllocTensor[^;]*;[^f]*for|AllocTensor.*Process|AllocTensor.*Compute" modified_files/op_kernel/*.cpp
# 期望: >= 1（AllocTensor 在循环外）

# 检查 2: FreeTensor 在循环后
grep -cE "FreeTensor[^;]*;[^f]*for|for.*FreeTensor" modified_files/op_kernel/*.cpp
# 期望: == 0（FreeTensor 不在循环内）

# 检查 3: DataCopy/Cast 在循环外
grep -cE "DataCopy[^;]*;[^f]*for|Cast[^;]*;[^f]*for" modified_files/op_kernel/*.cpp
# 期望: >= 1（初始化在循环外）

# 检查 4: 循环内无 DataCopy 搬运常驻数据
grep -cE "for.*DataCopy.*weight|for.*DataCopy.*bias|for.*Cast.*weight" modified_files/op_kernel/*.cpp
# 期望: == 0

# 检查 5: 有循环结构
grep -cE "for\s*\(|while\s*\(" modified_files/op_kernel/*.cpp
# 期望: >= 1
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：AllocTensor 在循环内 | 将 AllocTensor 提到循环外。UB buffer 不能在循环内反复分配释放 |
| 运行时：UB 溢出 | 常驻 buffer 过大，挤占主数据 tile 空间。计算常驻占比，若 >20% 考虑改用 TBuf 或缩小 tile |
| 运行时：数据不一致 | TBuf 常驻缺少 PipeBarrier。在 Cast 后和 Compute 后各加 `PipeBarrier<PIPE_V>()` |
| 性能不如预期 | 复用次数太少。若循环次数 < 4，初始化开销可能大于收益。需具体分析 |
| FreeTensor 忘记调用 | 必须 `this->inQueueW.FreeTensor(weightFp32)` 在循环后，否则 UB 泄漏 |
| 循环内误改常驻数据 | 若 Compute 会修改 weightFp32（如 in-place 更新），常驻策略不适用。需用副本 |
| 多核场景数据竞争 | 常驻数据若为全局共享（如 running_mean），需确保各核独立 buffer |
| TBuf 与 UB 混淆 | TBuf 不受队列保护，需手动同步；UB 受 TQue 队列保护。不要混用管理语义 |
| Cast 在循环内 | 检查是否所有 DataCopy+Cast 都提到了循环外。遗漏一个就会抵消收益 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P81 Playbook Completion]
Step 1: done (/tmp/p81_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta, canonical/variant applied
Step 4: constraints: resident_size=? ub_capacity=? reuse_count=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
