# P1 Playbook: 双缓冲机制实操流程

> 本 Playbook 为**强制流程**。采纳 P1 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步，禁止"看起来改了"就声明完成。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p1_locations.txt`：

```bash
# 文件路径以 shared/original/ 为根目录（若 parent_solution_ref 非空，改为父变体的 modified_files/）
grep -n "BUFFER_NUM\|bufferNum" shared/original/op_kernel/*.h shared/original/op_host/*.h > /tmp/p1_locations.txt
grep -n "InitBuffer" shared/original/op_kernel/*.cpp shared/original/op_host/*.cpp >> /tmp/p1_locations.txt
grep -n "for\s*(.*;.*;.*)\|while\s*(" shared/original/op_kernel/*.cpp >> /tmp/p1_locations.txt
grep -n "CopyIn\|CopyOut\|Compute" shared/original/op_kernel/*.cpp >> /tmp/p1_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **BUFFER_NUM 定义**：文件 + 行号
- **主计算循环**：文件 + 行范围（通常在 `Process()` 或类似函数内）
- **InitBuffer 调用**：文件 + 行号（所有 TQue 的初始化）
- **循环内的 CopyIn / Compute / CopyOut 三元组**：文件 + 行号

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| BUFFER_NUM | `?` | `2` | `?.h:L?` |
| tileSize / ub_factor | `?` | 减半为 `?` | `?_tiling.cpp:L?` |
| 主循环结构 | 串行 CopyIn→Compute→CopyOut | Prologue + Steady + Epilogue | `?_kernel.cpp:L?-L?` |
| 新 UB 占用 | `? bytes` | `? bytes`（≤ UB_TOTAL × 0.8） | — |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 循环结构转换（核心）

### 3A. 形态识别

读 Step 1 定位的主循环，判断你的代码属于以下哪种形态：

- **形态 α — 扁平循环**（最常见）：单层 for，每次迭代做 CopyIn → Compute → CopyOut
- **形态 β — 嵌套循环**：外层 block 循环 + 内层 tile 循环，CopyIn/Compute/CopyOut 在内层
- **形态 γ — 带 tail block**：主循环 + 尾块特殊处理分支

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 α）

```cpp
// === 改造前（串行）===
for (int32_t i = 0; i < loops; i++) {
    CopyIn(i);        // 搬入 tile i
    Compute(i);       // 计算 tile i
    CopyOut(i);       // 搬出 tile i
}

// === 改造后（双缓冲 + 流水）===
constexpr int32_t BUFFER_NUM = 2;
pipe.InitBuffer(inQueue, BUFFER_NUM, tileSize * sizeof(T));
pipe.InitBuffer(outQueue, BUFFER_NUM, tileSize * sizeof(T));

CopyIn(0);                                 // Prologue: 预取第一块
for (int32_t i = 0; i < loops - 1; i++) {  // Steady: 预取下一块 + 算当前 + 搬出当前
    CopyIn(i + 1);
    Compute(i);
    CopyOut(i);
}
Compute(loops - 1);                        // Epilogue: 算最后一块 + 搬出
CopyOut(loops - 1);
```

### 3C. Variant Notes（若是形态 β 或 γ）

- **形态 β（嵌套）**：**只对内层循环做变换**。外层 block 循环保持原样；内层按 3B 改造为 Prologue+Steady+Epilogue。
- **形态 γ（带 tail）**：主循环按 3B 改造；**tail block 保持单缓冲不变**（tile 数量 < 2 时双缓冲无收益）。
- **如果 UB 紧张**：减半 tileSize 同步 BUFFER_NUM=2（总 UB 占用不变）；若减半后仍溢出，放弃 P1，换 P22 TQueBind 复用。

## Step 4: UB 容量复核（防崩溃）

**公式**：
```
新 UB 占用 = BUFFER_NUM × tileSize × sizeof(dtype) × queue_count
```

**约束**：`新 UB 占用 ≤ UB_TOTAL × 0.8`（留 20% 给栈 + 临时变量）。

若不满足，回到 Step 3C 减半 tileSize。**在 implementation_note.txt "Playbook Step 4" 中报告具体计算**（实际数值 + 是否通过）。

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。

```bash
# 检查 1: BUFFER_NUM 已改为 2
grep -c "BUFFER_NUM\s*=\s*2\|bufferNum\s*=\s*2" modified_files/op_kernel/*.h modified_files/op_host/*.h
# 期望: >= 1

# 检查 2: InitBuffer 使用了 BUFFER_NUM
grep -c "InitBuffer.*BUFFER_NUM\|InitBuffer.*bufferNum\|InitBuffer.*,\s*2\s*," modified_files/op_kernel/*.cpp
# 期望: >= 1（每个双缓冲队列一次）

# 检查 3: Prologue 存在（循环外有 CopyIn(0) 或 CopyIn(prefetchIdx)）
grep -c "CopyIn(0)\|CopyIn(startIdx)\|CopyIn(prefetchIdx)" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: Steady 预取模式（CopyIn(i+1) 类模式）
grep -En "CopyIn\([a-zA-Z_]+\s*\+\s*1\)|CopyIn\(next" modified_files/op_kernel/*.cpp | wc -l
# 期望: >= 1

# 检查 5: Epilogue 存在（循环外有 Compute(last) 或 CopyOut(last)）
grep -En "Compute\(.*-\s*1\)|CopyOut\(.*-\s*1\)|Compute\(loops\s*-\s*1\)" modified_files/op_kernel/*.cpp | wc -l
# 期望: >= 1
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：UB overflow | 回 Step 4 复算；tileSize 需要减半 |
| 编译失败：TQue template argument mismatch | InitBuffer 模板参数 `<TQue, N>` 的 N 要对应 BUFFER_NUM（例如 `TQue<QuePosition::VECIN, 2>`） |
| 运行时：first iteration 结果错 | Prologue CopyIn(0) 必须在 pipe.InitBuffer 之后、主循环之前 |
| 运行时：最后一块数据错 | Epilogue 必须处理 `loops - 1`，且 loop 上限是 `loops - 1` 不是 `loops` |
| 精度退化：浮点累加顺序变了 | P1 不应改变计算顺序；检查是否误改了 Compute 内部 |
| 性能提升 < 10% | 查 profiling 看是不是 compute_bound；若是，P1 帮不上，换 P4/P8 |
| 数据竞争：UB tensor 内容被覆盖 | 检查 PipeBarrier<PIPE_V>() 是否在每次 Compute 后存在 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P1 Playbook Completion]
Step 1: done (/tmp/p1_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: UB calc: 新占用=BUFFER_NUM×tileSize×sizeof(dtype)×queue_count ≤ UB_TOTAL×0.8: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
