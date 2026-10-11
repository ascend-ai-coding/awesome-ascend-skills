# P91 Playbook: Softmax 梯度预计算独立阶段

> 本 Playbook 为**强制流程**。采纳 P91 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P91 的核心是**在 Flash Attention Backward 算子中，将 `sum(dY * O)` 计算提取为独立的 VecSfmg 阶段，使用独立 TPipe + 双缓冲，预计算结果写入 sfmgWorkspace 供 Main 阶段读取**。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p91_locations.txt`：

```bash
# 1. dY * O 逐元素乘法（softmax grad 的信号）
grep -n "Mul.*Vec\|Mul.*dy\|Mul.*dY\|Mul.*output\|Mul.*O" shared/original/op_kernel/*.cpp > /tmp/p91_locations.txt
# 2. ReduceSum/ReduceMax（归约操作）
grep -n "ReduceSum\|ReduceMax\|WholeReduceSum" shared/original/op_kernel/*.cpp >> /tmp/p91_locations.txt
# 3. S2/S1 循环（确认在 Main 循环内重复计算）
grep -n "for.*s[12]\|s1Loop\|s2Loop\|nLoop\|nloops" shared/original/op_kernel/*.cpp >> /tmp/p91_locations.txt
# 4. workspace 引用（sfmgWorkspace 或类似）
grep -n "workspace\|Workspace\|workspaceGm\|sfmg" shared/original/op_kernel/*.cpp shared/original/op_host/*_tiling.cpp >> /tmp/p91_locations.txt
# 5. TPipe 定义（确认是否已有独立 TPipe）
grep -n "TPipe\|pipe\s*=" shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h >> /tmp/p91_locations.txt
```

**交付物**（记录到 `implementation_note.txt` "Playbook Step 1"）：
- **dY*O 计算点**：Mul + ReduceSum 的位置 + 所在循环层级
- **当前循环结构**：Main 循环中是否每次 S2 迭代都重算
- **现有 workspace 分配**：sfmgWorkspace 或等价 workspace 的大小和位置
- **现有 TPipe 数量**：是否已有独立 TPipe 或共用主 TPipe

## Step 2: 改造计划表（强制填写）

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| dY*O 计算位置 | Main 循环内 L? | 独立 VecSfmg 阶段 | `?_kernel.cpp` |
| TPipe | 共用主 pipe | 新增 `pipeSfmg` | `?_kernel.cpp` |
| input buffer | — | 24K × 2 (ping-pong) | `?_kernel.cpp:L?` |
| cast buffer | — | 48K × 2 (fp16→fp32) | `?_kernel.cpp:L?` |
| sfmgWorkspace 大小 | `?` | `headNum * s1Loops * sizeof(float)` | `?_tiling.cpp:L?` |
| 循环重复次数 | 每 S2 迭代 `?` 次 | 0（预计算消除） | — |

## Step 3: 代码改造

### 3A. 形态识别

- **形态 α — FA Backward + dY*O 在 Main 循环内**：适合 P91。提取为独立 VecSfmg 阶段。
- **形态 β — FA Backward 但 dY*O 已预计算**：P91 已应用或类似优化已存在。不需要再改。
- **形态 γ — 非 FA Backward 或无 dY*O 计算**：不适用 P91。

### 3B. Canonical Template（形态 α）

```cpp
// === Step 1: 定义独立 VecSfmg 阶段 ===
class VectorSoftmaxGrad {
    TPipe pipeSfmg;
    TBuf<TPosition::VECCALC> inputBuf;   // 24K * 2 (ping-pong)
    TBuf<TPosition::VECCALC> castBuf;    // 48K * 2 (fp16 → fp32)
    TBuf<TPosition::VECCALC> outputBuf;
    TBuf<TPosition::VECCALC> tempBuf;

    void Init() {
        pipeSfmg.InitBuffer(inputBuf, 2, 24 * 1024);
        pipeSfmg.InitBuffer(castBuf, 2, 48 * 1024);
        // ...
    }

    void Process() {
        for (int s1 = 0; s1 < s1Loops; s1++) {
            // 搬入 dY 和 O
            CopyInSfmg(inputBuf, dyGm, oGm, s1);
            // FP16 → FP32
            Cast(castBuf, inputBuf, RoundMode::CAST_NONE);
            // dY * O
            Mul(castBuf, castDyBuf, castOBuf);
            // ReduceSum along D
            ReduceSum(outputBuf, castBuf, reduceParams);
            // 写入 sfmgWorkspace
            DataCopy(sfmgWorkspaceGm[s1 * headNum], outputBuf, params);
        }
    }
};

// === Step 2: Main 阶段读取预计算结果 ===
void VecMainProcess() {
    DataCopy(sfmgUb, sfmgWorkspaceGm[offset], params);
    Sub(dsUb, dyvUb, sfmgUb);  // dS = P * (dY*V^T - precomputedSum)
}
```

## Step 4: 同步与资源验证

- VecSfmg 和 Main 之间需要 `SyncAll()` 确保预计算完成后再进入 Main
- sfmgWorkspace 大小 = `headNum * s1Loops * sizeof(float)`，确认 tiling 分配足够
- 独立 TPipe 的 InitBuffer 总大小需 ≤ UB_TOTAL × 0.8
- 在 `implementation_note.txt` "Playbook Step 4" 中列出 UB/workspace 计算

## Step 5: 编码后自检

```bash
# 检查 1: 独立 VecSfmg 类或函数存在
grep -c "VectorSoftmaxGrad\|VecSfmg\|sfmgProcess\|SoftmaxGrad" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 独立 TPipe 定义
grep -c "TPipe.*sfmg\|pipeSfmg\|TPipe pipe" modified_files/op_kernel/*.cpp modified_files/op_kernel/*.h
# 期望: >= 1

# 检查 3: sfmgWorkspace 写入
grep -c "sfmgWorkspace\|DataCopy.*workspace" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: Main 阶段读取预计算结果
grep -c "DataCopy.*sfmg\|sfmgUb\|precomputed" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: VecSfmg 与 Main 之间有 SyncAll
grep -B2 -A2 "SyncAll" modified_files/op_kernel/*.cpp
# 人工检查 SyncAll 位于 VecSfmg 和 Main 之间
```

## Step 6: Known Pitfalls

| 现象 | 修复 |
|---|---|
| 结果错误（NaN） | sfmgWorkspace offset 计算错误，headNum/s1 索引不对 |
| 性能无提升 | s1Loops 太小（< 2），预计算开销 > 收益。恢复原方案 |
| UB overflow | VecSfmg 的 InitBuffer 太大。调整 buffer 大小或减少 BUFFER_NUM |
| SyncAll 后卡死 | VecSfmg 未正确完成所有 DataCopy，检查 ping-pong buffer 最后一块是否写出 |
| precision 退化 | Cast fp16→fp32 的 RoundMode 使用了 CAST_RINT 而非 CAST_NONE |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P91 Playbook Completion]
Step 1: done (/tmp/p91_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical applied
Step 4: workspace = headNum*s1Loops*sizeof(float) = ?, UB total ≤ limit: yes/no
Step 5: all 5 grep checks passed
Step 6: no pitfalls triggered / {列出触发的}
```
