# P9 Playbook: Deterministic Output via Workspace (确定性输出)

> 本 Playbook 为**强制流程**。采纳 P9 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P9 的核心是**用 workspace 隔离多核写冲突，并通过固定顺序的归并消除浮点累加顺序差异**，确保训练可复现。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p9_locations.txt`：

```bash
# 1. 多核写冲突点（AtomicAdd、直接输出写、累加）
grep -n "AtomicAdd\|AtomicMax\|outputGm.*=\|+=.*outputGm\|resultGm" \
    shared/original/op_kernel/*.cpp > /tmp/p9_locations.txt
# 2. 索引式访问（稀疏更新、embedding、gather/scatter）
grep -n "indices\|indexGm\|sparse\|embedding\|gather\|scatter" \
    shared/original/op_kernel/*.cpp shared/original/op_host/*_tiling.cpp >> /tmp/p9_locations.txt
# 3. 已有的 workspace 或同步机制
grep -n "workspace\|Workspace\|SyncAll\|sync_all\|deterministic" \
    shared/original/op_kernel/*.cpp shared/original/op_host/*_tiling.cpp >> /tmp/p9_locations.txt
# 4. 核间分块边界（start/end、rowsPerCore）
grep -n "startIdx\|endIdx\|startRow\|endRow\|blockIdx.*coreNum\|myRows" \
    shared/original/op_kernel/*.cpp shared/original/op_host/*_tiling.cpp >> /tmp/p9_locations.txt
# 5. 输出数据类型与精度相关
grep -n "float.*output\|half.*output\|quant\|smooth\|norm" \
    shared/original/op_kernel/*.cpp >> /tmp/p9_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **写冲突位置**：所有 AtomicAdd / 直接输出写 / 累加操作的文件 + 行号
- **索引访问模式**：是稀疏索引（embedding）、密集逐元素、还是归约累加
- **已有机制**：是否已有 workspace、SyncAll、或确定性标记
- **核边界**：各核处理的数据起止索引
- **输出类型**：FP32 / FP16 / INT8，是否涉及量化

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 冲突类型 | `?` (AtomicAdd / 直接写 / 无) | workspace 隔离 | `?_kernel.cpp:L?` |
| 数据模式 | `?` (稀疏索引 / 密集累加 / 双输出量化) | `alpha/beta/gamma` 见 3A | `?_kernel.cpp:L?` |
| workspace 大小 | `? bytes` | `coreNum × outputSlice × sizeof(dtype)` | `?_tiling.cpp:L?` |
| 同步点 | `?` (无 / SyncAll / 其他) | 至少 1 处 SyncAll | `?_kernel.cpp:L?` |
| 归并核 | `?` (分散写 / 指定核归并) | 核 0 或固定顺序归并 | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的写冲突模式和数据分布，判断你的代码属于以下哪种形态：

- **形态 α — 稀疏索引重叠（Optimizer / Embedding 更新）**：多个核通过 `indices` 数组更新同一输出行，AtomicAdd 顺序不定导致微差。
- **形态 β — 密集输出累加（Reduction / Norm / Softmax）**：各核产生部分结果，需累加到同一输出张量，顺序敏感。
- **形态 γ — 双输出量化中间态（SmoothNorm / Quantize）**：每个核产生两个中间输出（如 smooth1 / smooth2），需确定性写回后再二次处理。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 β — 密集累加，最通用）

```cpp
// === 改造前（多核直接写输出，竞争 + 顺序不定）===
__aicore__ inline void Compute(...) {
    for (uint32_t i = startIdx; i < endIdx; i++) {
        float val = ComputeLocal(i);
        AtomicAdd(outputGm[i], val);   // 竞争写，顺序不确定
    }
}

// === 改造后（per-core workspace → 固定顺序归并）===
// Host 侧：分配 workspace = coreNum × outputSize × sizeof(float)
// workspace[coreId] 为该核的私有累积区

__aicore__ inline void Compute(...) {
    // 1. 局部累积到 UB buffer（无竞争）
    LocalTensor<float> accLocal = accBuf.Get<float>();
    Duplicate(accLocal, 0.0f, outputSize);  // 清零
    for (uint32_t i = startIdx; i < endIdx; i++) {
        uint32_t row = indicesGm[i];
        float val = gradGm[i];
        accLocal[row] += val;  // UB 内局部累加，无竞争
    }
    
    // 2. 写回 GM workspace（每个核写自己的 slice）
    uint32_t wsOffset = blockIdx * outputSize;
    DataCopy(workspaceGm + wsOffset, accLocal, outputSize);
    
    SyncAll();  // 等待所有核完成 workspace 写入
    
    // 3. 核 0 按固定顺序归并到最终输出
    if (blockIdx == 0) {
        LocalTensor<float> outLocal = outBuf.Get<float>();
        DataCopy(outLocal, workspaceGm, outputSize);  // 拷贝核 0 结果作为初始值
        for (uint32_t c = 1; c < coreNum; c++) {
            LocalTensor<float> sliceLocal = sliceBuf.Get<float>();
            DataCopy(sliceLocal, workspaceGm + c * outputSize, outputSize);
            Add(outLocal, outLocal, sliceLocal, outputSize);  // 固定顺序：0→1→2→...
        }
        DataCopy(outputGm, outLocal, outputSize);
    }
}
```

### 3C. Variant Notes（若是形态 α 或 γ）

- **形态 α（稀疏索引重叠）**：
  数据特征为 `indices` 数组可能存在跨核重复。除了 workspace 方案外，也可用**反向扫描 + 边界去重**：
  ```cpp
  // 每个核从后向前处理，检测与前一个核的边界是否重叠
  for (int32_t i = endIdx - 1; i >= (int32_t)startIdx; i--) {
      uint32_t row = indicesGm[i];
      // 边界去重：若当前核起始索引等于前一个核的末尾索引，跳过
      if (i == startIdx && blockIdx > 0 && row == prevBoundaryRow) continue;
      // 写入本核 workspace（或局部 buffer），不直接写输出
      wsLocal[row] += gradGm[i];
  }
  ```
  反向扫描保证边界重复项由**编号更小**的核处理，消除顺序不确定性。
  - 若 `indices` 跨核重复严重（>10%），workspace 方案（形态 β）更优。
  - 若重复极少，反向扫描 + 边界去重更省内存。

- **形态 γ（双输出量化中间态）**：
  如 SmoothNorm 场景，每个核产生 `smooth1` 和 `smooth2` 两组 FP32 中间结果：
  ```cpp
  // workspace 布局：每个核占 2 × numLastDim
  uint32_t wsStride = 2 * numLastDim;
  workspaceGm.SetGlobalBuffer((__gm__ float*)workspace + blockIdx * wsStride);
  
  // 核内分别写入 smooth1 / smooth2
  CopyOutSmoothNorm(workspaceGm, 0, rowOffset, elementCount);           // smooth1
  CopyOutSmoothNorm(workspaceGm, numLastDim, rowOffset, elementCount);  // smooth2
  
  SyncAll();
  
  // 后续核（或核 0）按固定顺序读取各核 workspace 并做量化
  if (blockIdx == 0) {
      for (uint32_t c = 0; c < coreNum; c++) {
          CopyInSmoothNorm(xLocalFp32, 0, rowOffset, elementCount, localMax1);
          // 确定性处理...
      }
  }
  ```
  关键点：双输出场景 workspace 大小为 `coreNum × 2 × outputSlice`，Host 侧 tiling 必须正确计算。

- **与 P4 的边界**：P4 解决核间**数据量**均衡，P9 解决核间**写冲突**确定性。两者可叠加：先用 P4 均衡分核，再用 P9 workspace 消除竞争。

- **与 P12 的协同**：若输出涉及 mask/broadcast 后的写回，可先用 P12 向量 API 处理 mask，再经 P9 workspace 机制写回，避免 mask 分支引入额外非确定性。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: workspace 大小 = coreNum × outputSlice × sizeof(dtype) ≤ GM 可用空间（通常 ≤ 256MB）
约束 2: 每个核只写自己的 workspace slice（offset = blockIdx × sliceSize），严禁越界写其他核区域
约束 3: SyncAll 必须在所有核完成 workspace 写入后、归并前执行，不能省略
约束 4: 归并核（通常是核 0）的 output buffer 必须 ≥ outputSlice，且 32B 对齐
约束 5: 若 outputSlice 极大（> UB 容量），需分 tile 归并，不能一次性拷贝整个 slice
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `coreNum = ?`, `outputSlice = ?`, `wsTotalBytes = ?`
- workspace 是否在 GM 限制内：yes/no
- SyncAll 位置是否在所有写之后、读之前：yes/no
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 已引入 workspace 相关变量或函数
grep -cE "workspaceGm|workspace.*GlobalBuffer|wsLocal|wsBuf|workspace" \
    modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 存在 SyncAll（或等效全局屏障）
grep -cE "SyncAll|sync_all|__sync\b|barrier" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: 无裸 AtomicAdd（已被 workspace 替代）
grep -cE "AtomicAdd\s*\(|AtomicMax\s*\(" modified_files/op_kernel/*.cpp
# 期望: == 0（除非形态 α 保留 AtomicAdd 但加了边界去重，此时需在 note 中说明）

# 检查 4: 有明确的核 0 归并或固定顺序归并逻辑
grep -cE "if.*blockIdx.*==.*0|if.*blockIdx.*==.*0|for.*c.*coreNum|for.*core.*merge" \
    modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: workspace offset 计算使用 blockIdx，不是 hardcode
grep -cE "blockIdx.*\*.*outputSize|blockIdx.*\*.*wsStride|blockIdx.*\*.*numLastDim" \
    modified_files/op_kernel/*.cpp
# 期望: >= 1
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：workspaceGm 类型不匹配 | `SetGlobalBuffer` 要求 `__gm__` 指针。确认 Host 侧传入的 workspace 指针已正确转为 `__gm__ float*` |
| 运行时：输出全零或部分为零 | 检查 `Duplicate(accLocal, 0.0f, ...)` 是否遗漏。UB buffer 不会自动清零 |
| 运行时：仍偶尔出现微差（精度不对齐） | 确认 SyncAll 真的在所有核写入 workspace 之后执行。AscendC 的 SyncAll 是核间屏障，必须显式调用 |
| 运行时：核 0 明显慢于其他核 | 核 0 承担归并任务，若 `coreNum × outputSlice` 过大，归并会成为瓶颈。考虑多轮分阶段归并 |
| workspace 超出 GM 限制 | 若 `coreNum × outputSlice` 过大，改用形态 α（反向扫描 + 边界去重），不分配 workspace |
| 归并时 UB 不够放下整个 outputSlice | 将归并拆分为 tile：每次只拷贝一部分 slice 到 UB，累加后写回，循环直到覆盖完整 outputSlice |
| 稀疏索引跨核重复率极低（<1%） | 此时 workspace 方案的 overhead（额外 GM 读写 + SyncAll）可能超过收益。建议退化为形态 α 或不做 P9 |
| 双输出量化场景 workspace 布局错误 | 确认 stride = `2 × numLastDim`，不是 `numLastDim`。核内 offset 分别为 `0` 和 `numLastDim` |
| Host 侧未分配 workspace | 必须在 Host tiling 或算子接口中申请 workspace，并通过 TilingData 或 kernel 参数传入 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P9 Playbook Completion]
Step 1: done (/tmp/p9_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: constraints: coreNum=? outputSlice=? wsTotalBytes=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
