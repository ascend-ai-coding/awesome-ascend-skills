# P3 Playbook: Small-D Multi-Row Merging (小D多行合并优化)

> 本 Playbook 为**强制流程**。采纳 P3 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> P3 的核心是**当内维维度 D 较小时，将多个行合并到同一个 UB buffer 中一次性处理**，减少 GM 访问次数并提升内存局部性。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p3_locations.txt`：

```bash
# 1. 内维维度定义（D / embedding_dim / numLastDim）
grep -n "embedding_dim\|numLastDim\|innerDim\|featureDim\|D\\b.*=.*[0-9]" \
    shared/original/op_host/*_tiling.cpp shared/original/op_kernel/*.cpp > /tmp/p3_locations.txt
# 2. 逐行处理循环（重点改造对象）
grep -n "for.*row\|rowsPerCore\|rowIdx\|CopyIn.*row\|CopyOut.*row" \
    shared/original/op_kernel/*.cpp >> /tmp/p3_locations.txt
# 3. UB buffer 分配现状
grep -n "InitBuffer\|TBuf\|TQue\|BUFFER_NUM\|tileSize\|ubFactor" \
    shared/original/op_kernel/*.cpp >> /tmp/p3_locations.txt
# 4. 已有的多行或排序优化
grep -n "ReinterpretCast\|multiRow\|rowsPerUB\|Sort\|CreateVecIndex\|Extract" \
    shared/original/op_kernel/*.cpp >> /tmp/p3_locations.txt
# 5. GM 访问模式（判断是否为随机访问）
grep -n "indicesGm\|indexGm\|gather\|scatter\|random" \
    shared/original/op_kernel/*.cpp >> /tmp/p3_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **内维维度**：D / embedding_dim 的值或变量名、当前大小范围
- **逐行循环**：row loop 位置、内部是否有 tile loop
- **UB 现状**：当前 buffer 大小、数量、是否已用满
- **已有优化**：是否已有 ReinterpretCast / Sort / multiRow
- **GM 模式**：是顺序访问（row×D）还是随机索引访问（indices-based）

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 内维 D | `?` (值或范围) | 不变 | `?_host.cpp:L?` |
| 当前处理模式 | `?` (逐行 / 逐 tile / 其他) | 多行 batch | `?_kernel.cpp:L?` |
| UB 总容量 | `? bytes` | `MAXBUF` (通常 195584) | `?_kernel.cpp:L?` |
| 每 UB 可合并行数 | `?` (当前 =1) | `MAXBUF / (D × sizeof(T) × numBuf)` | `?_kernel.cpp:L?` |
| GM 访问模式 | `?` (顺序 / 随机索引) | `alpha/beta/gamma` 见 3A | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的 D 大小和 GM 访问模式，判断你的代码属于以下哪种形态：

- **形态 α — 顺序访问小D密集数据（Norm / Activation / Elementwise）**：输入是连续 memory（row-major），每行 D 较小，逐行 CopyIn/CopyOut 开销占比高。
- **形态 β — 随机索引小D稀疏数据（Embedding / Optimizer / Gather）**：通过 `indices` 数组访问，相同 index 分散在不同位置，随机访问多。
- **形态 γ — 极小 D 向量并行（D ≤ 32，如 vec4/vec8 场景）**：D 小到可用一条向量指令处理多行。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 α — 顺序访问多行合并，最常见）

```cpp
// === 改造前（逐行处理，多次 CopyIn/CopyOut，GM 开销大）===
for (uint32_t rowIdx = 0; rowIdx < rowsPerCore; rowIdx++) {
    for (uint32_t tileId = 0; tileId < nTiles; tileId++) {
        CopyIn(rowIdx, tileId);
        float tileSqSum = ComputePass1();
        rowSqSum += tileSqSum;
    }
}

// === 改造后（Single-N 大 buffer carve，多行 batch 处理）===
static constexpr uint32_t MAXBUF = 195584;  // (192KB - 1KB) margin，按实际 UB 容量调整
Pipe pipe;
TBuf<QuePosition::VECCALC> unitBuf;
pipe.InitBuffer(unitBuf, MAXBUF);

// 计算每 UB 可合并的行数
uint32_t bytesPerRow = D * sizeof(T);
uint32_t numBuffers = 4;  // 例如：输入x / 输入x2 / FP32中间 / 输出
uint32_t rowsPerUB = MAXBUF / (bytesPerRow * numBuffers);
if (rowsPerUB == 0) rowsPerUB = 1;  // 安全兜底

for (uint32_t rowBatch = 0; rowBatch < rowsPerCore; rowBatch += rowsPerUB) {
    uint32_t actualRows = (rowBatch + rowsPerUB <= rowsPerCore) 
                          ? rowsPerUB 
                          : (rowsPerCore - rowBatch);
    
    LocalTensor<float> ubLocal = unitBuf.Get<float>();
    // 用 ReinterpretCast 将大 buffer carve 成多行逻辑 buffer
    LocalTensor<T> xLocal  = ubLocal.ReinterpretCast<T>()[0];
    LocalTensor<T> x2Local = ubLocal.ReinterpretCast<T>()[actualRows * D];
    LocalTensor<float> fp32Local = ubLocal[actualRows * D * 2];
    LocalTensor<float> outLocal  = ubLocal[actualRows * D * 3];
    
    // 顺序拷贝多行（合并 GM 访问，提升带宽利用率）
    DataCopy(xLocal, inputGm + rowBatch * D, actualRows * D);
    
    // 多行一起 Compute（向量指令一次处理多行数据）
    ComputeMultiRow(xLocal, x2Local, fp32Local, outLocal, actualRows, D);
    
    // 顺序写回
    DataCopy(outputGm + rowBatch * D, outLocal, actualRows * D);
}
```

### 3C. Variant Notes（若是形态 β 或 γ）

- **形态 β（随机索引稀疏数据）**：
  当存在 `indices` 数组且相同 index 分散时，使用 **Sort + 重排** 将随机访问转为顺序访问：
  ```cpp
  // 1. 生成位置索引 [0, 1, 2, ..., numIndices-1]
  CreateVecIndex(vecIndexLocal, 0, 1, numIndices);
  
  // 2. 对 indices 排序，同时重排位置索引
  Sort(sortedIndicesLocal, vecIndexLocal, indicesLocal, numIndices);
  
  // 3. 按排序后顺序提取原始位置
  Extract(positionsLocal, vecIndexLocal, sortedIndicesLocal, numIndices);
  
  // 4. 按排序顺序处理：相同 index 连续出现，可合并累加
  uint32_t currentIdx = sortedIndicesLocal.GetValue(0);
  float acc = 0.0f;
  for (uint32_t i = 0; i < numIndices; i++) {
      uint32_t idx = sortedIndicesLocal.GetValue(i);
      uint32_t pos = positionsLocal.GetValue(i);
      if (idx != currentIdx) {
          // 写入上一个 index 的累加结果
          AtomicAdd(outputGm[currentIdx], acc);
          currentIdx = idx;
          acc = gradGm[pos];
      } else {
          acc += gradGm[pos];  // 同 index 连续累加
      }
  }
  AtomicAdd(outputGm[currentIdx], acc);  // 最后一个
  ```
  ⚠️ Sort 指令有最大长度限制（通常 8192），若 `numIndices` 更大，需分 chunk 排序。
  ⚠️ 若相同 index 的重复率 < 5%，排序 overhead 可能不划算，退化为形态 α 或直接逐行处理。

- **形态 γ（极小 D，D ≤ 32）**：
  D 极小意味着一行数据只占 1~2 个 vector block。此时用标准向量指令即可一次处理多行：
  ```cpp
  // D = 8 float，一行 = 32 byte = 1 block
  // 一次处理 16 行 = 16 blocks = 512 byte，远小于 UB
  uint32_t elemsPerRow = D;
  uint32_t rowsPerVec = 256 / elemsPerRow;  // 256B 是常用 vector block
  // 用 Duplicate / Mul / Add 直接处理拼接后的 multi-row tensor
  ```
  形态 γ 通常不需要显式的 Sort 或 ReinterpretCast，只需调整 loop stride。

- **与 P4 的协同**：P3 合并多行后，每核处理的数据量增加，可能改变 P4 的最优分核策略。建议先 P4 均衡分核，再 P3 合并行。

- **与 P11 的边界**：P3 的 `actualRows` 和 P11 的尾块处理可能冲突。确保：
  `rowBatch + actualRows ≤ rowsPerCore`，且最后一 batch 的 `actualRows` 是正确余数。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: D × rowsPerUB × numBuffers × sizeof(T) ≤ MAXBUF ≤ 实际 UB 容量（a3 约 192KB，a5 约 256KB）
约束 2: DataCopy 的 count 参数必须 32B 对齐。actualRows × D × sizeof(T) 需向上取整到 32B
约束 3: ReinterpretCast 后的各逻辑 buffer 不重叠：x2Offset ≥ x1Offset + actualRows×D, fp32Offset ≥ x2Offset + actualRows×D, ...
约束 4: rowsPerUB ≥ 2，否则多行合并没有收益（应退化为其他策略）
约束 5: 形态 β 的 Sort 长度 ≤ 指令最大支持（8192 或 16384，查具体芯片手册）
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `D = ?`, `sizeof(T) = ?`, `numBuffers = ?`, `MAXBUF = ?`
- `rowsPerUB = ?`, `actualBytes = ?`, `actualBytes 32B aligned = ?`
- 各逻辑 buffer offset：`x=0, x2=?, fp32=?, out=?`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 存在 ReinterpretCast 或多行 batch 变量
grep -cE "ReinterpretCast|rowsPerUB|actualRows|rowBatch" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 无逐行内层 CopyIn（row loop 内部不再有单行的 DataCopy）
grep -cE "for.*rowIdx.*\{[^}]*DataCopy|for.*row.*\{[^}]*CopyIn" modified_files/op_kernel/*.cpp
# 期望: == 0（形态 β 除外，若保留逐行索引访问需在 note 中说明）

# 检查 3: 有较大的 InitBuffer（≥ 64KB，表明使用了 Single-N 大 buffer）
grep -cE "InitBuffer.*[0-9]{5,}|MAXBUF|195584" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: DataCopy count 包含 actualRows（不是只拷贝一行）
grep -cE "DataCopy.*actualRows|DataCopy.*rowsPerUB|DataCopy.*rowBatch" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: 形态 β 时必须有 Sort/CreateVecIndex；形态 α/γ 时不应有 Sort
grep -cE "Sort|CreateVecIndex|Extract" modified_files/op_kernel/*.cpp
# 期望: 形态 β >= 1，形态 α/γ == 0
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：ReinterpretCast 类型不匹配 | `ReinterpretCast<T>()` 要求目标类型与源类型大小兼容。`float` 转 `half` 需确保 offset 按 half 计算 |
| 运行时：UB 越界导致随机崩溃 | 严格复核约束 3 的各 buffer offset 不重叠。建议用 `static_assert` 或 `assert` 检查 offset 序列单调递增 |
| 运行时：输出结果错位（行之间混叠） | 检查 `DataCopy` 的 src/dst offset 是否按 `rowBatch * D` 计算，不是按 `rowIdx * D`（rowIdx 已不存在） |
| 性能与 baseline 持平或更差 | 若 D 较大（>512）或 rowsPerUB 只能 =1，P3 无收益。检查 D 是否确实 ≤ 512 |
| Sort 指令编译失败 | 确认 `numIndices` ≤ Sort 最大长度。超长时拆分为多个 chunk 分别排序 |
| 32B 对齐导致实际拷贝量 > 有效数据量 | DataCopy 的 count 按对齐后计算，但 Compute 时只处理有效数据（`actualRows * D`）。不要将对齐 padding 参与计算 |
| 形态 β 排序后精度微差 | Sort 改变处理顺序，浮点累加顺序改变可能导致微差。若必须 bit-wise 确定，配合 P9 workspace 机制 |
| ReinterpretCast 后 buffer 类型混乱 | 建议给每个逻辑 buffer 显式定义 `LocalTensor<T>` 变量，不要直接操作 raw offset |
| MAXBUF 取值与芯片不匹配 | a3 可用约 192KB，a5 约 256KB。MAXBUF 不应 hardcode，建议从 TilingData 传入 `ubCapacity` |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P3 Playbook Completion]
Step 1: done (/tmp/p3_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: constraints: D=? rowsPerUB=? actualBytes=? aligned=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
