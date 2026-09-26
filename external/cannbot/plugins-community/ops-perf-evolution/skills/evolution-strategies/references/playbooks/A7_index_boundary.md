# A7 Playbook: 索引与边界安全处理

> 本 Playbook 为**强制流程**。采纳 A7 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> A7 的核心是**在 Gather/Scatter/索引访问等操作中，对索引值做边界检查（clamp）或使用 INT32 替代 INT64 以减少带宽，防止越界访问导致 segfault 或错误结果**。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/a7_locations.txt`：

```bash
# 1. 索引操作
grep -n "indices|Indices|index|Index|Gather|gather|Scatter|scatter|offset|Offset|argmax|argmin" \
    shared/original/op_kernel/*.cpp > /tmp/a7_locations.txt
# 2. 索引数据类型
grep -n "INT64|int64_t|INT32|int32_t|DT_INT64|DT_INT32" \
    shared/original/op_kernel/*.cpp >> /tmp/a7_locations.txt
# 3. 当前边界处理
grep -n "clamp|Clamp|clip|Clip|boundary|Boundary|out.*bound|min.*max.*index" \
    shared/original/op_kernel/*.cpp >> /tmp/a7_locations.txt
# 4. 数据搬运（涉及索引的 DataCopy）
grep -n "DataCopy.*index|DataCopy.*offset|Gather.*DataCopy|Scatter.*DataCopy" \
    shared/original/op_kernel/*.cpp >> /tmp/a7_locations.txt
# 5. Shape 推导
grep -n "InferShape|shape.*out|output.*shape|dim.*size" \
    shared/original/op_kernel/*.cpp >> /tmp/a7_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **索引位置**：所有 `indices`/`index`/`Gather`/`Scatter` 调用位置
- **索引类型**：INT64 / INT32
- **当前边界处理**：clamp / 无 / 其他
- **数据搬运方式**：DataCopy / Gather / Scatter
- **Shape 推导**：InferShape / 动态 shape

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 索引类型 | `?` (INT64/INT32) | `alpha/beta` 见 3A | `?_kernel.cpp:L?` |
| 索引操作 | `?` (Gather/Scatter/直接索引) | 不变 | `?_kernel.cpp:L?` |
| 当前边界处理 | `?` (无/clamp) | clamp | `?_kernel.cpp:L?` |
| 数据搬运 | `?` (DataCopy/Gather) | 不变 | `?_kernel.cpp:L?` |
| Shape 推导 | `?` (静态/动态) | 精确推导 | `?_kernel.cpp:L?` |
| 越界兜底 | `?` (无/有) | 有（输出 0 或 clamp） | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的索引类型和操作模式，判断你的代码属于以下哪种形态：

- **形态 α — INT32 索引 + 边界 clamp（最常见）**：将 INT64 索引改为 INT32，减少 50% 索引传输带宽；在索引使用前用 `Maxs`/`Mins` clamp 到有效范围。
- **形态 β — 动态 Shape 精确推导**：在 `InferShape` 中精确计算 Gather/Scatter 的输出 shape，避免运行时 shape 不匹配。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta`

### 3B. Canonical Template（形态 α — INT32 索引 + 边界 clamp）

```cpp
// === 改造前（INT64 索引，无边界检查）===
__aicore__ inline void GatherNaive(LocalTensor<int64_t> indices,
                                    LocalTensor<float> input,
                                    LocalTensor<float> output,
                                    uint32_t indexCount, uint32_t inputSize) {
    for (uint32_t i = 0; i < indexCount; i++) {
        int64_t idx = indices.GetValue(i);
        // ❌ 风险：idx 可能越界（idx < 0 或 idx >= inputSize）
        // ❌ 风险：INT64 索引占用 8 字节，带宽翻倍
        output.SetValue(i, input.GetValue(idx));
    }
}

// === 改造后（INT32 索引 + clamp）===
__aicore__ inline void GatherSafe(LocalTensor<int32_t> indices,
                                   LocalTensor<float> input,
                                   LocalTensor<float> output,
                                   uint32_t indexCount, uint32_t inputSize) {
    // 预计算边界
    int32_t maxIdx = static_cast<int32_t>(inputSize - 1);
    
    for (uint32_t i = 0; i < indexCount; i++) {
        int32_t idx = indices.GetValue(i);
        
        // Step 1: clamp 到 [0, inputSize-1]
        // Maxs: idx = max(idx, 0)
        Maxs(idx, idx, 0, 1);
        // Mins: idx = min(idx, maxIdx)
        Mins(idx, idx, maxIdx, 1);
        
        // Step 2: 安全访问
        output.SetValue(i, input.GetValue(idx));
    }
}
```

### 3C. Variant Notes（若是形态 β）

- **形态 β（动态 Shape 精确推导）**：
  在 `InferShape` 或 tiling 阶段精确推导输出 shape：
  ```cpp
  // InferShapeForMultiScaleDeformableAttentionGrad
  OP_CHECK(grad_value_shape == value_shape,
      "grad_value shape must match value shape");
  OP_CHECK(grad_sampling_loc_shape == sampling_loc_shape,
      "grad_sampling_loc shape must match sampling_loc shape");
  OP_CHECK(grad_attn_weight_shape == attn_weight_shape,
      "grad_attn_weight shape must match attn_weight shape");
  ```

- **与 A8 的边界**：A8（量化专用精度）处理量化边界 [quant_min, quant_max]，A7 处理索引边界 [0, size-1]。两者独立但常一起出现：A7 保证索引安全，A8 保证量化值安全。
- **与 D3 的协同**：D3（TilingKey 类型分发）决定索引的数据类型（INT32/INT64），A7 在已确定的类型上做边界检查。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: Maxs/Mins 的 scalar 参数必须是 int32_t，不能是浮点
约束 2: clamp 范围必须是 [0, size-1]，不是 [0, size]
约束 3: 若索引可能为负数，Maxs 的 lower bound 必须为 0
约束 4: 若使用 INT32 替代 INT64，需确认上层框架保证索引在 INT32 范围内（< 2^31-1）
约束 5: Gather 越界兜底可用 clamp 或输出 0。Scatter 越界应跳过写入
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `索引类型 = ?`, `索引范围 = [?, ?]`
- `inputSize = ?`, `clamp 范围 = [?, ?]`
- `越界处理 = ?`（clamp/输出0/跳过）
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 有 Maxs/Mins 或 clamp 操作
grep -cE "Maxs|Mins|clamp|Clip" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 有 INT32 索引（若形态 α）
grep -cE "int32_t|DT_INT32|INT32" modified_files/op_kernel/*.cpp
# 期望: >= 1（或 note 中说明 "形态 β，仅 InferShape"）

# 检查 3: 无裸露的 INT64 直接索引（kernel 内）
grep -cE "int64_t.*GetValue|INT64.*index|indices.*int64" modified_files/op_kernel/*.cpp
# 期望: == 0（或 note 中说明 "host 侧保留 INT64"）

# 检查 4: 有 Shape 推导或边界检查
grep -cE "InferShape|inputSize|maxIdx|size.*-.*1" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: Gather/Scatter 操作存在
grep -cE "Gather|gather|Scatter|scatter|GetValue|SetValue" modified_files/op_kernel/*.cpp
# 期望: >= 1
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：Maxs/Mins 不存在 | 确认 CANN 版本。旧版本用 `Compare` + `Select` 组合实现 clamp |
| 运行时：索引仍为 INT64 | 检查 DataType 配置。Host 侧 `.DataType({ge::DT_INT32})` 需同步修改 |
| 运行时：clamp 后仍越界 | 检查 Mins 的上界是否为 `size - 1` 而非 `size`。`size` 会越界 |
| INT32 范围溢出 | 确认索引 < 2^31-1（约 21 亿）。若可能超过，保留 INT64 并在 kernel 内截断 |
| Scatter 越界写入 | Scatter 的 clamp 应跳过写入（而非写入边界位置），避免污染有效数据 |
| 负索引未处理 | `Maxs(idx, idx, 0)` 必须存在。若遗漏，负索引会越界 |
| 动态 shape 不匹配 | 检查 InferShape 是否精确计算了 Gather/Scatter 的输出维度 |
| 性能下降 | INT32 减少 50% 带宽，通常提升性能。若下降，检查是否引入了额外同步 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[A7 Playbook Completion]
Step 1: done (/tmp/a7_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta, canonical/variant applied
Step 4: constraints: index_type=? clamp_range=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
