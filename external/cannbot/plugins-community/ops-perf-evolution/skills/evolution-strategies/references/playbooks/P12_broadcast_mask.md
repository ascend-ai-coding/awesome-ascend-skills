# P12 Playbook: Broadcast & Mask Operations (广播与掩码操作实操流程)

> 本 Playbook 为**强制流程**。采纳 P12 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步，禁止"看起来改了"就声明完成。
>
> P12 的核心是**用 AscendC 向量 API（Duplicate、Select、SelectWithBytesMask 等）替代手动的标量循环或 if-else 掩码逻辑**，减少标量开销并提升内存效率。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/p12_locations.txt`：

```bash
# 1. 手动 mask 逻辑（if-else、条件赋值、按元素 mask）
grep -n "if.*mask\|Mask\|Select\|maskGm\|maskLocal" shared/original/op_kernel/*.cpp > /tmp/p12_locations.txt
# 2. 标量广播痕迹（标量参与向量计算、重复赋值、Scalar→Vector）
grep -n "Duplicate\|GetValue\|Scalar.*Local\|Broadcast\|\.GetValue(" \
    shared/original/op_kernel/*.cpp >> /tmp/p12_locations.txt
# 3. 输入 shape 差异（broadcast 场景通常有多个不同 shape 的输入）
grep -n "inputShape\|outputShape\|GetShapeSize\|ndim\|dimCount" \
    shared/original/op_kernel/*.cpp shared/original/op_host/*.cpp >> /tmp/p12_locations.txt
# 4. 已有 SelectWithBytesMask / Duplicate / CompareScalar 等 API 使用
grep -n "SelectWithBytesMask\|CompareScalar\|Select\|Duplicate" \
    shared/original/op_kernel/*.cpp >> /tmp/p12_locations.txt
# 5. 循环内的标量操作（for + 标量 mask 判断是反模式）
grep -n "for.*mask\|for.*bool\|if.*maskLocal\[" shared/original/op_kernel/*.cpp >> /tmp/p12_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **Mask 逻辑位置**：所有 if-else / 条件赋值 / mask 数组使用的文件 + 行号
- **标量广播点**：标量参与向量计算的位置
- **输入 shape**：各输入 tensor 的 shape 定义
- **已有 API**：是否已使用 Select/Duplicate 等高阶 API
- **标量循环**：for 循环内按元素判断 mask 的位置（重点改造对象）

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| Mask 类型 | `?` (bool / float / int) | `bool`（1字节，最高效） | `?_kernel.cpp:L?` |
| Mask 应用方式 | `?` (if-else / Add / 无) | `SelectWithBytesMask` 或 `Select` | `?_kernel.cpp:L?` |
| 广播模式 | `?` (标量→向量 / 小维度→大维度) | `Duplicate` 或 `Broadcast` | `?_kernel.cpp:L?` |
| 输入 shape 对齐 | `?` | 广播维度对齐到 32B | `?_tiling.cpp:L?` |
| mask buffer 大小 | `? bytes` | `tileSize * sizeof(bool)` | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的 mask 和广播模式，判断你的代码属于以下哪种形态：

- **形态 α — 标量广播**：算子需要将一个标量（如学习率、scale）广播为向量参与计算
- **形态 β — bool mask 应用**：算子有 mask 输入，需要根据 mask 选择性替换值（如 Softmax mask、attention mask）
- **形态 γ — 多维度广播**：两个输入 tensor shape 不同，需要在一个或多个维度上广播对齐

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 β — bool mask 应用，最常见）

```cpp
// === 改造前（手动 if-else 或 float mask + Add，低效）===
__aicore__ inline void ApplyMask(LocalTensor<float> data, LocalTensor<float> mask) {
    for (uint32_t i = 0; i < tileSize; i++) {
        if (mask.GetValue(i) == 0.0f) {     // 标量循环，极慢
            data.SetValue(i, MASK_VAL);      // MASK_VAL = -10000.0f
        }
    }
}
// 或：float mask + Add（mask 占 4 字节，内存浪费）
Add(data, data, mask, tileSize);   // mask: 0 或 -inf

// === 改造后（SelectWithBytesMask，向量级一次完成）===
__aicore__ inline void ApplyMask(LocalTensor<float> data, LocalTensor<uint8_t> mask) {
    LocalTensor<float> maskValLocal = maskValBuf.Get<float>();
    Duplicate(maskValLocal, static_cast<float>(MASK_VAL), tileSize);
    
    // SelectWithBytesMask: mask[i] != 0 ? maskValLocal[i] : data[i]
    SelectWithBytesMask(data, maskValLocal, data, mask, tileSize);
}
```

### 3C. Variant Notes（若是形态 α 或 γ）

- **形态 α（标量广播）**：用 `Duplicate(dst, scalar, count)` 将标量广播为向量，然后参与向量计算。
  ```cpp
  LocalTensor<float> lrLocal = lrBuf.Get<float>();
  Duplicate(lrLocal, lrScalar, tileSize);   // 标量 → 向量
  Mul(gradLocal, gradLocal, lrLocal, tileSize);  // 向量-向量乘法
  ```
- **形态 γ（多维度广播）**：用 `TilingData` 在 Host 侧计算广播维度（哪个轴需要重复），Kernel 内用 `Duplicate` 或 `DataCopy` stride 模式实现。若维度差异复杂，考虑 P26 stride 搬运策略配套。
- **mask 是 float 而非 bool**：若输入 mask 是 float（0/1 或 0/-inf），先用 `CompareScalar` 转为 bool：`CompareScalar(maskBool, maskFloat, 0, CMPMODE::NE, tileSize)`，再应用 `SelectWithBytesMask`。这样 mask buffer 从 4 字节降到 1 字节。
- **与 P12  synergizes_with**：P12 无直接协同策略，但与 P67（Counter 模式）协同：mask 应用后可用 Counter 模式减少 repeat/tail 计算。
- **如果 mask 只在部分 tile 上有值**：不需要每次 Compute 都 ApplyMask。在 Host 侧判断 `hasMask` 标志，Kernel 内 `if constexpr` 或分支跳过。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: mask 长度 == data 长度（或能广播对齐到 data 长度）
约束 2: mask 类型为 bool / uint8_t（1 字节），不能是 float（4 字节）
约束 3: SelectWithBytesMask 的 dst/src0/src1 必须在同一块 UB 内且 32B 对齐
约束 4: Duplicate 的 count ≤ tileSize，不能超出 dst tensor 容量
约束 5: mask buffer 大小 = tileSize × 1 byte ≤ UB_TOTAL × 0.2（mask 不应占过大 UB）
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**（mask 大小、对齐状态、是否通过）。

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。

```bash
# 检查 1: 已使用 SelectWithBytesMask 或 Select 或 Duplicate（替代了手动循环）
grep -cE "SelectWithBytesMask|Select\s*\(|Duplicate\s*\(" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 无手动 for+if mask 循环（标量循环是反模式）
grep -cE "for\s*\([^)]*\)\s*\{[^}]*if[^}]*mask|for.*maskLocal|if.*mask.*GetValue" \
    modified_files/op_kernel/*.cpp
# 期望: == 0

# 检查 3: mask 类型为 bool / uint8_t（1 字节），不是 float
grep -cE "LocalTensor\s*<\s*(bool|uint8_t)\s*>.*mask|mask.*bool\b|mask.*uint8_t" \
    modified_files/op_kernel/*.cpp
# 期望: >= 1（若 mask 从外部传入 float，需有 CompareScalar 转换痕迹）

# 检查 4: 无 float mask 直接参与 Add（4 字节 mask 是内存浪费）
grep -cE "Add.*mask.*float|mask.*LocalTensor\s*<\s*float\s*>" modified_files/op_kernel/*.cpp
# 期望: == 0（除非形态 γ 的多维广播需要 float mask）

# 检查 5: mask buffer 通过 InitBuffer 分配（不是栈变量或硬编码指针）
grep -cE "InitBuffer.*mask|maskBuf\|mask.*Buf\.Get" modified_files/op_kernel/*.cpp
# 期望: >= 1
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：SelectWithBytesMask 类型不匹配 | `SelectWithBytesMask` 要求 mask 为 `uint8_t`，dst/src0/src1 为同类型向量。检查类型是否一致 |
| 运行时：mask 效果相反（该 mask 的位置没被 mask）| `SelectWithBytesMask(dst, src0, src1, mask)` 语义：mask != 0 ? src0 : src1。确认 src0/src1 的顺序 |
| 运行时：精度对不上（mask 位置不是预期值）| `Duplicate(maskValLocal, MASK_VAL, tileSize)` 的 MASK_VAL 必须是 scalar 变量，不能是常量表达式导致类型推导错误 |
| 性能与 baseline 持平 | 若 mask 应用次数很少（如只在 1% 的 tile 上），手动 if-else 可能比向量 API 更省。检查 mask 覆盖率 |
| float mask 转 bool 后内存没节省 | `CompareScalar` 输出仍是 uint8_t，但需要额外 buffer。确保旧 float mask buffer 被释放或复用 |
| 广播维度对不上 | Host 侧 shape 检查必须保证广播维度是合法的（如 1→N）。非法广播会导致越界 |
| mask 长度与 data 长度不一致 | 检查 Host 侧是否按最小公倍数分配了 mask buffer。若 mask shape 与 data shape 不同，需 tiling 阶段对齐 |
| Duplicate 的 scalar 是编译期常量导致模板推导失败 | 用 `static_cast<float>(MASK_VAL)` 显式指定类型，避免模板推导歧义 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[P12 Playbook Completion]
Step 1: done (/tmp/p12_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: constraints: mask_len==data_len; mask_dtype=bool/uint8; dst/src0/src1 32B_aligned; mask_buf ≤ UB_TOTAL×0.2: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
