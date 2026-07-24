# D1 Playbook: Mixed Precision Architecture (混合精度架构实操流程)

> 本 Playbook 为**强制流程**。采纳 D1 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步，禁止"看起来改了"就声明完成。
>
> D1 的核心是**在 Host 侧支持多 dtype 输入，在 Kernel 侧通过模板参数选择正确的计算精度链**。它不改变算子数学逻辑，只扩展数据类型支持范围并管理精度转换。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/d1_locations.txt`：

```bash
# 1. Host 侧当前支持的 DataType（通常是 op_host/*.cpp 或 op.cpp）
grep -n "DataType\|DT_FLOAT\|DT_FLOAT16\|DT_BF16\|DT_INT32" shared/original/op_host/*.cpp shared/original/op_host/*.h > /tmp/d1_locations.txt
# 2. Kernel 侧模板参数或硬编码类型
grep -n "template\|typename T\|half\|bfloat16_t\|float\|int32_t" shared/original/op_kernel/*.cpp shared/original/op_kernel/*.h >> /tmp/d1_locations.txt
# 3. 精度转换 Cast 点（FP16/BF16 ↔ FP32）
grep -n "Cast\|ToFloat\|ToHalf\|ToBfloat16\|calcType\|ComputeType" shared/original/op_kernel/*.cpp >> /tmp/d1_locations.txt
# 4. Host 侧 TilingKey 或 dtype 分发逻辑
grep -n "TilingKey\|dtype\|dataType\|GetInputDesc" shared/original/op_host/*.cpp >> /tmp/d1_locations.txt
# 5. 当前 UB 占用中是否依赖 sizeof(T)
grep -n "sizeof(T)\|sizeof(half)\|sizeof(float)\|sizeof(bfloat16_t)" shared/original/op_kernel/*.cpp >> /tmp/d1_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **Host DataType 定义**：当前支持的 dtype 列表 + 文件/行号
- **Kernel 类型定义**：模板参数 `T` 或硬编码 `float`/`half` 的位置
- **Cast 调用点**：所有精度转换的位置（文件 + 行号）
- **TilingKey / dtype 分发**：是否存在多 dtype 分支逻辑
- **sizeof(T) 使用**：UB 分配是否已参数化

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| Host DataType | `{DT_FLOAT}` | `{DT_FLOAT16, DT_FLOAT, DT_BF16}` | `?_host.cpp:L?` |
| Kernel 模板 | 无 / 硬编码 float | `template <typename T>` | `?_kernel.cpp:L?` |
| 计算类型 calcType | `float` | `std::conditional_t<std::is_same_v<T, half>, float, T>` | `?_kernel.cpp:L?` |
| Cast 模式 | 无 | FP16→FP32计算→FP16输出；BF16直接计算 | `?_kernel.cpp:L?` |
| UB 占用公式 | `tileSize * 4` | `tileSize * sizeof(T)` | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的 dtype 支持情况，判断你的代码属于以下哪种形态：

- **形态 α — 仅 FP32**：当前只支持 `DT_FLOAT`，需要全面添加 FP16/BF16 支持
- **形态 β — 已有 FP16 但精度链错误**：已有 `half` 支持，但计算时未提升到 FP32（直接用 FP16 计算导致精度损失）
- **形态 γ — 需多 dtype 同时支持**：Host 侧需要同时注册 FP16/FP32/BF16，Kernel 内用模板参数 `T` 统一处理

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 α — 最常见）

**Host 侧改造**（`op_host/*.cpp`）：
```cpp
// === 改造前（仅 FP32）===
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND});

// === 改造后（FP16 + FP32 + BF16）===
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .AutoContiguous();
```

**Kernel 侧改造**（`op_kernel/*.cpp`）：
```cpp
// === 改造前（硬编码 float）===
__aicore__ inline void Compute() {
    LocalTensor<float> xLocal = inQueue.DeQue<float>();
    LocalTensor<float> yLocal = outQueue.AllocTensor<float>();
    Add(yLocal, xLocal, xLocal, tileSize);   // float 计算
    outQueue.EnQue(yLocal);
}

// === 改造后（模板化 + 精度链管理）===
template <typename T>
__aicore__ inline void Compute() {
    using calcType = std::conditional_t<std::is_same_v<T, half>, float, T>;
    
    LocalTensor<T> xLocal = inQueue.DeQue<T>();
    LocalTensor<T> yLocal = outQueue.AllocTensor<T>();
    
    if constexpr (std::is_same_v<T, half>) {
        // FP16: 先 Cast 到 FP32 计算，再 Cast 回 FP16
        LocalTensor<calcType> xFp32 = tmpBuf.Get<calcType>();
        LocalTensor<calcType> yFp32 = tmpBuf.Get<calcType>();
        Cast(xFp32, xLocal, RoundMode::CAST_NONE, tileSize);
        Add(yFp32, xFp32, xFp32, tileSize);
        Cast(yLocal, yFp32, RoundMode::CAST_NONE, tileSize);
    } else {
        // FP32 / BF16: 直接以原类型计算
        LocalTensor<calcType> xCast = xLocal.template ReinterpretCast<calcType>();
        LocalTensor<calcType> yCast = yLocal.template ReinterpretCast<calcType>();
        Add(yCast, xCast, xCast, tileSize);
    }
    outQueue.EnQue(yLocal);
}
```

### 3C. Variant Notes（若是形态 β 或 γ）

- **形态 β（已有 FP16 但精度链错误）**：重点检查 `Add/Mul/Div` 等计算指令的输入类型。若输入是 `half` 但计算未提升 → 按 3B 的 `if constexpr (std::is_same_v<T, half>)` 分支插入 `Cast`。
- **形态 γ（多 dtype 需 TilingKey 分发）**：Host 侧在 `TilingFunc` 中读取输入 dtype，写入 `TilingData.dtype_flag`；Kernel 入口用 `if constexpr` 或函数重载分派。**不要**在 Kernel 内用运行时 `if (dtype == DT_FLOAT16)`，这会阻止编译器优化。
- **如果 UB 紧张**：`sizeof(half) = 2`，`sizeof(bfloat16_t) = 2`，`sizeof(float) = 4`。支持 FP16/BF16 后 tileSize 可翻倍（同 UB 下），但 Step 4 必须重新核算。
- **若算子含归约/累加**：FP16 累加必须提升到 FP32（否则误差累积）。在 `CalcType` 中强制 `std::conditional_t<std::is_same_v<T, half>, float, T>`。

## Step 4: 约束复核（防崩溃）

**精度链约束**：
```
FP16 输入 → 必须 Cast 到 FP32 计算 → Cast 回 FP16 输出
BF16 输入 → 可直接 BF16 计算（或可选 FP32 提升）
FP32 输入 → 直接 FP32 计算
INT32 输入 → 不参与混合精度，保持独立分支
```

**UB 占用变化**：
```
原 UB 占用 = tileSize × sizeof(float) = tileSize × 4
新 UB 占用 = tileSize × sizeof(T)
            = tileSize × 2  (FP16/BF16)
            = tileSize × 4  (FP32)
```
- 支持 FP16 后 **UB 占用减半**（同 tileSize 下），可支持更大 tile 或更多 buffer。
- 但形态 β 的 `Cast` 需要额外临时 buffer（`tmpBuf`），需预留 `tileSize × sizeof(float)` 空间。

**约束**：`新 UB 占用 + Cast 临时空间 ≤ UB_TOTAL × 0.8`

**在 implementation_note.txt "Playbook Step 4" 中报告具体计算**（每种 dtype 的占用 + 是否通过）。

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。禁止在 `implementation_note.txt` 中把失败 grep 标记为"不适用"。

```bash
# 检查 1: Host 侧 DataType 已扩展为至少 2 种（FP16 + FP32 + BF16）
grep -cE "DT_FLOAT16.*DT_FLOAT|DT_FLOAT.*DT_FLOAT16|DT_BF16" modified_files/op_host/*.cpp
# 期望: >= 1

# 检查 2: Kernel 侧存在模板参数 typename T
#（确保不是硬编码类型，而是参数化）
grep -cE "template\s*<.*typename T.*>" modified_files/op_kernel/*.cpp modified_files/op_kernel/*.h
# 期望: >= 1

# 检查 3: 存在精度链管理（Cast 或 std::conditional/calcType）
grep -cE "Cast.*CAST|calcType|std::conditional|ToFloat|ToHalf" modified_files/op_kernel/*.cpp
# 期望: >= 1（FP16 场景必须有 Cast 或 calcType 提升）

# 检查 4: UB 分配使用 sizeof(T) 而非硬编码 4
#（防止 UB 分配仍按 float 大小计算，导致 FP16 场景浪费或溢出）
grep -cE "sizeof\(T\)|sizeof\(half\)|sizeof\(bfloat16_t\)" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 5: 无裸 float/half 硬编码的 LocalTensor（应全用 T 或 calcType）
#（排除旧代码残留的硬编码类型）
grep -cE "LocalTensor\s*<\s*float\s*>|LocalTensor\s*<\s*half\s*>|LocalTensor\s*<\s*bfloat16_t\s*>" \
    modified_files/op_kernel/*.cpp
# 期望: == 0（形态 γ 下必须为 0；形态 α/β 允许 <= 2 处 Cast 相关的临时 tensor）
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：模板实例化错误 | 检查 `TilingData` 是否按 dtype 选择了正确的 kernel 特化；Host 侧 `REGISTER_OP` 需绑定对应模板实例 |
| 编译失败：UB overflow（仅 FP16 场景）| Cast 临时 buffer 额外占用空间；回 Step 4 复算，必要时 tileSize 减半 |
| 运行时：FP16 精度退化 | 检查 `calcType` 是否为 `float`；若直接用 `half` 做归约/累加，误差会快速累积 |
| 运行时：BF16 结果与 FP32 对不上（轻微差异）| BF16 尾数仅 7 位，与 FP32 对比时允许相对误差 1e-2 ~ 1e-3；若超出此范围，检查是否遗漏了 `calcType` 提升 |
| 性能提升 < 5% | 若算子原本是 compute_bound，dtype 变窄（FP16）可降低内存带宽但计算量不变；若瓶颈是 mte2_stall，FP16 应有显著收益 |
| Host 侧注册 dtype 后旧 case 报错 | `AutoContiguous()` 需同时注册；检查 `.DataType({...})` 和 `.Format({...})` 列表长度是否对齐 |
| TilingKey dtype 分发与 kernel 模板不匹配 | Host 侧 `TilingKey` 的 dtype 位必须与 Kernel 的 `template <typename T>` 实例一一对应；缺失的实例会导致运行时找不到 kernel |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[D1 Playbook Completion]
Step 1: done (/tmp/d1_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: UB calc: FP16=elements×2 FP32=elements×4 BF16=elements×2 ≤ UB_TOTAL×0.8: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
