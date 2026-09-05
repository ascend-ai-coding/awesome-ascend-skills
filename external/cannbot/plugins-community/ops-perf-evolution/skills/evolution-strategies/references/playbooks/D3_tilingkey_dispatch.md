# D3 Playbook: TilingKey 驱动类型分发

> 本 Playbook 为**强制流程**。采纳 D3 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> D3 的核心是**通过 TilingKey + 模板编程，在编译期根据数据类型（FP16/BF16/FP32）分发到最优计算路径，避免运行时类型判断开销**。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/d3_locations.txt`：

```bash
# 1. 模板与类型分发
grep -n "template|constexpr|is_same|if.*constexpr|std::is_same" \
    shared/original/op_kernel/*.cpp > /tmp/d3_locations.txt
# 2. 数据类型配置
grep -n "DataType|DT_FLOAT|DT_FLOAT16|DT_BF16|float|half|bf16" \
    shared/original/op_kernel/*.cpp >> /tmp/d3_locations.txt
# 3. TilingKey 设置
grep -n "TilingKey|tilingKey|SetTilingKeyMode|FP16_MODE|FP32_MODE|BF16_MODE" \
    shared/original/op_kernel/*.cpp >> /tmp/d3_locations.txt
# 4. 当前类型处理
grep -n "switch.*type|if.*float|if.*half|if.*bf16|else.*float|else.*half" \
    shared/original/op_kernel/*.cpp >> /tmp/d3_locations.txt
# 5. 算子注册与输入
grep -n "Input|Output|OpRegistration|OP_TYPE|OP_NAME" \
    shared/original/op_kernel/*.cpp >> /tmp/d3_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **模板结构**：模板类/函数、if constexpr 分支
- **数据类型**：当前支持的数据类型（FP16/BF16/FP32/其他）
- **TilingKey 设置**：SetTilingKeyMode / TilingKey 赋值
- **当前类型处理**：switch / if-else / 模板特化
- **算子注册**：输入输出 DataType 配置

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 当前类型 | `?` (FP32/FP16/BF16) | 不变 | `?_kernel.cpp:L?` |
| 当前分发方式 | `?` (switch/if-else/无) | TilingKey + 模板 | `?_kernel.cpp:L?` |
| TilingKey | `?` (无/有) | 有 | `?_kernel.cpp:L?` |
| 模板类 | `?` (无/有) | 有 | `?_kernel.cpp:L?` |
| 差异化 RoundMode | `?` (无/有) | 有 | `?_kernel.cpp:L?` |
| 输入 DataType | `?` (单类型/多类型) | 多类型 | `?_kernel.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的分发粒度和类型数量，判断你的代码属于以下哪种形态：

- **形态 α — TilingKey + 模板类（最常见）**：通过 `SetTilingKeyMode` 在 tiling 阶段设置类型对应的 key，kernel 入口用模板参数分发。
- **形态 β — 模板函数 + if constexpr（简单场景）**：不需要 TilingKey，直接用 `if constexpr (std::is_same_v<T, float>)` 做编译期分支。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta`

### 3B. Canonical Template（形态 α — TilingKey + 模板类）

```cpp
// === 改造前（仅支持 FP32）===
class FakeQuantKernel {
public:
    __aicore__ inline void Compute(...) {
        // 只有 FP32 路径
        DataCopy(outputLocal, sumBufLocal, len);
    }
};

// === 改造后（TilingKey + 模板类分发）===
// Host 侧：配置多数据类型
template <typename T>
class FakeQuantKernel {
public:
    __aicore__ inline void Compute(...) {
        if constexpr (std::is_same_v<T, float>) {
            // FP32 路径：直接搬运
            DataCopy(outputLocal, sumBufLocal, len);
        } else if constexpr (std::is_same_v<T, half>) {
            // FP16 路径：CAST_NONE
            Cast(outputLocal, sumBufLocal, RoundMode::CAST_NONE, len);
        } else {
            // BF16 路径：CAST_RINT
            Cast(outputLocal, sumBufLocal, RoundMode::CAST_RINT, len);
        }
    }
};

// Host 侧：TilingKey 分发
void TilingFunc(...) {
    if (inputType == ge::DT_FLOAT) {
        tilingData.set_fp32Mode(1);
        SetTilingKeyMode(1);  // FP32_MODE
    } else if (inputType == ge::DT_FLOAT16) {
        tilingData.set_fp16Mode(1);
        SetTilingKeyMode(2);  // FP16_MODE
    } else if (inputType == ge::DT_BF16) {
        tilingData.set_bf16Mode(1);
        SetTilingKeyMode(3);  // BF16_MODE
    }
}
```

### 3C. Variant Notes（若是形态 β）

- **形态 β（模板函数 + if constexpr）**：
  当算子逻辑简单，不需要 tiling 阶段分发时：
  ```cpp
  template <typename T>
  __aicore__ inline void ComputeTyped(LocalTensor<T> input, ...) {
      if constexpr (std::is_same_v<T, float>) {
          // FP32: 原生计算
          Add(output, input, bias, count);
      } else if constexpr (std::is_same_v<T, half>) {
          // FP16: upcast → compute → downcast
          Cast(fp32Buf, input, RoundMode::CAST_NONE, count);
          Add(fp32Buf, fp32Buf, bias, count);
          Cast(output, fp32Buf, RoundMode::CAST_RINT, count);
      }
  }
  ```

- **与 A3 的协同**：A3（Rounding Mode）定义了不同精度的 Cast 策略，D3 的模板分支中需按 A3 的规范选择 RoundMode（FP32: NONE, FP16: NONE→RINT, BF16: NONE→RINT）。
- **与 D4 的边界**：D4（FP8/INT4）处理更低比特的量化输出类型，D3 处理 FP16/BF16/FP32 的常规类型分发。两者可叠加：D3 分发到 FP16/BF16，D4 进一步分发到 FP8/INT4。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: TilingKey 值必须唯一（FP32=1, FP16=2, BF16=3），不能重复
约束 2: 模板分支必须覆盖所有注册的数据类型，不能有遗漏
约束 3: if constexpr 是编译期分支，不能使用运行时变量做条件
约束 4: FP16/BF16 分支必须包含正确的 Cast 链（A1 + A3 协同）
约束 5: Host 侧的 DataType 配置必须与 TilingKey 分发逻辑一致
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `支持类型 = ?`, `TilingKey 值 = ?`
- `模板分支数 = ?`, `差异化路径数 = ?`
- `RoundMode 配置 = ?`（FP32: ?, FP16: ?, BF16: ?）
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 有模板类或模板函数
grep -cE "template.*class|template.*typename|if.*constexpr" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 2: 有 TilingKey 或 SetTilingKeyMode
grep -cE "TilingKey|SetTilingKeyMode|tilingKey" modified_files/op_kernel/*.cpp
# 期望: >= 1（或 note 中说明 "形态 β，仅用 if constexpr"）

# 检查 3: 有 is_same 或类型判断
grep -cE "is_same|std::is_same|if.*float|if.*half|if.*bf16" modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 4: 有差异化 RoundMode（不同分支不同 RoundMode）
grep -cE "CAST_NONE|CAST_RINT|CAST_ROUND" modified_files/op_kernel/*.cpp
# 期望: >= 2（至少两种不同的 RoundMode）

# 检查 5: 无运行时类型判断（switch on dtype 在 kernel 内）
grep -cE "switch.*type|if.*dtype|if.*dataType.*==" modified_files/op_kernel/*.cpp
# 期望: == 0（或 note 中说明 "host 侧判断"）
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：is_same_v 不存在 | 确认 C++17 支持。旧版本用 `std::is_same<T, float>::value` |
| 编译失败：TilingKey 重复 | 检查每个类型的 TilingKey 值是否唯一。FP32/BF16/FP16 不能共用同一个 key |
| 运行时：类型分发错误 | 检查 Host 侧 `SetTilingKeyMode` 与 kernel 入口的模板参数是否匹配 |
| 某些类型路径未实现 | 模板分支必须覆盖所有注册的数据类型。若有遗漏，补充 `else` 分支或 static_assert |
| FP16 分支精度差 | 确认 FP16 分支有 FP32 中间计算（A1）和正确的 RoundMode（A3）。不能直接用 FP16 计算 |
| if constexpr 写成 if | `if constexpr` 是编译期分支，若写成 `if` 会导致所有分支都被编译，可能报错 |
| 模板膨胀导致编译慢 | 每种类型生成一份代码。若类型太多（>4），考虑用形态 β 的函数模板替代类模板 |
| Host 侧 DataType 未更新 | 检查算子注册文件的 `.DataType({...})` 是否包含了所有目标类型 |
| BF16 分支遗漏 | BF16 常被遗漏。确认三种类型（FP32/FP16/BF16）都已实现 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[D3 Playbook Completion]
Step 1: done (/tmp/d3_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta, canonical/variant applied
Step 4: constraints: types=? tilingKeys=? branches=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
