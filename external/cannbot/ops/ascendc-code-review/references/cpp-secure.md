# CANN C++ 安全编码规范

<适用>
语言: C++
侧别: All, Tiling
领域: true
默认启用: true
适用场景: Tiling 侧（Host 侧）和 Kernel 侧（Device 侧）的 C++ 安全编码规范
不适用场景: Python 代码（见 python-secure.md）、编译链接安全（见 compile-secure.md）
介绍: CANN C++ 安全编码规范，32 条条款覆盖 10 个安全类别
类别(All): 总体原则(类型安全/内存安全/未定义行为), 数值运算安全(整数溢出/回绕/除零), 内存与指针安全(未初始化变量/数组越界/sizeof误用/指针判空), 输入验证(外部输入合法性/内存操作长度校验), 类与对象安全(非trivially copyable对象位操作), 标准库安全(敏感信息清零/结构体兼容性)
类别(Tiling): 内存与指针安全(资源句柄释放后赋值/字符串空间), 资源管理(申请成功判断/泄露/new-delete配对/分配错误处理), 安全函数使用(安全函数库/destMax参数/返回值检查), 标准库安全(空指针string/c_str指针保存), LOG规范(空指针/格式化占位符)
> **说明**：安全编码红线规范，所有代码必须 100% 遵守。条款标注适用范围：`[适用: All]` / `[适用: Tiling]`
</适用>

## 快速索引

### 两者都适用 `[适用: All]`（17 条）

| 规范编号 | 规范名称 | 类别 | 严重级别 |
|---------|---------|------|---------|
| 1.1 | 保证静态类型安全 | 总体原则 | 高 |
| 1.2 | 保证内存安全 | 总体原则 | 高 |
| 1.3 | 禁止使用未定义行为 | 总体原则 | 高 |
| 2.1 | 有符号整数运算不溢出 | 数值安全 | 高 |
| 2.2 | 无符号整数运算不回绕 | 数值安全 | 高 |
| 2.3 | 除法/余数运算除零保护 | 数值安全 | 高 |
| 3.1 | 禁止使用未初始化的变量 | 内存安全 | 高 |
| 3.3 | 数组索引校验 | 内存安全 | 高 |
| 3.4 | 禁止 sizeof 指针 | 内存安全 | 中 |
| 3.5 | 指针使用前判空 | 内存安全 | 高 |
| 4.1 | 外部输入合法性校验 | 输入验证 | 高 |
| 4.2 | 内存操作长度校验 | 输入验证 | 高 |
| 9.1 | 禁止逐位操作非 trivially copyable 对象 | 类与对象 | 中 |
| 10.3 | 敏感信息使用后清零 | 标准库 | 高 |
| 10.4 | 结构体字段末尾添加 | 标准库 | 中 |
| 10.5 | 接口变更考虑兼容性 | 标准库 | 中 |

### 仅 Tiling 适用 `[适用: Tiling]`（15 条）

| 规范编号 | 规范名称 | 类别 | 严重级别 |
|---------|---------|------|---------|
| 3.2 | 资源释放后指针置新值 | 内存安全 | 中 |
| 3.6 | 字符串存储有足够空间 | 内存安全 | 高 |
| 5.1 | 资源申请后判断是否成功 | 资源管理 | 高 |
| 5.2 | 资源泄露防护 | 资源管理 | 高 |
| 5.3 | new/delete 配对使用 | 资源管理 | 高 |
| 5.4 | new 操作符错误处理 | 资源管理 | 高 |
| 8.1 | 使用安全函数替代危险函数 | 安全函数 | 高 |
| 8.2 | 正确设置安全函数 destMax 参数 | 安全函数 | 高 |
| 8.3 | 检查安全函数返回值 | 安全函数 | 高 |
| 10.1 | 禁止从空指针创建 std::string | 标准库 | 高 |
| 10.2 | 不要保存 c_str/data 指针 | 标准库 | 中 |
| 11.1 | LOG API 禁止传入空指针 | LOG API 安全 | 高 |
| 11.2 | LOG API 参数必须与格式化占位符逐位一致（数量、类型、顺序） | LOG API 安全 | 高 |
| 11.3 | LOG API 禁止传入已释放内存的指针 | LOG API 安全 | 高 |
| 11.4 | LOG 消息英语行文语法正确、表意清晰 | LOG API 规范 | 低 |

---

### 1. 总体原则

#### 1.1 保证静态类型安全 `[适用: All]`

> **Kernel 侧说明**：Ascend C 模板类需注意类型转换（如 half ↔ float）和范围错误（FP16 溢出）。

C++应该是静态类型安全的，这样可以减少运行时的错误，提升代码的健壮性。但是由于C++存在下面的特性，会破坏C++静态类型安全，针对这部分特性要仔细处理：

- 联合体
- 类型转换
- 缩窄转换
- 类型退化
- 范围错误
- void* 类型指针

可以通过约束这些特性的使用，或者使用C++的新特性，例如std::variant（C++17）、std::span（C++20）等来解决这些问题，提升C++代码的健壮性。

#### 1.2 保证内存安全 `[适用: All]`

> **Kernel 侧说明**：Ascend C 使用 UB（Unified Buffer）和 GM（Global Memory），需要通过 `DataCopy` API 安全访问，避免越界和未初始化访问。

C++语言的内存完全由程序员自己控制，所以在操作内存的时候必须保证内存安全，防止出现内存错误：

- 内存越界访问
- 释放以后继续访问内存
- 解引用空指针
- 内存没有初始化
- 把指向局部变量的引用或者指针传递到了函数外部或者其他线程中
- 申请的内存或者资源没有及时释放

建议使用更加安全的C++的特性，比如RAII，引用，智能指针等，来提升代码的健壮性。

#### 1.3 禁止使用编译器"未定义行为" `[适用: All]`

遵循ISO C++标准，标准中未定义的行为禁止使用。对于编译器实现的特性或者GCC等编译器提供的扩展特性也需要谨慎使用，这些特性会降低代码的可移植性。

---

### 2. 数值运算安全

#### 2.1 确保有符号整数运算不溢出 `[适用: All]`

> **Kernel 侧说明**：Kernel 中使用 `uint32_t` 等固定宽度类型进行循环索引和 Buffer 偏移计算，需防止溢出。

**【描述】**
有符号整数溢出是未定义的行为。出于安全考虑，对外部数据中的有符号整数值在如下场景中使用时，需要确保运算不会导致溢出：

- 指针运算的整数操作数(指针偏移值)
- 数组索引
- 变长数组的长度(及长度运算表达式)
- 内存拷贝的长度
- 内存分配函数的参数
- 循环判断条件

在精度低于int的整数类型上进行运算时，需要考虑整数提升。程序员还需要掌握整数转换规则，包括隐式转换规则，以便设计安全的算术运算。

**乘法示例（int32_t 乘法溢出）：**

```cpp
// 错误写法 — 两个 int32_t 相乘，结果可能超出 int32_t 范围
int32_t calcHeightAlign = GetAlignedSize(...);  // 对齐后高度，可达 65536
int32_t calcWidth = GetWidth(...);              // 宽度，可达 65536
int32_t size = calcHeightAlign * calcWidth;     // 65536 × 65536 = 4,294,967,296 溢出！

// 正确写法 — 提升为 int64_t 计算
int64_t size = static_cast<int64_t>(calcHeightAlign) * calcWidth;
```

**取反示例（INT64_MIN 取反溢出，红线问题）：**

```cpp
// 错误写法 — delta 取 INT64_MIN 时，-delta 溢出
int64_t delta = input2 - input1;       // 可能为 INT64_MIN = -9223372036854775808
int64_t absDelta = -delta;             // -(-9223372036854775808) = 9223372036854775808 > INT64_MAX!
// 有符号整数溢出是未定义行为（C++ 红线）

// 正确写法 — 转换为无符号类型后再求绝对值
uint64_t absDelta = (delta < 0) ? static_cast<uint64_t>(-delta) : static_cast<uint64_t>(delta);
```

**多维连乘示例（多维 shape 连续累乘溢出）：**

```cpp
// 错误写法 — 多维 shape 用 int32_t 连乘，极易溢出
int32_t totalSize = dim0 * dim1 * dim2 * dim3 * dim4;
// dim0=1024, dim1=1024, dim2=128, dim3=64 时积 ≈ 8.6 × 10^9 > INT32_MAX

// 正确写法 — 使用 int64_t 并提前提升
int64_t totalSize = static_cast<int64_t>(dim0) * dim1 * dim2 * dim3 * dim4;
```

**【检视策略 — 工具驱动】**

核心流程：运行 check_bounds.py → 读取敏感性分析 → 按行动指引验证关键边界 → 必要时重跑 → 收敛结论

**Step 1 — 提取表达式与类型**

扫描代码，提取每个有符号算术表达式。识别操作数的 C++ 类型。

**Step 2 — 首次工具运行**

为操作数设定初始边界后运行 check_bounds.py：

边界设定规则：
① 编译期常量 / 代码守卫 (if/assert) → 使用精确值
② 从赋值链推导 → 使用推导范围
③ 无代码证据 → 使用合理保守值（禁止用类型全范围——那必定违规，无意义）

禁止行为：
- 虚构变量关系作为安全证据（如声称 "X ≤ Y" 但找不到对应代码行）
- 用类型标签代替边界（"int64_t 所以够大不会溢出"——int64_t 的值可以是 1）

```bash
python3 {skill_base}/scripts/check_bounds.py \
  --expr "{表达式}" \
  --vars "a=int32_t:0:47" "b=int32_t:3:3" "c=int64_t:100:1000000" \
  --check overflow
```

表达式中的 C++ 写法（`func()`、`a->b`）直接用作变量名。

**Step 3 — 按工具输出行动**

工具输出包含「边界敏感性分析」逐变量标注安全临界值，以及「行动指引」分步指令。**严格按行动指引执行，不要跳过。**

【输出 SAFE】
  看「最敏感变量」及余量：找出余量最小的那个变量
    余量 > 10x 临界值 → 安全余量充足，PASS
    余量 ≤ 10x → 回代码核实该变量的边界是否来自 A/B 级代码证据
      有证据 → PASS。无证据 → 向不利方向放宽边界重跑，重跑后判断

【输出 VIOLATION】
  看反例中「触及上限/下限」的变量：
    来自 constexpr/守卫 (A 级) → 边界可靠，确认 FAIL
    来自推测 (B/C 级) → Grep 找该变量的真实限定值
      找到 → 修正边界重跑。找不到 → SUSPICIOUS + 标注边界不确定

**Step 4 — 收敛（最多 1 次重跑）**

重跑后按 Step 3 逻辑判断。仍不确定 → SUSPICIOUS + 标注关键变量及缺失的代码证据。

---

#### 2.2 确保无符号整数运算不回绕 `[适用: All]`

> **Kernel 侧说明**：Kernel 中大量使用 `uint32_t` 进行 tileLength、blockLength 计算，需防止回绕。

**【描述】**
涉及无符号操作数的计算永远不会溢出，因为超出无符号整数类型表示范围的计算结果会按照（结果类型可表示的最大值 + 1）的数值取模。这种行为更多时候被非正式地称为无符号整数回绕。

**乘法示例（uint32_t 乘法回绕后再 cast uint64_t——值已经错了）：**

```cpp
// 错误写法 — 乘法在 uint32_t 完成，回绕发生后才 cast 到 uint64_t，无法恢复
uint32_t blockSize = 65536;    // 来自 TilingData
uint32_t strideKV = 65536;     // 来自 TilingData
uint64_t result = blockSize * strideKV;
// blockSize * strideKV 在 uint32_t 空间计算：65536 × 65536 = 4,294,967,296 > UINT32_MAX
// 实际结果: (65536 × 65536) mod 2^32 = 0 → 回绕后的 0 再 cast 到 uint64_t = 0

// 正确写法 — 乘法前至少一个操作数提升为 uint64_t
uint64_t result = static_cast<uint64_t>(blockSize) * strideKV;
```

**减法示例（uint32_t 减法回绕——结果用作数组索引）：**

```cpp
// 错误写法 — aivIdx * singleCoreSize 可能大于 totalOutputSize，减法回绕
uint32_t tailSize = totalOutputSize - aivIdx * singleCoreSize;
// totalOutputSize=100, aivIdx=47, singleCoreSize=3:
//   47 × 3 = 141, 100 - 141 按 uint32_t 计算 = 4294967255（回绕）
//   tailSize 被误认为合法大小，后续 DataCopy 搬运 4GB 数据 → 越界崩溃

// 正确写法 — 先判断大小关系，或使用 int64_t 中间结果
int64_t tailSizeSigned = static_cast<int64_t>(totalOutputSize) - 
                         static_cast<int64_t>(aivIdx) * singleCoreSize;
uint32_t tailSize = (tailSizeSigned > 0) ? static_cast<uint32_t>(tailSizeSigned) : 0;
```

**类型混合示例（size_t 与 int64_t 混合运算——负数回绕成极大值）：**

```cpp
// 错误写法 — N_ALIGN 是 size_t 常量（无符号），numIters 是 int64_t
// 按 C++ 整型提升规则 int64_t → size_t，负数变成极大正数
constexpr size_t N_ALIGN = 128;
int64_t normSize = N_ALIGN * DOUBLE_SIZE * numIters * T * n0;
// 若 numIters 为 0 或负值，提升为 size_t 后回绕成 2^64-127 级别的极大值
// 再经 SetDim 传出，得到非预期的 shape，后续所有计算均错

// 正确写法 — 统一为有符号类型
constexpr int64_t N_ALIGN = 128;
int64_t normSize = N_ALIGN * DOUBLE_SIZE * numIters * T * n0;
```

**【检视策略 — 工具驱动】**

核心流程：运行 check_bounds.py → 读取敏感性分析 → 按行动指引验证关键边界 → 必要时重跑 → 收敛结论

**Step 1 — 提取表达式与类型**

扫描代码，提取每个无符号算术表达式（减法、乘法、混合运算）。识别操作数的 C++ 类型。

**Step 2 — 首次工具运行**

为操作数设定初始边界后运行 check_bounds.py：

边界设定规则：
① 编译期常量 / 代码守卫 (if/assert) → 使用精确值
② 从赋值链推导 → 使用推导范围
③ 无代码证据 → 使用合理保守值（禁止用类型全范围——那必定违规，无意义）

禁止行为：
- 虚构变量关系作为安全证据（如声称 "a ≥ b 恒成立" 但找不到对应代码行）
- 用类型标签代替边界（"uint64_t 所以够大不会回绕"——uint64_t 的值可以是 0）

```bash
python3 {skill_base}/scripts/check_bounds.py \
  --expr "{表达式}" \
  --vars "a=uint32_t:0:47" "b=uint32_t:3:3" "c=int64_t:100:1000000" \
  --check wraparound
```

表达式中的 C++ 写法（`func()`、`a->b`）直接用作变量名。

**Step 3 — 按工具输出行动**

工具输出包含「边界敏感性分析」逐变量标注安全临界值，以及「行动指引」分步指令。**严格按行动指引执行，不要跳过。**

【输出 SAFE】
  看「最敏感变量」及余量：找出余量最小的那个变量
    余量 > 10x 临界值 → 安全余量充足，PASS
    余量 ≤ 10x → 回代码核实该变量的边界是否来自 A/B 级代码证据
      有证据 → PASS。无证据 → 向不利方向放宽边界重跑，重跑后判断

【输出 VIOLATION】
  看反例中「触及上限/下限」的变量：
    来自 constexpr/守卫 (A 级) → 边界可靠，确认 FAIL
    来自推测 (B/C 级) → Grep 找该变量的真实限定值
      找到 → 修正边界重跑。找不到 → SUSPICIOUS + 标注边界不确定

**Step 4 — 收敛（最多 1 次重跑）**

重跑后按 Step 3 逻辑判断。仍不确定 → SUSPICIOUS + 标注关键变量及缺失的代码证据。

---

#### 2.3 确保除法和余数运算不会导致除以零的错误 `[适用: All]`

> **Kernel 侧说明**：
> - Kernel 中的除法/取余运算，按除数来源分类判定（见下方 Step 2）。
> - **硬件 API 返回值**（`GetBlockNum()`、`GetTaskRation()` 等）和 **TilingData 字段**（`tilingData->*`）作除数时，适用 Kernel 侧排除规则，无需零值守卫。
> - 仅对 Kernel 内部独立计算的运行时变量（非白名单 API、非 TilingData）要求零值守卫。

**【检视策略】**

**Step 1 — 识别除法/取余运算**

扫描代码中所有 `/` 和 `%` 运算符（含 `CeilDiv`/`CeilDivide` 工具函数调用），提取除数表达式。

**Step 2 — 除数来源分类**

| 优先级 | 除数来源 | 识别方法 | 信任等级 |
|--------|---------|---------|---------|
| P0 | 编译期常量 | `constexpr` 声明、字面量、`AscendC::BLOCK_CUBE` 等框架常量 | 自动 PASS |
| P1 | 硬件 API 返回值 | 白名单 API 直接调用或赋值链可追溯 | Kernel 侧自动 PASS |
| P2 | TilingData 字段 | `tilingData->xxx` / `tilingData_.xxx` | Kernel 侧自动 PASS |
| P3 | 外部输入 | `shape->GetDim()`、`context->GetAttr()`、`GetActualSeqLen()` | 严格：必须有守卫 |
| P4 | 设计过程参数 | Tiling/Kernel 内部多步计算的中间值 | 严格：必须有守卫 |

**硬件 API 白名单（P1）**：

| 侧别 | 白名单 API | 典型变量名 |
|------|-----------|-----------|
| Kernel | `AscendC::GetBlockNum()` | `coreNum`, `blockNum_` |
| Kernel | `AscendC::GetTaskRation()` | `taskRation`, `coreRation` |
| Kernel | `AscendC::GetSubBlockNum()` | `bn`, `subBlockNum` |
| Tiling | `ascendcPlatform.GetCoreNumAic()` | `aicNum` |
| Tiling | `ascendcPlatform.GetCoreNumAiv()` | `aivNum` |
| Tiling | `ascendcPlatform.GetCoreNum()` | `coreNum` |
| Tiling | `ascendcPlatform.GetCoreMemSize(...)` | `ubSize`, `l1Size` 等 |

> **适用条件**：除数直接来自上述 API 返回值，或赋值链可追溯到上述 API 的变量。若除数经过算术运算（如 `GetBlockNum() - 1`），需另行分析运算结果是否可能为零。

**Step 3 — 按来源判定**

- **P0（编译期常量）**：值非零 → PASS。值为零 → FAIL。
- **P1（硬件 API）**：Kernel 侧自动 PASS（见排除规则）。Tiling 侧必须有零值守卫（见严格模式）。
- **P2（TilingData）**：Kernel 侧自动 PASS。Tiling 侧作为中间值按 P3/P4 处理。
- **P3（外部输入）**：必须有有效守卫模式之一 → 无守卫则 FAIL。
- **P4（设计过程参数）**：必须有有效守卫，或可追溯到 P0/P1 的非零值 → 否则 FAIL。

**边界收集**（P3/P4 需要时）：按 SEC-2.1 的 Step 2 方法收集除数边界，按 Step 4 判定表做判定。

**【Kernel 侧排除规则】**

以下情况在 Kernel 侧自动排除，无需零值守卫：

| 排除条件 | 参数模式示例 | 排除原因 |
|---------|-------------|----------|
| 除数来自硬件 API 白名单 | `GetBlockNum()`, `GetTaskRation()` | 芯片出厂固定非零，异常场景由 Tiling 侧兜底 |
| 除数来自 TilingData | `tilingData->tileSize`, `tilingData->coreNum` | Tiling 阶段已校验非零 |
| 编译期常量 | `constexpr uint32_t BLOCK = 32` | 编译期固定非零 |

**判定方法**：
- 除数表达式直接匹配白名单 API 调用 → 直接判定 PASS
- 除数变量赋值链可追溯到白名单 API 或 `tilingData->xxx` → 直接判定 PASS
- 除数为 `constexpr` 且值非零 → 直接判定 PASS

**【Kernel 侧需校验场景】**

以下情况在 Kernel 侧仍需零值守卫：

| 校验条件 | 参数来源 | 代码模式 |
|---------|---------|----------|
| Kernel 内部计算的中间值 | 非 TilingData、非硬件 API | `if (computedDivisor == 0) { return; }` |
| 动态序列长度 | `GetActualSeqLen()` 运行时获取 | `if (actS1Size == 0) { return; }` |
| 条件分支中的计算值 | 依赖运行时条件的派生值 | `if (curMode == X && div != 0) { ... }` |

**【有效守卫模式】**

以下 6 种模式视为有效的零值守卫（任一存在 → PASS）：

| 模式 | 名称 | 代码形式 | 适用侧别 |
|------|------|---------|---------|
| A | OP_CHECK_IF | `OP_CHECK_IF(div == 0, LOG, return FAIL)` | Tiling |
| B | if-guard+return | `if (div == 0) return;` | 两侧 |
| C | std::max 保底 | `safeDiv = std::max(div, 1U)` | 两侧 |
| D | 三元运算符 | `safe = (div > 0) ? div : 1` | 两侧 |
| E | zero-flag+skip | `if (div==0) flag=true; if(!flag) { a/b }` | 两侧 |
| F | ASSERT | `ASSERT(div != 0)` | Kernel（仅 moe/ 族） |

> **ASSERT 注意**：ASSERT 在 Release 编译中可能被移除，仅在 moe/ 算子族的 Kernel 代码中视为有效守卫。其他场景的 ASSERT 降级为 SUSPICIOUS。

**【CeilDiv/CeilDivide 特殊说明】**

`CeilDiv(a, b)` / `CeilDivide(a, b)` 是算子仓最广泛使用的除法工具函数（3,000+ 处），但其标准实现 `(a + b - 1) / b` **本身不提供零值保护**。

- **禁止**将 `CeilDiv` 调用视为守卫模式
- `CeilDiv` 的除数参数（第二个参数）仍需按 P0-P4 分类判定：
  - 来自 P0/P1/P2 → PASS
  - 来自 P3/P4 且有守卫 → PASS
  - 来自 P3/P4 无守卫 → FAIL

**【Tiling 侧硬件参数校验 — 严格模式】**

Tiling 侧负责所有硬件参数的校验（业务约定）。当硬件 API 返回值（P1）用作除数时，**必须**在 Tiling 代码中有显式零值守卫，否则判定为 FAIL。

| 除数来源 | 校验方式 | 示例 |
|---------|---------|------|
| `GetCoreNumAic/Aiv()` | `OP_CHECK_IF(aicNum == 0, return GRAPH_FAILED)` | 核数获取后立即校验 |
| `GetCoreMemSize()` | `OP_CHECK_IF(ubSize == 0, return GRAPH_FAILED)` | 内存大小获取后立即校验 |
| `context->GetBlockDim()` | `if (blockDim == 0) return GRAPH_FAILED` | 使用前校验 |

**【Tiling 侧校验示例】**

```cpp
// Tiling 阶段校验外部输入非零（P3）
OP_CHECK_IF(keyShape->GetStorageShape().GetDim(DIM_2) == 0,
           OP_LOGE(context_, "dim N2 is 0."), return ge::GRAPH_FAILED);
fBaseParams.g = queryShape->GetStorageShape().GetDim(DIM_2) /
                keyShape->GetStorageShape().GetDim(DIM_2);
OP_CHECK_IF(fBaseParams.g == 0, OP_LOGE(context_, "g is 0"), return ge::GRAPH_FAILED);

// Tiling 阶段校验硬件参数非零（P1 严格模式）
totalCoreNum_ = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
if (totalCoreNum_ == 0UL) {
    OP_LOGE(context_->GetNodeName(), "coreNum is 0");
    return ge::GRAPH_FAILED;
}

// Tiling 阶段校验设计过程参数非零（P4）
uint32_t tileSize = ComputeTileSize(totalSize, coreNum);
OP_CHECK_IF(tileSize == 0, OP_LOGE(context_, "tileSize is 0"), return ge::GRAPH_FAILED);
uint32_t loopTimes = totalSize / tileSize;
```

**【Kernel 侧校验示例】**

```cpp
// ✅ Kernel 侧排除规则 — 硬件 API 除数，自动 PASS（P1）
uint32_t coreIdx = GetBlockIdx();
uint32_t coreNum = GetBlockNum();     // P1 白名单
uint32_t taskIdx = coreIdx / coreNum; // 无需守卫

// ✅ Kernel 侧排除规则 — TilingData 除数，自动 PASS（P2）
uint32_t tileSize = tilingData->tileSize;  // P2 TilingData
uint32_t loops = totalSize / tileSize;     // 无需守卫

// ✅ Kernel 侧需校验 — 运行时动态值（P3）
GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
if ((actS1Size == 0) || (actS2Size == 0)) {
    curActSeqLenIsZero = true;
    return;  // 早期退出，避免后续除法
}
// 后续计算：loopTimes = actS1Size / mBaseSize（actS1Size 已确保非零）
```

**【描述】**
整数的除法和取余运算的第二个操作数值为0会导致程序产生未定义的行为，因此使用时要确保整数的除法和余数运算不会导致除零错误。

---

### 3. 内存与指针安全

#### 3.1 禁止使用未初始化的变量 `[适用: All]`

> **Kernel 侧说明**：Kernel 模板类的成员变量必须在 `Init()` 函数中初始化，UB Buffer 通过 `AllocTensor` 获取后才能使用。

这里的变量，指的是局部动态变量，并且还包括内存堆上申请的内存块。因为他们的初始值都是不可预料的，所以禁止未经有效初始化就直接读取其值。

```cpp
void foo(...)
{
    int data;
    bar(data); // 错误：未初始化就使用
    ...
}
```

#### 3.2 指向资源句柄或描述符的变量，在资源释放后立即赋予新值 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无动态资源管理，Buffer 由 `InitBuffer` 静态分配，无需释放后置空。

**【描述】**
指向资源句柄或描述符的变量包括指针、文件描述符、socket描述符以及其它指向资源的变量。

以指针为例，当指针成功申请了一段内存之后，在这段内存释放以后，如果其指针未立即设置为NULL，也未分配一个新的对象，那这个指针就是一个悬空指针。如果再对悬空指针操作，可能会发生重复释放或访问已释放内存的问题，造成安全漏洞。

**【正确代码示例】**

```cpp
int foo(void)
{
    SomeStruct *msg = NULL;
    ... // 初始化msg->type，分配 msg->body 的内存空间

    if (msg->type == MESSAGE_A) {
        ...
        free(msg->body);
        msg->body = NULL;
    }

    ...
EXIT:
    ...
    free(msg->body);
    return ret;
}
```

#### 3.3 外部数据作为数组索引时必须确保在数组大小范围内 `[适用: All]`

> **Kernel 侧说明**：Kernel 中使用 blockIdx、tileLength 等变量访问 GM/UB，需确保索引不越界。

**【Kernel 侧排除规则】**

以下情况在 Kernel 侧自动排除，无需校验：

| 排除条件 | 参数模式示例 | 排除原因 |
|---------|-------------|----------|
| 索引来自 TilingData | `constInfo.*`, `baseInfo.*` | Tiling 阶段已校验范围（如 Shape 维度校验） |
| 循环边界内索引 | `for (i = 0; i < bound; i++)` 内的 `arr[i]` | 循环条件保证索引在范围内 |
| GM/UB Buffer 内偏移 | `gmTensor[offset]`，offset 来自 Tiling | Tiling 阶段计算偏移范围 |

**判定方法**：
- 识别索引变量名匹配 `constInfo.*|baseInfo.*` 时，直接判定为 PASS
- 识别索引在循环边界内使用时，直接判定为 PASS

**【Kernel 侧需校验场景】**

以下情况在 Kernel 侧仍需校验：

| 校验条件 | 参数来源 | 代码模式 |
|---------|---------|----------|
| aiCoreIdx 核索引 | `GetBlockIdx()` 运行时获取 | `if (aiCoreIdx >= usedCoreNum) { return; }` |
| bIdx batch 累积差值边界 | TND 布局 `actualSeqLen[bIdx] - actualSeqLen[bIdx-1]` | `if (bIdx > 0) { ... } else { return actualSeqLen[0]; }` |
| 动态计算的偏移 | 运行时计算值 | 边界判断逻辑 |

**【Tiling 侧校验示例】**

```cpp
// Tiling 阶段校验 Shape 维度范围
OP_CHECK_IF(shape->GetDimNum() != expectedDim, 
           OP_LOGE(context_, "dim num mismatch"), return ge::GRAPH_FAILED);
OP_CHECK_IF(shape->GetDim(i) > MAX_SIZE,
           OP_LOGE(context_, "dim %d exceeds limit", i), return ge::GRAPH_FAILED);
```

**【Kernel 侧校验示例】**

```cpp
// Kernel 核索引范围校验
if (aiCoreIdx >= tilingData->baseParams.usedCoreNum) {
    if ASCEND_IS_AIV {
        SyncAll();  // superkernel 同步
    }
    return;  // 超范围核退出
}

// Kernel TND 布局累积差值边界处理
if (bIdx > 0) {
    return actualSeqLen[bIdx] - actualSeqLen[bIdx - 1];  // 累积差值
} else {
    return actualSeqLen[0];  // 首元素，避免访问 bIdx-1
}
```

**【描述】**
外部数据作为数组索引对内存进行访问时，必须对数据的大小进行严格的校验，确保数组索引在有效范围内，否则会导致严重的错误。

**【正确代码示例】**

```cpp
#define DEV_NUM 10
static Dev devs[DEV_NUM];

int set_dev_id(size_t index, int id)
{
    if (index >= DEV_NUM) {
        ... // 错误处理
    }
    devs[index].id = id;
    return 0;
}
```

#### 3.4 禁止通过对指针变量进行sizeof操作来获取数组大小 `[适用: All]`

> **Kernel 侧说明**：Kernel 中 `LocalTensor<T>` 通过 API（如 `GetSize()`）获取大小，不能用 sizeof。

**【描述】**
将指针当做数组进行sizeof操作时，会导致实际的执行结果与预期不符。

**【错误代码示例】**

```cpp
char path[MAX_PATH];
char *buffer = (char *)malloc(SIZE);
...
(void)memset(path, 0, sizeof(path));
// sizeof与预期不符，其结果为指针本身的大小而不是缓冲区大小
(void)memset(buffer, 0, sizeof(buffer));
```

**【正确代码示例】**

```cpp
char path[MAX_PATH];
char *buffer = (char *)malloc(SIZE);
...
(void)memset(path, 0, sizeof(path));
(void)memset(buffer, 0, SIZE); // 使用申请的缓冲区大小
```

#### 3.5 指针操作，使用前必须要判空 `[适用: All]`

> **Kernel 侧说明**：Kernel 中 `GlobalTensor` 和 `LocalTensor` 通过 API 获取，一般不需要判空，但 GM 地址偏移需校验。

**【描述】**
解引用空指针会导致程序产生未定义行为，通常会造成程序异常终止。

- 指针变量在使用前，一定要做好初始化的赋值，严禁对空指针进行访问
- 对于指针所代表的地址空间的任何操作，一定要保证空间的有效性
- 指针指向的内存释放后，需要调用者将指针显式置为NULL，防止"野指针"

#### 3.6 确保字符串存储有足够的空间容纳字符数据和null结束符 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无 C 风格字符串处理。但 GM 数据搬运时需确保目标 Buffer 有足够空间。

**【描述】**
将数据复制到不足以容纳数据的缓冲区，会导致缓冲区溢出。

---

### 4. 输入验证

#### 4.1 外部输入数据需要做合法性校验 `[适用: All]`

> **Kernel 侧说明**：Kernel 中的 `TilingData` 参数（如 `constInfo.*`、`baseInfo.*`）已在 Tiling 阶段校验，无需重复校验。校验职责归属 Tiling 层。

**【Kernel 侧排除规则】**

以下情况在 Kernel 侧自动排除，无需校验：

| 排除条件 | 参数模式示例 | 排除原因 |
|---------|-------------|----------|
| 参数来自 TilingData | `constInfo.*`, `baseInfo.*`, `tilingData->*` | Tiling 阶段已校验（Shape、Dtype、范围、存在性） |
| __aicore__ 函数入参 | 模板类 Init/Process 参数 | 架构约定：尽量减少校验，有效性由调用者保证 |
| GM 指针可选输入 | `actualSeqLengths` 可能为 nullptr | 通过标志位 fallback 处理 |

**判定方法**：
- 识别参数变量名匹配 `constInfo.*|baseInfo.*|tilingData->*` 时，直接判定为 PASS
- 识别参数赋值来源为 `tilingData->xxx` 时，直接判定为 PASS
- 识别参数在 `__aicore__` 函数签名中时，不报告"输入验证缺失"

**【Kernel 侧需校验场景】**

以下情况在 Kernel 侧仍需处理（非"校验"，而是"分支处理"）：

| 处理条件 | 参数来源 | 代码模式 |
|---------|---------|----------|
| actualSeqLengths 可选输入 | GM 指针可能为 nullptr | `if (ptr != nullptr) { SetGlobalBuffer(ptr); }` |
| isActualLenDimsNull 标志位 | Tiling 传递 | `if (flag == 1) { return staticSize; } else { return gm[bIdx]; }` |
| 空 Tensor 专用 Kernel | ShapeSize == 0 | 专用模板 `FiaKernelEmptyTensor`，InitOutput 为 0 |

**【Tiling 侧校验示例】**

```cpp
// Tiling 阶段校验 Shape、Dtype、范围
OP_CHECK_IF(context_->GetInputDesc(QUERY) == nullptr,
           OP_LOGE(context_, "query desc is null"), return ge::GRAPH_FAILED);
OP_CHECK_IF(shape->GetDimNum() != expectedDim,
           OP_LOGE(context_, "dim num mismatch"), return ge::GRAPH_FAILED);
OP_CHECK_IF(headDim == 0,
           OP_LOGE(context_, "headDim is 0"), return ge::GRAPH_FAILED);

// Tiling 阶段校验参数组合存在性
ge::graphStatus FiaTilingCheck::CheckExists(const void *pointer, const std::string &name) const
{
    OP_CHECK_IF(pointer == nullptr,
        OP_LOGE(opName_, "%s should not be null", name.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
```

**【Kernel 侧处理示例】**

```cpp
// Kernel 可选 GM 指针条件处理（非"校验"，而是"分支处理"）
if (actualSeqLengthsQ != nullptr) {
    actualSeqQlenAddr = (__gm__ int32_t *)actualSeqLengthsQ;
}

// Kernel 标志位 fallback（Tiling 已传递 isActualLenDimsNull）
if (constInfo.isActualLenDimsNull == 1) {
    return constInfo.s1Size;  // 静态值 fallback
} else {
    return actualSeqQlenAddr[bIdx];  // 动态值
}
```

**【描述】**

- 外部输入数据需要做合法性校验且确保校验范围正确
- 边界接口需要对传入的地址做合法性校验避免任意地址读写
- 需要对入参进行合法性校验避免数组越界
- 需要对地址偏移校验避免任意地址读写
- 外部传入指针需要判空后使用
- 外部入参参与循环、递归条件的运算，必须严格校验边界和终止条件
- 文件路径来自外部数据时，必须对其做合法性校验

#### 4.2 外部输入作为内存操作相关函数的复制长度时，需要校验其合法性 `[适用: All]`

> **Kernel 侧说明**：Kernel 中 `DataCopy` 的搬运长度需校验，确保不超过 UB 容量和 GM 数据范围。

**【描述】**
将数据复制到容量不足以容纳该数据的内存中会导致缓冲区溢出。必须根据目标容量的大小限制被复制的数据大小，或者必须确保目标容量足够大以容纳要复制的数据。

---

### 5. 资源管理

#### 5.1 资源申请后必须判断是否成功 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无动态资源申请（malloc/new），Buffer 由 `InitBuffer` 静态分配，编译期确定。

**【描述】**
内存、对象、stream、notify等资源申请分配一旦失败，那么后续的操作会存在未定义的行为风险。

**【正确代码示例】**

```cpp
struct tm *make_tm(int year, int mon, int day, int hour, int min, int sec)
{
    struct tm *tmb = (struct tm *)malloc(sizeof(*tmb));
    if (tmb == NULL) {
        ... // 错误处理
    }
    tmb->year = year;
    ...
    return tmb;
}
```

#### 5.2 资源泄露（内存、句柄、锁等） `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无动态内存、无锁、无句柄，Buffer 静态分配无需释放。

**【描述】**

- 资源申请和释放必须匹配，包括：内存类的malloc/free/alloc_page/free_page, 锁lock/unlock、文件open/close等
- 释放结构体/类/数组/各类数据容器指针前，必须先释放成员指针
- 对外接口处理涉及资源申请但未释放，引起资源泄露，导致拒绝服务
- C++捕获异常时确保恢复程序的一致性; 建议使用RAII模式，确保资源在异常发生时自动释放

#### 5.3 new和delete配对使用，new[]和delete[]配对使用 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 禁止 new/delete。

#### 5.4 使用恰当的方式处理new操作符的内存分配错误 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 禁止 new。

---

### 8. 安全函数使用

#### 8.1 使用社区提供的安全函数库的安全函数，禁止使用内存操作类危险函数 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无 memcpy_s/memset_s，使用 Ascend C API（如 `Duplicate`、`DataCopyPad`）。

| 函数类别 | 危险函数 | 安全替代函数 |
|---------|---------|------------|
| 内存拷贝 | memcpy或bcopy | memcpy_s |
| 内存拷贝 | memmove | memmove_s |
| 字符串拷贝 | strcpy | strcpy_s |
| 字符串串接 | strcat | strcat_s |
| 格式化输出 | sprintf | sprintf_s |
| 格式化输出 | snprintf | snprintf_s |
| 格式化输入 | scanf | scanf_s |
| 内存初始化 | memset | memset_s |

#### 8.2 正确设置安全函数中的destMax参数 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无安全函数。

#### 8.3 必须检查安全函数返回值，并进行正确的处理 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无安全函数。

原则上，如果使用了安全函数，需要进行返回值检查。如果返回值!=EOK, 那么本函数一般情况下应该立即返回，不能继续执行。

```cpp
{
    ...
    err = memcpy_s(destBuff, destMax, src, srcLen);
    if (err != EOK) {
        MS_LOG("memcpy_s failed, err = %d\n", err);
        return FALSE;
    }
    ...
}
```

---

### 9. 类与对象安全

#### 9.1 禁止逐位操作非trivially copyable对象 `[适用: All]`

> **Kernel 侧说明**：Kernel 模板类都是 POD 类型，可以使用 `Duplicate` 进行内存操作。

---

### 10. 标准库安全

#### 10.1 禁止从空指针创建std::string `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无 std::string。

#### 10.2 不要保存std::string类型的 `c_str`和 `data`成员函数返回的指针 `[适用: Tiling]`

> **Kernel 侧不适用**：Kernel 无 std::string。

#### 10.3 内存中的敏感信息使用完毕后立即清0 `[适用: All]`

> **Kernel 侧说明**：Kernel 中 UB 数据可通过 `Duplicate` 清零，GM 数据需在 Host 侧处理。

口令、密钥等敏感信息使用完毕后立即清零，避免被攻击者获取。

#### 10.4 对外结构体接口新增字段时必须在结构体最后添加 `[适用: All]`

> **Kernel 侧说明**：`TilingData` 结构体新增字段需在末尾添加，保持 ABI 兼容性。

为了最大程度上在ABI层面的兼容，对外结构体接口添加新字段时必须在结构体最后添加。

#### 10.5 外部接口或数据结构变更必须考虑兼容性 `[适用: All]`

> **Kernel 侧说明**：Kernel 接口（如 TilingData 结构体）变更需考虑版本兼容性。

外部接口、接口参数、返回值、数据结构、消息字段等变更会引起版本兼容性问题，非必要不建议变更。

---

### 11. LOG 规范

> **适用范围**：仅 Tiling 侧（Host 侧）。Kernel 侧使用 `AscendC::PRINTF`，无下列风险。

Tiling 侧使用 `OP_LOGE` / `OP_LOGD` / `OP_LOGW` 等格式化 LOG 宏。11.1–11.3 为安全强制要求（防段错误/未定义行为），11.4 为质量建议。

LOG 宏签名（业务代码标准调用形式）：

```cpp
OP_LOGE(context->GetNodeName(), "format string %s %ld", arg1, arg2);
OP_LOGD(context->GetNodeName(), "format string %lu", arg1);
```

---

#### 11.1 LOG API 禁止传入空指针作为字符串参数 `[适用: Tiling]`

**【问题说明】**

`%s` 会解引用传入指针，若指针为 `nullptr`，将访问地址 0（受 OS 保护），导致段错误。Tiling 侧常见场景：从 `context` 获取 Desc/Attr 后未判空直接传入 LOG。

**错误示例**

```cpp
// 来自 quant_grouped_matmul_dequant_tiling.cpp 同类风险
auto inputDesc = context->GetInputDesc(0);
// 若 inputDesc 为 nullptr，GetDataType() 返回的字符串描述也可能为空
OP_LOGE(context->GetNodeName(),
        "input dtype: %s", ge::TypeUtils::DataTypeToSerialString(inputDesc->GetDataType()).c_str());
// 风险：inputDesc 未判空就调用成员函数
```

**正确示例**

```cpp
auto inputDesc = context->GetInputDesc(0);
if (inputDesc == nullptr) {
    OP_LOGE(context->GetNodeName(), "GetInputDesc(0) returned nullptr, skip dtype log.");
    return ge::GRAPH_FAILED;
}
OP_LOGE(context->GetNodeName(),
        "input dtype: %s", ge::TypeUtils::DataTypeToSerialString(inputDesc->GetDataType()).c_str());
```

---

#### 11.2 LOG API 参数必须与格式化占位符逐位一致（数量、类型、顺序） `[适用: Tiling]`

**【问题说明】**

LOG 宏的格式化占位符与实际参数之间必须满足三个维度的一致性：

1. **数量一致**：参数少于占位符时，从栈上读取垃圾值，若被解释为 `%s` 将触发段错误
2. **类型匹配**：类型大小不匹配时（如 `uint64_t` 误用 `%d`），按说明符宽度截断，后续参数全部错位
3. **顺序对应**：参数顺序与格式符位置不对应时（如 `%s` 位置收到整数），整数被当作地址读字符串 → **段错误(SIGSEGV)**

> **⚠️ 禁止仅凭 grep 单行分析 LOG 调用。** 算子仓中大量 LOG 语句跨越多行（2-35 行），且常嵌套在 `OP_CHECK_IF` 等外层宏内。grep 命中后**必须 Read 前后至少 10 行**获取完整的格式字符串和全部参数，否则分析的是截断的不完整调用，结论无效。多行字符串拼接（`"a" "b"`）需先合并再解析。

**错误与正确示例**

```cpp
// ❌ 数量不一致：2 个占位符，1 个参数
OP_LOGD(ctx, "M: %ld, K: %ld", m);           // 缺少 k
// ✅
OP_LOGD(ctx, "M: %ld, K: %ld", m, k);

// ❌ 类型不匹配：uint64_t 用了 %d
OP_LOGE(ctx, "n = %d, ubSize = %d\n", n, ubSize); // n/ubSize 均为 uint64_t
// ✅
OP_LOGE(ctx, "n = %llu, ubSize = %llu\n", n, ubSize);

// ❌ 顺序错位：数量=5 格式符=5，但位置1和3的参数放反了
//   格式符: %u(1) %u(2) %s(3) %u(4) %u(5)
//   参数:   inputName.c_str()(1) ... d0Size/NUM8(3) ...
//   → 位置1: %u 收到 const char*，位置3: %s 收到 uint → 段错误
OP_CHECK_IF(tempD0 != d0Size,
    OP_LOGE(opName, "...kvCache(%u)...%s(%u)...",
        inputName.c_str(), tempD0/NUM8, d0Size/NUM8, tempD0, d0Size),
    return ge::GRAPH_FAILED);
// ✅ 参数顺序与格式符逐位对应
OP_CHECK_IF(tempD0 != d0Size,
    OP_LOGE(opName, "...kvCache(%u)...%s(%u)...",
        tempD0/NUM8, d0Size/NUM8, inputName.c_str(), tempD0, d0Size),
    return ge::GRAPH_FAILED);
```

**类型与说明符速查**

| 类型 | 正确 | 常见错误 | 后果 |
|------|------|---------|------|
| `uint64_t` | `%llu` | `%u`, `%lu`, `%d` | 截断为 32 位，后续参数错位 |
| `int64_t` | `%lld` | `%d`, `%ld` | 同上 |
| `uint32_t` | `%u` | `%d` | 大值显示为负数 |
| `size_t` | `%zu` | `%d`, `%u` | 64 位系统上截断 |
| `bool` | `%d` 或 `? "true":"false"` + `%s` | `%s` 直传 | 未定义行为 |
| `void*` | `%p` | `%x` | 不可移植 |

**【检视方法】**

1. grep `OP_LOGE\|OP_LOGD\|OP_LOGW\|OP_LOGI` 找到所有 LOG 调用
2. Read 完整调用后，提取格式符序列和参数序列，逐位比对：数量是否一致 → 每个位置的参数类型是否兼容格式符
3. 高风险标记：`%s` 收到整数（段错误）、`uint64_t`/`int64_t` 配 `%d`（截断错位）

---

#### 11.3 LOG API 禁止传入已释放内存的指针 `[适用: Tiling]`

**【问题说明】**

Tiling 侧手动管理的堆内存（`new` / `malloc`）释放后若仍传入 `%s`，行为未定义，大概率触发段错误。典型场景：在函数末尾统一释放资源，但 LOG 语句写在释放之后。

**错误示例**

```cpp
char* errMsg = new char[256];
snprintf(errMsg, 256, "tiling failed, M=%ld", _Params.originM);
delete[] errMsg;
OP_LOGE(context->GetNodeName(), "error: %s", errMsg);   // 野指针，已释放
```

**正确示例**

```cpp
char* errMsg = new char[256];
snprintf(errMsg, 256, "tiling failed, M=%ld", _Params.originM);
OP_LOGE(context->GetNodeName(), "error: %s", errMsg);   // 先记录
delete[] errMsg;
errMsg = nullptr;
```

---

#### 建议 11.4 LOG 消息的英语行文应语法正确、表意清晰 `[适用: Tiling]`

**【问题说明】**

LOG 消息是排障的第一手线索。语法错误或含义模糊的日志会显著增加定位问题的时间成本。

**检视要点**：
- 主谓一致、时态统一（LOG 消息惯用一般现在时或过去时）
- 避免中英文混杂（变量名除外）
- 避免无意义占位（如 "error error"、"fail to fail"）
- 关键数值应包含在消息中，而非仅靠格式符

**提醒示例**

```cpp
// "is not support" → "is not supported"（仓内高频错误模式，5+ 文件）
OP_LOGE(op_name, "scale shape is not support");          // → is not supported
OP_LOGE(opName_, "...layout BNSD/BNSD_NBSD is not support"); // → is not supported
OP_LOGE(ACLNN_ERR_PARAM_INVALID, "...the soc verison is not support"); // → version; is not supported

// "do not support" → "does not support"（主谓不一致）
OP_LOGE(opName_, "...key layout do not support PA_BSND."); // → does not support

// 拼写错误
OP_LOGE(opName_, "...cu_seqlens_q's dtype msut be DT_INT32."); // msut → must

// 缺少主语
OP_LOGD("GetBlockInfoOfBNS4TND", " Not support BN2S2."); // → BN2S2 is not supported
```

> **检视级别**：仅标记 SUSPICIOUS，不标记 FAIL。