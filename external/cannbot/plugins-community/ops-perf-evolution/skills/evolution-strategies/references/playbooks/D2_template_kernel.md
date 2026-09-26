# D2 Playbook: Template Kernel Type Dispatch (模板化内核类型分发)

> 本 Playbook 为**强制流程**。采纳 D2 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> D2 的核心是**用 C++ 模板参数化 Kernel 的存储类型 (U) 和计算类型 (T)，在 Host 侧根据运行时 dtype 做编译期分发**，一套代码支持 FP16/BF16/FP32/INT8 等多种数据类型。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/d2_locations.txt`：

```bash
# 1. 当前硬编码的数据类型
grep -n "LocalTensor\s*<\s*float\s*>\|LocalTensor\s*<\s*half\s*>\|LocalTensor\s*<\s*bfloat16_t\s*>\|GlobalTensor\s*<\s*float\s*>\|GlobalTensor\s*<\s*half\s*>" \
    shared/original/op_kernel/*.cpp > /tmp/d2_locations.txt
# 2. Host 侧的 dtype 参数与校验
grep -n "DT_FLOAT\|DT_BF16\|DT_HALF\|DataType\|dtype\|xDtype\|outDtype" \
    shared/original/op_host/*_tiling.cpp shared/original/op_host/*.cpp >> /tmp/d2_locations.txt
# 3. 已有的模板或泛型代码
grep -n "template\s*<.*typename\|template\s*<.*class\|typename\s*T\|typename\s*U" \
    shared/original/op_kernel/*.cpp >> /tmp/d2_locations.txt
# 4. Cast / RoundMode 使用（与 D1/A1 的交界）
grep -n "Cast\|RoundMode\|CAST_NONE\|CAST_RINT" \
    shared/original/op_kernel/*.cpp >> /tmp/d2_locations.txt
# 5. 计算核心（判断 compute type 是否应与 storage type 分离）
grep -n "Add\|Mul\|Sub\|Div\|Sqrt\|Exp\|Log" \
    shared/original/op_kernel/*.cpp >> /tmp/d2_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **硬编码类型**：当前 Kernel 中所有 `LocalTensor<T>` / `GlobalTensor<T>` 的具体类型
- **Host dtype 参数**：dtype 变量名、校验位置、支持的类型集合
- **已有模板**：是否已有 template、Bind、泛型 DAG
- **Cast 现状**：是否已有精度转换 Cast 链
- **计算核心**：哪些 API 参与计算，是否精度敏感

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| Storage type (U) | `?` (float / half / bf16) | `template <typename U>` | `?_kernel.cpp:L?` |
| Compute type (T) | `?` (与 U 相同 / 固定 float) | `template <typename T>` | `?_kernel.cpp:L?` |
| Host dtype 校验 | `?` (无 / 部分) | 完整 supported set | `?_host.cpp:L?` |
| Cast 链 | `?` (无 / 有) | U→T entry, T→U exit | `?_kernel.cpp:L?` |
| 分发方式 | `?` (无) | `alpha/beta/gamma` 见 3A | `?_host.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的 dtype 复杂度和分发需求，判断你的代码属于以下哪种形态：

- **形态 α — 单类型 + 显式 Cast（类型少、精度不敏感）**：仅需支持 2 种类型（如 FP32 + FP16），且无需计算类型分离。直接在 Kernel 内用 `Cast` 硬编码处理。
- **形态 β — 模板类 + Host if-else 实例化（类型中等、2~4 种）**：Kernel 定义为 `template <typename U, typename T>` 的类，Host 用 `if (dtype == DT_X)` 实例化对应类型。这是最常见、最可维护的模式。
- **形态 γ — 模板 + TilingKey 或宏批量注册（类型多、≥4 种）**：需要支持 INT8/INT16/INT32/FP16/BF16/FP32 等大量组合。用 TilingKey 编码 dtype 信息，或宏展开批量注册模板实例。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 β — 模板类 + Host 分发，最常见）

```cpp
// === 改造前（仅 float32）===
class MyKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t size) {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        yGm.SetGlobalBuffer((__gm__ float*)y);
        this->tileSize = size;
    }
    
    __aicore__ inline void Compute(uint32_t count) {
        LocalTensor<float> xLocal = xBuf.Get<float>();
        LocalTensor<float> yLocal = yBuf.Get<float>();
        Add(yLocal, xLocal, biasLocal, count);
    }
    
private:
    GlobalTensor<float> xGm, yGm;
    TBuf<QuePosition::VECCALC> xBuf, yBuf, biasBuf;
    uint32_t tileSize;
};

// === 改造后（模板化 U=storage, T=compute）===
template <typename U, typename T = float>
class MyKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t size) {
        xGm.SetGlobalBuffer((__gm__ U*)x);
        yGm.SetGlobalBuffer((__gm__ U*)y);
        this->tileSize = size;
    }
    
    __aicore__ inline void Compute(uint32_t count) {
        // Buffer 大小按类型调整
        LocalTensor<U> xLocal = xBuf.Get<U>();
        LocalTensor<U> yLocal = yBuf.Get<U>();
        LocalTensor<T> xCalc = calcBuf.Get<T>();
        LocalTensor<T> yCalc = calcBuf.Get<T>();
        
        // U → T（upcast，用 CAST_NONE 保持数值）
        Cast(xCalc, xLocal, RoundMode::CAST_NONE, count);
        Cast(yCalc, yLocal, RoundMode::CAST_NONE, count);
        
        // 在 T 精度下计算（避免低精度累积误差）
        Add(yCalc, xCalc, biasCalc, count);
        
        // T → U（downcast，用 CAST_RINT 四舍五入）
        Cast(yLocal, yCalc, RoundMode::CAST_RINT, count);
    }
    
private:
    GlobalTensor<U> xGm, yGm;
    TBuf<QuePosition::VECCALC> xBuf, yBuf, biasBuf;
    TBuf<QuePosition::VECCALC> calcBuf;  // 额外 buffer 供 T 类型计算
    uint32_t tileSize;
};
```

**Host 侧分发**：
```cpp
// Host tiling / launch 代码
if (xDtype == ge::DT_FLOAT16) {
    // U=half, T=float：FP16 存储，FP32 计算
    MyKernel<half, float> kernel;
    kernel.Init(x, y, size);
    kernel.Compute(size);
} else if (xDtype == ge::DT_BF16) {
    // U=bfloat16_t, T=float
    MyKernel<bfloat16_t, float> kernel;
    kernel.Init(x, y, size);
    kernel.Compute(size);
} else if (xDtype == ge::DT_FLOAT) {
    // U=float, T=float：无精度转换开销
    MyKernel<float, float> kernel;
    kernel.Init(x, y, size);
    kernel.Compute(size);
} else {
    // 不支持的类型应在此处报错，不要落入默认分支
    return ge::GRAPH_FAILED;
}
```

### 3C. Variant Notes（若是形态 α 或 γ）

- **形态 α（单类型 + 显式 Cast，不引入模板）**：
  若只需支持 FP32 + FP16 两种类型，且代码简单，可不用模板，直接在 Kernel 内用 `if constexpr` 或宏：
  ```cpp
  __aicore__ inline void Compute(uint32_t count, DataType dtype) {
      if (dtype == DT_FLOAT16) {
          LocalTensor<half> xLocal = xBuf.Get<half>();
          // ... half path
      } else {
          LocalTensor<float> xLocal = xBuf.Get<float>();
          // ... float path
      }
  }
  ```
  ⚠️ 形态 α 代码重复度高，超过 2 种类型时必须升级为形态 β。

- **形态 γ（大量类型，TilingKey / 宏注册）**：
  当支持类型 ≥4 种时，Host 侧 if-else 链过长。改用 TilingKey 编码 dtype：
  ```cpp
  // TilingKey 低 8 位编码 in_dtype，高 8 位编码 out_dtype
  constexpr uint32_t DTYPE_FLOAT = 0;
  constexpr uint32_t DTYPE_HALF  = 1;
  constexpr uint32_t DTYPE_BF16  = 2;
  constexpr uint32_t DTYPE_INT8  = 3;
  // ...
  uint32_t tilingKey = (outDtypeId << 8) | inDtypeId;
  
  // 注册时批量实例化（宏展开）
  #define REGISTER_KERNEL(U, T, KEY) \
      REGISTER_KERNEL_API(KEY, MyKernel<U, T>)
  REGISTER_KERNEL(half, float, 0x0100);
  REGISTER_KERNEL(bfloat16_t, float, 0x0200);
  REGISTER_KERNEL(float, float, 0x0000);
  ```
  形态 γ 需要框架支持 TilingKey 分发，若当前算子注册体系不支持，退化为形态 β。

- **与 D1 的边界**：D2 解决**类型分发**（一套代码支持多 dtype），D1 解决**精度链**（低精度存储 + 高精度计算 + 再转回）。两者常一起使用：D2 模板参数 `<U, T>` 天然承载 D1 的精度链。
  - 若算子已有 D1 的 `calcType` 机制，D2 只需把 `calcType` 提升为模板参数 `T`。
  - 若算子精度不敏感（如纯 memcpy/reshape），D2 可设 `T = U`，跳过 Cast 链。

- **与 A1 的协同**：A1 要求 FP32 中间计算。在 D2 模板中，统一设 `T = float` 即可满足 A1 的精度要求，无需额外改造。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: calcBuf 大小 = tileSize * sizeof(T)。若 T = float, U = half，calcBuf 是 xBuf 的 2 倍
约束 2: UB 总用量（xBuf + yBuf + biasBuf + calcBuf）≤ UB 容量（a3: ~192KB, a5: ~256KB）
约束 3: Cast 的 RoundMode 必须正确：U→T upcast 用 CAST_NONE，T→U downcast 用 CAST_RINT
约束 4: Host 侧必须校验 dtype 在 supported set 内，禁止未定义类型落入默认分支
约束 5: 模板实例化后代码体积可控：≤4 种 (U,T) 组合时，形态 β 最优；>4 种时考虑形态 γ
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `sizeof(U) = ?`, `sizeof(T) = ?`, `tileSize = ?`
- `xBuf = ? bytes, calcBuf = ? bytes, totalUB = ? bytes`
- `supported dtypes = [?, ?, ?]`
- Cast RoundMode: entry = ?, exit = ?
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: Kernel 中存在 template 声明
grep -cE "template\s*<.*typename\s+U|template\s*<.*typename\s+T" modified_files/op_kernel/*.cpp
# 期望: >= 1（形态 α 除外，若用 if-else 硬编码需有 dtype 分支）

# 检查 2: 存在 Cast 链（U→T 和 T→U），形态 β/γ 必须有
grep -cE "Cast\s*\(.*CAST_NONE|Cast\s*\(.*CAST_RINT" modified_files/op_kernel/*.cpp
# 期望: >= 2（若 T == U 无 Cast，需在 note 中说明并跳过此检查）

# 检查 3: Host 侧有 dtype 校验或分发
grep -cE "DT_FLOAT|DT_BF16|DT_HALF|dtype.*==|supported.*dtype" modified_files/op_host/*_tiling.cpp modified_files/op_host/*.cpp
# 期望: >= 1

# 检查 4: 无硬编码 float/half 在计算核心（应使用模板参数 T/U）
grep -cE "LocalTensor\s*<\s*float\s*>.*xLocal|LocalTensor\s*<\s*half\s*>.*xLocal|Add.*float.*float" modified_files/op_kernel/*.cpp
# 期望: == 0（旧代码残留）

# 检查 5: calcBuf / 额外 buffer 已正确按 sizeof(T) 分配
grep -cE "InitBuffer.*calcBuf|calcBuf.*Get|sizeof\s*\(\s*T\s*\)|tileSize.*sizeof" modified_files/op_kernel/*.cpp
# 期望: >= 1
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：模板参数推断失败 | `Cast` 的模板参数需显式指定类型：`Cast<T>(dst, src, ...)` 而不是依赖推断。确保 `T` 和 `U` 都是 AscendC 支持的类型 |
| 编译失败：`bfloat16_t` 未定义 | 确认 CANN 版本 ≥ 8.0。旧版本可能没有 `bfloat16_t`，需用 `uint16_t` + 自定义转换替代 |
| 运行时：UB 越界 | 检查约束 2：T = float 时 calcBuf 是 U = half 时的 2 倍。`InitBuffer` 的大小必须按 `tileSize * sizeof(T)` 计算，不是硬编码 |
| 运行时：FP16 输入输出但精度明显差于 FP32 baseline | 确认 `T = float`（不是 `T = half`）。若 T = U = half，则失去了 D1/A1 的精度保护 |
| Host 侧遗漏了某种 dtype | 在 supported set 中枚举所有目标 dtype。常见遗漏：`DT_INT8`、`DT_UINT8`、`DT_BOOL` |
| 模板导致编译时间剧增 | 若 (U,T) 组合 > 8 种，考虑形态 γ（TilingKey）或精简 supported set。形态 β 建议最多 4 种组合 |
| RoundMode 错误导致数值偏差 | U→T upcast 用 `CAST_NONE`（保留精确值），T→U downcast 用 `CAST_RINT`（四舍五入）。不要反过来 |
| 多类型测试覆盖不全 | 每种 (U,T) 组合必须跑编译 + 精度测试。常见遗漏：BF16 路径 |
| 模板类中没有默认构造函数 | AscendC Kernel 类需要默认构造函数或 Init 函数来接收 GM 指针。确保模板类与普通类有相同的生命周期 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[D2 Playbook Completion]
Step 1: done (/tmp/d2_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: constraints: sizeof(U)=? sizeof(T)=? totalUB=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
