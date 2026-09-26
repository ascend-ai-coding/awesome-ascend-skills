# D5 Playbook: BF16 & Platform-Specific Handling (BF16/多平台处理)

> 本 Playbook 为**强制流程**。采纳 D5 策略的子 agent 必须逐步执行，每步填写/验证后才能进入下一步。禁止跳步。
>
> D5 的核心是**在 Host 侧识别芯片平台能力，动态选择 BF16 原生路径或 FP32 fallback，并针对不同平台调整 tiling 约束**。

## Step 1: 定位关键结构

执行下面的 grep，把结果写入 `/tmp/d5_locations.txt`：

```bash
# 1. 当前平台配置（重点改造对象）
grep -n "AddConfig\|aicore_config\|chip\|platform\|ascend910\|kirinx90\|310P" \
    shared/original/op_host/*.cpp shared/original/op_host/*_tiling.cpp > /tmp/d5_locations.txt
# 2. BF16 相关代码
grep -n "BF16|bf16|bfloat16|DT_BF16" \
    shared/original/op_kernel/*.cpp shared/original/op_host/*.cpp >> /tmp/d5_locations.txt
# 3. 芯片特性检测
grep -n "GetChipFeature\|HasBf16Support\|chipFeature\|platformId\|GetChipName" \
    shared/original/op_host/*.cpp shared/original/op_kernel/*.cpp >> /tmp/d5_locations.txt
# 4. Shape 校验与平台约束
grep -n "shape.*check\|dim.*limit\|maxShape\|minShape\|supported.*shape\|2D.*shape" \
    shared/original/op_host/*.cpp >> /tmp/d5_locations.txt
# 5. 已有的 FP32 fallback 或 Cast 链
grep -n "CAST_NONE\|CAST_RINT\|useFp32\|fallback\|Fp32Compute" \
    shared/original/op_kernel/*.cpp shared/original/op_host/*.cpp >> /tmp/d5_locations.txt
```

**交付物**（必须记录到 `implementation_note.txt` 的 "Playbook Step 1" 段落）：
- **当前平台配置**：`AddConfig` 位置、当前支持的平台列表
- **BF16 代码**：是否已有 bf16 类型使用、DT_BF16 校验
- **特性检测**：是否已有 `GetChipFeature` 等动态检测
- **Shape 约束**：当前 shape 校验逻辑、是否有平台差异化限制
- **Fallback 现状**：是否已有 FP32 fallback 路径

## Step 2: 改造计划表（强制填写）

**不填完此表不得进入 Step 3**。子 agent 必须在 `implementation_note.txt` 的 "Playbook Step 2" 段落贴出填满的表格：

| 元素 | 当前值 | 目标值 | 修改位置 |
|---|---|---|---|
| 目标平台 | `?` (当前仅 1 个) | 910B/910_93/910_95/Kirin/310P | `?_host.cpp:L?` |
| BF16 支持 | `?` (无 / 部分) | 按平台动态检测 | `?_host.cpp:L?` |
| Shape 约束 | `?` (1D / 固定) | 平台差异化（如 910_95 支持 2D） | `?_host.cpp:L?` |
| Fallback 策略 | `?` (无) | 拒绝 / FP32 compute / Cast | `?_host.cpp:L?` |
| Tiling 参数 | `?` (固定) | 平台自适应 | `?_tiling.cpp:L?` |

**如果你填不出任何一格**，说明 Step 1 定位不完整，回到 Step 1。

## Step 3: 代码改造（核心）

### 3A. 形态识别

读 Step 1 定位的平台需求和 BF16 支持情况，判断你的代码属于以下哪种形态：

- **形态 α — 多平台注册（无 BF16，纯平台适配）**：算子需部署到多种芯片，但只用 FP32/FP16，主要差异是 shape 支持和 tiling 参数。
- **形态 β — BF16 原生 + 不支持平台 Fallback（最常见）**：部分平台原生支持 BF16 向量指令，部分不支持，需要 FP32 compute fallback。
- **形态 γ — 平台特定优化路径（深度定制）**：不同平台有完全不同的最优实现（如 910_95 的 2D shape 支持），需要平台分支内核。

**必须在 implementation_note.txt "Playbook Step 3A" 明确声明**：`form: alpha | beta | gamma`

### 3B. Canonical Template（形态 β — BF16 原生 + Fallback，最常见）

```cpp
// === 改造前（单平台硬编码，无 BF16 处理）===
this->AICore().AddConfig("ascend910b");
// Kernel 内直接处理，不区分平台能力
LocalTensor<bfloat16_t> inLocal = inBuf.Get<bfloat16_t>();
Add(outLocal, inLocal, biasLocal, count);  // 在不支持 bf16 vec 的平台上编译或运行失败

// === 改造后（Host 侧平台检测 + BF16 fallback 决策）===
// Host 侧：平台识别与能力检测
std::string chipType = GetChipName();  // 从 CANN 运行时获取
OpAICoreConfig config;
bool supportsBf16Vec = false;
bool supports2DShape = false;

if (chipType == "ascend910_95") {
    config = Get910_95Config();
    supportsBf16Vec = true;
    supports2DShape = true;
} else if (chipType == "ascend910_93") {
    config = Get910_93Config();
    supportsBf16Vec = true;
    supports2DShape = false;
} else if (chipType == "ascend910b") {
    config = Get910BConfig();
    supportsBf16Vec = GetChipFeature("Intrinsic_data_move_l12bt_bf16");
    supports2DShape = false;
} else if (chipType == "kirinx90") {
    config = GetKirinConfig();
    supportsBf16Vec = false;  // Kirin 通常无原生 BF16 向量支持
    supports2DShape = false;
} else if (chipType == "ascend310p") {
    config = Get310PConfig();
    supportsBf16Vec = false;
    supports2DShape = false;
} else {
    // 未知平台：保守 fallback
    config = GetDefaultConfig();
    supportsBf16Vec = false;
    supports2DShape = false;
}

this->AICore().AddConfig(chipType, config);

// 将平台能力传入 TilingData
this->tilingData.supportsBf16Vec = supportsBf16Vec;
this->tilingData.supports2DShape = supports2DShape;
this->tilingData.chipType = EncodeChipType(chipType);

// BF16 不支持时的决策
if (inputDtype == ge::DT_BF16 && !supportsBf16Vec) {
    // 方案 A：拒绝（严格模式）
    // return ge::GRAPH_FAILED;
    
    // 方案 B：FP32 compute fallback（推荐，配合 D1/D2）
    this->tilingData.computeDtype = ge::DT_FLOAT;
    this->tilingData.needsCast = true;
    this->tilingData.workspaceSize = totalElems * sizeof(float);
}
```

**Kernel 侧平台分支**（形态 β 的 Kernel 实现）：
```cpp
__aicore__ inline void Compute(uint32_t count) {
    if (this->tilingData.needsCast) {
        // Fallback 路径：BF16 存储 + FP32 计算
        LocalTensor<bfloat16_t> inBf16 = inBuf.Get<bfloat16_t>();
        LocalTensor<float> inFp32 = calcBuf.Get<float>();
        LocalTensor<float> outFp32 = calcBuf.Get<float>();
        
        Cast(inFp32, inBf16, RoundMode::CAST_NONE, count);
        Add(outFp32, inFp32, biasFp32, count);
        Cast(inBf16, outFp32, RoundMode::CAST_RINT, count);
    } else {
        // 原生路径：BF16 存储 + BF16 计算（或 FP32 计算）
        LocalTensor<bfloat16_t> inLocal = inBuf.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> outLocal = outBuf.Get<bfloat16_t>();
        Add(outLocal, inLocal, biasLocal, count);
    }
}
```

### 3C. Variant Notes（若是形态 α 或 γ）

- **形态 α（纯平台适配，无 BF16）**：
  仅需扩展 `AddConfig` 列表，并在 Host 侧根据平台调整 tiling 参数：
  ```cpp
  this->AICore().AddConfig("ascend910_95", Get910_95Config());
  this->AICore().AddConfig("ascend910_93", Get910_93Config());
  this->AICore().AddConfig("ascend910b", Get910BConfig());
  this->AICore().AddConfig("kirinx90", GetKirinConfig());
  this->AICore().AddConfig("ascend310p", Get310PConfig());
  ```
  形态 α 通常不需要 Kernel 侧改动，只需 Host 侧配置和 shape 校验差异化。

- **形态 γ（平台特定优化路径）**：
  当不同平台的最优实现差异很大（如 910_95 支持 2D grouped matmul，其他平台不支持）：
  ```cpp
  // Host 侧选择不同 TilingKey 或 Kernel 模板
  if (supports2DShape && inputShape.ndim() == 2) {
      // 910_95 专用 2D 路径
      tilingData.tilingKey = TILING_2D;
      tilingData.blockDimM = M / tileM;
      tilingData.blockDimN = N / tileN;
  } else {
      // 通用 1D 路径
      tilingData.tilingKey = TILING_1D;
      tilingData.blockDim = BLOCK_DIM;
  }
  ```
  形态 γ 需要维护多套 Kernel 逻辑，只有当性能收益 > 20% 时才值得。

- **与 D1/D2 的协同**：
  D5 的 BF16 fallback 天然复用 D1（Mixed Precision）的 Cast 链和 D2（Template Kernel）的类型分发。
  - 若算子已用 D2 模板 `<U, T>`，D5 只需在 Host 侧设置 `T = float` 当 `supportsBf16Vec == false`。
  - 若算子已有 D1 的 `calcType`，D5 将 `calcType` 与平台能力关联即可。

- **与 P4 的边界**：某些平台（如 310P）核数较少，P4 的分核策略需要随平台调整。建议在 Host 侧根据 `GetBlockNum()` 实际返回值动态调整 `BLOCK_DIM`。

## Step 4: 约束复核（防崩溃）

**约束**：
```
约束 1: 未知平台必须有保守 fallback（不支持 BF16、不支持 2D shape），不能崩溃
约束 2: BF16 fallback 的 workspace 大小 = totalElems × sizeof(float)，必须 ≤ GM 可用空间
约束 3: Cast 链的 RoundMode：BF16→FP32 用 CAST_NONE，FP32→BF16 用 CAST_RINT
约束 4: 平台检测必须在编译期或 kernel launch 前完成，不能在 kernel 内做字符串比较
约束 5: 每新增一个平台，必须跑完整编译 + 精度 + 性能测试
```

**在 implementation_note.txt "Playbook Step 4" 中报告具体数值**：
- `目标平台 = [?, ?, ?]`
- `supportsBf16Vec = [true/false per platform]`
- `fallback workspace = ? bytes`
- `未知平台 fallback 策略 = ?`
- 是否通过：yes/no

## Step 5: 编码后自检（5 条 grep，全部必须过）

**严格度**：任一失败 → 回到 Step 3 重做。

```bash
# 检查 1: 存在多平台 AddConfig 或等效注册
grep -cE "AddConfig.*910|AddConfig.*kirin|AddConfig.*310" modified_files/op_host/*.cpp
# 期望: >= 2（至少 2 个平台）

# 检查 2: 存在平台能力检测或 BF16 相关逻辑
grep -cE "supportsBf16Vec|GetChipFeature|chipType|platform" modified_files/op_host/*.cpp modified_files/op_kernel/*.cpp
# 期望: >= 1

# 检查 3: BF16 fallback 时有 Cast 链（形态 β）
grep -cE "CAST_NONE.*bf16|CAST_RINT.*bf16|needsCast" modified_files/op_kernel/*.cpp
# 期望: >= 2（若形态 α 无 BF16，此检查应为 0，需在 note 中说明）

# 检查 4: 无裸 bfloat16_t 计算（在不支持平台上会编译失败）
grep -cE "Add.*bfloat16_t|Mul.*bfloat16_t" modified_files/op_kernel/*.cpp
# 期望: == 0（应走模板参数或 fallback 分支）

# 检查 5: Host 侧有未知平台兜底处理
grep -cE "else\s*\{[^}]*default|unknown.*chip|unsupported.*platform|GRAPH_FAILED" modified_files/op_host/*.cpp
# 期望: >= 1
```

**在 implementation_note.txt "Playbook Step 5" 写入每条检查的实际命令输出** 和通过/未通过标记。

## Step 6: Known Pitfalls + 修复建议

| 现象 | 修复 |
|---|---|
| 编译失败：bfloat16_t 在不支持平台上未定义 | 用 `#ifdef __BFloat16__` 或模板隔离 BF16 路径。不支持的平台只编译 FP32/FP16 实例 |
| 运行时：BF16 结果精度差 | 确认 fallback 路径使用了 FP32 compute（不是 FP16 compute）。`T = float` 是底线 |
| 运行时：310P/Kirin 上性能暴跌 | 这些平台核数少、无 BF16 向量单元。考虑禁用 BF16 输入，强制 FP32 |
| 未知平台导致 segfault | 必须有兜底分支。`GetChipFeature` 返回失败时，默认 `supportsBf16Vec = false` |
| 多平台测试矩阵爆炸 | 形态 β 建议核心测试 910B + 910_95 + Kirin 三个平台。310P 和 910_93 可与 910B 共享配置 |
| workspace 申请失败 | fallback workspace 在 Host 侧通过 `TilingData` 传入，大小按 `totalElems * sizeof(float)`。确认未超过 GM 限制 |
| AddConfig 参数不匹配 | 不同平台的 `OpAICoreConfig` 字段可能不同。使用平台专用 `GetXXXConfig()` 工厂函数，不要硬编码通用 config |
| TilingKey 冲突 | 若用 TilingKey 编码平台信息，确保与已有 key 不冲突。建议平台占高 4 位，原 key 占低 12 位 |
| BF16 的 Cast 到 FP32 后忘记再 Cast 回去 | fallback 路径必须是 U→T→compute→T→U 完整链。漏掉最后一步会导致输出 dtype 错误 |

---

**完成 Step 1-6 后**，在 `implementation_note.txt` 末尾贴上清单：
```
[D5 Playbook Completion]
Step 1: done (/tmp/d5_locations.txt written)
Step 2: plan table filled
Step 3: form = alpha/beta/gamma, canonical/variant applied
Step 4: constraints: platforms=[?] bf16_support=[?] workspace=? passed: yes/no
Step 5: all 5 grep checks passed (详见下文)
Step 6: no pitfalls triggered / {列出触发的}
```
