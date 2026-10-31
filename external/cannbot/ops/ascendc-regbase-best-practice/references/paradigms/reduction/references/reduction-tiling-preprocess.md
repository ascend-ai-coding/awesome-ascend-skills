# reduction 范式 Tiling 预处理

> ub 切分前的预处理阶段。所有模板（Base/Group/Empty）共用同一套预处理逻辑。

依赖输入：[reduction-template-overview.md](reduction-template-overview.md)、[common/value-depend-process.md](../../common/value-depend-process.md)

## 1 算子信息获取

### 1.1 平台信息获取

```cpp
coreNum      = ascendcPlatform.GetCoreNumAiv();
ubSize       = ascendcPlatform.GetCoreMemSize(UB, ubSize);
blockSize    = Ops::Base::GetUbBlockSize(context);      // 不要写 32
cacheLineSize = Ops::Base::GetCacheLineSize(context);   // 不要写 256/512
vectorSize   = Ops::Base::GetVRegSize(context);
```

**约束**：
- 每个返回值必须校验非 0，报错用 `OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON`（param 取 `platform.<参数名>`，reason 注明查询接口名）
- 严禁将平台参数写死，必须通过接口获取

### 1.2 算子输入输出获取及校验

**约束**：
- 标量输入按算子语义处理（示例 euclidean_norm：rank-0 视作 [1] 单元素张量；标量判断用 `gert::Shape::IsScalar()` 官方 API，勿手写 `GetDimNum() == 0`；不支持的算子显式报错）
- 获取可选输入要使用 `GetOptionalInputShape` 和 `GetOptionalInputDesc`
- 可选输入为空时要特殊处理，不能直接返回失败
- 可选 attr 为空时要使用默认值，不能直接返回失败
- shape/dtype 要严格校验，不支持的 shape/dtype 必须报错。**错误日志分层**：参数值错误用 `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON`、shape 维度错误用 `OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON`、dtype 错误用 `OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON`；
- 必须使用 INFO 级别日志打印算子每个输入参数/属性，包括但不限于 shape、dtype、attr；

### 1.3 值依赖分析

> `axes` 只是个标识，含义是 reduce 的轴信息，算子中有可能叫 axes/axis/dim/dims 等。

reduce 算子的 host 端要做 pattern 预处理（合轴/切分），依赖 `axes` 的**实际数值**。如果 `axes` 是个 `tensor`，就需要获取 tensor 的值。framework 默认只传 shape/dtype，此时必须显式声明"值依赖"。三处契约详见 [common/value-depend-process.md](../../common/value-depend-process.md)。

reduce 系列通常 `x` 在 0、`axes` 在 1 → 填 `{1}`：

| # | 位置 | 声明 |
|---|------|------|
| 1 | `op_def.cpp` 输入定义 | `this->Input("axes").ParamType(REQUIRED).ValueDepend(OPTIONAL)...` |
| 2 | `IMPL_OP_INFERSHAPE` 注册块 | `.InputsDataDependency({1})` |
| 3 | `IMPL_OP_OPTILING` 注册块 | `.TilingInputsDataDependency({1})` |

> 空 tensor 短路同样依赖 axes 数值（判断 A/R 类型），需要值依赖。

## 2 合轴

**流程**：axes 归一化 + 空 tensor 短路 → 去轴 → 合轴 → 补轴

### 前置：axes 归一化 + 空 tensor 短路

在进入合轴四步之前，先完成两项前置：

**axes 归一化**：将用户传入的 reduce 轴参数（`axes`/`axis`/`dim`/`dims` 等，名称因算子而异）解析为合法、规范化的 reduce 轴下标集合。包含以下步骤（顺序不可换）：

1. **空 axes 语义判定**：`axes=[]`（axesNum==0）的语义按算子公式判定——多数算子（对标 TF/PyTorch）按 "all reduce" 展开为 `[0..xRank-1]`；少数算子（`noopWithEmptyAxes=true`）按 "不 reduce" 处理。
2. **dtype 解析**：axes 输入可能是 int32 或 int64（`IndexNumberType`），按实际 dtype 读取数值。
3. **越界校验**：对原始值校验 `v < -xRank || v >= xRank` → 报错（须在转正之前——如 `-xRank-1` 转正后变成 `-1`，会漏检）。
4. **负数转正**：`v < 0` → `v + xRank`（对标 torch.sum，负数表示从末尾倒数）。归一化后所有轴下标落在 `[0, xRank)`。
5. **去重校验**：重复轴下标 → 报错。
6. **排序**：去重后升序排序，便于后续合轴按轴顺序处理。

归一化完成后，遍历 xShape，按下标是否在 reduce 集合中标记 A/R 类型，构建初始 A/R 类型表（`axisShape` + `isReduceAxis`），作为合轴四步的输入。

> infershape 与 tiling 两端的归一化逻辑必须完全一致。

**空 tensor 短路**（`HandleEmptyTensor()`）：检查 axisShape（必须在合轴四步之前）：
- 任一 A 轴 size == 0 → **EMPTY_A**（output 空 tensor），不进入合轴，跳转 Empty 模板
- 任一 R 轴 size == 0 且所有 A 轴 size > 0 → **EMPTY_R**（output = `empty_r_output_value`），不进入合轴，跳转 Empty 模板
- 否则（所有 axisShape ≥ 1）→ 进入合轴四步

**优先级**：**EMPTY_A > EMPTY_R**（A=0 时 output 必空，无论 R 是否 0）。

> 空 tensor 短路后不进入合轴四步，直接跳转 Empty 模板 tiling 处理。

**两子模板共用 `isEmptyTensor=1` tilingkey + 同一份 `<op>_empty.h`**，靠 `usedCoreNum` 在 kernel 内区分：
- EMPTY_A：`usedCoreNum = 0`，所有核早退
- EMPTY_R：`usedCoreNum` = 切分算出的核数

### 步骤 1：去 1 轴

**对应函数**: `DropSizeOneAxes()`

无论 A 轴或 R 轴，`size == 1` 的整根轴直接删除。

**不变量（非空保留）**：若整张图所有轴 size 都为 1（∏shape == 1），删除会清空轴列表，此时**保留 1 根占位 `A=1`** 作为结果，不允许列表为空。

### 步骤 2：合轴

**对应函数**: `FuseAxis()`

连续 A 轴合并为 1 根 A 轴，连续 R 轴合并为 1 根 R 轴。

### 步骤 3：补 leading A 轴

**对应函数**: `PadLeadingOneA()`

若第 0 轴是 R 轴，前置一根大小为 1 的 A 轴。保证最高轴必为 A。

**stride 计算**：合成 A=1 的 stride = `∏所有轴 size`（整段总元素数），不是 1。

### 步骤 4：补 R 增广（纯 A 退化路径）

**对应函数**: `PadRIfPureA()`

若上一步完成后仍无 R 轴（纯 A 场景），按 A 轴 size 分两种：

- **A=1**（全 1 退化，∏shape==1）：**末尾补** `[R=1]`，pattern 由 `A` 变 `AR`（tail-R）。
  此时 A=R=1，单元素 reduce，无 burst 性能问题。
  好处：全 1 退化统一走 tail-R 路径，某些算子可因此省去 TailA 模板。
- **A>1**（纯 A 但非全 1，如无 reduce 轴的 passthrough）：**前置** `[A=1, R=1]`，pattern 由 `A` 变 `ARA`（tail-A）。
  **约束**：此场景必须前置 ARA，**绝不能末尾补 R=1 的 AR**。`AR` 路径下 tail R=1 而 A>1 会导致 UB burst 长度塌到底，性能崩。

### 合轴后的 pattern 形态

四步执行完后，pattern 必然形如 `A R A R … A R …`，A 起头，A/R 严格交替，每根轴 size ≥ 2（首部补 1 的 A、紧随其后补 1 的 R 除外）。

**支持的 pattern 范围**：

| pattern | 轴数 | tail 类型 |
|---------|------|-----------|
| AR | 2 | R |
| ARA | 3 | A |
| ARAR | 4 | R |
| ARARA | 5 | A |
| ARARAR | 6 | R |
| ARARARA | 7 | A |
| ARARARAR | 8 | R |
| ARARARARA | 9 | A |

**tail 类型判定**：`axisNum` 偶数 → tail R；`axisNum` 奇数 → tail A。kernel 侧由 axisNum 奇偶现算。

**轴类型判定**：合轴后 A 起头 + 严格交替，`axisType[i]` 完全由位置 `i` 的奇偶决定：`i` 偶 → A，`i` 奇 → R。

### 算子级 pattern 上界分析（MAX_PATTERN_RANK）

MAX_PATTERN_RANK = 该算子合轴后 pattern 的**最大可能轴数**，用于 TilingData `axisShape` 数组分配和合法性校验。

**推导方法**：

1. 按算子自身语义确定 reduce 轴信息——来源因算子而异：可能是 axes/axis/dim 等显式输入，可能是 attr，也可能由 format 隐式决定（如 NCHW 格式下固定 C 为 A 轴、其余为 R 轴），还可能是算子公式固化的（如固定尾轴 reduce 的算子恒有 1 根 R 轴）。无论来源如何，最终都要得到输入张量各轴的 A/R 标记
2. 在最坏情况下，将连续的 R 轴视为 1 段，统计 R 的连续段数，记为 `rseg`
3. 合轴后 A 与 R 严格交替、A 起头，A 段数最多 = `rseg + 1`，因此合轴后最大轴数 = `2 × rseg + 1`（A/R 类型标记以 A 结尾时达上限；以 R 结尾（含全 R）时为 `2 × rseg`）；纯 A（`rseg=0`，即不 reduce）补 R 增广后为 ARA = 3（A=1 全 1 退化时为 AR = 2）。标记由算子的 reduce 轴语义确定：固定 reduce 轴按固定标记取精确 MAX（如 NCHW reduce N/HW，标记 `R A R` → MAX = ARAR = 4，即使 HW=1 去 1 后成 ARA 也只是变小）；axes 由输入决定时按最坏情形（以 A 结尾）取 `2 × rseg + 1`，rseg 不定则兜底 9
4. 框架上界为 9，取 `min(推导值, 9)`

**典型例子**：

| reduce 轴语义 | 合轴后 pattern | MAX_PATTERN_RANK |
|------|------|------|
| 固定标记，以 R 结尾（全 R 为其 rseg=1 特例；如 NCHW reduce N/HW：`R A R`） | `(AR) × rseg` | `2 × rseg` |
| 固定标记，以 A 结尾（如 `R A R A`） | `(AR) × rseg + A` | `2 × rseg + 1` |
| axes 任意（rseg 不定） | 兜底 | 9 |

> 例：`RARA` → R0 是 1 段、R2 是 1 段 → `rseg=2` → MAX=5。合轴四步：R A R A → 补 leading A → `A R A R A`（5 轴，A 结尾达上限）。

推导出的 MAX_PATTERN_RANK 定义在各算子 `tiling_data.h` 中（无前缀，直接 `constexpr int32_t MAX_PATTERN_RANK = N;`），host 和 kernel 共享。

### Tiling 端合轴代码规范

1. **封装为独立函数**：去 1、合轴、补 leading A、补 R 增广 各自独立函数，可独立单元测试
2. **每函数有效代码行数 ≤ 50**（不含空行、注释）
3. **shape/dtype 校验必不可少**
4. **空 tensor 短路在合轴之前**

## 3 合轴产出校验

> 为了校验 agent 是否真正理解了合轴逻辑，需要给出一个复杂输入的样例来校验。固定输入，合轴输出必定相同。

**样例 1：通用 reduce，axes=[1,3]**

输入 shape `(2, 3, 4, 5, 6)`，axes=[1, 3]，reduce 轴 = {1, 3}。

- 初始 A/R 类型表：A0=2, R0=3, A1=4, R1=5, A2=6
- 步骤 1 去 1 轴：无 size==1 的轴，不变
- 步骤 2 合轴：无连续同类型轴，不变
- 步骤 3 补 leading A：第 0 轴是 A，不变
- 步骤 4 补 R 增广：已有 R 轴，不变
- 结果：pattern = `A0 R0 A1 R1 A2`（ARARA，axisNum=5，tail-A）

**Golden**：`axisShape = [2, 3, 4, 5, 6]`，`axisNum = 5`，`isTailR = false`（奇数轴）。

**样例 2：全 1 退化**

输入 shape `(1, 1, 1)`，axes=[0, 1, 2]，reduce 轴 = {0, 1, 2}。

- 初始 A/R 类型表：R0=1, R1=1, R2=1（全 R）
- 步骤 1 去 1 轴：全删，列表空 → 保留 1 根占位 `A=1`
- 步骤 2 合轴：仅 1 根 A，不变
- 步骤 3 补 leading A：第 0 轴是 A，不变
- 步骤 4 补 R 增广：无 R 轴，A=1 → 末尾补 [R=1]，pattern 变 `AR`
- 结果：pattern = `A R`（AR，axisNum=2，tail-R）

**Golden**：`axisShape = [1, 1]`，`axisNum = 2`，`isTailR = true`（偶数轴）。

**样例 3：EMPTY_R 短路**

输入 shape `(3, 0, 4)`，axes=[1]：
- A 轴 = {0, 2}，R 轴 = {1}
- axisShape[1] = 0（R 轴为 0），但 axisShape[0] = 3 > 0、axisShape[2] = 4 > 0 → **EMPTY_R**（不是 EMPTY_A）

**Golden**：`isEmptyTensor=1, usedCoreNum=按 aTotal=12 切分`，output = `empty_r_output_value`。

**样例 4：EMPTY_A 短路**

输入 shape `(0, 3, 4)`，axes=[1]：
- A 轴 = {0, 2}，R 轴 = {1}
- axisShape[0] = 0（A 轴为 0）→ **EMPTY_A**

**Golden**：`isEmptyTensor=1, usedCoreNum=0, SetBlockDim(1)`，output = 空 tensor。

**校验要点**：
1. 空 tensor 短路在合轴四步之前
2. EMPTY_A 优先级 > EMPTY_R（A=0 时 output 必空）
3. EMPTY_A: `usedCoreNum=0, SetBlockDim(1)`（框架要求 ≥ 1）
4. EMPTY_R: `usedCoreNum` = 按 aTotal 切分算出的核数
