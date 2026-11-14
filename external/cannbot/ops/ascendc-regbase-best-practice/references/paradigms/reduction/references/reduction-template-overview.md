# reduction 模板总览

> 描述本范式 kernel 模板划分、 TilingKey 划分、TilingData 划分及三者之间的联系

| 概念 | 说明 |
|------|------|
| **模板** | Kernel 的编译期特化实例。同一份 kernel 代码根据 TilingKey 字段值的不同组合，编译出不同的执行路径 |
| **TilingKey** | Host 端 tiling 阶段算出的编译期开关（一组 bool/int 等字段），决定运行时框架加载哪个编译实例。一个模板可对应多个 TilingKey（一对多），一个 TilingKey 只对应一个模板 |
| **TilingData** | Host 端 tiling 阶段算出的运行时参数（struct）。一个模板只对应一个 TilingData，一个 TilingData 可被多个模板共用 |

**三者协作流程**：TilingKey 决定"走哪条路"，TilingData 提供"路上的参数"。

## 1 范式术语

### 1.1 术语表

| 术语 | 含义 |
|------|------|
| **A 轴** | 非 reduce 轴（preserve） |
| **R 轴** | reduce 轴（沿此轴规约） |
| **pattern** | 合轴预处理后轴的类型序列，形如 `A R A R …` |
| **TailA** | 合轴预处理后尾轴是A，如 `ARA` |
| **TailR** | 合轴预处理后尾轴是R，如 `ARAR` |
| **Base 模板** | 仅A轴参与分核，核间无依赖，核内可处理完对应的R轴 |
| **Group 模板** | A轴和R轴都参与分核，核间有依赖，需要二次reduce |
| **空 tensor** | Empty Tensor, shape中含有0的tensor, 如 (3, 0)，分 EMPTY_A（A 轴含 0）和 EMPTY_R（R 轴含 0） |
| **二分缓存树** | 所有 reduce 算子统一使用二分缓存树累加，提升累加精度（对 sum/mean 有精度收益，对 max/min/any/all 结果不变；prod 因 fp 舍入不满足结合律，结果可能有差异） |
| **All Reduce** | 所有轴都 reduce 的算子，合轴后 pattern 恒为 `AR` |
| **post_reduce_input** | PostReduce 阶段的额外 GM 输入（如 mean/variance 等统计量）。shape 与输出 y 相同（A 轴，size = aTotal） |
| **D_T** | 算子 tensor 数据的 dtype（模板参数，如 fp16/bf16/fp32） |

## 2 kernel 模板分类

### 2.1 kernel 模板分类

所有 reduce 算子（sum/mean/max/min/prod/any/all）统一使用二分缓存树累加。对 sum/mean 有精度收益，对 max/min/any/all 结果不变（满足结合律），prod 因 fp 舍入不满足结合律、结果可能有差异，但统一路径简化了模板设计。

**按多核切分轴分为三类**:
- 空 Tensor 模板：不需要做 reduce 计算，是一个简化模板
- Base 模板：仅A轴参与分核，核间无依赖，核内可处理完对应的R轴
- Group 模板：A轴和R轴都参与分核，核间有依赖，需要二次reduce。A较小时借R开多核可以提升性能。

**模板分类汇总**:
| # | 模板 | 使用场景 | 准入条件 |
| ---- | ---- | ------- | ------ |
| 1 | Base | 所有 reduce 算子（sum/mean/max/min/prod/any/all）的 Base 模板 | 非空tensor, A轴较大，足够开多核 |
| 2 | Group | 所有 reduce 算子的 Group 模板 | 非空tensor, A轴较小，需要借R开多核 |
| 3 | Empty | 所有 reduce 算子的空 Tensor 模板 | 空tensor（EMPTY_A / EMPTY_R 共用） |

每个算子选择 1/2/3 中的 1~3 个模板。

## 3 TilingKey

### 3.1 TilingKey 定义

reduction 范式 TilingKey 采用 **2 个 bool** 字段：`isGroup` / `isEmptyTensor`。

```cpp
ASCENDC_TPL_ARGS_DECL(
    ReduceGeneric,
    ASCENDC_TPL_BOOL_DECL(isGroup, 0, 1),   // 0=base/empty, 1=group
    ASCENDC_TPL_BOOL_DECL(isEmptyTensor, 0, 1),  // 空 tensor 模板
);
```

**字段含义**：
- `isGroup`：0=base/empty，1=group（A×R 2D 分核）
- `isEmptyTensor`：0=非空，1=空 tensor（EMPTY_A/EMPTY_R 共用）


**TPL_SEL 结构（3 个 tilingkey 组合）**：

```cpp
ASCENDC_TPL_SEL(
    // base 模板
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(isGroup, 0),
        ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0),
    ),
    // 空 tensor 模板
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(isGroup, 0),
        ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 1),
    ),
    // group 模板（isEmptyTensor 固定 0，与 empty 互斥）
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(isGroup, 1),
        ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0),
    ),
);
```

**设计决策**：
- **dtype 不进 tilingkey**：框架按 REG_OP 中注册的输入名，为每种 dtype 自动生成独立编译实例（通过 `DTYPE_<INPUT_NAME>` 宏，如输入名为 `x` 则框架定义 `DTYPE_X`）。因此 dtype 分支在编译期已由框架展开，不需要进 tilingkey


### 3.2 TilingKey与kernel模板对应关系

| 模板 | isGroup | isEmptyTensor |
|---|---|---|
| Base | 0 | 0 |
| Group | 1 | 0 |
| EMPTY | 0 | 1 |

**约束**：
- `isGroup=1` 与 `isEmptyTensor=1` 互斥（group 不与 empty 组合）

**实例数**：

| | isEmptyTensor=0 (base) | isEmptyTensor=1 (empty) | isGroup=1 (group) | 单 dtype 总计 |
|---|---|---|---|---|
| 组合数 | 1 | 1 | 1 | **3** |

## 4 TilingData

### 4.1 TilingData 定义

reduction 范式有 **2 份** TilingData struct：
- Base / Group 模板共用 `ReduceGenericTilingData`
- Empty 模板使用独立的 `ReduceEmptyTilingData`

> 命名约定：`ReduceGenericTilingData` / `ReduceEmptyTilingData` 为范式占位名，开发时替换为实际算子名（如 `EuclideanNormTilingData` / `EuclideanNormEmptyTilingData`）。

#### 4.1.1 Base / Group TilingData

```cpp
constexpr int32_t MAX_PATTERN_RANK = {RANK}; // 本算子最大rank值,替换为实际值
struct ReduceGenericTilingData {
    // ─── pattern 描述 ───
    int32_t axisNum;                  // 2~MAX_PATTERN_RANK
    int64_t axisShape[MAX_PATTERN_RANK];  // 合轴后每根轴的 size
    int64_t axisStride[MAX_PATTERN_RANK]; // 每根轴的 GM stride（按 element 计）
    // axisType[i] 不需要：i 偶→A，i 奇→R（A 起头+严格交替）

    // ─── 多核切分（外层 A loop 扁平为线性计数，按 coreNum 均匀分核）───
    int64_t aLoopCntTotal;            // ∏(outer A 整根) × aSplitChunkCnt
    int64_t aSplitChunkCnt;           // CeilDiv(axisShape[aSplitIdx], aUbFactor)
    int64_t aBigCoreLoopCnt;          // 大核处理的块数（= aSmallCoreLoopCnt + (aBigCoreCnt > 0 ? 1 : 0)）
    int64_t aSmallCoreLoopCnt;        // 小核处理的块数（= aLoopCntTotal / coreNum，floor；
                                      //   == 0 等价 aLoopCntTotal < coreNum，只有前 aBigCoreCnt 核工作）
    int32_t aBigCoreCnt;              // 大核个数（= aLoopCntTotal % coreNum）
    int32_t usedCoreNum;              // 实际使用核数

    // ─── UB 切分 ───
    int32_t aSplitIdx;            // UB 内被切分的 A 轴下标
    int32_t rSplitIdx;            // UB 内被切分的 R 轴下标
    int64_t aUbFactor;                // valid：A 维实际元素数（可能非 block 对齐，UB 行 stride 由 postBufSize 兜底）
    int64_t rUbFactor;                // valid：R 维实际元素数
    int64_t rUbFactorAlign;           // padded：UB 行 stride（tail-R 且 UB 切尾轴 且 尾轴全载 且 尾轴非对齐时 > rUbFactor，其余相等）
    int64_t innerAProdAlign;          // padded：含最内 burst-tail A 的 CeilAlign
    int64_t innerRProdAlign;          // padded：含最内 burst-tail R 的 CeilAlign

    // ─── 外层 R loop 扁平化 ───
    int64_t rLoopCntTotal;            // ∏(外层 R 轴 size) × CeilDiv(axisShape[rSplitIdx], rUbFactor)

    // ─── UB buffer 字节数 ───
    int64_t preBufSize;               // pre 阶段单份 buffer 大小（含 R 维，按 maxDtypeSize），blocksize 对齐
    int64_t postBufSize;              // post 阶段单份 buffer 大小（不含 R 维，按 maxDtypeSize），blocksize 对齐
    int64_t cacheBufUbSize;           // 固定 16 × 1024 (= 16 KB)

    // ─── group 模板 ───
    int64_t rGroupCnt;                // Phase 1 分组数 = Phase 2 workspace R 维大小（base 不读）

    // ─── 算子自定义字段 ───
    // 由具体算子按需追加，如 mean 算子追加 `float invRTotal`（= 1/R_total）
};

```

**为什么只有 `rUbFactorAlign` 而没有 `aUbFactorAlign`**：

- **tail-R 需要 `rUbFactorAlign`**：pattern 最简为 `AR`，前面无其他 R 轴可切。R 全载非对齐时不能 FloorAlign（R 小于 1 个 block 时 FloorAlign 会变 0），改用 CeilAlign，故需独立存 padded 版
- **`aUbFactorAlign` 不需要**：三种情形（aSplitIdx==LastA / aSplitIdx!=LastA / tail-R）均直接使用 `aUbFactor`，理由见 [reduction-binary-base-tiling.md](reduction-template-binary-base/reduction-binary-base-tiling.md) §1.2 Step 1 约束

**`rUbFactor` / `rUbFactorAlign`**：
- valid（`rUbFactor`）：GM stride、valid 元素数、partial chunk 判定
- padded（`rUbFactorAlign`）：UB 行 stride、ReduceXxx srcShape、UB 占用预算
- Align > valid 条件：tail-R 且 UB 切尾轴 且 尾轴全载 且 尾轴非对齐，其余相等

#### 4.1.2 Empty TilingData

```cpp
struct ReduceEmptyTilingData {
    // ─── 多核切分 ───
    int32_t usedCoreNum;            // EMPTY_A: 0（所有核早退）；EMPTY_R: 切分算出的核数
    int64_t aTotal;                 // ∏(所有 A 轴 axisShape)，EMPTY_A 不读
    int64_t aUbFactor;              // 单 chunk a 元素数（4 约束取 min）
    int32_t aBigCoreCnt;            // 大核个数
    int64_t aBigCoreLoopCnt;        // 每大核 chunk 数
    int64_t aSmallCoreLoopCnt;      // 每小核 chunk 数

    // ─── UB buffer ───
    int64_t postBufSize;            // post 阶段单份 buffer 字节数
};
```

### 4.2 TilingData与kernel模板对应关系

| TilingData | 使用的 kernel 模板 |
|------------|------------------|
| `ReduceGenericTilingData` | Base / Group |
| `ReduceEmptyTilingData` | Empty |

**约束**：
- Group 模板 Phase 2 的大小核字段（`aSmallCoreLoopCnt_p2` / `aBigCoreCnt_p2` / `aBigCoreLoopCnt_p2` / `usedCoreNum_p2`）由 kernel 侧用 `usedCoreNum` 现算，不单独存入 TilingData

## 5 总结

| 模板名 | 准入条件 | TilingKey 字段值 | TilingData 名 |
|--- |---|---|---|
| Base | 非空, A轴够分核 | isGroup=0, isEmptyTensor=0 | ReduceGenericTilingData |
| Group | 非空, A轴不够分核且R有并行度 | isGroup=1, isEmptyTensor=0 | ReduceGenericTilingData |
| Empty | 空 tensor（EMPTY_A / EMPTY_R） | isGroup=0, isEmptyTensor=1 | `ReduceEmptyTilingData`（独立 7 字段） |


## 6 产出

> 通过前面知识，使不同算子产出对应的模板、TilingKey、TilingData

### 6.1 产出结果

结合算子公式，参考前面章节，逐步产出算子实际使用的模板、TilingKey、TilingData：

**1. 输出模板**：参考 [§2.1 kernel 模板分类](#21-kernel-模板分类)，按 `isEmptyTensor`（空 tensor 短路）、`isGroup`（Group 触发判定）取值，输出算子实际使用的模板（Base / Group / Empty 中的 1~3 个）。

**2. 输出 TilingKey**：参考 [§3.1 TilingKey 定义](#31-tilingkey-定义) 和 [§3.2 TilingKey与kernel模板对应关系](#32-tilingkey与kernel模板对应关系)，按所选模板输出 `isGroup` / `isEmptyTensor` 各字段取值。

**3. 输出 TilingData**：参考 [§4.1.1 Base / Group TilingData](#411-base--group-tilingdata) 和 [§4.1.2 Empty TilingData](#412-empty-tilingdata)，按所选模板确定对应的 TilingData struct：Base/Group 用 `ReduceGenericTilingData`，Empty 用 `ReduceEmptyTilingData`。按算子需要确定 `MAX_PATTERN_RANK`（影响 `axisShape`/`axisStride` 数组长度）。算子自定义字段（如 mean 的 `invRTotal`）追加在 `ReduceGenericTilingData` 末尾。

### 6.2 产出校验

每个算子完成三要素确定后，按以下清单校验：

| # | 校验项 | 规则 |
|---|--------|------|
| 1 | `isGroup=1` 与 `isEmptyTensor=1` 互斥 | group 不与 empty 组合 |
| 2 | dtype 不进 TilingKey | 走 `DTYPE_<INPUT_NAME>` 编译期实例化 |
| 3 | Empty 使用独立 `ReduceEmptyTilingData` | 不复用 Base/Group struct |
