# broadcast 模板总览

> 描述本范式 kernel 模板划分、 TilingKey 划分、TilingData 划分及三者之间的联系

## 1 范式术语

### 1.1 术语表

| 术语 | 含义 |
|------|------|
| **effective_shape** | 去 1 补 1 后的归一化 shape，所有输入/输出的统一坐标系 |
| **maximumBroShape** | broadcast 上界 shape，作为 UB/多核切分的坐标系 |
| **PadAndSqueeze** | 输入预处理：补 1 拉 rank、去 1 消废维、标量归一 |
| **split ubSplitIdx** | UB 单切分轴，将 effective_shape 分为 UB 内轴和 UB 外轴 |
| **perBufElems** | 单个 UB buffer 可容纳的元素数 = UB / P |
| **P (物理存活节点)** | UB buffer 层面的峰值同时驻留数，决定 TBuf 槽位数 |
| **L (逻辑存活节点)** | 计算图 tensor 层面的峰值同时驻留数 |
| **RANK** | effective_shape 的维度数，TilingKey 的唯一参数 |
| **NDDMA** | 多维数据搬运（DataCopy<T, NDDMA_DIMS, cfg>），支持随路 broadcast |
| **TBuf** | 单缓冲 UB 管理模型，禁止 TQue |
| **RegBase** | 寄存器级编程，中间结果在 VF 寄存器链传递，不落 UB |

## 2 kernel 模板分类

## 2.1 kernel 模板分类

> Broadcast 范式只有一个场景模板（standard），内部按 RANK 分两档编译期实例化。

| 序号 | 模板 | 使用场景 | 准入条件 |
| ---- | ---- | ------- | ------ |
| 1 | Standard | 输入输出 shape 不一致需广播对齐的 element-wise 算子 | 计算公式不含 Reduce，多输入间存在 broadcast |

## 3 TilingKey

### 3.1 TilingKey 定义

**参考源码**: `example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_struct.h`

> Broadcast 范式的 TilingKey 只与 RANK 有关，与数据类型无关。
> dtype 由 CANN 框架 `DTYPE_IN0` 注入，Kernel 侧编译期 `is_same_v<T, float>` 判定路径，不在 TilingKey 中编码。

**范式开发者约束**：
- 严禁使用 `TILING_KEY_IS` 定义 TilingKey，必须使用 `ASCENDC_TPL_ARGS_DECL` 模板编程方式
- 严禁 TilingKey 包含 dtype
- 必须有完整的注释，明确 TilingKey 各字段的含义
- 产出完整的 `ASCENDC_TPL_SEL` 结构
- 其他

TilingKey 只有一个参数 `RANK`，两个穷举值：

```cpp
#include "ascendc/host_api/tiling/template_argument.h"

#define BROADCAST_RANK_4 4
#define BROADCAST_RANK_8 8

ASCENDC_TPL_ARGS_DECL(Broadcast,
    ASCENDC_TPL_UINT_DECL(RANK, 8, ASCENDC_TPL_UI_LIST,
        BROADCAST_RANK_4, BROADCAST_RANK_8)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, BROADCAST_RANK_4)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, BROADCAST_RANK_8))
);
```

RANK ≤ 4 走 RANK_4，RANK ≥ 5 走 RANK_8。建议 Rank=4 是常态，Rank=8 是保底。

### 3.2 TilingKey与kernel模板对应关系

| TilingKey | 模板 | 说明 |
|---|---|---|
| RANK_4 (4) | Standard | effective_shape 维度 ≤ 4 |
| RANK_8 (8) | Standard | effective_shape 维度 5-8 |

**约束**:
- 一个模板包含两个 TilingKey（RANK_4 / RANK_8）
- dtype 不进 TilingKey（统一通过 `DTYPE_IN0` 编译宏实例化）

## 4 TilingData

### 4.1 TilingData 定义

**参考源码**: `example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_tiling_struct.h`

**产出**：
- 完整 TilingData struct 定义（含注释）
- 字段分类：公共字段 vs 扩充字段
- 如果需要有多个 tilingdata，全部列出，并注明每个 tilingdata 使用场景

Broadcast 范式按 RANK 模板化 TilingData，RANK=4 和 RANK=8 各一个 struct：

```cpp
// 公共常量（按算子需求替换）
constexpr int64_t MAX_INPUT_SLOTS  = {MAX_INPUTS};   // 本算子最大输入张量数
constexpr int64_t MAX_OUTPUT_SLOTS = {MAX_OUTPUTS};  // 本算子最大输出张量数
constexpr int64_t PHYS_NODES      = {PHYS_NODES};   // 物理存活节点 P（= TBuf 槽位数）

// 防御性初值（NSDMI）：计数/块数类字段默认 1（单块、单核退化配置），索引默认 0，
// 避免 Host 侧异常路径（如空 shape 导致 FindSplitAxis 未写入）时 Kernel 读到未定义值
struct SplitResult {
    int64_t ubSplitIdx = 0;  // UB 切分轴
    int64_t ubFactor    = 1;  // UB 主块元素数
    int64_t ubOuter     = 1;  // UB 块数（外层次数）
    int64_t ubTail      = 1;  // UB 尾块元素数
};

struct MultiCoreResult {
    int64_t usedCoreNum = 1;  // 实际使用的核数
    int64_t totalTiles  = 1;  // 总 tile 数
    int64_t mainTiles   = 1;  // 每核主块数 = CeilDiv(totalTiles, usedCoreNum)
    int64_t mainCoreNum = 1;  // 主核数（前 mainCoreNum 核各处理 mainTiles 块，其余核各处理 mainTiles-1 块）
};

// 使用场景：RANK ≤ 4 的算子
template<int64_t RANK>
struct BroadcastTilingData {
    SplitResult     split;            // UB 切分结果（来源：FindSplitAxis）
    MultiCoreResult multicore;        // 多核切分结果（来源：MultiCoreSplit）
    int64_t         rank              = 1;  // 实际 rank (1~8)；Host 侧写入供 dump 对齐，Kernel 不读取（用 RANK 模板参数）
    int64_t         perBufBytes       = 0;  // 单 buffer 字节数 = (UB/P) & ~(ubBlockSize-1)，ubBlockSize 经 GetUbBlockSize 获取（来源：P 结论表）
    int64_t         maxBroShape[RANK] = {0}; // 广播后各维大小（坐标系）
    int64_t         numInputs         = 0;  // 输入张量数
    int64_t         numOutputs        = 0;  // 输出张量数
    int64_t         inputShapes [MAX_INPUT_SLOTS][RANK]  = {0}; // 各输入补 1 后的 shape
    int64_t         inputStrides[MAX_INPUT_SLOTS][RANK]  = {0}; // 各输入的 GM stride（broadcast 轴 = 0）
    int64_t         outputShapes [MAX_OUTPUT_SLOTS][RANK] = {0}; // 各输出补 1 后的 shape（稠密，= maximumBroShape）
    int64_t         outputStrides[MAX_OUTPUT_SLOTS][RANK] = {0}; // 各输出的 GM stride
};
```

RANK=4 和 RANK=8 的 struct 结构相同，仅 `RANK` 模板参数不同，导致 `maxBroShape` / `inputShapes` / `outputShapes` 等数组维度不同。

### 4.2 TilingData与kernel模板对应关系

| TilingData | 模板 | 使用场景 |
|---|---|---|
| BroadcastTilingData\<4\> | Standard | effective_shape 维度 ≤ 4 |
| BroadcastTilingData\<8\> | Standard | effective_shape 维度 5-8 |

**约束**:
- 不同TilingData一定是不同模板
- 不同模板不一定是不同TilingData

## 5 总结

| 模板名 | 准入条件 | TilingKey 字段值 | TilingData 名 |
|---|---|---|---|
| Standard | 计算公式不含 Reduce，多输入间存在 broadcast | RANK_4 (4) / RANK_8 (8) | BroadcastTilingData\<4\> / BroadcastTilingData\<8\> |

**参考源码**（Adam 为 Broadcast 范式唯一参考实现）：

| 层级 | 路径 | 内容 |
|------|------|------|
| Kernel | `example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_kernel.h` | 完整 Process/Init/CopyInBrc/CopyOut |
| Tiling | `example/examples-adam-code/op_host/arch35/adam_apply_one_assign_tiling_arch35.cpp` | PadAndSqueeze/FindSplitAxis/MultiCoreSplit |
| TilingData | `example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_tiling_struct.h` | TilingData 结构体定义 |
| TilingKey | `example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_struct.h` | TilingKey 模板参数定义 |
| 设计文档 | `example/examples-adam-design/adam_apply_one_assign_design.md` | 完整 §1-§6 设计案例 |

**代码模板**（占位符替换是代码生成的基础）：

TilingKey 模板（`op_struct.h.templ`，占位符：`{OP}` = 算子名 CamelCase，`{OP_UPPER}` = UPPER_SNAKE）：

```cpp
#ifndef {OP_UPPER}_STRUCT_H_
#define {OP_UPPER}_STRUCT_H_
#include "ascendc/host_api/tiling/template_argument.h"
#define {OP_UPPER}_RANK_4 4
#define {OP_UPPER}_RANK_8 8
ASCENDC_TPL_ARGS_DECL({OP},
    ASCENDC_TPL_UINT_DECL(RANK, 8, ASCENDC_TPL_UI_LIST,
        {OP_UPPER}_RANK_4, {OP_UPPER}_RANK_8)
);
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, {OP_UPPER}_RANK_4)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, {OP_UPPER}_RANK_8))
);
#endif
```

TilingData 模板（`tiling_data.h.templ`，占位符：`{OP}` / `{MAX_INPUTS}` / `{MAX_OUTPUTS}` / `{PHYS_NODES}`）：

```cpp
constexpr int64_t MAX_INPUT_SLOTS  = {MAX_INPUTS};
constexpr int64_t MAX_OUTPUT_SLOTS = {MAX_OUTPUTS};
constexpr int64_t PHYS_NODES      = {PHYS_NODES};
struct SplitResult { int64_t ubSplitIdx, ubFactor, ubOuter, ubTail; };
struct MultiCoreResult { int64_t usedCoreNum, totalTiles, mainTiles, mainCoreNum; };
template<int64_t RANK>
struct {OP}TilingData {
    SplitResult split; MultiCoreResult multicore;
    int64_t rank, perBufBytes, maxBroShape[RANK];
    int64_t numInputs, numOutputs;
    int64_t inputShapes[MAX_INPUT_SLOTS][RANK], inputStrides[MAX_INPUT_SLOTS][RANK];
    int64_t outputShapes[MAX_OUTPUT_SLOTS][RANK], outputStrides[MAX_OUTPUT_SLOTS][RANK];
};
```

Kernel 入口模板（原 `apt.cpp.templ`，已内联；入口文件命名为 `{OP_SNAKE}.cpp`，须与 def 注册的 `opFile.value` 一致。占位符：`{OP_SNAKE}` / `{OP}` / `{OP_UPPER}` / `{PRIMARY_INPUT_NAME}` / `{INPUT_GMADDRS}` / `{OUTPUT_GMADDRS}` / `{INPUT_ARRAY}` / `{OUTPUT_ARRAY}`。`RANK_4/RANK_8`、`MAX_INPUT_SLOTS/MAX_OUTPUT_SLOTS` 均为命名常量，定义在 struct.h / tiling_struct.h 中，禁止裸数字）：

```cpp
using TilingData4 = {OP}TilingData<{OP_UPPER}_RANK_4>;
using TilingData8 = {OP}TilingData<{OP_UPPER}_RANK_8>;
template<int RANK>
__global__ __aicore__ void {OP_SNAKE}(
    GM_ADDR {INPUT_GMADDRS}, GM_ADDR {OUTPUT_GMADDRS},
    GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR ins[MAX_INPUT_SLOTS] = {INPUT_ARRAY};
    GM_ADDR outs[MAX_OUTPUT_SLOTS] = {OUTPUT_ARRAY};
    REGISTER_NONE_TILING;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (RANK == {OP_UPPER}_RANK_4) {
        GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling);
        {OP}Kernel<DTYPE_{PRIMARY_INPUT_NAME}, {OP_UPPER}_RANK_4> kernel;
        kernel.Init(ins, outs, &td); kernel.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(TilingData8, td, tiling);
        {OP}Kernel<DTYPE_{PRIMARY_INPUT_NAME}, {OP_UPPER}_RANK_8> kernel;
        kernel.Init(ins, outs, &td); kernel.Process();
    }
}
```

Kernel 类声明模板（`kernel.h.templ`，占位符：`{OP_SNAKE}` / `{OP}` / `{PHYS_NODES}` / `{MAX_INPUTS}` / `{MAX_OUTPUTS}`）：

```cpp
template <typename T, int64_t RANK>
class {OP}Kernel {
    static constexpr int64_t NDDMA_DIMS = (RANK <= 5) ? RANK : 5;
    AscendC::TPipe pipe_;
    const {OP}TilingData<RANK> *td_ = nullptr;
    AscendC::GlobalTensor<T> gmIn_[MAX_INPUT_SLOTS], gmOut_[MAX_OUTPUT_SLOTS];
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[PHYS_NODES];
    AscendC::MultiCopyParams<T, NDDMA_DIMS> nddmaParams_[MAX_INPUT_SLOTS];
    int64_t nddmaOuterIters_[MAX_INPUT_SLOTS], nddmaDims_ = 0;
public:
    __aicore__ inline void Init(GM_ADDR inputs[MAX_INPUT_SLOTS], GM_ADDR outputs[MAX_OUTPUT_SLOTS],
                                const {OP}TilingData<RANK> *td);
    __aicore__ inline void Process();
};
```

## 6 能力边界

> 当前 Broadcast 范式的能力范围与不支持场景

### 6.1 不支持 UB 内 broadcast

本范式的 broadcast 发生在 **GM → UB 搬运阶段**，由 NDDMA（`DataCopy<T, NDDMA_DIMS, cfg>`）的随路 broadcast 完成。**不支持在 UB 内对已驻留的 buffer 做 broadcast**。
