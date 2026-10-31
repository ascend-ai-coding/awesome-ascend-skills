# broadcast 范式 Kernel 入口

> Kernel 入口函数注册、模板分发、TilingData 获取的统一规范

## 1 kernel 入口

**参考源码**: `example/examples-adam-code/op_kernel/adam_apply_one_assign.cpp`, `example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_struct.h`

**约束**:
- `DTYPE_IN0` 编译期实例化（dtype 不走 TilingKey）
- TPipe pipe; 统一在入口申请 TPipe，通过 Init 以指针的方式传入各模板中
- kernel 入口函数的模板参数，必须与 TilingKey 字段一一对应，顺序一致
- 入口文件命名为 `{op_snake}.cpp`，kernel 入口函数名须与 def 注册的 `opFile.value` 一致（受框架约束使用 snake_case，文件名不带 `_apt` 等后缀）
- 其他
  - `using TilingData4 = BroadcastTilingData<BROADCAST_RANK_4>` — 交叉编译器不能在宏里处理 `<>`，用 using 别名；`RANK_4/RANK_8` 用命名常量，不用裸数字 4/8
  - `ins`/`outs` 数组大小用 `MAX_INPUT_SLOTS`/`MAX_OUTPUT_SLOTS` 命名常量（定义在 tiling_struct.h），不用裸数字
  - `REGISTER_NONE_TILING` 必须写在 `GET_TILING_DATA_WITH_STRUCT` 之前，否则宏不认类型名
  - `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` 声明核类型，模板化算子必须
  - kernel.h 必须 include struct.h（含 `ASCENDC_TPL_ARGS_DECL`），否则构建系统走 `#define` 包装器路径，与 `template<int RANK>` 不兼容
  - 交叉编译器限制：宏参数里不能有 `<>`（模板尖括号被误解析为比较运算符）；`reinterpret_cast` 不允许（`__gm__` 到非 `__gm__`）；`__gm__` 到 `__gm__` 的 C 风格转换允许

### 1.1 kernel 入口知识

**内容**：
- 入口文件命名：`{op_snake}.cpp`，须与 def 注册的 `opFile.value` 一致（不带 `_apt` 等后缀）
- 模板参数与TilingKey的对应关系：`template<int RANK>` 模板函数，RANK 与 TilingKey 的 RANK 字段一一对应
- `REGISTER_NONE_TILING` / `REGISTER_TILING_DEFAULT` 选择：多 TilingData 场景用 `REGISTER_NONE_TILING`（RANK=4 和 RANK=8 各一个 struct，模板参数不同值时 TilingData 结构体大小不同）
- `GET_TILING_DATA_WITH_STRUCT` 获取 TilingData：`if constexpr (RANK == 4)` 分支内 `GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling)`
- `if constexpr` 分发到不同 kernel 类：编译期 `if constexpr (RANK == 4)` 替代废弃的 `TILING_KEY_IS`
- dtype 通过 `DTYPE_IN0` 编译宏注入，永远不硬编码 `float` 或 `half`

**多 TilingData 关键坑汇总**：

| 坑 | 说明 |
|----|------|
| `<>` 在宏里 | 交叉编译器把 `BroadcastTilingData<4>` 的 `<` 当小于号。用 `using TilingData4 = ...` 别名 |
| `REGISTER_NONE_TILING` | 必须写在 `GET_TILING_DATA_WITH_STRUCT` 之前，否则宏不认类型名 |
| `KERNEL_TASK_TYPE_DEFAULT` | 声明核类型（AIV），模板化算子必须 |
| kernel.h 必须 include struct.h | 构建系统读到 `ASCENDC_TPL_ARGS_DECL` 才知道这是模板化算子，走模板编译路径而不是 `#define` 包装器 |
| 模板函数与 `#define` 不兼容 | 构建系统对非模板函数生成 `#define func func_0_tilingkey` 包装器。模板化算子不触发这个 |

**编译系统关键约束**：

| 错误 | 阶段 | 原因 |
|------|------|------|
| `unexpected type name` | INFERCHANNEL | `GET_TILING_DATA_WITH_STRUCT` 缺少 `REGISTER_NONE_TILING` |
| `no matching function for call to 'func_X_tilingkey'` | FATBIN | kernel.h 缺 `#include struct.h`，构建系统走了 `#define` 包装器 |
| `cast not allowed in aicore` | INFERCHANNEL | 试图 `reinterpret_cast` 跨地址空间 |
| `undeclared identifier 'GetBlockIdx'` | INFERCHANNEL | 缺 `AscendC::` 命名空间前缀 |

**TilingKey bitwidth 选择**：`ASCENDC_TPL_UINT_DECL(NAME, BITWIDTH, ...)` — 数值型（RANK=4/8）→ `8`；布尔型 → `1`。bitwidth 一经定义不可更改（影响后续参数编码偏移）。

**必需配置文件**：
- `simplified_key.ini`：`[{OpType}]\ndefault=0`
- `binary.json`：每种 dtype 组合一个 `bin_filename`，列出每个 input/output 的 name/index/dtype/format
- CI 算子列表：按字母序加入算子名

### 1.2 kernel入口代码

以 arch35 为例，举例如下：

```cpp
#include "kernel_operator.h"
#include "arch35/broadcast_kernel.h"
#include "arch35/broadcast_tiling_struct.h"

// 交叉编译器不能在宏里处理 <>, 用 using 别名
// RANK_4/RANK_8 命名常量与 ins/outs 数组大小 MAX_INPUT_SLOTS/MAX_OUTPUT_SLOTS 均定义在 struct.h / tiling_struct.h 中
using TilingData4 = BroadcastTilingData<BROADCAST_RANK_4>;
using TilingData8 = BroadcastTilingData<BROADCAST_RANK_8>;

template<int RANK>
__global__ __aicore__ void op_broadcast(
    GM_ADDR in0, GM_ADDR in1, GM_ADDR out0,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR ins[MAX_INPUT_SLOTS]  = {in0, in1};
    GM_ADDR outs[MAX_OUTPUT_SLOTS] = {out0};

    REGISTER_NONE_TILING;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (RANK == BROADCAST_RANK_4) {
        GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling);
        BroadcastKernel<DTYPE_IN0, BROADCAST_RANK_4> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(TilingData8, td, tiling);
        BroadcastKernel<DTYPE_IN0, BROADCAST_RANK_8> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    }
}
```
