# Tiling（`*_tiling.h` / `*_tiling.cpp`）

tiling文件负责输入输出shape的校验与device侧所需常量的获取。该文档中，只解释相关逻辑如何在catlass中查找，具体实现请参考ascendc-operator-tiling-code-gen skill。
**重要提示**：`tiling.cpp` 中必须进行函数注册，参考 **ascendc-operator-tiling-code-gen** 中的 `example_tiling.cpp`。

## `*_tiling.h`：Tiling 数据结构

- **字段来源**：catlass `kernel/*_kernel.hpp` 的 `Params` 结构体中，除 `gmA`/`gmB`/`gmC`（内存指针）和 `layoutA`/`layoutB`/`layoutC`（布局对象）外的字段，即为 tiling 所需常量值。
- **典型字段**：`m`、`n`、`k`（GemmCoord 维度）。
- **编写方式**：使用 `BEGIN_TILING_DATA_DEF` / `TILING_DATA_FIELD_DEF` / `REGISTER_TILING_DATA_CLASS` 宏。

## `*_tiling.cpp`：Workspace

固定写法（与 [kernel-rules.md](./kernel-rules.md)「Workspace 固定写法」一致；说明见 [如何使用 workspace](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_0092.html)）：

```cpp
#include "tiling/platform/platform_ascendc.h"
// ...
size_t userSize = /* Catlass GetWorkspaceSize 或设计文档 */;
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
uint32_t sysSize = ascendcPlatform.GetLibApiWorkSpaceSize();
size_t *ws = context->GetWorkspaceSizes(1);
ws[0] = static_cast<size_t>(userSize) + static_cast<size_t>(sysSize);
```

## `*_tiling.cpp`：TilingKey

- **Host 侧**：根据 dtype、**转置布尔属性**（与 OpDef 一致，勿用 shape 猜）等设计并设置 key，调用 `context->SetTilingKey(key值)`。
- **Kernel 侧**：通过 `TILING_KEY_IS(key)` 分支实例化不同模板（与 [kernel-rules.md](./kernel-rules.md) 对应）。
- **Key 编码**：根据实际需求设计。
- **函数注册**：必须在 tiling.cpp 中使用 `IMPL_OP_OPTILING` 宏注册 Tiling 函数。
    - 假设函数叫做 `CatlassTilingFunc`，算子名为`CatlassOp`
    - 注册语句应写为`IMPL_OP_OPTILING(CatlassOp).TilingFunc(CatlassTilingFunc)`
    - 可参考 **ascendc-operator-tiling-code-gen** 中的 `example_tiling.cpp`
    - 示例中可能存在`TilingParse`函数的注册，可以构建一个空结构体和直接返回默认值的函数来注册。

## 检查清单（tiling）

- [ ] Tiling 数据结构字段与 catlass kernel 的 Params 一致
- [ ] `*_tiling.cpp` 中使用 `IMPL_OP_OPTILING` 宏注册 Tiling 函数
- [ ] TilingKey 覆盖范围与设计文档一致
- [ ] Workspace：按上文固定写法 `userSize + GetLibApiWorkSpaceSize()`
- [ ] 对于是否转置的状态的获取，直接从attrs获取，而不是从shape推导；如果是方阵情况，无法推导。

文件落盘路径见 [code-structure.md](./code-structure.md)；端到端顺序中 op_host 步骤见上级目录 [SKILL.md](../SKILL.md)「Step 4」。
