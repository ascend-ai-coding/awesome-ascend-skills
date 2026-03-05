# Tiling 实现指导

Host 侧 Tiling 有两种实现方式：**标准 C++ 定义**与**宏定义**，二者在取值、赋值与写回 context 上差异较大。**参考以 CANN 官方文档为准，本页仅做要点归纳。**

## 官方文档参考（示例与规范来源）

- **Tiling 基本流程（宏定义方式）**：[Host侧Tiling实现 - 基本流程](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00021.html)（TilingData、SaveToBuffer、SetDataSize、SetBlockDim 等）
- **标准 C++ 定义 Tiling**：[使用标准C++语法定义Tiling结构体](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00024.html)（GetTilingData、直接赋值、REGISTER_TILING_DEFAULT、约束与对比）
- **标准 C++ 完整样例**：[Gitee - MatmulCustomMultiCore](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/10_matmul_frameworklaunch/MatmulCustomMultiCore)

---

## 一、标准 C++ 定义 Tiling

- **头文件位置**：Tiling 结构体定义放在**算子工程的 op_kernel 目录下**（该目录会打包进算子包，供在线编译）。
- **结构体形态**：POD（基本类型、数组、结构体/结构体数组、列表初始化）；可引用高阶 API 的 Tiling（如 `AscendC::tiling::TCubeTiling`）通过 `kernel_tiling/kernel_tiling.h`。
- **禁止**：成员函数、指针/引用、虚函数/虚继承/模板类；继承时 `GetTilingData<T>` 的 T 必须与实际写入类型一致。
- **无默认初值**：`GetTilingData<T>()` 不保证初始化，需**显式对所有需下发字段赋值**（或 memset 后赋值）。
- **Kernel 侧**：`REGISTER_TILING_DEFAULT(YourTilingData)` + `GET_TILING_DATA(tilingData, tiling)` 解析使用。

**Host 侧**：`TilingData *tiling = context->GetTilingData<TilingData>();` 得到的是 **context 内部缓冲区指针**，直接对成员赋值即可，**无需** SaveToBuffer/SetDataSize。

```cpp
// 头文件：Tiling 结构体定义在 op_kernel 目录下，见官方「使用标准C++语法定义Tiling结构体」
#include "../op_kernel/xxx_tiling.h"
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    XxxTilingData *tiling = context->GetTilingData<XxxTilingData>();
    memset(tiling, 0, sizeof(XxxTilingData));
    tiling->field1 = value1;
    tiling->field2 = value2;
    context->SetBlockDim(...);
    return ge::GRAPH_SUCCESS;
}
```

---

## 二、宏定义 Tiling

- **头文件与依赖**：结构体一般在 **op_host** 下的 `*_tiling.h`，需包含 `register/tilingdata_base.h`（BEGIN_TILING_DATA_DEF、TILING_DATA_FIELD_DEF、REGISTER_TILING_DATA_CLASS 等）。
- **访问**：仅通过生成的 **set_字段名** / **get_字段名**，不能直接对成员赋值。嵌套时用 TILING_DATA_FIELD_DEF_STRUCT，子结构也需 BEGIN/END_TILING_DATA_DEF 并 REGISTER_TILING_DATA_CLASS。
- **同名冲突**：多算子若用宏定义同名但结构不同的 Tiling，会注册到全局，按名字查找可能拿错；标准 C++ 无此问题。
- **Host 侧写回**：必须先**创建该宏生成类的实例**（栈或成员），set 完所有字段后**显式写回 context**：
  - `tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());`
  - `context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());`
否则 kernel 收不到本次 tiling。

```cpp
// 宏定义方式：先创建 TilingData 实例，set 后必须 SaveToBuffer + SetDataSize，见官方「基本流程」
TilingData tiling;
tiling.set_field1(value1);
tiling.set_field2(value2);
tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
context->SetBlockDim(blockDim);
```

---

## 三、两种方式对比

| 维度 | 标准 C++ | 宏定义 |
|------|----------|--------|
| 定义位置 | op_kernel 下头文件，POD | op_host 下头文件，宏展开 |
| Host 获取 | `context->GetTilingData<TilingData>()` 得 context 内缓冲区指针 | 自己创建 TilingData 实例 |
| 赋值 | 直接 `tiling->field = value` | 仅 set_xxx |
| 写回 context | **不需要**，赋值即生效 | **必须** SaveToBuffer + SetDataSize |
| Kernel 侧 | REGISTER_TILING_DEFAULT(TilingData) + GET_TILING_DATA | GET_TILING_DATA 与注册类对应 |

**参考示例**：标准 C++ 见官方文档「使用标准C++语法定义Tiling结构体」及 Gitee [MatmulCustomMultiCore](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/10_matmul_frameworklaunch/MatmulCustomMultiCore)；宏定义见官方「基本流程」中的 Add 算子 Tiling 示例。
