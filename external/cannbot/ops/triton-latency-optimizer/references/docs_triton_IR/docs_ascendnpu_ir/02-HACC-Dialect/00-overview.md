# HACC 方言总览

## 1. 简介

HACC（Heterogeneous Async Computing Call，异构异步计算调用）方言是 AscendNPU-IR 中用于描述 Host-Device 异构计算模型的核心方言。它定义了函数的 Host/Device 归属、NPU 设备规格参数、Kernel 参数类型以及 Host-Device 函数绑定关系，为后续的编译流程提供了异构语义基础。

- **方言名称**：`hacc`
- **C++ 命名空间**：`::mlir::hacc`
- **依赖方言**：`mlir::DLTIDialect`

## 2. 核心概念

| 概念 | 说明 |
|------|------|
| HACCFuncType | 函数的 Host/Device 归属分类 |
| DeviceSpecEnum | NPU 硬件规格参数枚举 |
| TargetDeviceSpecAttr | 目标设备规格属性，映射具体 NPU 型号 |
| KernelArgType | Kernel 参数类型分类 |
| HostFuncType | Host 端函数角色分类 |
| HACCFunctionInterface | 异构函数接口，提供查询与设置方法 |

## 3. 源码位置

| 文件 | 路径 |
|------|------|
| 方言基类 | [HACCBase.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/IR/HACCBase.td) |
| 属性与枚举 | [HACCAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td) |
| 接口定义 | [HACCInterfaces.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/IR/HACCInterfaces.td) |
| 变换 Pass | [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Transforms/Passes.td) |

## 4. 方言定义

```tablegen
def HACC_Dialect : Dialect {
  let name = "hacc";
  let description = [{
    Heterogeneous Async Computing Call (HACC) dialect.
  }];
  let cppNamespace = "::mlir::hacc";
  let useDefaultAttributePrinterParser = 1;
  let dependentDialects = ["mlir::DLTIDialect"];
}
```

## 5. 与其他方言的关系

```
HACC 方言
  ├── 被 HFusion 方言依赖（HFusion 依赖 hacc::HACCDialect）
  ├── 被 Scope 方言间接使用（scope.scope 可携带 tcore_type 等属性）
  ├── 与 DLTIDialect 协作（设备规格通过 DLTI 机制存储）
  └── 为 HIVM 层提供 Host-Device 调用约定基础
```

## 6. 典型 IR 示例

```mlir
module {
  func.func @host_kernel(%arg0: tensor<?x?xf16>)
    attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
    return
  }

  func.func @device_kernel(%arg0: tensor<?x?xf16>,
                           %arg1: i64 {hacc.arg_type = #hacc.arg_type<tiling_data>})
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>,
               hacc.tiling_function = #hacc.tiling_function<@tiling_func>} {
    return
  }
}
```

## 7. 文档索引

| 文档 | 内容 |
|------|------|
| [01-function-management.md](01-function-management.md) | HOST/DEVICE 函数类型与 HACCFunctionInterface |
| [02-device-specification.md](02-device-specification.md) | NPU 设备规格参数与型号映射 |
| [03-kernel-args.md](03-kernel-args.md) | Kernel 参数类型完整列表 |
| [04-host-device-binding.md](04-host-device-binding.md) | Host-Device 函数绑定关系 |
| [05-transforms.md](05-transforms.md) | HACC 变换 Pass |
