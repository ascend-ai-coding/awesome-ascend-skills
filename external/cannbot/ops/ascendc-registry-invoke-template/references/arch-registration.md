# 注册算子工程的平台注册规范

**使用时机**：新增目标支持、维护 `_def.cpp`、调整平台注册方式或排查目标配置缺失时读取。适用于注册算子工程，架构与 SDK 范围按实际依赖核实。

## 1. 确认工程采用的注册方式

读取 `_def.cpp`、构建依赖与工程约定，确认采用直接 `AddConfig` 还是架构 helper。使用 `opbase::ArchConfigHelper` 的工程，检查实际依赖中 `op_arch_helper.h` 的声明、实现及架构映射，再调用 `AddConfigByArch`。

以下仅示意 helper 的调用形式，使用前须核对实际依赖的声明与映射：

```cpp
#include "op_common/op_host/util/op_arch_helper.h"

opbase::ArchConfigHelper::AddConfigByArch(
    this->AICore(), Ops::Base::DAV_3510, aicConfig);
```

示例中的枚举不是所有目标的通用配置；目标枚举与映射需读取所用 helper 确认，并与目标 SDK/平台接口识别结果对应。目标支持由实际展开配置证明，验收时检查：

- 目标架构对应的配置完整，重复注册按工程规则处理。
- 原算子输入输出的 dtype、format、属性与动态 shape 配置保持一致。
- helper 的头文件与链接实现来自匹配版本，构建依赖可解析。

## 2. 与其他工程接入点协同

平台注册、构建 compute unit、kernel 源选择、binary 配置、API/图/tiling 分发是各自独立的接入点。按 [多芯片开发指南](npu-arch-adapt.md) 检查实际工程与依赖链；运行时记录加载库、kernel 与目标分支，证明实际调用到预期实现。

## 核查依据

- 平台身份和架构：以目标 CANN SDK、配套官方文档与平台接口为主要依据，按 `npu-arch` 的 `references/npu-hardware-params.md` 核验。
- 注册 API 与 helper：从实际构建 include/链接依赖定位声明和实现；`ArchConfigHelper` 属于工程依赖时，不将其假定为所有 CANN SDK 均提供的接口。
- 使用案例：可在实际工程或兼容版本的 [ops-nn 官方仓](https://gitcode.com/cann/ops-nn) 检索 `mat_mul_v3_def.cpp` 和 `AddConfigByArch`。仅辅助理解调用方式，不绑定提交或行号；实际接口、可用架构与展开结果仍以本工程依赖为准。
