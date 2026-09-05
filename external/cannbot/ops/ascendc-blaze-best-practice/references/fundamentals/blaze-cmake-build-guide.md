# Blaze 项目 CMake 编写指导

> **定位**：Step 3 在生成包含 CMake configure/build action 的 PLAN 时读取本文，用于形成可执行的初始构建方案。本文不选择 Blaze 组装方案，不要求 Step 3 预判实际编译时才会暴露的配置问题，也不增加专用编译合同字段。

## 1. 先执行的结论

1. **优先使用项目内同源副本。** Kernel 构建使用由 Step 1 建立，或由 PLAN create-only 初始化 action 从当前 Blaze 源码原样物化到项目内只读区的 `blaze/`、`tensor_api/` 和确有依赖的同源公共头，不引用其他项目、历史副本或项目外源码目录。
2. **CANN 只承担必要职责。** CANN 提供编译器、运行时，以及当前 Blaze/tensor_api 没有等价接口时确有必要的头文件和库；不要把宽泛 CANN include 或固定库清单作为默认配置。
3. **配置绑定到具体 target。** 使用 `target_include_directories`、`target_compile_options` 和 `target_link_libraries`，避免全局 include/link 配置掩盖依赖来源。
4. **PLAN 只冻结构建目标和验收结果。** 将项目构建文件登记为可修改目标，建立 configure/build action 与构建 checkpoint；include 顺序、编译选项和链接细节可在 Step 4 根据真实编译证据调整。
5. **保持项目自包含。** 项目源码、项目内副本、生成物和自有依赖位于 `<project-root>/operators/<operator_name>/`；项目外位置只作为明确的工具链或只读资料来源。

## 2. PLAN 中的最小构建信息

若项目需要 CMake 构建，Step 3 使用现有 PLAN 结构记录：

- 在 `target_file_manifest` 中列出项目 `CMakeLists.txt` 或实际构建文件，作为 Step 4 的初始构建基线；
- 在 `ordered_actions` 中写明 configure/build 目标和预期产物；
- 在 `validation_checkpoints` 中写明构建成功判据；
- 在 `failure_rollback` 中要求保留错误和实际命令，并进入 Step 4 排障。

不新增编译状态、环境 manifest 或专用构建合同。Step 3 无需通过预处理、include trace 或试编译提前证明全部构建细节。

## 3. 目标级 CMake 骨架

以下骨架只表达配置归属关系。尖括号内容按当前项目、工具链和 PLAN action 替换，不是固定库清单或可直接复制的 recipe。

```cmake
cmake_minimum_required(VERSION <current-project-minimum>)
project(<project-name> LANGUAGES CXX)

# 仅在当前工具链和项目入口使用 ASC CMake language module 时启用。
find_package(ASC REQUIRED)
enable_language(ASC)

set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(PROJECT_INCLUDE_ROOT "${PROJECT_ROOT}/op_kernel/include")

add_executable(<build-target>
    <authorized-host-source>
    <authorized-kernel-or-wrapper-source>
    <authorized-tiling-source>
)

set_source_files_properties(
    <authorized-asc-source>
    PROPERTIES LANGUAGE ASC
)

target_include_directories(<build-target> PRIVATE
    "${PROJECT_INCLUDE_ROOT}"
    <project-local-blaze-public-include-root>
    <project-local-tensor-api-public-include-root>
    <other-required-project-local-include-root>
)

target_compile_options(<build-target> PRIVATE
    "$<$<COMPILE_LANGUAGE:ASC>:<required-asc-option>>"
    "$<$<COMPILE_LANGUAGE:ASC>:<required-target-option>>"
)

target_link_libraries(<build-target> PRIVATE
    <libraries-required-by-current-project>
)
```

使用骨架时：

- 显式列出 Kernel、Wrapper、Tiling 和 Launcher 源文件，不用目录通配符隐藏 target 输入；
- 只对设备侧源文件设置 ASC 语言和设备编译选项；
- 只加入当前符号和运行时确实需要的库；
- 当前工具链不使用示例中的 ASC CMake module 时，沿用项目实际可用的构建入口，不为匹配示例重写工具链。

### 3.1 Host、ASC Host 与 Device 的分层核验

同一工程可能同时包含普通 CXX Host、由 ASC 联合编译产生的 Host 部分和
Device entry。`__CCE_AICORE__` 等设备宏只描述当前编译上下文，不能作为
跨上下文 API 的唯一边界。把 Host 声明或定义包在设备专用宏内，可能使普通
CXX 看不到它们；把 Host-only 实现暴露给 Device，又可能产生错误的符号或
语言依赖。

因此，PLAN 必须为实际存在的每种上下文写明 source、include 和语言归属，且
至少分别验证：

1. 普通 CXX Host 编译能看到所需 Host 声明和定义；
2. ASC Host（工具链提供该编译阶段时）能看到联合入口需要的 Host 部分；
3. Device entry 能看到 `__aicore__`/设备 helper 及其真实 ABI。

共享头应按上下文拆分声明，或用当前 target 的真实预处理结果证明宏分支在三种
上下文中都闭合。仅看到某个 `.asc` 文件存在 Host 分支，或仅通过普通 CXX
编译，均不能证明联合编译和 Device entry 的符号闭合。

### 3.2 独立 Host/CTest 的可选平台探针

独立 Host 或 CTest target 可能没有运行时注入的完整平台对象；普通 Host
预检中调用平台查询时，`PlatformAscendCManager` 或等价 probe 可能为空。可选
平台探针必须先判空并记录 `probe skipped`，不能把空指针解引用伪装成 Tiling 或
Device kernel 失败。

Host 侧需要的平台事实分为两类：

1. **必需事实**：当前 Host Tiling 算法没有它就无法构造合法 Params，应在该
   target 的构建/运行合同中显式提供并在缺失时阻断；
2. **诊断/优化探针**：只用于打印或选择非冻结路径，可在独立 CTest 中跳过，
   但必须保留跳过原因，并单独验证 Host contract、target link/CTest 注册和
   设备路径。

不能为了让 CTest 通过而伪造平台库、探测零维资源或声明设备 PASS；具体 manager
符号和 fallback 值仍以当前源码、DESIGN/PLAN 和实际 target 为准。

### 3.3 CTest 与生成数据的前置闭合

依赖 `data/input`、Golden 或 manifest 的 CTest 不能假设这些文件已经存在，也
不能从旧目录补读。PLAN/Step 4 必须把数据生成和完整性校验作为 CTest 前置
checkpoint，或者让测试自身在隔离临时目录中完成同一生成动作；生成器必须校验
shape、dtype、`group_list`/metadata 和文件字节数，定向谓词还要在 manifest 中
可复算并由测试入口断言。

运行 CTest 前先用 `ctest -N`（或等价方法）确认预期测试已注册且测试数量非零，
再使用本次 clean build 产物执行。缺失、截断、拒绝生成或 metadata 不匹配是
Host/test-setup 失败，不是 Kernel、Tiling 或设备精度失败；应保留生成命令、
实际输入目录和返回码，修复前不得继续汇总为 `device_verified`。不能用
`set -e`、批处理返回码或历史输出掩盖单 case generator/verifier 失败。

## 4. Blaze/tensor_api 与 CANN 的边界

项目内编译输入通常位于：

```text
<project-root>/ops-tensor/                                      # Blaze 源码调查根，只读
operators/<operator_name>/op_kernel/include/blaze/              # 同源项目副本，只读
operators/<operator_name>/op_kernel/include/tensor_api/         # 同源项目副本，只读
```

构建使用项目内副本，不直接依赖 `ops-tensor/` 调查根。若 Blaze/tensor_api 的真实 include 链需要其他同源公共头，将其复制到项目只读区并以 target 级路径接入；不要引用其他工程或本机历史目录。

复制或适配时不能只按文件名平铺。若源文件使用相对路径或项目根路径
include，目标副本必须保留能使这些 include 解析的相对目录拓扑；有意迁移目录时，
PLAN 必须同时冻结源到目标映射、include root 和必要的 include 改写。复制完成后，
使用目标实际 compiler、语言模式、架构参数和 include roots 执行 include-closure
预检或最小编译探针，确认直接及递归 include 都来自授权项目副本或工具链例外。

CANN 头文件或库只有在以下情况接入对应 target：

- ASC 编译器和设备语言本身需要；
- Launcher、运行时或设备调用没有 Blaze/tensor_api 等价入口；
- 选定 Blaze 组装方案的真实源码依赖直接要求。

接入后仍应避免把整个 CANN include 或 link 目录全局暴露给所有 target。

## 5. 编译期调整原则

初始配置不能保证覆盖所有工具链差异。Step 4 应以实际 configure、compile 和 link 输出为准，在 `<project-root>/operators/<operator_name>/` 内修复或补充实现和构建文件，并重新执行受影响 checkpoint。

静态库目标还必须核对当前 CANN/toolchain 的完整链接闭包：不仅要记录直接
调用的库，还要从实际 link command 和符号解析结果确认其传递依赖、库顺序和
版本来源。某个库“在头文件中可见”不代表它的实现依赖已经进入 target；清洁
链接仍出现未解析符号时，应归类为当前 target 的依赖闭包缺口，不能复制其他
工程或其他 CANN 版本的固定库清单。需要新增外部二进制依赖时回 Step 3 更新构建合同。

允许的实现期调整包括：

- 修正 ASC/host 源文件归属；
- 调整项目内 include 路径和顺序；
- 增减当前 target 的必要编译选项；
- 修正当前项目和既有工具链/运行时依赖的链接配置。

`-iquote` 不是默认选项。只有真实错误和头文件解析结果表明引号形式 include 选中了错误同名头，而项目内同源副本包含所需定义时，才对受影响的 ASC target 增加项目内 quote 路径。它不能修复尖括号 include、ABI 不一致或源码版本不兼容。

如果修复需要改变 Blaze 组装方案、接口/ABI、数据语义、支持范围或验证范围，停止实现并返回 Step 3；需要新的 Blaze 源码事实时先回 Step 2。项目内新增文件不单独触发回退，新的外部依赖仍需回 Step 3 评估。

## 6. 编译器与 include 原理

### 6.1 ASC 语言与设备编译

CMake 的语言启用、源文件 `LANGUAGE ASC` 属性和 target 编译选项共同决定设备侧命令。不同工具链可能以不同方式注入语言、平台和内置头路径，因此最终行为以实际 target 命令为准，不能只看 `CMakeLists.txt`。

### 6.2 引号、尖括号与搜索路径

`#include "..."` 与 `#include <...>` 使用的搜索路径类别不同。普通 `-I`、系统路径、编译器内置路径和 `-iquote` 的顺序取决于实际 compiler 与语言模式。同名头影响编译时，应查看当前 target 的真实命令和必要的预处理/include trace，再做最小调整。

### 6.3 为什么优先同源副本

Blaze 与 tensor_api 的模板、内部 include 和编译期约束通常需要同一 Blaze 源码版本下的定义协同工作。混用项目内副本、CANN 同名头和其他版本源码，可能产生声明、宏、模板或 constexpr 语义不一致。优先使用项目内同源副本，可以将工具链头限制在确有必要的边界内。
