# Launcher 开发指导

本文是按需加载的 host 侧方法，供 Step 3 设计 PLAN、Step 4 实现冻结设计时使用。它不提供 Basic、Grouped、MX 或融合 recipe，也不规定固定工程树；参数语义、Tiling 合同和验证标准以当前 DESIGN/PLAN 为准，具体项目文件可由 Step 4 在实现中补充。

## 1. Launcher 职责

Launcher 负责：

1. 从项目合同解析逻辑 shape、dtype、layout、属性和运行时 dispatch；
2. 按 DESIGN/PLAN 的 Tiling/Params 合同生成或装载 TilingData；
3. 分配和释放 host/device buffer，按物理数据合同执行 H2D/D2H；
4. 用 DESIGN 冻结的 Kernel entry、GM 参数顺序、grid/core、workspace 和 stream 启动设备 Kernel；
5. 输出 PLAN 指定的设备结果和运行记录。

Launcher 不生成随机输入、不计算 CPU Golden、不判定精度、不选择候选、不切换备选、不读取 golden 文件决定行为。输入生成、物理转换、Golden 和比较由 PLAN 指定的项目资料负责。

## 2. 输入合同与文件清单

PLAN 必须先冻结：

```text
argument_schema
logical_to_physical_conversion
buffer_size_and_alignment_rules
input/output artifact paths
tiling provider
kernel ABI and optional arguments
dispatch/specialization rules
workspace lifecycle
```

`target_file_manifest` 与 `data_and_golden_wiring` 提供 Launcher、数据和 CLI 的初始计划，不由本文默认固定名称或数量。Step 4 可以在项目根内补充文件和脚本，并将实际路径记录到 `execution_record`。读取时按设计字节数执行 exact-size 校验，拒绝缺失、截断、非有限或越界输入。

### 2.1 调用方 cwd 与项目根闭合

Runner 必须从脚本自身位置解析本次项目的 `PROJECT_ROOT`，并在数据生成、CMake、
二进制启动和结果读取前统一使用该根目录下的绝对 artifact path，或显式进入该
根目录。不能假设调用方当前 cwd 已经是项目根，也不能让 source CANN 环境脚本或
子 shell 改变业务路径语义。至少要从项目根外的 cwd 做一次路径闭合回归。

启动前检查 PLAN 中登记的输入文件、Golden/输出目录和可执行文件；缺失、截断或
写入失败必须在 Host/Launcher I/O 边界报告，并保留实际 cwd、解析后的根目录、完整
命令和返回码。不能把相对路径找不到归因于 Tiling、Kernel 或设备精度，也不能用
旧目录中的同名文件补证据。

### 2.2 零工作量输入

Launcher 必须先完成输入文件、shape、dtype、group/batch metadata 和输出合同
校验，再判断是否为合法零工作量输入。

若合同允许逻辑行数为零，必须选择以下一种已证明路径：

1. 在调用 tiler、分配 device buffer 和启动 kernel 前产生合同规定的空输出；
2. 使用源码和设备证据证明支持零维的 tiler/kernel 路径。

不得通过把零维输入试探性传入只支持正维度的 tiler 来判断其行为。Host 空路径
仍须校验 group endpoint、输入字节数和输出 shape，不能因不启动 kernel 而跳过
合同校验。测试记录必须区分空 Tensor 与 `R>0` 的全零数值行，并记录实际处理
路径是 Host 空路径、tiler specialization 还是正式 kernel entry。

## 3. ACL 会话与资源生命周期

按项目的设备上下文包装创建/销毁：

```cpp
ACL_CHECK(aclInit(nullptr));
ACL_CHECK(aclrtSetDevice(deviceId));
ACL_CHECK(aclrtCreateContext(&context, deviceId));
ACL_CHECK(aclrtCreateStream(&stream));

// H2D -> kernel launch -> synchronize -> D2H

ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtDestroyContext(context));
ACL_CHECK(aclrtResetDevice(deviceId));
ACL_CHECK(aclFinalize());
```

真实工程必须对每个成功分配的 host/device/workspace 资源建立异常路径和释放顺序。`deviceId`、stream 数量和内存 flag 由项目合同决定，不能作为场景默认。

## 4. 逻辑数据到物理 buffer

对每个 Tensor 分开计算：

```text
logical shape/dtype
storage dtype/packed bytes
layout pattern (ND/NZ/other)
transpose/packing/padding
alignment and physical shape
row/plane stride and offset units
valid/tail range
```

ND buffer 通常按有效逻辑元素计算，块化/packed layout 必须按物理块、C0、padding 和打包单位计算。不要把逻辑 shape、Copy extent、UB pitch 和 GM row bytes 混为一个 size。量化 scale、group metadata、broadcast operand、workspace 和输出分别建立 size/offset 合同。

### 4.1 逻辑到物理物化的独立 witness

当 Launcher 或其数据 producer 需要把逻辑 Tensor 转成另一种设备物理表示（包括
paired view、pack、transpose、alignment 或 padding）时，逻辑 ABI 仍然是对外合同，
物理 buffer 是设备实现细节。PLAN 必须登记转换步骤、物理 shape、storage span、
offset/stride 单位、padding 规则和首个设备消费者，并把它们接入同一
`abi_crosswalk`；只替换地址而保留逻辑步长或 gate offset 不算闭合。

对可以预先生成的物理输入，使用独立于 Launcher 适配代码的 data path 生成
physical witness。Launcher 从本次 case 的逻辑输入重新生成 H2D buffer，在 ACL H2D
之前执行 shape、字节数和逐字节比较；比较失败必须在 Host ABI/I/O 边界返回错误，
不能启动 Kernel，也不能用设备精度失败反推布局。Golden 和 manifest 记录最终实际
H2D 的 physical bytes、转换后的地址/顺序和 hash，而不是只记录逻辑文件。若没有
逻辑到物理转换，逻辑 buffer 直接作为 physical fact source，不需要额外 witness。

独立 witness 的目的，是避免 generator 与 Launcher 共享同一个错误的转换公式而
同时“自洽通过”；具体 paired 宽度、tile、padding 值和 byte order 仍由当前 DESIGN/
PLAN 与目标 CANN witness 冻结，不能从本节推导。

## 5. Tiling 与 Params

host Tiling 先逐字段对照 Investigation 的 Scheduler Params Semantic Contract：

1. 若已有项目 Tiling Engine 的类型、字段、单位、合法域、交叉约束和输出 ABI 完全兼容，PLAN 可授权复用或最小适配；
2. 若 Blaze 源码没有 host tiling，但 device 侧语义、合法域和 ABI 已闭合，PLAN 可授权项目 Engine 直接返回当前 partition 已证明的固定合法控制值；
3. 需要改变 partition、增加 specialization 或缺少合法域时回 Step 2/3；Launcher 不猜值；
4. TilingData、Kernel Params、Block/Scheduler/Epilogue Params 和 host 参数逐字段检查类型、顺序、单位和生命周期；
5. `usedCoreNum`、grid、workspace 大小和 dispatch flag 只能来自 DESIGN/PLAN 的冻结合同。

## 6. ABI 合同、签名骨架与启动绑定

Step 3 先在接口合同中冻结 `abi_mapping_draft`，再以每个选定 Blaze 组装方案的同一真实 witness 在 `matmul_base_analysis.abi_bindings[]` 中冻结准确 `kernel_abi_contract`。Launcher 不重新解释这些合同，只把与当前 `design_binding_ref` 对应的映射编译为 PLAN action。

### 6.1 合同表和交叉表

每个 `abi_bindings[]` 记录至少包含 `design_binding_ref`、`partition_ids`、`assembly_witness_ref` 及其 `kernel_abi_contract`：入口修饰符/链接、入口符号、模板与具体 specialization、有序 GM 参数及方向/可空语义、TilingData/Params、workspace、grid/usedCore、Wrapper、dispatch、final/partial 生命周期和每项 `source_ref`。

`abi_crosswalk` 对每个逻辑参数、输出和必要辅助 ABI 对象建立一行，并为每行提供稳定 `crosswalk_row_id`：

| 合同列 | 必填内容 |
|---|---|
| 逻辑对象 | `crosswalk_row_id`、参数或辅助 ABI 对象的 ID、角色、必选/可选语义 |
| 物理表示 | buffer、storage dtype/layout、字节范围、offset/stride 单位、有效范围 |
| host 接线 | Launcher 参数、所有者、分配/释放生命周期、H2D/D2H 方向 |
| device 接线 | Wrapper 参数、有序 Kernel GM 参数或入口绑定、可空条件 |
| 控制接线 | TilingData/Params 字段，或有证据的 `not_applicable`；workspace、grid/usedCore、stream/dispatch |
| 消费者 | Block、Kernel、Epilogue 或最终输出消费者，以及 final/partial 时机 |
| 证据 | 同一真实 witness 的 `source_refs` 和 DESIGN 合同 ID |

不同 `design_binding_ref` 的 crosswalk 行不能互换或拼接。场景只可用 `abi_crosswalk_delta` 追加自身 operand、输出、Params、Wrapper 或同步接线；不能重写基础 MatMul 行。所有实际 buffer、启动和 Wrapper action 都必须引用对应 binding 的 crosswalk 行。

若 Launcher 通过物化或预打包把逻辑 Tensor 适配为另一物理表示，必须更新同一
`abi_crosswalk` 行中的 buffer 地址、Pattern/layout、shape、`lda`/stride、
storage span、Copy specialization 和最终消费者；不能只替换地址而保留原物理
步长。PLAN 还必须登记一组可观察的正负回归：完整重绑定应通过，packed 内容
正确但下游使用 stale `lda`/stride 的对照必须失败。

### 6.2 从真实 witness 形成签名骨架

签名骨架是 DESIGN 中的可审计声明形态，不是可直接复制的实现代码。它的结构固定为：

```cpp
<observed linkage and entry modifier>
void <observed entry symbol>(
    <ordered GM parameter 0>,
    ...,
    <observed TilingData/Params parameter(s)>,
    <observed workspace or control parameter(s)>) {
  <observed Wrapper or selected Kernel invocation>;
}
```

只有真实 witness 已明确的 token 才能替换尖括号中的描述。不得因本文骨架自行加入 `__cube__`、`__mix__`、`GM_ADDR`、模板参数、GM 参数顺序或 Wrapper 调用方式；任何一个未闭合的 token 都是源码事实缺口，不是 Launcher 的自由选择。

### 6.3 启动绑定和闭合门禁

PLAN 的启动绑定必须以合同表逐项写明：

```text
entry modifier and symbol
selected template/specialization
ordered GM arguments and physical byte rules
TilingData address/size and Params mapping
workspace address/size/lifetime
grid/usedCore and dispatch/stream
Wrapper/entry invocation binding
source-backed signature skeleton ref
```

启动前必须证明每个必需逻辑对象都可追到设备消费者，且输出路径、optional/null 语义、物理字节/offset 单位、Tiling/Params、workspace、grid/usedCore 和 Wrapper/entry 全部闭合。Blaze 源码事实缺失时返回 Step 2 的一次补充调查；接口或冻结合同本身缺失时回 Step 3；仅 PLAN 文件映射或实现不完整时由 Step 4 在项目内补齐并记录。需求本身不明确时等待用户澄清。在任何一种语义或 ABI 缺口未解决前，不允许启动 Kernel。

### 6.4 启动 ABI 的最小穿刺

在运行完整矩阵前，先用当前真实 entry、GM 参数顺序、Tiling/Params 传递方式、
grid 和 stream 编译一个最小启动 probe。probe 必须覆盖 concrete binding 的
入口修饰符、符号、参数类型，以及 TilingData 是按值、指针还是引用传递；不能只
检查一个函数名能否被二进制查找到。

若项目选择直接按二进制符号查找作为诊断，符号查找成功只能说明 lookup 成功，不能
证明 typed Kernel ABI、POD 布局或设备指令路径正确。lookup 后仍须完成同一版本
ASC witness 的 typed launch 编译和一次 device-visible 运行；若首个运行错误落在
ABI 或设备指令边界，保留该首错并修正启动绑定，不要为了迎合猜测的 ABI 修改
Kernel。每次绑定修复都要 clean rebuild，并在 `execution_record` 记录实际 entry、
命令、目标架构、CANN 版本和返回码。

## 7. Kernel 启动与 ABI

启动前依次校验：

- required/optional GM 参数顺序、方向、空值语义和物理字节数；
- TilingData/Params 地址和大小、workspace、grid/core、stream 和 entry modifier；
- layout/transpose/dtype/shape/dispatch 是否在 DESIGN 支持域；
- Buffer 不重叠、offset/stride 不越界，融合额外输入和输出满足 broadcast/生命周期合同；
- custom 文件、wrapper 和 build target 都在 PLAN 白名单。

不要从 Kernel 名称或相似工程猜测 `__cube__`、`__mix__`、模板参数、ratio 或参数顺序。实际 entry/Wrapper 必须来自 Investigation 和 DESIGN 的 concrete witness。

本 Skill 生成 direct-invoke launcher。若 concrete binding 已证明为 MIX entry，
入口声明固定使用 witness 对应的 `__global__ __mix__(aicCount, aivCount)`，并由
Launcher 通过 `entry<<<blockDim, workspace, stream>>>(...)` 直调；不得再叠加
ACLNN 路线的 `KERNEL_TASK_TYPE_DEFAULT(...)`。是否使用 MIX 以及具体 ratio 仍由
binding 决定，不能把 MIX 强加给 Cube-only entry。

自定义 POD 从 `GM_ADDR` 读取时，设备侧 pointer cast 必须保留 `__gm__` 地址空间
限定；主机普通指针类型即使字段布局相同，也不能作为设备 GM pointer 类型使用。
把这一点纳入真实 ASC 编译门禁，避免只通过 Host C++ 静态检查。

## 8. Dispatch 设计

只有需求要求 runtime 变化时才建立 dispatch。Step 3 必须决定：

```text
dispatch_axes
compile_time_specializations
runtime flags/enums
partition coverage per specialization
rejection for unsupported combinations
```

Launcher 可以采用分层 dispatch 或显式表，但不能自动生成所有 layout × transpose × dtype 的笛卡尔积。每个 specialization 的 Tiling/Params/ABI 和支持域要有独立 source refs；无证据的组合返回明确错误。

## 9. 数据生成、Golden 和结果输出

Launcher 与数据/Golden 组件的边界固定为：

```text
data producer: logical values -> physical buffers
launcher: buffers -> device Kernel -> raw output
golden/verifier: logical Golden + raw output -> threshold/nonfinite decision
```

五模式、known-C、SplitM、slot、broadcast、量化 metadata 或诊断输出只有在 DESIGN/PLAN 声明时才进入 Launcher 参数。Launcher 不现场生成 Golden，也不因 golden 失败自行调整阈值或重试备选。

### 9.1 输出文件归属

当 DESIGN/PLAN 只有一个正式输出时，Launcher 可以采用
`data/output/output.bin` 作为单输出默认文件名；该默认值不代表所有算子都只有
一个输出。存在 `y`、`yScale` 或其他多输出时，DESIGN 必须冻结每个输出的文件名、
顺序和 dtype/shape 元数据，所有正式输出统一写入 `data/output/`。Launcher
不得按参数名临时猜测后缀、覆盖另一个输出或从旧目录补读结果；实际路径必须
写入执行记录，并由 verifier 按同一份输出清单读取。

### 9.2 Launch 与 verifier 的证据隔离

Launcher 只负责把当前输入送入设备、等待并保存 raw output；verifier 是独立的
进程或明确的后续 action，读取同一 case 的输入、Golden、输出清单和元数据。一个
case 的交付结论必须同时具备 launch return code 和 verifier return code；只启动
成功、没有 verifier 结果或 verifier 没有执行，状态是 `incomplete`，不是 PASS。

矩阵 runner 必须为每个 case 保留输入、raw output、stdout/stderr、两个返回码和
超时/终止状态。`set -e`、设备重置、环境脚本或批处理循环不能吞掉 verifier 的结果；
批量汇总只能消费已经单 case 闭合的记录。直接启动和批处理启动的结果应先分别闭合，
再做汇总比较，不能用旧目录输出、历史 verifier 或 wrapper 的零返回码补证据。

超时数值、文件名和具体 shell 实现由项目 PLAN 冻结；本节只规定证据边界和状态
语义。

## 10. 构建和安全边界

PLAN 提供项目 include、CMake target、Tiling library、Kernel wrapper 和 launcher source 的初始计划；Step 4 可以在项目根内调整或补充实现。三个官方源码区和 Skill Asset 原文件只读；编译错误涉及官方区时只读采集证据，并按版本问题回 Step 1、源码事实问题回 Step 2 或设计问题回 Step 3，不能直接打补丁。

### 10.1 CANN 根目录发现

Standalone 工程不得假定某个固定 CANN 环境变量始终存在。CMake 必须先读取当前
CANN package/toolchain 实际导出的安装根信息，再归一成项目唯一的
`CANN_ROOT`，所有 include、library、编译器和工具路径都从该根派生。若当前环境
同时提供旧、新变量，可显式规定优先级；若解析结果为空、目录不存在或不同来源
指向不同安装，配置阶段立即失败，不能让路径静默退化为 `/include` 或混用两套
CANN。

例如，目标环境可能提供 `ASCEND_HOME_PATH`，也可能由 CMake package 提供
`ASCEND_CANN_PACKAGE_PATH`；它们是发现候选，不是跨版本固定 ABI。构建记录至少
包含最终解析的 `CANN_ROOT`、ASC 编译器和目标 SoC，clean configure 必须验证
这些对象来自同一安装。

### 10.2 Host、Tiling 与 Device 的符号所有权

同一 Host/ASC 联合翻译单元中，项目 helper 必须有明确且互不重叠的所有者：

- Host 资源、文件和 ACL helper 放入项目 Host 命名空间；
- 项目 Tiling helper 放入项目 Tiling 命名空间，并以限定名调用外部 Tiler
  helper；
- Device helper 放入项目 Device 命名空间或具体 Kernel/Block 类；
- Kernel entry 保持 Investigation 和 ABI 合同要求的 linkage 与作用域，不为
  统一风格擅自包入命名空间；
- 禁止翻译单元级 `using namespace` 把 Host、Tiling 或 Device helper 重新注入
  同一查找域。

命名空间隔离不能替代 include 隔离。未经审计不得直接引入在全局命名空间定义
`CeilDiv`、`Align`、错误检查宏或其他通用符号的 example/helper 头文件；Launcher
只实现当前 PLAN 所需的最小 Host helper。也不得用
`namespace project { #include "external.h" }` 包裹外部头文件，因为这会改变
外部声明所属命名空间并可能破坏类型或 ABI。最终门禁必须编译真实
Host+Tiling+Device 翻译单元和真实 launch 表达式，仅编译 Host helper 或 Device
函数体不能证明符号查找与 ABI 闭合。

每次构建/运行记录：

```text
project contract IDs
source/源码版本一致性
CANN_ROOT/ASC compiler/target SoC
Tiling/Params values and units
actual grid/usedCoreNum
entry specialization
buffer sizes/addresses (non-sensitive summary)
result artifact and checkpoint
```

清理和最终回归按照 PLAN `cleanup_contract`、`validation_checkpoints` 和 `execution_record` 中的实际临时产物执行。本文只提供 host 方法，不定义交付命令、固定脚本、固定数据目录或固定文件名。

### 10.3 环境初始化与业务参数隔离

Runner 或 shell launcher 在 source CANN 环境脚本前后必须保持业务 `argv` 和已解析
的 shape/dtype/group metadata 不变。环境脚本可能在当前 shell 中改写位置参数；包装
脚本应先保存并在 source 后恢复 `argv`，再重新解析或校验实际传给 generator、CMake
和 executable 的参数。变量命名不得覆盖 shell 的特殊变量；参数缺失、group endpoint
截断或 shape 改写必须在生成输入和启动构建前失败。具体环境脚本路径和 shell 实现
属于项目恢复记录，不构成 Blaze 算子 ABI 规则。

### 10.4 跳过构建时的产物绑定

`SKIP_BUILD` 或等价的跳过构建开关只能在 runner 已验证默认构建目录存在、且该
目录由当前 source/design contract clean-build 产生时使用。若工程允许多个构建目录，
必须显式传入并记录被执行的 binary 路径、构建目录和 source identity；不能把
`build_final`、历史目录或其他 contract 的 binary 默认为 runner 的 `build/`。
缺少 launcher、目录不匹配或产物 identity 无法证明时，首个失败边界是
Host/test setup，状态为 `incomplete`，不得启动设备或归因到 Kernel 精度。恢复后
必须通过同一公开 runner 重新生成/绑定产物，再保存 launch RC、verifier RC 和 raw
output；该规则只约束证据绑定，不规定具体目录名称。
