# Blaze 编译问题排查

## 适用范围

用于 Step 4 中实际发生的 CMake configure、ASC/C++ 编译、链接、头文件解析和项目构建边界问题。不处理运行时精度、layout、同步或定制场景专属问题。

## 触发信号

- CMake configure/generate 失败；
- ASC 源文件未按设备语言编译，或设备编译选项缺失；
- Blaze/tensor_api 头缺失，或同名符号、宏、模板、constexpr 语义冲突；
- `undefined reference`、入口符号或模板实例无法解析；
- 实际命令引用项目外历史目录、错误源码副本或旧构建路径；
- 构建结果与当前 action/checkpoint 的预期不一致。

## 先做什么

先保留完整错误输出，并查看失败 target 的实际 configure、compile 或 link 命令。根据错误只检查相关信息：源文件语言、include 来源、头文件解析路径、声明/定义、链接输入或缓存路径。不要在没有实际命令和错误证据时猜测库名、include 顺序或设备选项。

## 构建前 preflight

第一次完整构建前，先对目标工程做一次最小闭合检查。它的目的不是提前证明算子正确，
而是避免把构建配置、共享头和 ABI 的问题误报成 Kernel 或设备问题：

1. **目标架构进入真实设备编译命令。** 从 clean configure 产生的实际 ASC command
   中确认目标 SoC/架构和设备语言选项，而不是只看 CMake cache、环境变量或 README。
   目标选项缺失时先修 target 配置，再看后续诊断。
2. **宏的定义语义与取值语义分开。** 对 `#ifdef`、`#if` 和命令行
   `-DNAME=0/1` 做最小预处理检查；值为 `0` 的宏仍然是“已定义”，不能据此判断
   分支被关闭。CPU debug、Host selector 和 Device selector 需要分别记录实际宏集。
3. **共享头的来源和工程 include 根闭合。** 用同一语言、编译器和参数查看 include
   trace/预处理结果，确认项目副本和当前工具链头来自同一版本；缺少项目根 include
   时补 target 级路径，不用系统头或历史工程副本碰运气。
4. **符号先在真实 link 闭合。** 对 `undefined reference` 或设备 link 缺符号，
   同时检查定义是否进 target、实际 archive/shared library 是否含该符号（可用
   `nm` 等只读工具核对）以及该库是否出现在实际 link command；头文件可见不等于
   链接闭合。
5. **当前 SDK 的签名和命名空间先做最小 compile probe。** 对入口修饰符、Params
   引用限定、RegBase/PIPE 等名称和 Host/Device helper，直接以当前头文件做最小
   调用编译。不要从相似版本、类型 alias 或命名猜测 API 所属命名空间。
   自定义 RegBase Epilogue 还应分别 probe 实际使用的 elementwise、reduction、
   rounding 和 store API；相似的动词名或 Host 可见声明不能证明 device 语义、
   lane 结果或地址空间已经闭合。
6. **三种上下文分别 clean 验证。** 普通 CXX、ASC Host 和 ASC Device/link
   可能走不同宏、地址空间和符号实例；其中一侧通过不能关闭另外两侧门禁。每次
   修改目标、宏、include 或 link 后重新 configure，避免旧缓存掩盖结果。

这些检查来自实际出现过的“设备架构未进入 ASC 命令”“`-D...=0` 仍触发
`#ifdef` 分支”“项目头未被解析”“库中有符号但未进入 link”以及“当前 SDK
命名空间不同”等原始故障。具体 SoC、库名、错误码和某一版本的宏名只写入项目
问题记录，不上升为本 Skill 的固定配置。

### 本次 unary/MIX 负向 probe 的首错模式

下面几类错误都曾在同一 fresh GMM custom 生成中出现；它们是 API/地址空间边界，
不是通过换路线或关闭诊断来规避的理由：

- `MakeTensor` 产生的 `NDExt` `LocalTensor` 传给 legacy `AscendC::Muls` 时，SDK
  报 `LocalTensor::PrimType` 缺失和静态断言失败。根因是 tensor_api layout 类型与
  legacy vector API 的类型约束不兼容，不是“float 不支持”。应让 tensor_api 和
  legacy/RegBase API 在各自已证明的 layout 边界内使用，并用当前 SDK 的同 layout
  最小 probe 选择调用面；不能靠类型别名或伪造 `PrimType`。
- 初次写成 `__global__ __aicore__ __mix__` 或调用未在当前 source 中存在的同步 helper
  时，ASC 首错是未知入口/未解析 helper。应先以当前版本的 typed entry spelling、
  Params 地址空间和 source-backed CvSync helper 做最小 compile witness，再实现
  body；compile witness 不能被当成设备可运行证据。
- 直接把 `const __gm__` TilingData 绑定到普通 C++ 引用、猜测 `DataCopyPad` 方向/参数，
  或把 `UpdateMask`/RegBase mask 保存成 `const`，会分别触发地址空间、重载和可变引用
  错误。应使用当前 SDK 已证明的 GM loader、Copy specialization 和可变 mask lvalue，
  分别 clean 验证普通 CXX、ASC Host、ASC Device/link。

| 触发信号 | 首要检查方向 |
|---|---|
| ASC 源文件被按普通 C++ 处理，或缺少预期设备语言诊断 | target 的语言、源文件属性和设备侧编译选项 |
| 普通 CXX、ASC Host 与 Device 对同一共享头看到不同符号 | 三种编译上下文的 source/宏分支和实际预处理结果 |
| 项目内 Blaze/tensor_api 头缺失，或同名符号、宏、模板、constexpr 语义异常 | 实际头文件来源、include 形式和搜索顺序 |
| `undefined reference`、入口符号或模板实例无法解析 | source 是否进入 target、声明/定义、二进制库和链接输入 |
| 源文件和 ABI 已闭合但静态链接仍有未解析符号 | 当前 CANN/toolchain 的传递库闭包、库顺序和版本来源 |
| 编译或链接命令出现项目外历史目录、外部源码或旧构建目录 | 项目自包含边界和缓存路径 |
| Blaze/tensor_api 副本、submodule 或源码位置缺失/不一致 | Step 1 的 Blaze 源码工作区和项目内副本完整性 |

排障和修复以 `<project-root>/operators/<operator_name>/` 为项目边界，不受 PLAN manifest 或当前 action 文件清单限制。修改后更新 PLAN 第 2、4--8 章，重新执行受影响 action/checkpoint，并把错误、实际修改和结果追加到第 11 章 `execution_record`。

## ASC 语言和编译选项

当 ASC 源文件被按普通 C++ 处理，或设备侧选项没有进入命令时：

1. 查看实际 target 的源文件列表和语言属性；
2. 确认设备侧源文件使用当前工具链要求的 ASC 构建入口；
3. 确认设备架构和 ASC 选项只作用于相应 target/source；
4. 修改已授权的项目构建文件，重新生成并构建。

不要为快速通过而把所有 host 源文件都标为 ASC，也不要把设备选项设置成全局编译选项。

## 头文件来源和同名冲突

当报错涉及 Blaze/tensor_api 头、同名声明、模板或 constexpr 语义时：

1. 从实际编译命令区分项目内 include、工具链内置 include 和其他外部 include；
2. 必要时用同一 compiler、语言和参数查看预处理/include trace，确认真正解析的头文件；
3. 对照项目内 Blaze/tensor_api 副本的 include 链，判断是否被 CANN 或历史副本中的同名头抢占；
4. 在受影响 target 上做最小 include 调整并重新核验解析结果。

一个已知类型是：项目内 tensor_api 与 CANN 工具链同时提供同名头，但其中函数的 constexpr、设备侧可用性或模板声明不同。此时不能通过包装同名函数、宏重定义或修改工具链头掩盖冲突。若源码使用引号形式 include，且 trace 证明工具链头抢占了项目内同源头，可以只对受影响的 ASC target 增加项目内 `-iquote` 路径；这是一种证据驱动的修复，不是默认配置。

在普通 Step 4 排障中，禁止修改编译器、CANN 安装目录、Blaze 源码、tensor_api
源码或未授权的 Skill Asset 原文件，也不要用强制预包含或伪声明绕过真实 API。
若当前动作是维护本 Skill 明确授权的版本化 `assets/`，应按该维护计划更新
asset，并以当前 SDK/target 的编译回归闭合，而不是把修改散落到生成算子或外部
工具链目录。

## `__VEC_SCOPE__` induction 类型闭合

当 ASC 在 `__VEC_SCOPE__` 中报告 `Induction variable must have a type
uint16_t`，或同时给出“expected type of ops in vector loop cond”时，这是 ASC
向量循环 DSL 的语言约束，不是可以忽略的 C++ 兼容性 warning：

1. 保留完整诊断，确认首错位于 `__VEC_SCOPE__` 的哪一个循环，以及实际的
   induction 声明和 bound 类型。
2. 将该向量循环的 induction variable 声明为 `uint16_t`，并让循环 bound、
   loop count 在进入 scope 前转换到可证明的 `uint16_t` 范围；用独立的更宽
   类型保存 byte offset、shape 和剩余元素，避免把地址算术窄化。
3. 不要用 `const_cast`、关闭诊断、普通 C++ 外层循环或伪声明绕过 ASC DSL
   约束。若支持域可能超过 `uint16_t`，应拆分向量循环并在每轮重新计算 mask，
   而不是让 induction 溢出。
4. 用同一 ASC compiler、架构和 target 做 clean rebuild，再继续真实设备
   精度/同步回归；Host/C++ 编译通过不能关闭该语言边界。

具体 induction 名称、tile 大小和某一版工具链的诊断文字只记录在项目
`WALKTHROUGH.md`，不写成 Skill 常量。

## Host/device helper 符号闭合

当 ASC device link 报告某个项目命名空间 helper 的 `undefined symbol`，而普通
Host 编译已通过时，先把它当作 Host/device 上下文边界问题处理，而不是立即
添加运行时库或复制另一份实现：

1. 保留 device link 的完整符号、参数类型和失败 target；确认调用点是否在
   `__global__`/`__aicore__`/`__mix__` entry 的设备实例化路径。
2. 检查 helper 的真实声明是否具有目标工具链要求的 device 可见限定。若同一
   ASC 翻译单元确实需要一个 helper 同时生成 Host 和 Device 实体，双限定必须
   由当前工具链 witness 证明；若代码按上下文分支编译，则 Host helper 可用
   `inline constexpr`、纯 device helper 用 `__aicore__ inline`。普通 `inline`
   或 `constexpr` 只说明 C++ 内联/常量语义，不能证明它会进入 device 编译上下文。
3. 对需要 runtime 参数的 helper，优先在同一项目头中补齐已证明的
   Host/device 声明；若只是设备侧简单推导，也可在 device 路径使用等价的
   局部表达式。不得用 dummy 外部定义、伪 runtime 库或历史源码掩盖缺失的
   device 定义。
4. 用相同 ASC compiler、架构和 target 做 clean device link，再继续设备
   运行验证；“Host 编译通过”不能关闭该边界。

这条规则由以下可复用原始故障模式归纳：普通 Host 可见的 runtime helper
被 ASC device 实例化后，因缺少 device 限定而留下未解析设备符号；最小修复
是闭合 helper 的 Host/device 声明或使用等价 device-local expression，回归
判据是 device link clean 且下游真实设备用例仍通过。具体函数名、tile、flag
和某次工具链的符号名只保留在项目 `WALKTHROUGH.md`，不写成 Skill 常量。

## 当前 ASC/RegBase 签名闭合

当 RegBase、mask 或向量 helper 报参数类型、可变性或声明顺序错误时，以当前
目标 CANN/ASC 头文件和实际编译命令为唯一事实源：先核对完整声明及调用上下文，
再让实参的所有权和生命周期满足该声明（例如需要可变引用时使用可变的局部
值），不要用 `const_cast`、伪声明或工具链补丁绕过错误。若 helper 使用返回类型
推导，必须在首次调用前提供当前翻译单元可见的声明或定义；“Host 能编译”不等于
ASC device 路径已闭合。修复后必须 clean rebuild 受影响 target，并把首个诊断、
实际声明、改动和下游 device/link 回归写入 `execution_record`。具体函数名和
某一版 SDK 的签名只保留在项目问题记录，不写成固定 Skill API。

## 设备地址空间与编译期分支闭合

当 `Reg::DataCopy`、UB load/store 或 GM/UB tensor view 报指针类型不匹配，或
代码同时包含 AIC/AIV 路径时：

1. 先按当前 CANN 头文件确认实参的地址空间。`LocalTensor::GetPhyAddr()` 得到
   的 UB 地址必须在后续表达式中保持 `__ubuf__`，GM 入口和输出地址必须保持
   `__gm__`；不要先声明成普通 `float*`/`const T*` 再期待 compiler 恢复地址空间。
   可以用当前工具链接受的 `auto*` 继承地址空间，或显式写目标 address-space
   pointer，并重新检查 DataCopy 的完整模板签名。
2. `ASCEND_IS_AIC`、`ASCEND_IS_AIV` 等是编译期语法分区，不是可与运行时布尔
   表达式拼接的普通值。把需要的运行时判断放在对应 compile-time block 内，不能
   写成 `if ASCEND_IS_AIC && condition`，也不能让另一侧访问未实例化的对象。
3. device GM POD/tiling 由 `__global__ __aicore__` entry 使用当前项目
   生成的 `GET_TILING_DATA_WITH_STRUCT`/`GET_TILING_DATA_MEMBER` 或同一工具链
   已证明的 typed loader 解包，再把本地 POD 传给 Kernel。公共 Kernel
   资产不得直接持有 raw tiling `GM_ADDR`、逐字段复制或用
   `reinterpret_cast`/伪声明掩盖生命周期和地址空间问题。
4. 若当前 Tensor API/source witness 的 `MakeMemPtr` 接口接收 raw `GM_ADDR`，
   Params 应保留 raw GM address；只在该 source-compatible 调用点形成 typed
   `__gm__` view。typed `__gm__` 指针能够通过 Host 编译但设备挂起时，应先做
   当前 SDK 的最小 GM-copy probe，不得把 typed pointer 形式当作 ABI 等价物。
5. 修复后必须用同一 ASC compiler、架构和 target 做 clean configure/build，再
   继续真实设备回归；只看到 Host/C++ 编译通过不能关闭该边界。

这条规则由 MX MatMul+SwiGLU 的实际首错归纳：普通指针声明丢失 UB 地址
空间、编译期宏与运行时 `&&` 混写、以及未经生成 loader 的 GM tiling
aggregate helper 均会在设备
编译边界失败。具体指针、tile 和函数名保留在项目问题记录，不写成固定 API。

## 链接和项目边界

遇到未解析符号时，依次确认：

1. 定义所在源文件是否进入当前 target；
2. 声明、定义、Wrapper 和入口 ABI 是否一致；
3. 所需项目库或既有工具链/运行时库是否进入实际 link command；
4. 路径是否错误指向其他项目、历史构建目录或不同版本副本。

可以在项目根内修正 source、target 级链接配置和既有依赖路径，也可以补充项目内 helper 或构建文件，但不要照搬其他工程的库清单或全局链接目录。若修复需要新的外部依赖，返回 Step 3。

## Blaze 源码工作区问题

如果实际命令表明项目内副本、`ops-tensor/`、submodule 或缓存来源不一致：

- 只读核对 Step 1 建立的 Blaze 源码工作区和递归 submodule；
- 确认项目内 `blaze/`、`tensor_api/` 副本来自当前 Blaze 源码；
- 确认构建没有读取其他项目或历史目录；
- 源码位置、submodule 或项目副本不完整时停止修改并返回 Step 1；
- 当前 Blaze 源码的类型、入口、Params 或 API 事实不足时返回 Step 2。

不要用旧二进制、历史副本或项目外源码临时替代当前项目输入。

## 修复边界和验证

Step 4 可以在项目根内调整源码、include 顺序、target 编译选项和当前项目/既有工具链依赖的链接配置，以实现冻结设计。修复后重新执行失败的 checkpoint 及受影响的下游 checkpoint；构建通过后继续执行 PLAN 中的功能、精度和回归 checkpoint，不能把“编译成功”当作算子验证完成。实现问题应继续留在 Step 4 诊断和修复，除非证据要求改变冻结设计或补充上游源码事实。

以下情况停止修复并返回 Step 3：

- 需要改变实现路线、Blaze 组装方案或定制场景；
- 需要改变算子接口、Kernel ABI 或 Tiling/Params 语义；
- 需要改变数据语义、地址/ABI、支持范围或验证范围；
- 需要新的 Blaze 源码事实或新的外部依赖。
