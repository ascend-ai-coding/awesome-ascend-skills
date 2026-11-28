# 模板使用指南

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

本文件说明如何将 `templates/` 下的 SoftmaxV2 Kernel 模板集成到实际算子项目中。

## 0. 模板文件格式与使用约定

`templates/` 下的所有源码模板均以 `.template` 为后缀（如 `softmax_v2_base.template`）。这些是**参考模板**，不是可直接编译的源码。

**使用时必须将 `.template` 重命名/复制为对应的 `.h`（或 `.cpp`）文件，并拷贝到项目 `op_kernel/` 目录使用**（与算子 kernel 源码同级，即构建脚本 `ascendc_library` 的源码目录，见下文 CMake 示例），否则最终工程里的 `#include` 引用的是 `.h`，代码不易读也无法编译。转换规则很简单，去掉 `.template` 后缀并按文件头 `\file` 注释中标注的名字落地：

| 模板文件 | 转换后落地文件 |
|----------|---------------|
| `dav310/softmax_v2_base.template` | `dav310/softmax_v2_base.h` |
| `dav310/softmax_v2_ar_small_r.template` | `dav310/softmax_v2_ar_small_r.h` |
| `dav310/softmax_v2_ar_full_load.template` | `dav310/softmax_v2_ar_full_load.h` |
| `dav310/softmax_v2_ar_recompute.template` | `dav310/softmax_v2_ar_recompute.h` |
| `dav310/softmax_v2_ara_full_load.template` | `dav310/softmax_v2_ara_full_load.h` |
| `dav310/softmax_v2_ara_recompute.template` | `dav310/softmax_v2_ara_recompute.h` |
| `dav310/softmax_v2_ara_online.template` | `dav310/softmax_v2_ara_online.h` |
| `dav310/kernel_utils.template` | `dav310/kernel_utils.h` |
| `softmax_v2_tiling_data.template` | `softmax_v2_tiling_data.h` |
| `softmax_v2_tiling.template` | `softmax_v2_tiling.h` |
| `dav310/euclidean_norm_interleaved_contiguous_tailr.template` | `dav310/euclidean_norm_interleaved_contiguous_tailr.h` |
| `dav310/euclidean_norm_interleaved_contiguous_tail_a.template` | `dav310/euclidean_norm_interleaved_contiguous_tail_a.h` |
| `euclidean_norm_tiling_data.template` | `euclidean_norm_tiling_data.h` |
| `euclidean_norm_tiling.template` | `euclidean_norm_tiling.h` |

转换命令示例（保留目录结构）：

```bash
# 在 templates/ 目录下，把所有 .template 复制成同名 .h 到工程目录
find . -name '*.template' | while read f; do
  cp "$f" "${f%.template}.h"
done
```

> **架构目录说明**：Kernel 模板位于 `dav310/` 子目录（对应目标架构 `dav-3510`/Ascend950），`kernel_utils.template` 也在 `dav310/` 下，与其它 Kernel 模板同级。

> **禁止从零重写**：模板已包含实测验证过的结构与参数组合，查到匹配模板后必须**复用**——转换成同名 `.h` 拷贝到项目 `op_kernel/` 目录直接集成/裁剪，不得凭模板思路从零重写 kernel；只有模板确实不覆盖的场景才新增/改造模板，并按消融纪律重新验证。

## 1. 文件清单

### Kernel 文件（7 个，均在 `dav310/` 下）

| 模板文件 | 转换后 | 作用 |
|----------|--------|------|
| `dav310/softmax_v2_base.template` | `softmax_v2_base.h` | 公共基类 `SoftmaxV2OpsBase`：CastTrait、VF 工具、CopyIn/Out、NlastReduceSum、UpdateCache 等 |
| `dav310/softmax_v2_ar_small_r.template` | `softmax_v2_ar_small_r.h` | AR-SmallR：`SoftmaxV2ArSmallR<Tx,Ty>`，R 极小转置向量化 |
| `dav310/softmax_v2_ar_full_load.template` | `softmax_v2_ar_full_load.h` | AR-FullLoad：`SoftmaxV2AR<T_in,T_out>`，R 全量载入 UB |
| `dav310/softmax_v2_ar_recompute.template` | `softmax_v2_ar_recompute.h` | AR-Recompute：`SoftmaxV2ArRecompute<Tx,Ty>`，R 超 UB 三阶段重读 |
| `dav310/softmax_v2_ara_full_load.template` | `softmax_v2_ara_full_load.h` | ARA-FullLoad：`SoftmaxV2ARA<T1,T2>`，3D R 全量载入 |
| `dav310/softmax_v2_ara_recompute.template` | `softmax_v2_ara_recompute.h` | ARA-Recompute：`SoftmaxV2ARARecompute<T1,T2>`，3D R 超 UB |
| `dav310/softmax_v2_ara_online.template` | `softmax_v2_ara_online.h` | ARA-Online：`SoftmaxV2ARAOnline<T1,T2>`，3D 在线 max/sum |

### 公共依赖（3 个）

| 模板文件 | 转换后 | 作用 |
|----------|--------|------|
| `softmax_v2_tiling_data.template` | `softmax_v2_tiling_data.h` | 六套 TilingData 结构体定义（packed），host/device 共用 |
| `softmax_v2_tiling.template` | `softmax_v2_tiling.h` | 六个独立 Host tiling 计算函数 |
| `dav310/kernel_utils.template` | `dav310/kernel_utils.h` | Device 侧工具模板（CeilDiv、CeilAlign、FloorDiv、Aligned 等） |

## 2. 六类可独立选择的模板

每个模板可独立选择、独立包含和独立实例化，不使用 TilingKey、templateCode 或统一模板分发器。

| 模板 | 类名 | TilingData | Tiling 函数 | 构造方式 |
|------|------|-----------|------------|---------|
| AR-SmallR | `SoftmaxV2ArSmallR` | `SoftmaxV2ArSmallRTilingData` | `TilingArSmallR` | AR 系：构造传 `TPipe*` |
| AR-FullLoad | `SoftmaxV2AR` | `SoftmaxV2ARTilingData` | `TilingArFullLoad` | AR 系：构造传 `TPipe*` |
| AR-Recompute | `SoftmaxV2ArRecompute` | `SoftmaxV2ArRecomputeTilingData` | `TilingArRecompute` | AR 系：构造传 `TPipe*` |
| ARA-FullLoad | `SoftmaxV2ARA` | `SoftmaxV2ARATilingData` | `TilingAraFullLoad` | ARA 系：构造传 TilingData* |
| ARA-Recompute | `SoftmaxV2ARARecompute` | `SoftmaxV2ARARecomputeTilingData` | `TilingAraRecompute` | ARA 系：构造传 TilingData* |
| ARA-Online | `SoftmaxV2ARAOnline` | `SoftmaxV2ARAOnlineTilingData` | `TilingAraOnline` | ARA 系：构造传 TilingData* |

## 3. 每个模板的最小依赖集合

> 下面的依赖关系均以**转换后的 `.h`** 描述。集成前请先按第 0 节把 `.template` 落地为 `.h`。

```
具体模板（如 dav310/softmax_v2_ar_small_r.h）
├── dav310/softmax_v2_base.h（按实际依赖）
│   └── dav310/kernel_utils.h（同目录）
├── softmax_v2_tiling_data.h
└── CANN SDK headers（kernel_tiling/kernel_tiling.h, kernel_operator.h, op_kernel/platform_util.h）
```

使用者不需要包含其他不相关模板。每个模板显式选择，不依赖模板注册表或自动路由。

## 4. 如何显式选择一个模板

在 kernel 入口文件中 `#include` 需要的模板头文件和 TilingData 头文件：

```cpp
#include "softmax_v2_tiling_data.h"
#include "dav310/softmax_v2_base.h"
#include "dav310/softmax_v2_ar_full_load.h"  // 只包含需要的模板
```

## 5. 如何构造对应的 TilingData

在 Host 侧使用 `softmax_v2_tiling.h` 中的对应函数：

```cpp
#include "softmax_v2_tiling.h"

softmax_tiling::PlatformParam plat;
// 运行时查询平台参数
plat.ubSize = static_cast<int64_t>(ubSize);
plat.numBlocks = static_cast<int64_t>(coreNumAiv);

softmax_tiling::CaseShape shape;
shape.a1 = a1; shape.r = r; shape.a0 = a0; shape.dtypeCode = 0; // 0=FP32, 1=FP16, 2=BF16

SoftmaxV2ARTilingData td;
int64_t blockDim = softmax_tiling::TilingArFullLoad(shape, plat, td);
```

## 6. 如何实例化类并调用 Process

### AR 系模板（构造传 `TPipe*`，Init 传 TilingData 指针）

```cpp
TPipe pipe;
SoftmaxV2AR<float, float> op(&pipe);       // 构造传 TPipe*
op.Init(x, y, &tiling);                     // Init 传 TilingData*
op.Process();
```

### ARA 系模板（构造传 TilingData 指针，Init 传 `TPipe*`）

```cpp
TPipe pipe;
SoftmaxV2ARA<float, float> op(&tiling);     // 构造传 TilingData*
op.Init(x, y, &pipe);                        // Init 传 TPipe*
op.Process();
```

> **注意**：AR 和 ARA 系的构造/Init 参数顺序不同，不能混用。

## 7. FP32、FP16、BF16 实例化方法

```cpp
// FP32
SoftmaxV2AR<float, float> op_fp32(&pipe);
// FP16
SoftmaxV2AR<half, half> op_fp16(&pipe);
// BF16
SoftmaxV2AR<bfloat16_t, bfloat16_t> op_bf16(&pipe);
```

输入输出 dtype 必须一致（`Tx == Ty`）。内部计算始终在 FP32 进行。

## 8. Reg API、Kernel 入口、CANN SDK include 和目标架构要求

- **Reg API**：模板使用 `AscendC::Reg::RegTensor`、`MaskReg` 等 VF 类型。寄存器操作必须位于目标工程有效的 VF 区域；类型/trait 的声明位置由最终构建契约决定，不能一概规定为文件级或 `__VEC_SCOPE__` 内。
- **Kernel 入口按构建契约选择**：目标工程为 `xasc_combined` 且已有 Reg VF 成功实现时，保持同工程的普通 `__global__ __aicore__` 入口和 VF 边界；不得无条件添加 `__vector__`。device-only 或 `__simd_vf__ + asc_vf_call` 工程按各自成功契约处理。
- **CANN SDK include**：
  - `softmax_v2_tiling_data.h` — 本模板随附的 Host/Device 共用 TilingData
  - `kernel_operator.h` — AscendC 高层 API 和 Reg API
  - `op_kernel/platform_util.h` — 平台工具
- **目标架构**：`dav-3510`（Ascend950），VRegSize=256B，VL_FP32=64。

## 9. 构建方式

改代码前先运行 `scripts/vf_preflight.py` 判断最终构建契约；契约不同/未知时先做最小编译探针。首次探针、编译或集成失败后，再阅读 [CANN / Reg VF 编译问题 FAQ](../../common/arch35/regbase/cann_version_compat_faq.md)。目标工程为 `xasc_combined` 时，不得直接套用后文 device-only、`.asc` 或 `ascendc_library` 的声明位置和构建约束。

### 方式零：`.cpp -xasc` 直调

复用工程 `compile_commands.json` 中已成功算子的真实 flags，典型关键项为 `-xasc --npu-arch=dav-3510 -c *.cpp`。这种模式下 Reg API 不自动要求 `-dc`，入口也不自动要求 `__vector__`；未保护的文件/类级 Reg 类型可能被 host pass 解析，须先对照同工程已编译成功的 VF 边界，再做单分支最小探针。

### 方式一：ASC 语言单 .asc 文件（host + kernel 同一翻译单元）

参考源工程 `softmax.asc`，host main 与 kernel 入口写在同一个 `.asc` 文件中，一趟 ASC 编译，`<<<blockDim, nullptr, stream>>>` 直调。

关键 CMakeLists.txt 配置：

```cmake
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "")
find_package(ASC REQUIRED)
project(softmax LANGUAGES ASC CXX)

add_executable(softmax softmax.asc)
target_include_directories(softmax PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)
target_compile_options(softmax PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:-dc>
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
target_link_options(softmax PRIVATE
    --npu-arch=${CMAKE_ASC_ARCHITECTURES}
)
```

### 方式二：`ascendc_library(kernellaunch)` 框架

```cmake
set(ASCEND_KERNEL_LAUNCH_ONLY ON)
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
ascendc_library(ascendc_kernels_npu STATIC op_kernel/softmax_kernel.cpp)
ascendc_include_directories(ascendc_kernels_npu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
```

Host 侧通过 `ACLRT_LAUNCH_KERNEL` 调用。

## 10. 构建注意事项

### `-dc` 和 device link `--npu-arch`

dav310 头文件使用 `AscendC::Reg::RegTensor` 等 reg_compute 类型，仅在 device arch 下声明。**单 `.asc` 且需要可重定位 device code 的模式**需：

1. **`-dc`**：生成可重定位 device 代码。
2. **`--npu-arch=dav-3510`**：device 编译和链接阶段均需显式指定架构，否则默认落到 `dav-m100` 导致 device 符号无法解析。
3. **host pass 守卫**：只在同一 `.asc` 工程的已成功源码和编译命令明确使用 `__ASC_NPU_HOST__` 时，才按该工程契约复用守卫。不得把它作为所有 Reg 类型报错的通用补丁。

### VF fusion

`--cce-simd-vf-fusion` 选项控制 VF 融合模式（`true`/`false`），按需开启。

### 两种方式的选择

- **`.cpp -xasc` 直调**：复用目标工程已有普通入口和真实编译 flags，不套用 `.asc` 的 `-dc` 结论。
- **单 .asc**：适合快速验证和直调场景，host+kernel 同一翻译单元。
- **`ascendc_library`**：适合正式算子工程，自动生成 `aclrtlaunch_*.h`。

> 三种方式均可使用。关键是先识别目标工程模式，再复用该模式的入口、device 编译/链接和 include 约束；不能把 `-dc` 或 `__vector__` 泛化成所有 Reg VF 工程的必选项。

## 11. 参考模板与完整可运行工程之间的边界

`templates/` 目录提供的是**参考模板**：

- **包含**：7 个 Kernel 模板（`dav310/*.template`）、公共 TilingData 定义、Host tiling 参考实现、`dav310/kernel_utils.template`、文档。集成前按第 0 节将 `.template` 转成 `.h`/`.cpp`。
- **不包含**：完整的 CMakeLists.txt、`.asc` 入口文件、数据生成/验证脚本、input/output I/O 工具。
- 完整可运行工程参考：`softmaxNewKernel/` 源工程（含 `softmax.asc`、`CMakeLists.txt`、`data_utils.h`）。

集成时需自行编写：
1. Kernel 入口（`.asc` 或 `.cpp`），按本文档第 4-6 节的方式实例化模板。
2. Host main 或 op_host，调用 `softmax_v2_tiling.h` 中的 tiling 函数。
3. CMakeLists.txt，按本文档第 9 节的方式配置构建。
4. 数据 I/O 工具（如 `data_utils.h`）。

## 11.5 EuclideanNorm 多轴交错归约模板集成要点

两个交错归约模板（`euclidean_norm_interleaved_contiguous_tailr` 路径A / `euclidean_norm_interleaved_contiguous_tail_a` 路径B）与 softmax_v2 系列集成方式一致（`.template` → `.h`），但有几处专属注意事项：

### 依赖与文件

```
dav310/euclidean_norm_interleaved_contiguous_tailr.h   （或 _tail_a.h）
├── kernel_operator.h                    CANN SDK：AscendC 高层 API + Reg API
├── adv_api/reduce/reduce.h              CANN SDK：ReduceSum + Pattern::Reduce::AR / RA
└── euclidean_norm_tiling_data.h         EuclideanNormTilingData / MAX_PATTERN_RANK
Host 门控：euclidean_norm_tiling.h        TryComputeFastTiling / TryComputeFastRARATiling
```

- 两个 kernel 类均 **无参构造**（`TPipe pipe_` 是类成员），`Init(x, y[, workspace], td)` 传 TilingData 指针。两个类均为模板类 `EuclideanNormFastKernel<DType>` / `EuclideanNormFastRARAKernel<DType>`，`DType ∈ {float, half, bfloat16_t}`。
- 路径 A 的 `Init` 有 `workspace` 形参（需 host 分配 `workspaceBytes` 字节）；路径 B **无 workspace**。`coresPerGroup==1` 时路径 A kernel 不使用 workspace（直接 sqrt→yGm），但 host 仍按公式预留。
- kernel 入口需 `LoadTilingData` 把 GM 上 TilingData 逐字段搬进核内局部结构（等价框架 `GET_TILING_DATA_WITH_STRUCT`），示例见各 `.md` 第 10 节。

### arch35 `ReduceSum` 只支持 AR/RA 两级

这是本模式族最重要的 SDK 约束：`ReduceSum<float, Pattern::Reduce::AR/RA>` **只接受两级 shape**（`{outer, inner}`），传入多维 `ARAR` 会**编译报错**。故内块 `[A,R,A,R]` / `[R,A,R,A]` 的规约必须拆成两步链式规约：

- 路径 A（`[A0,R0,A1,R1]`）：step1 `AR` 规约 R1 → step2 逐 `(a0,a1)` **标量累加** R0（不用 adv_api `ReduceSum`：其 AR 规约在 `dimR==vlSize(64)` 边界处结果偏大，实测 szR0=64 的 case 高估 ~4.5%；step2 非 MTE2 热点，标量规约开销可忽略）。
- 路径 B（`[R0,A0,R1,A1]`）：step1 由 VF 逐行 `Add` 吃掉最外 R0 → step2 单次 `RA` 规约 R1 保留 A1（RARA 连续，**免 gather**）。

> 因 adv_api `ReduceSum` 以整 VL（64 float）粒度 Load/Store，会在源/目的末尾多读多写最多 1 VL，路径 A 参与该 API 的 `acc`/`red` buffer 末尾各留 64 元素余量（`kPadElems`）防越界。

### 门控硬阈值（必须上板校准，勿仅凭 mte2_ratio）

`euclidean_norm_tiling.h` 的两个 `Try*` 函数是窄门控，命中才用、否则 `return false` 回退。支持 dtype ∈ {fp32, fp16, bf16}（计算全程升 fp32）。核心阈值：

| 模板 | 关键门控 |
|------|---------|
| 路径 A | `dtype∈{fp32,fp16,bf16}` + `axisNum` 偶数≥6 tail-R + 内块 stride 连续 + `innerElems×dtypeSize` 32B 对齐 + **`R1×dtypeSize ≤ 16B`（burst 病态）** + `rowsSize ≥ 8` + **内块整块 fits UB（`chunkA0==szA0`，需分块则回退）** + `groupCnt≤coreNum` |
| 路径 B | `dtype∈{fp32,fp16,bf16}` + `axisNum` 奇数≥7 tail-A + 内块 stride 连续 + `innerRowElems×dtypeSize` 32B 对齐 + `slabCnt≥coreNum/2` + **`innerRowElems≥512`（每行≥2KB）** |

> 详细理由见 [guide.md](../guide.md)「多轴交错归约模式族 → 关键洞察」：中等 burst / 小内块下模板固定开销会反超 MTE2 增益，实测慢 28%~10×，故用 burst / 内块大小硬阈值而非 `mte2_ratio`。

### 数值语义

fp32 直通；fp16/bf16 载入即升 fp32（`DIST_UNPACK_B16` + Cast），全程 fp32 累加/规约；`sqrt` 在最终 `part` 上一次性 `Reg::Sqrt` in-register 完成，输出 Cast 降回 `D_T`（`DIST_PACK_B32`），无需 TilingData 传系数（与 mean 需 `invRTotal` 不同）。workspace 恒为 fp32（部分和精度）。

## 12. 常见编译问题

### 问题 1：Reg 类型找不到

**现象**：`error: no template named 'RegTensor'` 或 `error: 'MicroAPI' has not been declared`

**根因**：常见于实际目标架构、include 链或构建模式与候选代码不匹配。`MicroAPI` 在 CANN 9.x 又是按目标架构声明的旧名兼容别名，该报错不足以证明 Reg API 缺失。

**解决**：先从同工程成功 kernel 和 `compile_commands.json` 对比 `--npu-arch`、入口形式和 include 根目录。只有同一 `.asc` 构建契约明确在 host pass 使用 `__ASC_NPU_HOST__` 时才复用该守卫；不得为消除报错手工添加命名空间别名或修改 SDK。

### 问题 2：device 链接符号未定义

**现象**：链接阶段报 device 符号无法解析。

**根因**：`--npu-arch` 未在链接阶段指定，默认架构不匹配。

**解决**：`target_link_options` 中添加 `--npu-arch=dav-3510`。

### 问题 3：compile_commands.json 不存在

**现象**：`error: No such file or directory: 'compile_commands.json'`

**解决**：`export CMAKE_EXPORT_COMPILE_COMMANDS=ON` 后在同一 shell 执行 cmake 和 make。

## 13. 无效输入与资源不足

所有 tiling 函数以返回 0 表示输入或资源无效；调用方必须停止启动该配置。全载入路径放不下一个最小 tile 时，应选择资源允许的重算/在线路径，不能强制把因子改为 1。
