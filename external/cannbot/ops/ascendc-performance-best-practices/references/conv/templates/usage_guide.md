# Conv 模板使用指南

本目录（`references/conv/`）是**自包含知识库**：模板源码、接口说明、接入方式、未实现能力扩展落点均在本树内完成，不依赖外部工程文档或示例目录。

Conv2D、Depthwise、Conv3DDX KS、Conv3DDW StreamK 均使用五文件结构：公共 tiling 头、Host 实现、Kernel、入口、launch 声明。bisheng 编 Kernel，g++ 编 Host / plugin，以 runtime Case 直接调用。

Conv3DDW 的 launch 内部封装 schedule mode 1，外部调用方式与其他模板一致；工作区按 Case 分配。详细示例见[模板说明](conv3ddw_streamk_deterministic.md)，机制见[流程](../conv3ddw_streamk_tuning_guide.md)。

## 0. `.template` → 源文件

推荐**按表落地**（去掉 `.template` 后缀，按用途命名）：

| 模板 | 落地文件 | 编译器 |
|------|----------|--------|
| `conv2d_tiling.template` | `op_kernel/conv2d_tiling.h` | 共用 |
| `conv2d_tiling_impl.template` | `op_kernel/conv2d_tiling.cpp` | **g++**（plugin / host） |
| `dav310/conv2d_kernel.template` | `op_kernel/conv2d_kernel.h` | bisheng |
| `dav310/conv2d_kernel_entry.template` | `op_kernel/conv2d_kernel.cpp` | **bisheng** |
| `dav310/conv2d_launch.template` | `op_kernel/conv2d_launch.h` | 共用 |
| `dwconv2d_tiling.template` | `op_host/dwconv2d_tiling.h` | 共用 |
| `dwconv2d_tiling_impl.template` | `op_host/dwconv2d_tiling.cpp` | **g++** |
| `dav310/dwconv2d_kernel.template` | `op_kernel/dwconv2d_kernel.h` | bisheng |
| `dav310/dwconv2d_kernel_entry.template` | `op_kernel/dwconv2d_kernel.cpp` | **bisheng** |
| `dav310/dwconv2d_launch.template` | `op_kernel/dwconv2d_launch.h` | 共用 |
| `conv3ddx_ks_tiling.template` | `op_kernel/conv3ddx_ks_tiling.h` | 共用 |
| `conv3ddx_ks_tiling_impl.template` | `op_host/conv3ddx_ks_tiling.cpp` | **g++** |
| `dav310/conv3ddx_ks_kernel.template` | `op_kernel/conv3ddx_ks_kernel.h` | bisheng |
| `dav310/conv3ddx_ks_kernel_entry.template` | `op_kernel/conv3ddx_ks_kernel.cpp` | **bisheng** |
| `dav310/conv3ddx_ks_launch.template` | `op_kernel/conv3ddx_ks_launch.h` | 共用 |
| `conv3ddw_stream_k_tiling.template` | `op_kernel/conv3ddw_stream_k_tiling.h` | 共用 |
| `conv3ddw_stream_k_tiling_impl.template` | `op_host/conv3ddw_stream_k_tiling.cpp` | **g++** |
| `dav310/conv3ddw_stream_k_kernel.template` | `op_kernel/conv3ddw_stream_k_kernel.h` | bisheng |
| `dav310/conv3ddw_stream_k_kernel_entry.template` | `op_kernel/conv3ddw_stream_k_kernel.cpp` | **bisheng** |
| `dav310/conv3ddw_stream_k_launch.template` | `op_kernel/conv3ddw_stream_k_launch.h` | 共用 |

> `dav310/` 表示目标微架构（Ascend950 / `dav-3510`）。落地时可**摊平**到 `op_kernel/` / `op_host/`，不必保留子目录。

```bash
REF=<本目录>   # .../references/conv/templates
DST=<你的工程>/csrc/ops/conv_2d
mkdir -p "$DST/op_kernel"
cp "$REF/conv2d_tiling.template"              "$DST/op_kernel/conv2d_tiling.h"
cp "$REF/conv2d_tiling_impl.template"         "$DST/op_kernel/conv2d_tiling.cpp"
cp "$REF/dav310/conv2d_kernel.template"       "$DST/op_kernel/conv2d_kernel.h"
cp "$REF/dav310/conv2d_kernel_entry.template" "$DST/op_kernel/conv2d_kernel.cpp"
cp "$REF/dav310/conv2d_launch.template"       "$DST/op_kernel/conv2d_launch.h"
# 文件头「集成说明」注释可删；Depthwise 同理：kernel→op_kernel，tiling→op_host
```

Depthwise 模板内 `#include "dwconv2d_tiling.h"`：把 `op_host` 与 `op_kernel` 都加入 include path（见 §3）。

## 1. 工程接入（direct-launch 形态）

### 1.1 推荐目录

```
csrc/ops/<op_name>/
├── CMakeLists.txt
├── op_kernel/     # bisheng：*_kernel.cpp + *.h
├── op_host/       # 可选；depthwise tiling 放这里
└── op_plugin/     # g++：torch.library + tiling 调用 + launch
```

### 1.2 编译分工

| 源 | 编译器 | 说明 |
|----|--------|------|
| `*_kernel.cpp`（含 `__cube__` / `__mix__`） | bisheng（`-xasc`，`--npu-arch=dav-3510`） | Device kernel + `extern "C"` launch |
| `*_tiling.cpp` | **g++** | Host tiling；**禁止**进 bisheng TU |
| `*_plugin.cpp` | g++ | Schema / Meta / NPU 实现，调用 launch |

CMake 需把 kernel 源与 plugin/tiling 源分到不同目标（或不同 `OBJECT` 库），再链成同一 `.so`。若工程已有「注册宏 + 自动 `add_subdirectory`」脚手架，按该脚手架的 kernel/plugin 列表注册即可；**本知识库不绑定某一示例树**。

### 1.3 CMake 片段（自包含算子目录）

**Conv2D：**

```cmake
set(CONV2D_KERNEL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/conv2d_kernel.cpp)
set(CONV2D_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/conv2d_plugin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/conv2d_tiling.cpp)

# 将 KERNEL_SRCS → bisheng 目标；PLUGIN_SRCS → g++ 目标；include 含 op_kernel
```

**Depthwise（源也在本算子目录内）：**

```cmake
set(DWCONV2D_KERNEL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/dwconv2d_kernel.cpp)
set(DWCONV2D_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/dwconv2d_plugin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/dwconv2d_tiling.cpp)

# include 同时含 op_kernel 与 op_host
```

### 1.4 Plugin 最小调用

**Conv2D：**

```cpp
Conv2DTilingIn tin = /* shape/attrs */;
tin.aicoreNum = platform->GetCoreNumAic(); // 目标 SoC 可用核数；platform 由调用方初始化
Conv2DTilingOut tout{};
TORCH_CHECK(Conv2DDoTiling(tin, tout), "tiling failed");
Conv2DCase tiling = ToCaseView(tin, tout);
launch_conv2d(x, w, b, y, tiling, stream);
```

**Depthwise：**

```cpp
DwTilingIn tin = /* shape/attrs/aicoreNum */;
DwTiling tiling{};
TORCH_CHECK(DwConv2dGetTiling(tin, tiling), "tiling failed");
launch_dwconv2d(x, w, b, y, tiling, stream);
```

**Conv3DDW：**

```cpp
Conv3ddwSkTilingIn tin = /* shape/attrs/dtype */;
tin.aicoreNum = platform->GetCoreNumAic(); // 目标 SoC 可用核数；platform 由调用方初始化
platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, tin.ubBytes);
Conv3ddwSkTilingOut tout{};
TORCH_CHECK(Conv3ddwSkDoTiling(tin, tout), "unsupported Conv3DDW case");
auto tiling = ToCaseView(tin, tout);
// 分配 tiling.workspaceBytes 字节，保持存活至当前 stream 完成。
TORCH_CHECK(launch_conv3ddw_sk(x, dy, dw, workspace, tiling, stream)==0, "launch failed");
```

Tiling 结构**按值**进入 kernel 参数缓冲，launch 路径无 H2D tiling copy。

Conv3DDX KS 调用 `Conv3ddxKsDoTiling` 前同样必须填入 `aicoreNum = platform->GetCoreNumAic()`，
并通过 `platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, tin.ubBytes)` 填入单核 UB 容量。
查询失败时停止 tiling；上述数值仅适用于查询到的目标 SoC，不能由模板示例或环境旋钮猜测。

### 1.5 Plugin 骨架要点

- `TORCH_LIBRARY_FRAGMENT` 注册 schema（如 `conv_2d` / `depthwise_conv_2d`）
- Meta：算 Ho/Wo，返回 `empty` 输出
- NPU：按模板校验连续 NCHW/NCDHW、dtype、属性长度 → tiling → `OpCommand`/`acl_call` 里调 launch
- 可选：环境变量读核数（见 §5）

## 2. 构建与架构

- 目标架构：**dav-3510**（Ascend950）
- Kernel：bisheng + `-xasc` + `--npu-arch=dav-3510`
- Host / plugin：与 PyTorch / torch_npu 链接的普通 C++17
- 轮子 / 扩展名随你的工程约定（例如 `cann_bench._C`）

```bash
# 在你的工程根目录按该工程的 build 入口构建并安装
bash build.sh --soc=ascend950 --install   # 若工程提供该入口
```

## 3. Include / 依赖边界

```
conv2d_kernel.cpp
├── conv2d_kernel.h → conv2d_tiling.h（Conv2DCase + ValidateCase）→ kernel_operator.h
├── conv2d_launch.h
└── （tiling 不进 bisheng TU）

conv2d_plugin.cpp
├── conv2d_launch.h
└── conv2d_tiling.h → conv2d_tiling.cpp

dwconv2d_kernel.cpp
├── dwconv2d_kernel.h → dwconv2d_tiling.h
└── dwconv2d_launch.h

conv3ddw_stream_k_kernel.cpp（bisheng）
├── conv3ddw_stream_k_kernel.h → conv3ddw_stream_k_tiling.h
├── conv3ddw_stream_k_launch.h
└── acl/acl.h（schedule mode 1）
conv3ddw_stream_k_tiling.cpp（g++）→ conv3ddw_stream_k_tiling.h

conv3ddx_ks_kernel.cpp（__mix__(1,2)，bisheng）
├── conv3ddx_ks_kernel.h → conv3ddx_ks_tiling.h
└── conv3ddx_ks_launch.h
    （conv3ddx_ks_tiling.cpp 只进 g++ / plugin TU）
```

CANN SDK：`kernel_operator.h` 等由工具链 / 环境提供。

## 4. 模板实现现状与选型

### 4.1 已实现模板

| 模板 | 核类型 | 适用场景 | 编译期实例 | 说明文档 |
|------|--------|---------|-----------|----------|
| **Conv2D Cube** | `__cube__` | groups=1，NCHW，必带 bias | 3（dtype） | [conv2d_cube.md](conv2d_cube.md) |
| **Depthwise mix** | `__mix__(1,2)` | groups=C，NCHW，channel multiplier=1 | 6（dtype×PP） | [dwconv2d_mix.md](dwconv2d_mix.md) |
| **Conv3DDX kernel split** | `__mix__(1,2)` | Conv3D 输入梯度，strideH=strideW=2，kh/kw∈[1,4]，dilation=1，groups=1，NCDHW | 2（dtype） | [conv3ddx_kernel_split.md](conv3ddx_kernel_split.md) |
| **Conv3DDW StreamK + 确定性** | `__mix__(1,2)` | NCDHW groups=1 权重梯度；尾轮拆 BatchDout/Ho、FP32 输出 | 3（dtype）；runtime `Conv3ddwSkCase` | [conv3ddw_streamk_deterministic.md](conv3ddw_streamk_deterministic.md) |

各套模板互相独立；Host 按问题形态选一套落地。各模板已实现能力与使用条件详见对应文档 §3。

### 4.2 未实现模板

以下场景在现有模板上硬改风险大（循环轴 / 布局假设不兼容），建议新开模板：

| 场景 | 现有模板为何不够 | 建议 |
|------|-----------------|------|
| **通用 Group Conv**（1 < groups < C） | Cube 假设 groups=1（无 group 循环轴）；mix 假设 groups=C（weight 按组全载） | 新模板：复用 Cube Mmad 框架，外层加 group 维循环；weight 按 group 切 BL1；tiling 增加 groupDim 核切分 |
| **NHWC Conv2D / Depthwise** | 两套模板的 Load3D / Fixpipe 均假设 NCHW 布局 | 新模板：改 DataCopy / Load3d 布局参数；或做 NCHW↔NHWC 适配层（优先级低，多数场景可 transpose 前置） |
| **Channel multiplier > 1 Depthwise** | mix 的 enlarge 打包和 weight 布局假设 cin/groups=1 | 新模板：weight `[C*m, m, Kh, Kw]` 打包；enlarge 逻辑重写；BL1 容量按 m 重新评估 |
| **Conv + 融合 epilogue** | 两套模板 Fixpipe 直写 GM，无后处理路径 | 新模板：L0C→GM 间插激活 / 量化 / add 等 epilogue；或复用现有模板 + 独立后处理 kernel 两步走 |
| **通用 Conv3DDX 模板**（stride=1 或不满足 kernel split 准入的反卷积） | kernel split 只覆盖 stride≥2 kernel 拆分域 | 新模板：按传统前放大三段式通路（GM→L1 NZ 留洞 / Load3D im2col / 逆序 Load2d）实现；基座方法已内联在 [conv3ddx_kernel_split_tuning_guide.md §11](../conv3ddx_kernel_split_tuning_guide.md) |

> 能力级扩展（可选 bias、真 HW `woDim>1`、独立 L1 乒乓等）不需新模板，在现有模板上改契约即可，见 [§6](#6-未实现能力怎么扩展模板运用)。

### 4.3 选型决策

| 问题特征 | 选择 |
|---------|------|
| Conv3DDW / Conv3DBackpropFilterV2 / filter 梯度，arch35，需 StreamK 或确定性归约 | [Conv3DDW StreamK + 确定性](conv3ddw_streamk_deterministic.md)；先区分 DW 与 depthwise/DX |
| `groups==1`，filter `[Co,Ci,Kh,Kw]` | Conv2D Cube |
| `groups==C`，weight `[C,Kh,Kw]`，multiplier=1 | Depthwise mix |
| Conv3DDX（反卷积输入梯度），strideH=strideW=2 且满足 [conv3ddx_kernel_split.md §4.3](conv3ddx_kernel_split.md) 准入 | Conv3DDX kernel split |
| Conv3DDX，strideH≥2 但不满足 SplitHW 准入（如 Kh>4、strideW=1） | **SplitH 模板已抽取**（splitMode=2，同份模板），准入与设计见 [§10 专章](conv3ddx_kernel_split.md) |
| Conv3DDX 但 stride=1 / 不满足任何拆分准入 | 无现成模板（传统前放大路线，见 §4.2 反卷积行） |
| `1 < groups < C` | 无现成模板（见 §4.2 通用 Group Conv） |
| NHWC layout | 无现成模板（见 §4.2 NHWC） |
| channel multiplier > 1 | 无现成模板（见 §4.2） |
| 需融合 epilogue | 无现成模板（见 §4.2 融合） |

## 5. 环境旋钮（约定）

在 plugin 中自行解析（名称可按工程调整，语义如下）：

| 变量 | 算子 | 作用 |
|------|------|------|
| `CONV2D_AICORE_NUM` | Conv2D | 限制实际使用的 Cube 核数，不得超过目标平台查询值 |
| `DWCONV2D_AICORE_NUM` | DW | 限制实际使用的核数，不得超过目标平台查询值 |
| `DWCONV2D_ENLARGE_LIMIT` | DW | 限制 enlarge（±inf 场景常用 1） |
| `DWCONV2D_HF32` | DW | FP32 可选 HF32 |
| `CONV3DDX_KS_AICORE_NUM` | Conv3DDX KS | 限制实际使用的核数，不得超过目标平台查询值 |
| `CONV3DDX_KS_DISABLE` | Conv3DDX KS | 强制关闭 kernel split（回退传统路径，用于 A/B） |

## 6. 未实现能力：怎么扩展（模板运用）

两套模板都是评测向子集，有意裁剪。扩展时**不要另起无关骨架**——在本目录对应模板上改契约与循环。

**落点表以本节为准**（本知识库内完整自洽，无需外链 STATUS）。

### 6.1 通用步骤

1. 选模板（标准卷积 → Cube；`groups==C` → mix）  
2. 先改契约：`ValidateCase` / `ValidateTiling` / tiling 字段  
3. 再改 kernel：循环轴或编译期实例（如 `L1_PINGPONG`）  
4. 策略对照 [../conv_common_perf_experience.md](../conv_common_perf_experience.md)

### 6.2 Conv2D Cube — 能力落点

仍用 `Conv2DKernel<T>` + `Conv2DCase`：

| 能力 | 模板侧改法 | Host / API |
|------|------------|------------|
| **可选 bias**（P0） | `hasBias` 时跳过 `LoadBias`/`LoadBiasToBt`，`DoMmad` 关 `cmatrixSource` | plugin 允许无 bias；tiling 入参传标志 |
| **AL1 全载 + BL1 切分**（P1） | kernel 机制已支持；补自测组合 | tiling 勿过滤该组合 |
| **非对称 L1 乒乓**（P1） | 确认 `l1PingPong` 为 `1`/`2` 时预取正确 | 产出非 `{0,3}` 并测带宽 |
| **groups≠1**（P2） | **勿硬改本模板**；depthwise 用 mix；通用 group 需新循环轴 | 另开模板 |
| **NHWC / 融合**（P2） | Load3D/Fixpipe / epilogue 新路径 | 新 Validate + plugin schema |

当前已支持（勿重复造）：dtype 特化；`batchDim×mDim×nDim` 多核；AL1/BL1 全载与切分；L1/L0 乒乓字段；splitM；**splitN**；FP32 `kAccum`；HF32；必带 bias；NCHW groups=1。

**splitN 已落地（勿再按"未做"处理）**：`nBL1 > nL0` 时 `ComputeWindow` 在 M 子循环外再套一层
`nL0` 子循环，`LoadBl0` 以 `MStartPosition = nOffset/16` 取 BL1 窗内 fractal、`SetDstStride`
按子块宽度打包，BT 每个 N 子块从同一份 bias L1 副本重载。契约见 `ValidateCase`：
`nBL1 % nL0 == 0` 且 `CinTileA == AlignedCi` 且 AL1 不乒乓——AL1 必须常驻，否则更宽的 BL1 窗
换不回 fmap 重读。tiling 在 `FillL0L1` 里按 `coreN` 的因子搜 `nBL1`，打分键新增 `-nBL1`
（排在 `-mAL1` 之后）。A/B 开关：`CONV2D_SPLIT_N=0` 把 `nBL1` 钉回 `nL0`。

### 6.3 Depthwise mix — 能力落点

仍用 `DepthwiseConv2dKernel<DTYPE, L1_PINGPONG>` + `DwTiling`：

| 能力 | 模板侧改法 | Host / API |
|------|------------|------------|
| **可选 bias**（P0） | kernel 已有 `hasBias`；false 时不碰 BT | plugin 放开强制 bias |
| **Load3D / 单行 AL1 硬校验**（P0） | 可不改 device | `ValidateTiling` / `BuildGeom` 增加检查 |
| **真 HW / `woDim>1`**（P1） | A：解除 `woDim==1`；B：外层按 Wo 多 launch/再切 | 先摸底失败 shape |
| **AL1/BL1 独立乒乓**（P2） | 编译期扩到 `{0,1,2,3}` 或运行期半区 | `DecideAl1` 代价模型选型 |
| **Weight BL1 再切**（P2） | 需新 K/BL1 循环，等同新模板 | 新字段 `kBL1 < K` |
| **channel multiplier>1 / NHWC**（P3） | 重写布局假设 | schema / Validate 同步 |

当前已支持：weight 按 opt-group 全载 BL1；AL1 尽量全载/切分；L1 ping-pong `{0,3}`；`__mix__(1,2)` AIV NZ + AIC Cube；纯 depthwise `groups==C`。

### 6.4 Conv3DDX kernel split — 能力落点

仍用 `Conv3ddxKsKernel<T>` + `Conv3ddxKsCase`：

| 能力 | 模板侧改法 | Host / API |
|------|------------|------------|
| **bias**（P0） | `ComputePhase` 首 K 的 `cmatrixSource=true` + BT 装载（复用 Conv2D Cube 的 LoadBias/LoadBiasToBt） | tiling 加 `hasBias`；准入放开 |
| **stride≥3 / strideH≠strideW**（P1） | 子核表扩为 `strideH×strideW` 个（`splitIndex = sh*strideW+sw` 通用化）；`KsSubBackpadBegin/End` 公式去 s=2 特化 | `ValidateCase` 放开 stride 约束 |
| **H 单拆模式** | **已实现**（splitMode=2：相位维度进核间切分 + 尾相位 M 收缩 + 三段直写，AIV 早退） | 设计见 [conv3ddx_kernel_split.md §10](conv3ddx_kernel_split.md) + [tuning guide §10A](../conv3ddx_kernel_split_tuning_guide.md)；源码对照 [§15 索引](../conv3ddx_kernel_split_tuning_guide.md) |
| **NDHWC filter**（P2） | `LoadToB1Full` 的 Dn2Nz 参数换 spatial-major 视图；`kernelSplitStrideB` 路径 | `ValidateCase` 放开 filter layout |
| **kSCoutFullLoad**（P2） | A1 也全载驻留（L0C 常驻 + 步数下降）；ops-nn 有独立模板（优先级 97） | tiling 加 `a1FullLoad` 判定 |
| **SplitK（kSegment）**（P3） | K 过长时 Cout 分段 + workspace fp32 累加（传统路径已有同款方案） | tiling 加 `kSegment` 字段 |

当前已支持（勿重复造）：SplitHW（s=2、kernel 1~4）+ **SplitH（strideH≥2、Kh∈[sH,16]）双模式**；
SplitHW 子核轮转 + 每子核 stepK/pad/M 动态重算；B1 全载 + Load2d `MStartPosition` 相位选行；
A1 连续无洞装载（H 模式 W 留洞）；三段对齐 fixpipe（HW→UB/ws + AIV Interleave，H→GM 直写隔行）；
workspace 中转；尾相位 M 收缩（HW 核内 / H 核间 kSTail*）；1×1 零相位特判。

### 6.5 缺口速查

| 缺口 | 用哪套 | 优先动作 |
|------|--------|----------|
| 可选 bias | 三套 | Conv2D 加字段；DW 放开 plugin；KS 见 §6.4 |
| AL1 全载 + BL1 切分 / 非对称乒乓 | Conv2D | 补 tiling 产出与自测 |
| 真 HW `woDim>1` | Depthwise | §6.3 真 HW 行 |
| Weight BL1 再切 / multiplier>1 / NHWC | Depthwise 或新模板 | 超出简化模板，工作量大 |
| KS bias / stride≥3 | Conv3DDX KS | §6.4 落点表 |

## 7. 常见问题

| 现象 | 处理 |
|------|------|
| bisheng 编进了 `*_tiling.cpp` | tiling 只进 g++ / plugin |
| DW include 找不到 `dwconv2d_tiling.h` | include 同时加 `op_host`+`op_kernel` |
| Conv2D tiling 失败 | 查 `ValidateCase`；L1 装不下或非法 Ho/Wo |
| case13 DW NaN | `DWCONV2D_ENLARGE_LIMIT=1`（仅兜底，全局开会伤分） |
| 要扩 §6 未列能力 | 按 §6.1 改契约+kernel，并回写本节落点表 |

## 8. 本目录文档地图

| 文档 | 用途 |
|------|------|
| [../guide.md](../guide.md) | 模板索引与策略速查 |
| [../conv_common_perf_experience.md](../conv_common_perf_experience.md) | 卷积优化经验 |
| [conv2d_cube.md](conv2d_cube.md) | Conv2D 实现现状、接口与数据流 |
| [dwconv2d_mix.md](dwconv2d_mix.md) | Depthwise 实现现状、接口与流水线 |
| [conv3ddx_kernel_split.md](conv3ddx_kernel_split.md) | Conv3DDX kernel split 的 kernel 拆分理论、实现现状与数据流 |
| **本文件** | 落地、接入、**§4 模板实现现状**、**§6 能力扩展** |
| `*.template` | 可复用源码 |
