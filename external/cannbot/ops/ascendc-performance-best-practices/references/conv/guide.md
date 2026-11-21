# Conv 类算子高性能实现索引

> 适用架构：arch35（A5 / DAV_3510）；具体型号、SDK 与验证范围以正文为准。

Conv 族（标准 Conv2D、Depthwise Conv2D、Conv3DDX 反卷积核拆分、Conv3DDW 权重梯度 StreamK）在 Ascend Cube / 混合核上的高性能实现，沉淀为**可复用 Kernel 模板**（按各模板的接口和适用范围接入）。

**本目录（`references/conv/`）自包含**：各套模板提供独立 Host tiling、Kernel 与 launch；调用方负责匹配输入布局、设备资源、内存分配和构建。超出已实现范围时再扩展，不以占位函数代替核心计算。

> 策略经验见 [conv_common_perf_experience.md](conv_common_perf_experience.md)。
> Conv2D Cube 模板翻译（循环/API/流水控制）见 [conv2d_cube_tuning_guide.md](conv2d_cube_tuning_guide.md)。
> Depthwise 模板翻译（循环/API/流水控制）见 [depthwise_conv_tuning.md](depthwise_conv_tuning.md)。
> Conv3DDX kernel split 模板翻译（kernel 拆分/循环/API/事件）见 [conv3ddx_kernel_split_tuning_guide.md](conv3ddx_kernel_split_tuning_guide.md)。
> Conv3DDW StreamK + 确定性计算见 [流程说明](conv3ddw_streamk_tuning_guide.md)；使用与 Conv3DDX KS 一致的五文件 direct-launch 结构。
> 落代码与扩展见 [templates/usage_guide.md](templates/usage_guide.md)。
> 官方实现对标（ops-nn 映射 + TTK 竞品 tiling 采集）：正向见 [conv2d_opsnn_mapping.md](conv2d_opsnn_mapping.md)、反向输入梯度见 [conv3ddx_opsnn_mapping.md](conv3ddx_opsnn_mapping.md)、反向权重梯度见 [conv3ddw_opsnn_mapping.md](conv3ddw_opsnn_mapping.md)（三者映射互不通用，禁止互套）；TTK 采集操作见 [TTK_Conv2D_Usage.md](TTK_Conv2D_Usage.md)，tiling 字段释义见 [TILING_FIELDS.md](TILING_FIELDS.md)，解码/汇总脚本在 `scripts/`。

## 文档结构

```
guide.md（本文件）
├── conv_common_perf_experience.md          卷积性能优化策略集（经验，侧重 Depthwise）
├── conv2d_cube_tuning_guide.md             Conv2D Cube 模板翻译（循环/API/流水控制）
├── depthwise_conv_tuning.md                Depthwise 模板翻译（循环/API/流水控制）
├── conv3ddx_kernel_split_tuning_guide.md   Conv3DDX kernel split 模板翻译（kernel 拆分/子核轮转/AIC-AIV 事件）
├── conv3ddw_streamk_tuning_guide.md       Conv3DDW 尾轮调度、workspace、固定归约树和同步
├── conv2d_opsnn_mapping.md                正向 Conv2D 官方实现映射（tiling_key → ops-nn 行为）
├── conv3ddx_opsnn_mapping.md              反向 ConvDx 官方实现映射（M/N/K 定义与正向相反）
├── conv3ddw_opsnn_mapping.md              反向 ConvDw 官方实现映射（小 M/N 大 K 规约 + atomic）
├── TTK_Conv2D_Usage.md                    TTK 竞品 tiling 采集操作手册
├── TILING_FIELDS.md                       tiling_data 字段释义
├── scripts/
│   ├── decode_tiling.py                   tiling_data 解码为命名字段
│   ├── summarize_results.py               TTK 结果汇总表
│   └── run_test.sh                        一键采集（环境变量驱动）
└── templates/
    ├── usage_guide.md                      落地 / 接入 / 未实现能力扩展
    ├── conv2d_cube.md                      标准 Conv2D Cube 说明
    ├── dwconv2d_mix.md                     Depthwise __mix__ 说明
    ├── conv3ddx_kernel_split.md            Conv3DDX kernel split（stride≥2 kernel 拆分）说明
    ├── conv3ddw_streamk_deterministic.md  Conv3DDW 五文件模板准入与接入
    ├── conv3ddw_stream_k_tiling.template         Conv3ddwSkCase + Host API（→ .h）
    ├── conv3ddw_stream_k_tiling_impl.template    StreamK Host tiling（→ .cpp，g++）
    ├── conv2d_tiling.template                Conv2DCase + ValidateCase + Host tiling API（→ .h）
    ├── conv2d_tiling_impl.template           Host tiling 实现（→ .cpp，g++）
    ├── dwconv2d_tiling.template              DwTiling + ValidateTiling + Host tiling API（→ .h）
    ├── dwconv2d_tiling_impl.template         DW Host tiling（→ .cpp）
    ├── conv3ddx_ks_tiling.template           Conv3ddxKsCase + ValidateCase + Host tiling API（→ .h）
    ├── conv3ddx_ks_tiling_impl.template      KS Host tiling（→ .cpp，g++）
    └── dav310/
        ├── conv2d_kernel.template          Conv2DKernel<T>（→ .h）
        ├── conv2d_kernel_entry.template    __cube__ + launch（→ .cpp，bisheng）
        ├── conv2d_launch.template          launch 声明（→ .h）
        ├── dwconv2d_kernel.template        DepthwiseConv2dKernel（→ .h）
        ├── dwconv2d_kernel_entry.template  __mix__(1,2) + launch（→ .cpp）
        ├── dwconv2d_launch.template        launch 声明（→ .h）
        ├── conv3ddx_ks_kernel.template     Conv3ddxKsKernel<T>（→ .h，bisheng）
        ├── conv3ddx_ks_kernel_entry.template __mix__(1,2) + launch（→ .cpp）
        ├── conv3ddx_ks_launch.template     launch 声明（→ .h）
        ├── conv3ddw_stream_k_kernel.template     Conv3ddwSkKernel<T>（→ .h）
        ├── conv3ddw_stream_k_kernel_entry.template __mix__(1,2) + launch（→ .cpp）
        └── conv3ddw_stream_k_launch.template     launch 声明（→ .h）
```

**阅读顺序**：本指南策略表 → `conv2d_cube.md` / `dwconv2d_mix.md` / `conv3ddx_kernel_split.md`（§3 实现现状） → 同名 `.template` → 模板翻译（`conv2d_cube_tuning_guide.md` / `depthwise_conv_tuning.md` / `conv3ddx_kernel_split_tuning_guide.md`）→ [usage_guide.md](templates/usage_guide.md)（**§4 模板实现现状**、接入、**§6 能力扩展**）。

## 可复用 Kernel 模板

| 模板 | 场景 | 核类型 | 代码 | 说明 | 模板翻译 |
|------|------|--------|------|------|---------|
| **Conv2D Cube** | groups=1，NCHW，必带 bias；runtime `Conv2DCase` | `__cube__` | [.template](templates/dav310/conv2d_kernel.template) | [.md](templates/conv2d_cube.md) | [翻译](conv2d_cube_tuning_guide.md) |
| **Depthwise mix** | groups=C，简化组卷积；runtime `DwTiling` | `__mix__(1,2)` | [.template](templates/dav310/dwconv2d_kernel.template) | [.md](templates/dwconv2d_mix.md) | [翻译](depthwise_conv_tuning.md) |
| **Conv3DDX kernel split** | Conv3D 输入梯度，空间 stride≥2（kernel 拆分消插零；SplitHW/SplitH 双模式）；runtime `Conv3ddxKsCase` | `__mix__(1,2)` | [.template](templates/dav310/conv3ddx_ks_kernel.template) | [.md](templates/conv3ddx_kernel_split.md) | [翻译](conv3ddx_kernel_split_tuning_guide.md)（含 §15 源码索引） |
| **Conv3DDW StreamK + 确定性** | Conv3D 权重梯度，最后一轮按 BatchDout/Ho 拆分；FP32 输出 | `__mix__(1,2)` | [Kernel](templates/dav310/conv3ddw_stream_k_kernel.template) · [Host](templates/conv3ddw_stream_k_tiling_impl.template) | [模板契约](templates/conv3ddw_streamk_deterministic.md) | [流程](conv3ddw_streamk_tuning_guide.md) |

四套模板均按 Host tiling → runtime Case → launch 接入，互相独立。Conv3DDW 的 launch 已封装 schedule mode 1 和 workspace 契约。

## 策略速查（经验层）

| 策略 | 适用场景 | 核心手段 | 详细文档 |
|------|---------|---------|---------|
| **loop_axis_simplification** | Depthwise（weight 全载、cin/groups=1） | 消除 kAL1/khAL1/kwAL1；SetupGroup 一次载 L1 | [conv_common_perf_experience.md](conv_common_perf_experience.md) |
| **scalar_minimization** | 单次 Mmad 短（kL0 ≤ 48） | 去内循环 TQue、Load3D 预计算、tiling 字段局部化 | 同上 |
| **weight_nz_transform** | cin/groups < K0 | AIV Gather：NCHW→NZ + enlarge | 同上 |
| **aiv_aic_pipeline** | `__mix__` | AIV 写 BL1，AIC Cube；CrossCore；BL1 PP | 同上 |
| **bias_cmatrix_source** | 含 bias | bias→C2，`cmatrixSource=true` | 同上 |
| **kernel_split_phase** | 反卷积 stride≥2 | 相位拆 kernel 为 stride² 个子核：A1 无洞连续搬（MTE2 ×s²）、K 缩短消无效 MMAD、子核 B1 全载 mStartPosition 选行、AIV interleave 重排输出 | [conv3ddx_kernel_split_tuning_guide.md](conv3ddx_kernel_split_tuning_guide.md) |
| **streamk_deterministic** | Conv3DDW 尾轮输出块数不超过半核且累加轴可分 | 单核主轮 + 尾轮 BatchDout/Ho 分片 + 私有 workspace + AIV 固定树归约 | [conv3ddw_streamk_tuning_guide.md](conv3ddw_streamk_tuning_guide.md) |

## 未实现能力

- **模板级**（通用 Group Conv、NHWC、multiplier>1、融合 epilogue 等需新模板的场景）：→ [templates/usage_guide.md §4.2](templates/usage_guide.md)
- **能力级**（可选 bias、真 HW `woDim`、独立 L1 乒乓等在现有模板上扩展；SplitN 已落地）：→ [templates/usage_guide.md §6](templates/usage_guide.md)
