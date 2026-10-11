# Reduction 归约类算子高性能实现索引

Reduce 族算子（Softmax、LayerNorm、RMSNorm、ReduceSum、ReduceMax 等）在 Ascend Vector 核心上的高性能实现，沉淀为一组**可复用 Kernel 模板**。本指南是这些模板的入口，介绍每个模板解决的问题、关键技术与适用条件，以及如何适配到同族其他算子（如 Norm）。

> 本指南是**优秀实践目录**，而非选型决策树：每个模板列出其适用条件与解决的问题，由实现者根据实际 shape、UB 容量与带宽情况判断采用。Reduce 族算子形态多样，不存在普适的"唯一最优"，理解各模板的权衡是正确适配的前提。

## 文档结构

```
guide.md（本文件）
└── templates/
    ├── usage_guide.md                              模板集成指南
    ├── softmax_v2_tiling_data.template             六套 TilingData 公共定义（→ .h）
    ├── softmax_v2_tiling.template                  六个独立 Host tiling 参考实现（→ .h）
    ├── dav310/
    │   ├── kernel_utils.template                   Device 侧公共工具模板（→ .h）
    │   ├── softmax_v2_base.template                公共基类 SoftmaxV2OpsBase（→ .h）
    │   ├── softmax_v2_ar_small_r.template          AR-SmallR Kernel（→ .h）
    │   ├── softmax_v2_ar_full_load.template        AR-FullLoad Kernel（→ .h）
    │   ├── softmax_v2_ar_recompute.template        AR-Recompute Kernel（→ .h）
    │   ├── softmax_v2_ara_full_load.template       ARA-FullLoad Kernel（→ .h）
    │   ├── softmax_v2_ara_recompute.template       ARA-Recompute Kernel（→ .h）
    │   └── softmax_v2_ara_online.template          ARA-Online Kernel（→ .h）
    ├── softmax_v2_ar_small_r.md                    AR-SmallR 说明文档
    ├── softmax_v2_ar_full_load.md                  AR-FullLoad 说明文档
    ├── softmax_v2_ar_recompute.md                  AR-Recompute 说明文档
    ├── softmax_v2_ara_full_load.md                 ARA-FullLoad 说明文档
    ├── softmax_v2_ara_recompute.md                 ARA-Recompute 说明文档
    ├── softmax_v2_ara_online.md                    ARA-Online 说明文档
    ├── fused_attention_online_softmax_design.md    融合 Attention Online Softmax 设计
    └── state_resident_design.md                    融合算子 UB 状态常驻设计
```

**阅读顺序**：先读本指南「通用范式」「公共框架」理解骨架 → 读对应模板 `.md` 了解流程与关键技术 → 落代码时对照同名 `.template`（使用时去掉 `.template` 后缀转成 `.h`，见 [usage_guide.md](templates/usage_guide.md) 第 0 节）→ 集成时读 [usage_guide.md](templates/usage_guide.md) → 在线场景读 `fused_attention_online_softmax_design.md` / `state_resident_design.md`。

> ⚠️ **集成前必读**：[usage_guide.md](templates/usage_guide.md) 说明了如何将模板集成到实际项目中，包括构建配置、kernel 入口写法、host 调用方式、常见编译问题及解决方案。

## Reduce 计算的通用范式

Reduce 族算子均可拆为两段，模板的差异主要在"如何切分 R 以适配 UB"，而两段数学是可替换的：

```
① 归约统计（reduce statistic）：沿 R 轴求一个标量/向量统计量
    Softmax     → max、sum(exp)
    LayerNorm   → mean、variance
    RMSNorm     → mean(square)
    ReduceSum   → sum
    ReduceMax   → max

② 归一化/输出（normalize）：用统计量逐元素变换
    Softmax     → (x - max) → exp → div sum
    LayerNorm   → (x - mean) / sqrt(var + eps) → scale + shift
    RMSNorm     → x / sqrt(mean_sq + eps) → scale
```

正因如此，6 个 Softmax 模板的**骨架（切分 + 流水 + 归约统计的组织方式）对 Norm 等同族算子通用**，适配时只需替换两段数学（见下文「适配同类算子」）。

## 公共框架与工具

所有模板继承 [dav310/softmax_v2_base.template](templates/dav310/softmax_v2_base.template)（转换后 `softmax_v2_base.h`）中的 `SoftmaxV2OpsBase`，提供跨模板复用的 VF 工具：

| 工具 | 说明 |
|------|------|
| `CastTrait` | MicroAPI Cast 必需配置（饱和模式、舍入、掩码合并），`castTraitFp16ToFp32`/`castTraitFp32ToFp16` |
| `CastToFp32From<T>` / `CastFromFp32To<T>` | 载入即升 FP32、输出即降回原精度，FP32 直通、FP16/BF16 自动 Cast |
| `UpdateCache` | **跨 chunk/bin 二分累加树**核心：按 `cacheID` 层级归并局部 sum，recompute 类模板共用 |
| `GetCacheID` / `FindNearestPower2` | 二分树层级计算，确定配对关系与根节点位置 |
| `NlastReduceSum` | N-last 方向规约求和：小 R 用 `NlastDichotomyAdd` 编译期二分展开，大 R 用 `NlastReduceSumLargeR` 8 行分组折叠 |
| `CopyIn` / `CopyOut` | 带 stride 的批量 DMA（rows×cols），支持非对齐尾块 |
| 常量 | `VL_FP32=64`（256B VReg / 4B float）、`CONST_EIGHT=8`、`BLOCK_SIZE=32` |

> `UpdateCache` 和 `NlastReduceSum` 是 recompute 系模板（AR-Recompute / ARA-Recompute）的关键，把"逐 chunk 串行累加 sum"变为"二分树对数层合并"，显著降低长 R 的归约串行延迟。

## 离线计算模板

离线模板针对**独立 Kernel** 的 Reduce 算子，输入已完整存在于 GM，按 R 能否载入 UB 与输入维数分为 5 个。各模板互相独立，Host 侧 Tiling 根据 shape 与 UB 容量选用。

> 注：下表"适用条件"是模板的设计前提，非强制决策规则。边界 shape（如 R 刚好接近 UB 上限）可由实现者权衡带宽与空间。

| 模板 | 输入 | R 与 UB 关系 | 核心策略 | 代码 | 说明 |
|------|------|-------------|---------|------|------|
| **AR-SmallR** | 2D `[A1,R]` | R 极小（FP32≤16/FP16≤32） | 转置为 `(R,A1)`，沿 A1 向量化 | [.template](templates/dav310/softmax_v2_ar_small_r.template) | [.md](templates/softmax_v2_ar_small_r.md) |
| **AR-FullLoad** | 2D `[A1,R]` | R 全量载入 UB | 多行批量载入，单 pass VF 计算 | [.template](templates/dav310/softmax_v2_ar_full_load.template) | [.md](templates/softmax_v2_ar_full_load.md) |
| **AR-Recompute** | 2D `[A1,R]` | R 超出 UB | R 切 chunk，3 阶段重读 GM + 二分累加树 | [.template](templates/dav310/softmax_v2_ar_recompute.template) | [.md](templates/softmax_v2_ar_recompute.md) |
| **ARA-FullLoad** | 3D `[A1,R,A0]` | R 全量载入 UB | 沿 A0 切 tile，跨 R 逐列 VF | [.template](templates/dav310/softmax_v2_ara_full_load.template) | [.md](templates/softmax_v2_ara_full_load.md) |
| **ARA-Recompute** | 3D `[A1,R,A0]` | R 超出 UB | R 切 bin，3 阶段跨 bin 二分折叠 | [.template](templates/dav310/softmax_v2_ara_recompute.template) | [.md](templates/softmax_v2_ara_recompute.md) |

各模板的「解决的问题 / 适用条件 / 执行流程 / 关键技术 / 关键代码索引」详见对应 `.md` 文档。

### 维度命名约定

- **A1**：最外层非归约轴（行方向），2D 中即行数。
- **R**：归约轴（reduce axis），Softmax/Norm 沿此求统计量。
- **A0**：3D 中 R 内侧的非归约轴（列方向），与 R 共同决定一个 tile 的数据量。
- **AR**：2D `(A1, R)`；**ARA**：3D `(A1, R, A0)`。

## 在线计算模板

### 独立 ARA Online Softmax

| 模板 | 输入 | 核心策略 | 代码 | 说明 |
|------|------|---------|------|------|
| **ARA-Online** | 3D `[A1,R,A0]` | 沿 R 分 chunk 在线更新 running max/sum，输入读 2 次 | [.template](templates/dav310/softmax_v2_ara_online.template) | [.md](templates/softmax_v2_ara_online.md) |

ARA Online Softmax 是**独立的 Online Softmax**，不包含 `QK^T`、`P×V` 或 FlashAttention 融合。与 ARA Recompute（输入读 3 次）相比，减少一次 R 全量搬入。

### 融合 Attention Online Softmax 设计

| 文档 | 解决的问题 | 关键思想 |
|------|-----------|---------|
| [fused_attention_online_softmax_design.md](templates/fused_attention_online_softmax_design.md) | FlashAttention 中 Softmax 需 O(S²) 中间内存 | 逐 tile 生成 `QK^T`，维护 running max/sum，融合 `P×V`，内存降至 O(S) |
| [state_resident_design.md](templates/state_resident_design.md) | 融合算子跨 S2 循环重复分配状态 buffer | 状态 buffer 一次性常驻 UB + 双缓冲索引 |

> **重要区分**：[softmax_v2_ara_online.md](templates/softmax_v2_ara_online.md) 是独立的 Online Softmax Kernel 实现（输入读 2 次，输出完整概率）。[fused_attention_online_softmax_design.md](templates/fused_attention_online_softmax_design.md) 是 FlashAttention 融合场景的设计参考（逐 tile 生成 score 即消费，融合 `P×V`，通常不生成完整概率矩阵）。两者共享在线 max/sum 数学基础，但不是同一个实现。当前目录未提供与融合设计完全对应的 FlashAttention Kernel。

## 关键技术总览

以下技术横跨多个模板，是 Reduce 族高性能实现的公共经验：

| 技术 | 涉及模板 | 核心要点 |
|------|---------|---------|
| **DOUBLE/TRIPLE BUFFER 乒乓** | 全部 | `TQue` 多缓冲让 CopyIn/Compute/CopyOut 重叠；recompute 用 3 缓冲支持 Main/Fold 双 chunk 并行载入 |
| **MicroAPI VF（`__VEC_SCOPE__`）** | 全部 | `RegTensor<float>` + `MaskReg` 直接操作 256B VReg，逐 `VL_FP32=64` 块处理，尾块用 `UpdateMask` 精确掩码 |
| **FP16/BF16 ↔ FP32 计算** | 全部 | 载入即升 FP32 保证数值稳定，输出即降回，`CastTrait` 控饱和/舍入 |
| **批量 strided DMA** | 全部 | `DataCopyPad` + `DataCopyExtParams`，`blockCount` 一次搬多行，`srcStride/dstStride` 跨维跳跃，对齐到 32B block |
| **二分累加树（UpdateCache）** | AR/ARA-Recompute | 跨 chunk/bin 的局部 sum 按 `GetCacheID` 层级二分合并，O(log N) 层而非 O(N) 串行 |
| **NlastReduceSum** | ARA-FullLoad/Recompute | N-last 方向规约：小 R 用 `NlastDichotomyAdd` 编译期二分展开，大 R 用 8 行分组折叠 |
| **MainBlock/FoldBlock 配对** | AR/ARA-Recompute | 两 chunk 的 exp 在 tmp 上 `Add` 合并后一次 ReduceSum，归约次数减半 |
| **布局转置向量化** | AR-SmallR | 短归约轴转置到外层，长伴生轴转成向量化方向，VReg 利用率从 R/64 拉到 ~100% |
| **BinaryAddVF** | ARA-FullLoad | R>8 时按 8 行分组二分折叠，`TwoRowAddWithTail` 处理余数尾块 |
| **按 R 大小分支归约** | ARA-FullLoad | R≤2/≤4/≤8 用 VF `Add` 直累，R>8 用 `BinaryAddVF` 二分折叠 |
| **在线 max/sum 更新** | ARA-Online | 沿 R 分 chunk，running max/sum 单遍在线更新，减少一次输入读取 |

## 适配同类算子

Softmax 模板的骨架（切分 + 流水 + 归约统计组织）对同族算子通用，适配只需替换两段数学。下面以 **LayerNorm / RMSNorm** 为例说明映射关系。

### 适配要点

1. **维度映射**：Norm 的归约轴（hidden_size / norm_size）对应模板的 R；其余轴对应 A1（2D）或 A1+A0（3D）。
   - 2D LayerNorm `[N, H]` 沿 H 归约 → AR 模板（R=H, A1=N）。
   - 3D LayerNorm `[B, S, H]` 沿 H 归约 → 可视为 ARA（R=H, A0=S, A1=B），或展平为 AR。
2. **数学替换**：把 Softmax 的 `max/sub/exp/sum/div` 替换为 Norm 的统计量计算，见下表。
3. **阶段数调整**：
   - Softmax 是三段（max → exp-sum → div）。
   - LayerNorm 可两遍扫描（mean → var）或 Welford 单遍；recompute 模板的阶段①可同时累加 mean，阶段②算 var。
   - RMSNorm 单段即可（平方累加 → div sqrt），比 Softmax 更简单。
4. **affine 参数**：Norm 通常带 gamma/beta（scale/shift），可在输出阶段追加 `Muls`+`Add`。
5. **精度**：mean/variance 建议用 FP32 累加（与 Softmax 的 FP32 中间一致），即使输入输出为 FP16。

### 数学替换对照

| 算子 | 阶段①（reduce statistic） | 阶段②（sum/second pass） | 阶段③（normalize/output） |
|------|------------------------|------------------------|--------------------------|
| **Softmax** | global max | exp 累加求 sum | sub max→exp→div sum |
| **LayerNorm（两遍）** | 累加求 mean | sub mean→平方→累加求 var | sub mean→div sqrt(var+eps)→scale+shift |
| **RMSNorm** | 平方累加 | （可并入①） | div sqrt(mean_sq+eps)→scale |
| **ReduceSum** | 累加求 sum | — | （直接输出） |
| **ReduceMax** | max 累加 | — | （直接输出） |

### 模板选择参考

| Norm 变体 | R（norm_size）特征 | 倾向模板 |
|-----------|-------------------|---------|
| LayerNorm/RMSNorm，hidden 小 | 极小 | AR-SmallR（转置向量化收益高） |
| LayerNorm/RMSNorm，hidden 中等 | 全量载入 | AR-FullLoad / ARA-FullLoad |
| LayerNorm/RMSNorm，hidden 很大 | 超出 UB | AR-Recompute / ARA-Recompute |

> 以上是"特征—模板"对应关系，非决策树。实际选择还需结合 A1/A0 规模、UB 预算、是否需 affine、是否与 matmul 融合等综合判断。

## Softmax 独有优化（在线融合）

以下两个设计文档主要服务于 Softmax 在 Attention 中的融合场景，对独立 Norm 算子参考价值有限，列出供完整性：

| 文档 | 说明 |
|------|------|
| [fused_attention_online_softmax_design.md](templates/fused_attention_online_softmax_design.md) | 融合 Attention Online Softmax 设计：逐 tile 生成 `QK^T`，running max/sum + `P×V` 融合，内存 O(S²)→O(S) |
| [state_resident_design.md](templates/state_resident_design.md) | 融合算子状态 buffer 跨循环常驻 + 双缓冲设计建议 |
