# Pipeline 选项完整参考

本文档列出 HIVMPipelineOptions 和 HFusionPipelineOptions 的所有字段及默认值，从 Passes.h 精确提取。

源码参考：
- [HIVM/Pipelines/Passes.h](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Pipelines/Passes.h)
- [HFusion/Pipelines/Passes.h](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/Pipelines/Passes.h)

## 1. HIVMPipelineOptions

### 1.1 特性控制选项

| 选项名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enable-triton-kernel-compile` | bool | `false` | 启用 Triton kernel 编译 |
| `enable-dot-scaled-compile` | bool | `false` | 启用 dot_scaled 编译 |
| `enable-mixed-cv` | bool | `false` | 启用混合 Cube-Vector 编译 |
| `enable-layout-optimization` | bool | `false` | 启用布局优化 |
| `pure-simt` | bool | `false` | 启用纯 SIMT 编译（不使用 Cube） |

### 1.2 优化控制选项

| 选项名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enable-auto-multi-buffer` | bool | `false` | 启用自动多缓冲优化 |
| `limit-auto-multi-buffer-only-for-local-buffer` | bool | `false` | 限制自动多缓冲仅对 local buffer 生效 |
| `limit-auto-multi-buffer-of-local-buffer` | MultiBufferStrategy | `no-l0c` | 限制 local buffer 自动多缓冲策略：`no-limit`/`no-l0c` |
| `limit-mix-auto-multi-buffer-buffer` | MultiBufferStrategy | `only-cube` | 限制 Mix 多缓冲策略：`no-limit`/`only-cube`/`only-vector` |
| `set-workspace-multibuffer` | unsigned | `2` | workspace 多缓冲数量 |
| `enable-auto-bind-sub-block` | bool | `true` | 启用自动绑定子块 |
| `enable-auto-cv-balance` | bool | `false` | 启用 CV 流水线化时的自动平衡 |
| `enable-auto-storage-align` | bool | `true` | 启用自动存储对齐标记/启用 |
| `enable-global-workspace-reuse` | bool | `false` | 启用全局 workspace 重用 |
| `enable-print-memory-allocated-size` | bool | `false` | 启用打印内存分配大小 |
| `disable-tightly-coupled-buffer-reuse` | bool | `false` | 禁用紧耦合缓冲重用 |

### 1.3 同步控制选项

| 选项名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enable-hivm-inject-barrier-all-sync` | bool | `false` | 启用 barrier-all 模式的同步注入 |
| `enable-hivm-inject-block-all-sync` | bool | `false` | 启用全块同步注入 |
| `disable-auto-inject-block-sync` | bool | `false` | 禁用自动生成 sync block wait/set |
| `enable-hivm-graph-sync-solver` | bool | `false` | 启用基于图的同步求解器 |
| `enable-hivm-unit-flag-sync` | bool | `false` | 启用 unit-flag 同步模式 |

### 1.4 编译策略选项

| 选项名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `target` | string | `Ascend910B1` | 目标设备名称 |
| `enable-code-motion` | bool | `true` | 启用代码移动/子集提升 |
| `enable-vf-merge-level` | int32 | `1` | VF 合并级别（0=不合并，1=无依赖合并，2=全部合并） |
| `enable-direct-hivm-lowering` | bool | `false` | 启用直接 HIVM lowering |
| `enable-fused-multiply-add` | bool | `false` | 启用融合乘加 |
| `enable-hivm-nd2nz-on-vector` | bool | `false` | 启用 HIVM ND→NZ 在 Vector 上执行 |
| `enable-auto-blockify-loop` | bool | `false` | 启用逻辑块数>物理块数时自动循环 |
| `enable-bishengir-simt-optimization` | int | `900101` | 启用 BishengIR SIMT 优化级别 |
| `use-dpx` | bool | `true` | 启用通过 DPX 方言的 SIMT lowering |

### 1.5 调优选项

| 选项名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `disable-decompose-reduction` | int | `0` (false) | 禁用归约分解 |
| `disable-reorder-instruction` | int | `0` (false) | 禁用指令重排 |
| `enable-simt-reorder-instruction` | int | `0` (false) | 启用 SIMT 指令重排模式 |
| `simt-vf-dynamic-size` | int | `216` | SIMT VF 动态 UB 大小（KB） |
| `max-reduction-split` | int | `1` | 归约循环最大分裂次数 |
| `inject-ir-from-file` | string | `""` | 注入 IR 文件路径（调试用） |

### 1.6 MultiBufferStrategy 枚举

| 枚举值 | 说明 |
|--------|------|
| `NO_LIMIT` | 无限制 |
| `CUBE_NO_L0C` | 禁止 L0C 多缓冲 |
| `ONLY_CUBE` | 仅 Cube 多缓冲 |
| `ONLY_VECTOR` | 仅 Vector 多缓冲 |

## 2. ConvertToHIVMPipelineOptions

| 选项名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enable-triton-kernel-compile` | bool | `false` | 启用 Triton kernel 编译 |
| `enable-auto-blockify-loop` | bool | `false` | 启用逻辑块数>物理块数时自动循环 |
| `enable-regbase-hivmpipe` | bool | `false` | 启用基于寄存器的 HIVM pipeline |

## 3. HFusionPipelineOptions

### 3.1 特性控制选项

| 选项名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enable-manage-host-resources` | bool | `false` | 启用 Host 函数资源管理 |
| `enable-triton-kernel-compile` | bool | `false` | 启用 Triton kernel 编译 |
| `disable-ffts` | bool | `false` | 强制禁用 FFTS |
| `disable-hfusion-vectorize` | bool | `false` | 禁用 HFusion 自动向量化 |
| `multi-kernel` | bool | `false` | 启用多 kernel 融合 |
| `enable-symbol-analysis` | bool | `false` | 启用符号分析 |
| `enable-auto-vectorize-v2` | bool | `true` | 启用第二代自动向量化器 |
| `enable-vf-fusion` | bool | `false` | 启用 VF 融合 |
| `enable-tree-reduce` | bool | `false` | 启用树形归约 |
| `enable-multiple-consumer-fusion` | bool | `false` | 启用 AutoVectorizeV2 中的多消费者融合 |
| `skip-scope` | bool | `true` | 存在 scope 时跳过 flattenOps 等 Pass |
| `target` | string | `Ascend910B1` | 目标设备名称 |
| `enable-high-precision` | bool | `true` | 启用 sin/cos 高精度计算 |

### 3.2 优化控制选项

| 选项名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enable-layout-optimization` | bool | `false` | 启用布局优化 |
| `enable-mixed-cv` | bool | `false` | 启用混合 Cube-Vector 编译 |
| `enable-fuse-reduction-into-loop` | bool | `false` | 启用将后循环归约融合到循环体 |
| `enable-auto-multi-buffer` | bool | `false` | 启用自动多缓冲优化 |
| `enable-drop-unit-dims` | bool | `true` | 启用移除单位维度 Pass |
| `enable-flatten` | bool | `true` | 启用展平 Pass |
| `enable-auto-bind-sub-block` | bool | `true` | 启用自动绑定子块 |
| `enable-deterministic-computing` | bool | `true` | 启用确定性计算（禁用可能提升性能但结果不确定） |
| `enable-ops-reorder` | bool | `true` | 启用操作重排序 |

### 3.3 调优选项

| 选项名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `max-horizontal-fusion-size` | int32 | `-1` | 最大水平融合大小（-1 无限制） |
| `max-fused-elementwise-ops` | int32 | `-1` | PreVectorizationFusion 最大融合逐元素算子数 |
| `max-fused-ops-in-auto-vectorize-v2` | int32 | `-1` | AutoVectorizeV2 最大融合算子数 |
| `max-buffer-count-tuning` | int64 | `0` | 调优最大缓冲计数 |
| `cube-tiling-tuning` | int64 list | `[]` | Cube tiling 参数调优 |
| `enable-count-buffer-dma-opt` | bool | `false` | DMA 缓冲不被 Vector 重用 |
| `external-tiling-func-path` | string | `"-"` | 外部 tiling 函数路径 |
| `inject-ir-from-file` | string | `""` | 注入 IR 文件路径（调试用） |
| `block-dim` | unsigned | `1` | 块维度大小 |

### 3.4 内部状态

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `insertFFTS` | bool | `true` | 是否插入 FFTS（非 CLI 选项，内部状态） |

## 4. Pipeline 入口函数

### 4.1 HFusion Pipeline

```cpp
void buildHFusionPipelines(OpPassManager &pm,
                           const HFusionPipelineOptions &options);

void buildHFusionRegBasePipeline(OpPassManager &pm,
                                 const HFusionPipelineOptions &options);

void registerLowerHFusionPipelines();

bool enableSIMDVFFusion(const HFusionPipelineOptions &options);
```

### 4.2 HIVM Pipeline

```cpp
void buildConvertToHIVMPipeline(OpPassManager &pm,
                                const ConvertToHIVMPipelineOptions &options);

void buildHIVMTensorOptimizations(
    OpPassManager &pm, const HIVMPipelineOptions &hivmPipelineOptions);

void buildLowerHIVMPipelines(OpPassManager &pm,
                             const HIVMPipelineOptions &hivmPipelineOptions);

void registerConvertToHIVMPipelines();
void registerLowerHIVMPipelines();

void canonicalizationHIVMPipeline(OpPassManager &pm);
```

## 5. 典型配置示例

### 5.1 纯 SIMD（Cube）模式

```
--enable-triton-kernel-compile=true
--pure-simt=false
--enable-mixed-cv=false
--target=Ascend910B1
```

### 5.2 纯 SIMT（Vector）模式

```
--pure-simt=true
--enable-mixed-cv=false
--use-dpx=true
--target=Ascend910B1
```

### 5.3 MIX（Cube+Vector）模式

```
--enable-mixed-cv=true
--enable-auto-multi-buffer=true
--enable-auto-cv-balance=true
--target=Ascend910B1
```

### 5.4 调试模式

```
--enable-hivm-inject-barrier-all-sync=true
--enable-print-memory-allocated-size=true
--inject-ir-from-file=/path/to/debug.ir
```
