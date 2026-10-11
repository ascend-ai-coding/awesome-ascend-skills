# HFusion 变换 Pass 总览

## 1. 概述

HFusion 方言提供了 40+ 个变换 Pass，覆盖操作融合、自动调度、自动向量化、类型合法化、Tiling 数据打包等多个编译阶段。本文档列出所有 Pass 的名称和功能简述。

> 源码参考：[Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/Transforms/Passes.td)、[Passes.h](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/Transforms/Passes.h)

## 2. 融合与调度 Pass

| Pass 名称 | 作用域 | 功能简述 |
|-----------|--------|----------|
| `hfusion-fuse-ops` | ModuleOp | 张量上的操作融合，将可融合操作 outline 为独立函数 |
| `hfusion-auto-schedule` | ModuleOp | 自动调度融合 Kernel，生成 Tiling 参数 |
| `hfusion-infer-func-fusion-kind` | FuncOp | 推断函数的融合类型 |
| `hfusion-infer-out-shapes` | ModuleOp | 生成输出张量形状推断函数 |
| `hfusion-compose-multi-reduce` | FuncOp | 组合多个归约为单个操作 |
| `hfusion-decompose-multi` | FuncOp | 将多操作分解为单操作 |
| `hfusion-pre-vectorization-fusion` | FuncOp | 向量化前的逐元操作融合 |
| `hfusion-eliminate-duplicate-funcs` | ModuleOp | 消除融合后的重复函数 |

## 3. 向量化 Pass

| Pass 名称 | 作用域 | 功能简述 |
|-----------|--------|----------|
| `hfusion-auto-vectorize` | ModuleOp | 自动向量化 Linalg 命名操作 |
| `hfusion-auto-vectorize-v2` | ModuleOp | 第二版自动向量化（Tile + Fuse + Vectorize） |
| `hfusion-generic-unroller` | FuncOp | 非可向量化 Linalg 操作的展开 |
| `hfusion-pull-slice-into-vector-function` | ModuleOp | 将 extract_slice/insert_slice 拉入 VF 函数 |
| `outline-vector-function` | ModuleOp | Outline 向量函数 |
| `hfusion-vectorize-ops` | ModuleOp | 向量化 HFusion 操作 |
| `remove-mask-from-unaligned-reduction-loop` | FuncOp | 移除未对齐归约循环的掩码 |

## 4. 类型合法化 Pass

| Pass 名称 | 作用域 | 功能简述 |
|-----------|--------|----------|
| `hfusion-legalize-scalar` | FuncOp | 将标量操作合法化为张量操作 |
| `hfusion-legalize-bf16` | FuncOp | 将 BF16 类型合法化为 FP32 |
| `hfusion-legalize-fp8` | FuncOp | 将 FP8 类型合法化为 FP32 |
| `hfusion-legalize-bool` | ModuleOp | 将 int8/int1 布尔类型合法化 |
| `hfusion-downgrade-fp64` | FuncOp | 将 FP64 常量降级为 FP32 |
| `hfusion-normalize-ops` | FuncOp | 规范化 HFusion 操作 |
| `hfusion-normalize-slice-ops` | FuncOp | 规范化 Slice 操作 |

## 5. Tiling 与数据打包 Pass

| Pass 名称 | 作用域 | 功能简述 |
|-----------|--------|----------|
| `hfusion-pack-tiling-data` | ModuleOp | 将动态 Tiling 信息打包为结构体 |
| `hfusion-constantize-tiling-data` | ModuleOp | 在 Tiling 和 Device 函数间传播常量 |
| `hfusion-add-ffts-addr` | ModuleOp | 添加 FFTS 基地址到函数参数 |

## 6. 形状与参数变换 Pass

| Pass 名称 | 作用域 | 功能简述 |
|-----------|--------|----------|
| `hfusion-tensor-results-to-out-params` | ModuleOp | 将张量结果移至函数输出参数 |
| `hfusion-hoist-tensor-empty` | ModuleOp | 将 tensor.empty 提升到函数参数并合并 |
| `hfusion-simplify-vf-arg` | ModuleOp | 简化 VF 函数参数 |
| `hfusion-merge-vf` | ModuleOp | 合并 VF 函数 |
| `hfusion-fold-unit-dims` | FuncOp | 折叠 Linalg 操作中的单位维度 |

## 7. 操作变换 Pass

| Pass 名称 | 作用域 | 功能简述 |
|-----------|--------|----------|
| `hfusion-convert-generic-to-named` | FuncOp | 将 Linalg generic 转换为命名操作 |
| `hfusion-flatten-ops` | FuncOp | 展平 Linalg 和 HFusion 操作 |
| `hfusion-inline-brc` | FuncOp | 内联广播类操作 |
| `hfusion-outline-single-op` | FuncOp | Outline 单个 Linalg 操作为 Kernel |
| `hfusion-simplify-ops` | FuncOp | 简化操作 |
| `hfusion-reorder-ops` | FuncOp | 按 BFS 重排操作 |
| `hfusion-generalize` | FuncOp | 将 HFusion 操作转换为 linalg.generic |
| `hfusion-decompose` | FuncOp | 分解实现 AggregatedOpInterface 的操作 |

## 8. 缓存与符号 Pass

| Pass 名称 | 作用域 | 功能简述 |
|-----------|--------|----------|
| `hfusion-cache-io` | FuncOp | 缓存输入输出参数 |
| `hfusion-cache-io-for-return-arg` | FuncOp | 缓存直接返回的参数 |
| `hfusion-recache-io` | FuncOp | 重新缓存 IO |
| `hfusion-remove-cache-io` | FuncOp | 移除缓存 IO |
| `hfusion-fold-symbolic-dim` | FuncOp | 用 hfusion::SymbolicDimOp 替换 tensor.dim |
| `hfusion-unfold-symbolic-dim` | FuncOp | 将 SymbolicDimOp 替换为符号参数 |
| `hfusion-drop-symbols` | FuncOp | 丢弃操作中的 ranked tensor 符号 |

## 9. Host 函数与 Triton 适配 Pass

| Pass 名称 | 作用域 | 功能简述 |
|-----------|--------|----------|
| `hfusion-wrap-host-func` | ModuleOp | 为 Host 辅助函数创建包装函数 |
| `adapt-triton-kernel` | ModuleOp | 适配 Triton 入口 Kernel |

## 10. 关键 Pass 选项

### 10.1 hfusion-fuse-ops

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output-mode` | OutputMode | Multiple | Outline 函数输出模式 |
| `fusion-mode` | FusionKind | Unknown | 融合类型 |
| `always-inline` | bool | false | 始终内联 outline 函数 |
| `move-out-to-param` | bool | true | 将张量输出移至参数 |
| `max-horizontal-fusion-size` | int | -1 | 最大水平融合大小 |
| `multi-kernel` | bool | false | 启用多 Kernel outline |
| `enable-symbol-analysis` | bool | false | 启用符号分析 |

### 10.2 hfusion-auto-vectorize

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vector-length` | unsigned | 256 | 向量长度（字节） |
| `peel-loops` | bool | false | 尝试剥离循环 |
| `max-vectorize-axes` | int64_t | -1 | 最大向量化轴数 |
| `tree-reduce` | bool | false | 使用树形归约 |

### 10.3 hfusion-flatten-ops

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `flatten-mode` | FlattenMode | Greedy | 展平模式 |
| `skip-host` | bool | false | 跳过 Host 函数 |
| `multi-dynamic-shape` | bool | true | 折叠多个动态形状 |
| `register-based` | bool | false | 使用基于寄存器的展平模式 |
| `skip-scope` | bool | true | 存在 scope 时跳过 |

### 10.4 hfusion-auto-vectorize-v2

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vector-length` | unsigned | 256 | 向量长度（字节） |
| `max-fused-ops` | unsigned | 15 | 单个融合节点最大操作数 |
| `enable-multiple-consumer-fusion` | bool | false | 启用多消费者融合 |
