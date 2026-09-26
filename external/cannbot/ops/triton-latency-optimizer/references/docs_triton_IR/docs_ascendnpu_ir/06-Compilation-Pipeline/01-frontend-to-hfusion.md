# 前端→HFusion 转换规则

本文档详细描述从前端方言到 HFusion 方言的转换 Pass。

源码参考：[Conversion/Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Conversion/Passes.td)

## 1. 转换总览

前端→HFusion 的转换将各种前端方言的操作转换为 HFusion 方言的操作，为后续的算子融合和调度优化做准备。

```
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  arith   │ │  math    │ │ linalg   │ │   gpu    │ │  tensor  │ │  torch   │
└────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │            │            │            │
     ▼            ▼            ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HFusion Dialect                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. ConvertArithToHFusion

将 `arith` 方言操作转换为 HFusion 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `convert-arith-to-hfusion` |
| 构造函数 | `mlir::createArithToHFusionConversionPass()` |
| 依赖方言 | `arith::ArithDialect`, `linalg::LinalgDialect`, `hfusion::HFusionDialect` |

**转换语义**：
- `arith.addf/addi` → `hfusion` 命名算子或 `linalg.generic`
- `arith.mulf/muli` → `hfusion` 命名算子或 `linalg.generic`
- 其他逐元素算术操作类似映射

## 3. ConvertMathToHFusion

将 `math` 方言操作转换为 HFusion 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `convert-math-to-hfusion` |
| 构造函数 | `mlir::createMathToHFusionConversionPass()` |
| 依赖方言 | `linalg::LinalgDialect`, `hfusion::HFusionDialect` |

**转换语义**：
- `math.exp` → `hfusion` 命名算子
- `math.log` → `hfusion` 命名算子
- `math.sqrt` → `hfusion` 命名算子
- `math.sin/cos` → `hfusion` 命名算子
- 其他数学函数类似映射

## 4. ConvertLinalgToHFusion

将 `linalg` 方言操作转换为 HFusion 操作。这是最核心的前端转换 Pass。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `convert-linalg-to-hfusion` |
| 构造函数 | `mlir::createLinalgToHFusionConversionPass()` |
| 依赖方言 | `arith::ArithDialect`, `linalg::LinalgDialect`, `hfusion::HFusionDialect`, `bufferization::BufferizationDialect` |

**转换语义**：
- `linalg.matmul` → `hfusion` 矩阵乘法命名算子
- `linalg.batch_matmul` → `hfusion` 批量矩阵乘法命名算子
- `linalg.generic` (逐元素) → `hfusion` 逐元素命名算子
- `linalg.generic` (归约) → `hfusion` 归约命名算子
- `linalg.fill` → `hfusion` 填充算子
- `linalg.conv_2d` → `hfusion` 卷积命名算子

此 Pass 依赖 `bufferization` 方言，因为部分 linalg 操作在转换前需要缓冲化。

## 5. ConvertGPUToHFusion

将 `gpu` 方言操作转换为 HFusion 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `convert-gpu-to-hfusion` |
| 构造函数 | `mlir::createGPUToHFusionConversionPass()` |
| 依赖方言 | `gpu::GPUDialect`, `hfusion::HFusionDialect` |

**转换语义**：
- `gpu.launch_func` → HFusion kernel 启动逻辑
- GPU 并程相关操作 → HFusion 调度操作

## 6. ConvertTensorToHFusion

将 `tensor` 方言操作转换为 Linalg/HFusion 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `convert-tensor-to-hfusion` |
| 构造函数 | `mlir::createTensorToHFusionConversionPass()` |
| 依赖方言 | `tensor::TensorDialect`, `hfusion::HFusionDialect`, `linalg::LinalgDialect` |

**转换语义**：
- `tensor.extract_slice` → HFusion 切片操作
- `tensor.insert_slice` → HFusion 切片写入操作
- `tensor.empty` → HFusion 空张量操作
- `tensor.expand_shape/collapse_shape` → HFusion 形状变换操作

## 7. ConvertTorchToHFusion

将 Torch 方言操作转换为 Linalg/HFusion 命名算子。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `convert-torch-to-hfusion` |
| 构造函数 | `mlir::createConvertTorchToHFusionPass()` |
| 依赖方言 | `linalg::LinalgDialect`, `math::MathDialect`, `func::FuncDialect`, `tensor::TensorDialect`, `arith::ArithDialect`, `hfusion::HFusionDialect` |
| 条件编译 | `BISHENGIR_ENABLE_TORCH_CONVERSIONS` |

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ensure-no-implicit-broadcast` | bool | `false` | 是否确保无隐式广播语义 |

**转换语义**：
- ATen 算子 → linalg 或 hfusion 命名算子
- 支持动态形状和符号维度

### ConvertTorchToSymbol

将 Torch 符号操作转换为 Symbol 方言。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `convert-torch-to-symbol` |
| 构造函数 | `mlir::createConvertTorchToSymbolPass()` |
| 依赖方言 | `func::FuncDialect`, `tensor::TensorDialect`, `symbol::SymbolDialect`, `arith::ArithDialect` |
| 条件编译 | `BISHENGIR_ENABLE_TORCH_CONVERSIONS` |

**转换语义**：
- `torch.symbolic_int` → `symbol.symbolic_int`

## 8. 辅助转换 Pass

### 8.1 ConvertArithToAffine

将 arith 操作转换为 affine 操作。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `convert-arith-to-affine` |
| 构造函数 | `mlir::createArithToAffineConversionPass()` |
| 依赖方言 | `arith::ArithDialect`, `affine::AffineDialect` |

### 8.2 ConvertTensorToHIVM

将 tensor 操作直接转换为 HIVM 操作（跳过 HFusion 层的路径）。

| 项目 | 内容 |
|------|------|
| CLI 参数 | `convert-tensor-to-hivm` |
| 构造函数 | `mlir::createTensorToHIVMConversionPass()` |
| 依赖方言 | `tensor::TensorDialect`, `hivm::HIVMDialect` |

## 9. 转换顺序与依赖

前端转换 Pass 的推荐执行顺序：

1. `ConvertTorchToSymbol`（如果使用 Torch 前端）
2. `ConvertTorchToHFusion`（如果使用 Torch 前端）
3. `ConvertLinalgToHFusion`
4. `ConvertArithToHFusion`
5. `ConvertMathToHFusion`
6. `ConvertTensorToHFusion`
7. `ConvertGPUToHFusion`

转换完成后，进入 HFusion 变换阶段（融合、调度、向量化）。
