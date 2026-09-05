# Triton IR → Linalg IR 转换

## 概述

Triton IR → Linalg IR 转换是 Triton-Ascend SIMD 编译路径的核心阶段，将 Triton Dialect 的操作转换为 MLIR 标准的 Linalg Dialect 操作。该转换由 [ttir_to_linalg()](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L95) 函数驱动，在 Ascend 特有 Pass 之后执行 `TritonToLinalg` Pass 完成。

Linalg IR 是连接 Triton 前端和 BiSheng Compiler 后端的桥梁，BiSheng Compiler 接收 Linalg IR 后执行 HFusion、HIVM 等优化并最终生成 NPU 二进制。

## 关键概念

| 概念 | 说明 |
|------|------|
| Linalg Dialect | MLIR 的线性代数 Dialect，表达结构化计算（映射、归约、矩阵乘等） |
| mix_mode | 混合执行模式，取值为 `aiv`（Vector）、`aic`（Cube）、`mix_simd_simt`（混合） |
| parallel_mode | 并行模式，取值为 `simd`、`simt`、`mix_simd_simt` |
| tensor_kind | 张量类型标记，0=输入，1=输出，用于 BiSheng Compiler 的内存规划 |
| named_ops | 是否使用命名 Linalg 操作（如 `linalg.matmul`），默认为 True |
| enable_nd2nz_on_vector | 是否在 Vector 核心上启用 ND→NZ 布局转换 |
| enable_select_analysis | 是否启用 select 分析 |
| _parse_linalg_metadata | Linalg IR 元数据解析函数 |

## ttir_to_linalg() 函数详解

[ttir_to_linalg()](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L95) 是 Ascend Pass 序列的入口函数，负责将 TTIR 转换为 Linalg IR：

```python
def ttir_to_linalg(mod, metadata, opt, *, named_ops=False):
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ttir.mlir")
        dst_path = os.path.join(tmpdir, "kernel.ttadapter.mlir")
        Path(src_path).write_text(ttir_code)

        # 读取编译选项
        enable_nd2nz_on_vector = metadata["enable_nd2nz_on_vector"]
        enable_select_analysis = metadata["enable_select_analysis"]
        compile_on_910_95 = metadata["compile_on_910_95"]
        force_simt_template = metadata["force_simt_template"]
        enable_mask_fallback_conversion = metadata["enable_mask_fallback_conversion"]
        optimize_dynamic_offset = metadata["optimize_dynamic_offset"]
        auto_blockify_size = metadata["auto_blockify_size"]

        # 创建 Pass Manager
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        # 1. AutoBlockify
        ascend.passes.ttir.add_auto_blockify(pm, auto_blockify_size)

        # 2. DAG 亲和性优化（可选）
        if metadata["add_auto_scheduling"]:
            ascend.passes.ttir.add_dag_sync(pm)
            ascend.passes.ttir.add_dag_scope(pm)
            passes.common.add_cse(pm)
            passes.common.add_canonicalizer(pm)
            ascend.passes.ttir.add_dag_ssbuffer(pm)
            passes.common.add_cse(pm)
            passes.common.add_canonicalizer(pm)

        # 3-12. Ascend Passes + TritonToLinalg
        ascend.passes.ttir.add_triton_to_structure(pm, ...)
        ascend.passes.ttir.add_discrete_mask_access_conversion(pm, ...)
        ascend.passes.ttir.add_triton_to_annotation(pm)
        ascend.passes.ttir.add_triton_to_unstructure(pm, ...)
        ascend.passes.ttir.add_triton_to_hivm(pm)
        ascend.passes.ttir.add_triton_to_hfusion(pm)
        ascend.passes.ttir.add_triton_to_llvm(pm)
        ascend.passes.ttir.add_bubble_up_operation(pm)
        ascend.passes.ttir.add_triton_to_structure(pm, ...)  # 二次
        ascend.passes.ttir.add_triton_to_linalg(pm, False, named_ops,
                                                 enable_nd2nz_on_vector,
                                                 enable_select_analysis,
                                                 compile_on_910_95)
        pm.run(mod)

        if opt.debug:
            dump_manager = get_dump_manager(metadata["hash"])
            dump_manager.put(str(mod), "kernel.ttadapter.mlir", binary=False)

        return str(mod)
```

### 关键步骤

1. **序列化 TTIR**：将 MLIR Module 序列化为字符串
2. **创建临时目录**：用于存放中间文件
3. **构建 Pass Manager**：注册所有 Ascend Pass
4. **执行 Pass 序列**：按顺序执行所有 Pass
5. **返回 Linalg IR**：将转换后的 Module 序列化为字符串返回

### 注意事项

- 函数返回的是**字符串**而非 MLIR Module 对象，后续阶段（`linalg_to_bin_*`）直接使用字符串作为输入
- `named_ops=True` 时使用命名 Linalg 操作（如 `linalg.matmul`），否则使用通用 `linalg.generic`
- 调试模式下，转换后的 IR 保存为 `kernel.ttadapter.mlir`

## TritonToLinalg Pass 的工作原理

### 转换框架

TritonToLinalg 使用 MLIR 的 DialectConversion 框架，为每个 Triton 操作注册对应的转换模式（ConversionPattern）。

### 核心转换器

| 转换器 | 描述 |
|--------|------|
| StoreConverter | `triton::StoreOp` → `memref::copy` |
| AddPtrConverter | `triton::AddPtrOp` → `memref::ReinterpretCast` |
| GetProgramIDConverter | `triton::GetProgramIdOp` → 函数参数 |
| GetNumProgramsConverter | `triton::GetNumProgramsOp` → 函数参数 |
| LoadConverter | `triton::LoadOp` → `memref::copy` + `bufferization::ToTensorOp` |
| AtomicRMWConverter | `triton::AtomicRMWOp` → `linalg::GenericOp` |
| AtomicCASConverter | `triton::AtomicCASOp` → `linalg::GenericOp` |
| MakeRangeConverter | `triton::MakeRangeOp` → `linalg::GenericOp` |
| SplatConverter | `triton::SplatOp` → `linalg::FillOp` |
| ClampFConverter | `triton::ClampFOp` → `tensor::EmptyOp` + `linalg::FillOp` |
| PreciseDivConverter | `triton::PreciseDivFOp` → `arith::DivFOp` |
| ArgMinConverter | `triton::ArgMinOp` → `linalg::ReduceOp` |
| ArgMaxConverter | `triton::ArgMaxOp` → `linalg::ReduceOp` |
| ReduceConverter | `triton::ReduceOp` → `linalg::ReduceOp` |
| ScanConverter | `triton::ScanOp` → `func::CallOp` |
| ReshapeConverter | `triton::ReshapeOp` → `tensor::ReshapeOp` |
| ExpandDimsConverter | `triton::ExpandDimsOp` → `tensor::ExpandShapeOp` |
| BroadcastConverter | `triton::BroadcastOp` → `linalg::BroadcastOp` |
| DenseConstantConverter | `arith::ConstantOp` → `linalg::FillOp` |
| ExternElementwiseClOpConverter | `triton::ExternElementwiseOp` → `linalg::MapOp` |
| TritonMulhiuiConverter | `triton::MulhiUIOp` → `arith::MulSIExtendedOp` |
| TritonPreciseSqrtConverter | `triton::PreciseSqrtOp` → `math::SqrtOp` |
| AdvanceConverter | `triton::AdvanceOp` → `memref::ReinterpretCastOp` |
| TransposeConverter | `triton::TransOp` → `linalg::TransposeOp` |
| SplitConverter | `triton::SplitOp` → `tensor::ExtractSliceOp` |
| JoinConverter | `triton::JoinOp` → `tensor::InsertSliceOp` |
| CatConverter | `triton::CatOp` → `tensor::InsertSliceOp` |
| BitcastConverter | `triton::BitcastOp` → `arith::BitcastOp` |
| LoopConverter\<scf::ForOp\> | `scf::ForOp` → `scf::ForOp` |
| LoopConverter\<scf::WhileOp\> | `scf::WhileOp` → `scf::WhileOp` |
| YieldConverter | `scf::YieldOp` → `scf::YieldOp` |
| GatherConverter | `triton::GatherOp` → `func::FuncOp` |
| GatherLoadConverter | `triton::GatherLoadOp` → `scf::ForOp` |
| DeviceAssertConverter | `triton::AssertOp` → `func::FuncOp` |
| DevicePrintConverter | `triton::PrintOp` → `func::FuncOp` |
| MatmulConverter | `triton::DotOp` → `linalg::MatmulOp` |
| SortOpConverter | `triton::SortOp` → `func::FuncOp` |
| DotScaledConverter | `triton::DotScaledOp` → `linalg::MatmulOp` |
| PtrToIntConverter | `triton::PtrToIntOp` | 
| MakeTensorPtrConverter | `triton::MakeTensorPtrOp` → `arith::IndexCastOp` |

### 编译选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `globalKernel` | bool | False | 是否为全局 Kernel |
| `namedOps` | bool | True | 是否使用命名 Linalg 操作 |
| `enableNd2nzOnVector` | bool | False | 是否在 Vector 核心启用 ND→NZ 转换 |
| `enableSelectAnalysis` | bool | True | 是否启用 select 分析 |
| `compileOn91095` | bool | 自动检测 | 是否在 910_95 平台编译 |

## Linalg IR 的结构和语义

### Linalg Dialect 概述

Linalg（Linear Algebra）Dialect 是 MLIR 中用于表达结构化计算的高级 Dialect，其核心思想是将计算描述为对多维张量的迭代和映射。

### 核心 Linalg 操作

| 操作 | 语义 | 示例 |
|------|------|------|
| `linalg.matmul` | 矩阵乘法 | C += A * B |
| `linalg.fill` | 填充张量 | 用标量填充张量 |
| `linalg.broadcast` | 广播 | 将低维张量广播到高维 |
| `linalg.transpose` | 转置 | 交换张量维度 |
| `linalg.reduce` | 归约 | 沿某维度归约 |
| `linalg.generic` | 通用结构化计算 | 自定义迭代和计算 |
| `linalg.map` | 逐元素映射 | 对每个元素应用函数 |
| `linalg.copy` | 复制 | 张量复制 |

### Linalg IR 示例

向量加法的 Linalg IR：

```mlir
func.func @add_kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>,
                       %arg2: memref<?xf32>, %arg3: i32) {
  %c0 = arith.constant 0 : index
  %0 = arith.index_cast %arg3 : i32 to index
  %1 = memref.load %arg0[%c0] : memref<?xf32>
  // ...
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                    affine_map<(d0) -> (d0)>,
                                    affine_map<(d0) -> (d0)>],
                   iterator_types = ["parallel"]}
    ins(%subview_0, %subview_1 : memref<?xf32>, memref<?xf32>)
    outs(%subview_2 : memref<?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
  }
  return
}
```

矩阵乘法的 Linalg IR：

```mlir
func.func @matmul_kernel(...) {
  linalg.matmul ins(%A, %B : memref<?x?xf32>, memref<?x?xf32>)
                outs(%C : memref<?x?xf32>)
  return
}
```

## mix_mode 参数

### mix_mode 取值

| 值 | 说明 | 适用场景 |
|----|------|----------|
| `aiv` | Vector 核心执行 | 逐元素运算、向量计算 |
| `aic` | Cube 核心执行 | 矩阵乘法、Cube 密集型计算 |
| `mix_simd_simt` | SIMD/SIMT 混合执行 | 包含离散访存和结构化计算的混合场景 |

### mix_mode 的确定

`mix_mode` 在 `_parse_linalg_metadata()` 中从 Linalg IR 解析得到：

```python
metadata["mix_mode"] = re.search(MIX_MODE_REGEX, linalg).group(1)
```

### mix_mode 对编译的影响

| mix_mode | BiSheng Compiler 行为 |
|----------|----------------------|
| `aiv` | 走 HFusion Vector 编译路径 |
| `aic` | 走 HFusion Cube 编译路径，禁用 HFusion 向量化 |
| `mix_simd_simt` | 走混合编译路径，SIMD 部分走 HFusion，SIMT 部分走 TritonGPU |

### parallel_mode

`parallel_mode` 与 `mix_mode` 类似，从 Linalg IR 中解析：

```python
metadata["parallel_mode"] = re.search(PARALLEL_MODE_REGEX, linalg).group(1)
```

| 值 | 说明 |
|----|------|
| `simd` | SIMD 并行模式 |
| `simt` | SIMT 并行模式 |
| `mix_simd_simt` | 混合并行模式 |

## Linalg 元数据解析

### _parse_linalg_metadata() 函数

[_parse_linalg_metadata()](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L288) 从 Linalg IR 字符串中提取编译所需的元数据：

```python
def _parse_linalg_metadata(linalg: str, metadata: dict):
    metadata["shared"] = 1
    metadata["auto_tile_and_bind_subblock"] = not re.search(
        DISABLE_AUTO_TILE_AND_BIND_SUBBLOCK_REGEX, linalg)
    metadata["mix_mode"] = re.search(MIX_MODE_REGEX, linalg).group(1)
    metadata["parallel_mode"] = re.search(PARALLEL_MODE_REGEX, linalg).group(1)
    metadata["kernel_name"] = re.search(KERNEL_NAME_REGEX, linalg).group(1)
    metadata["name"] = metadata["kernel_name"] + "_" + metadata["mix_mode"]
    metadata["tensor_kinds"] = [int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, linalg)]
    metadata["required_ub_bits"] = 0
    metadata["bitcodes"] = [val for group in re.findall(BITCODES_REGEX, linalg) for val in group if val]
    return linalg, metadata
```

### 解析的元数据字段

| 字段 | 正则表达式 | 示例 | 说明 |
|------|-----------|------|------|
| `mix_mode` | `mix_mode\s*=\s*"([^"]+)"` | `mix_mode = "aiv"` | 混合执行模式 |
| `parallel_mode` | `parallel_mode\s*=\s*"([^"]+)"` | `parallel_mode = "simd"` | 并行模式 |
| `kernel_name` | `func\.func\s+@(\w+)` | `func.func @add_kernel` | Kernel 函数名 |
| `tensor_kinds` | `tt\.tensor_kind\s*=\s*([^:\s}]+)` | `tt.tensor_kind = 0` | 张量类型（0=输入，1=输出） |
| `bitcodes` | `bitcode\s*=\s*"([^"]+)"` | `bitcode = "a.bc"` | 需要链接的 bitcode 路径 |
| `auto_tile_and_bind_subblock` | `hivm.disable_auto_tile_and_bind_subblock` | 存在则禁用 | 是否启用自动分块和绑定 |

### kernel_name 命名规则

Kernel 名称由 `kernel_name` + `_` + `mix_mode` 组成：

```
metadata["name"] = metadata["kernel_name"] + "_" + metadata["mix_mode"]
```

例如：`add_kernel_aiv`、`matmul_kernel_aic`。

运行时通过 `pack_metadata()` 处理名称长度限制（CANN 运行时限制 kernel name <= 49 字符）。

### _parse_ttir_metadata() 函数

[_parse_ttir_metadata()](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L342) 用于 SIMT 路径（TTIR 直接编译），从 TTIR 中解析元数据：

```python
def _parse_ttir_metadata(ttir: str, metadata: dict):
    metadata["shared"] = 1
    metadata["mix_mode"] = "aiv"  # TTIR 输入只支持 vector kernel
    metadata["kernel_name"] = re.search(KERNEL_NAME_REGEX, ttir).group(1)
    metadata["name"] = metadata["kernel_name"] + "_" + metadata["mix_mode"]
    metadata["tensor_kinds"] = [int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, ttir)]
    return metadata
```

## IR 转换示例

### 向量加法：TTIR → Linalg

TTIR 输入：

```mlir
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                              %arg2: !tt.ptr<f32>, %arg3: i32) {
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %2 = arith.muli %0, %1 : i32
    %3 = arith.addi %2, %1 : tensor<1024xi32>
    %4 = arith.cmpi slt, %3, %arg3 : tensor<1024xi1>
    %5 = tt.addptr %arg0, %3 : !tt.ptr<f32>, tensor<1024xi32>
    %6 = tt.load %5, %4 : tensor<1024xf32>
    %7 = tt.addptr %arg1, %3 : !tt.ptr<f32>, tensor<1024xi32>
    %8 = tt.load %7, %4 : tensor<1024xf32>
    %9 = arith.addf %6, %8 : tensor<1024xf32>
    %10 = tt.addptr %arg2, %3 : !tt.ptr<f32>, tensor<1024xi32>
    tt.store %10, %9, %4
  }
}
```

Linalg IR 输出（简化）：

```mlir
module {
  func.func @add_kernel(%arg0: memref<?xf32> {tt.divisibility = 16, tt.tensor_kind = 0},
                         %arg1: memref<?xf32> {tt.divisibility = 16, tt.tensor_kind = 0},
                         %arg2: memref<?xf32> {tt.divisibility = 16, tt.tensor_kind = 1},
                         %arg3: i32 {tt.divisibility = 16})
    attributes {mix_mode = "aiv", parallel_mode = "simd"} {
    // GetProgramId → 函数参数
    // Load → memref.copy + bufferization.to_tensor
    // Add → linalg.generic
    // Store → memref.copy
    return
  }
}
```

### 矩阵乘法：TTIR → Linalg

TTIR 输入：

```mlir
%dot = tt.dot %a, %b, %c : tensor<64x64xf32>
```

Linalg IR 输出：

```mlir
linalg.matmul ins(%A, %B : memref<64x64xf32>, memref<64x64xf32>)
              outs(%C : memref<64x64xf32>)
```

## NPU 适配要点

1. **named_ops 选项**：NPU 路径使用 `named_ops=True`，生成 `linalg.matmul` 等命名操作，便于 BiSheng Compiler 识别和优化
2. **mix_mode 编码**：`mix_mode` 被编码到函数属性和 kernel 名称中，运行时通过名称区分执行模式
3. **tensor_kind 标记**：输入/输出张量通过 `tt.tensor_kind` 属性标记，BiSheng Compiler 据此进行内存规划
4. **bitcode 链接**：Linalg IR 中可能包含 `bitcode` 属性，指定需要链接的 bitcode 文件
5. **910_95 差异**：910_95 平台使用 `linalg_to_bin_enable_npu_compile_910_95()`，其他平台使用 `linalg_to_bin_enable_npu_compile_A2_A3()`

## 常见问题

### Q: 为什么 ttir_to_linalg() 返回字符串而非 Module？

因为后续的 `linalg_to_bin_*()` 函数需要将 Linalg IR 作为文件传递给 BiSheng Compiler（外部进程），字符串格式便于写入临时文件。

### Q: mix_mode 是如何确定的？

`mix_mode` 由 TritonToLinalg Pass 在转换过程中根据操作类型确定。纯向量操作为 `aiv`，包含矩阵乘法为 `aic`，包含离散访存为 `mix_simd_simt`。

### Q: named_ops=True 和 named_ops=False 有何区别？

`named_ops=True` 使用 `linalg.matmul`、`linalg.fill` 等命名操作，BiSheng Compiler 可以直接识别。`named_ops=False` 使用 `linalg.generic`，表达更通用但优化空间较小。

### Q: TTIR 直接编译路径为何只支持 aiv？

SIMT 路径（`ttir_to_npubin()`）将 TTIR 直接传给 BiSheng Compiler 的 Triton IR 编译流水线，该流水线目前只支持 Vector 核心执行，因此 `mix_mode` 固定为 `aiv`。

## 相关文档

- [01-pipeline-overview.md](01-pipeline-overview.md) - 编译流水线总览
- [04-ascend-passes.md](04-ascend-passes.md) - Ascend 特有 Pass 详解
- [06-linalg-to-binary.md](06-linalg-to-binary.md) - Linalg IR → 设备二进制
