# HMAP 方言

## 1. 概述

HMAP（Hybrid Mesh Aware Parallelism，混合 Mesh 感知并行）方言提供集合通信操作，用于多卡/多设备间的数据交换。它基于 MLIR Mesh 方言构建，支持 Mesh 感知的并行策略。

- **方言名称**：`hmap`
- **C++ 命名空间**：`::mlir::hmap`
- **依赖方言**：`mesh::MeshDialect`

> 源码参考：[HMAPOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HMAP/IR/HMAPOps.td)、[HMAPBase.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HMAP/IR/HMAPBase.td)

## 2. 方言定义

```tablegen
def HMAP_Dialect : Dialect {
  let name = "hmap";
  let cppNamespace = "::mlir::hmap";
  let description = [{
    HMAP (Hybrid Mesh Aware Parallelism) dialect.
  }];
  let dependentDialects = ["mesh::MeshDialect"];
}
```

## 3. 操作基类

### 3.1 HMAP_Op

```tablegen
class HMAP_Op<string mnemonic, list<Trait> traits = []> :
    Op<HMAP_Dialect, mnemonic, traits>
```

### 3.2 HMAP_CollectiveCommunicationOpBase

集合通信操作基类，自动附加 `SymbolUserOpInterface` 和 `OpAsmOpInterface` traits。

| 公共参数 | 类型 | 说明 |
|----------|------|------|
| `mesh` | `FlatSymbolRefAttr` | Mesh 符号引用 |
| `mesh_axes` | `DenseI16ArrayAttr` (默认: {}) | Mesh 轴 |

## 4. 操作定义

### 4.1 hmap.all_to_allv

#### 功能

在设备间执行全互联向量（All-to-All Vector）通信操作，将张量分片发送到所有设备并接收来自所有设备的分片。

#### 操作签名

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `mesh` | `FlatSymbolRefAttr` | Mesh 符号引用 |
| `mesh_axes` | `DenseI16ArrayAttr` (默认: {}) | Mesh 轴 |
| `input` | `AnyNon0RankedTensor` | 输入张量 |
| `input_splits` | `AnyTensor` | 输入分片大小 |
| `output_splits` | `AnyTensor` | 输出分片大小 |
| `result` | `AnyNon0RankedTensor` | 输出张量 |

#### Traits

- `Pure`
- `SymbolUserOpInterface`
- `OpAsmOpInterface`

#### MLIR 示例

```mlir
%result = hmap.all_to_allv %input on @mesh
  input_splits = %in_splits : tensor<4xi32>
  output_splits = %out_splits : tensor<4xi32>
  : tensor<1024xf32> -> tensor<1024xf32>
```

带 mesh_axes 的示例：

```mlir
%result = hmap.all_to_allv %input on @mesh mesh_axes = [0, 1]
  input_splits = %in_splits : tensor<4xi32>
  output_splits = %out_splits : tensor<4xi32>
  : tensor<1024xf32> -> tensor<1024xf32>
```

## 5. 与 HACC 方言的协作

HMAP 操作与 HACC 方言的 `kMeshArg` 参数类型配合使用：

1. HMAP 集合通信操作在 Host 端编排
2. Mesh 参数通过 `hacc.arg_type = #hacc.arg_type<mesh_arg>` 传递给 Device Kernel
3. Device Kernel 使用 Mesh 参数进行多卡数据交换

## 6. 与 MLIR Mesh 方言的关系

HMAP 方言基于 MLIR Mesh 方言构建，但提供了更高级的抽象：

| HMAP | Mesh | 说明 |
|------|------|------|
| `hmap.all_to_allv` | `mesh.all_to_all` | HMAP 支持不均匀分片（v 表示 variable） |

HMAP 的 `mesh` 参数引用 MLIR Mesh 方言定义的 Mesh 资源，`mesh_axes` 指定通信使用的 Mesh 维度。
