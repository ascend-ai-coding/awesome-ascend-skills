# HIVM 二元向量运算

> 关键词：HIVM, Binary, vadd, vsub, vmul, vdiv, vmax, vmin, vor, vand, vxor, vmod, vmodui, vpow

## 概述

HIVM 二元向量运算继承自 `HIVM_ElementwiseBinaryOp`，对两个输入向量的对应元素执行运算，产生一个相同形状的结果向量。所有二元运算满足 `ElementwiseNaryOpTrait<2>`，即 2 个输入操作数、1 个输出结果。

二元运算可分为以下类别：
- **算术运算**：vadd, vsub, vmul, vdiv, vmod, vmodui, vpow
- **极值运算**：vmax, vmin
- **位运算**：vor, vand, vxor

> Python API 对应：`tl.abs()`, Triton 的 `+`, `-`, `*`, `/`, `%`, `//` 等运算符，以及 `tl.minimum()`, `tl.maximum()` 等。

## IR 操作定义

### 基类：HIVM_ElementwiseBinaryOp

```tablegen
class HIVM_ElementwiseBinaryOp<string mnemonic, list<Trait> traits = []> :
  HIVM_ElementwiseNaryOp<mnemonic,
                         !listconcat([ElementwiseNaryOpTrait<2>], traits)>;
```

源码参考：[HIVMVectorOps.td#L506-L508](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L506-L508)

---

### hir.vadd — 逐元加法

#### TableGen 定义

```tablegen
def VAddOp : HIVM_ElementwiseBinaryOp<"vadd",
    [SameOperandsElementType, StaticMaxRankTrait<3>,
     OperElemTypeConstraints<[0, 1], [I8, I16, I32, F16, F32, I64]>,
     CommutativeOpTrait, VectorOnlyTrait<0>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
     DeclareOpInterfaceMethods<VectorizableOpInterface>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Binary Vector Addition Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
      2. Support both Vector-Vector and Vector-Scalar operation.
  }];
  let arguments = (ins Variadic<AnyType>:$src, Variadic<AnyShaped>:$dst,
      Optional<AnyMemRef>:$temp_buffer,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast);
}
```

源码参考：[HIVMVectorOps.td#L510-L542](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L510-L542)

#### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| $src | Variadic\<AnyType\> | 是 | 两个输入操作数（支持向量+向量或向量+标量） | 元素类型 I8/I16/I32/F16/F32/I64 |
| $dst | Variadic\<AnyShaped\> | 是 | 输出向量 | 与 src 相同元素类型 |
| $temp_buffer | Optional\<AnyMemRef\> | 否 | 临时缓冲区 | - |
| $transpose | DenseI64ArrayAttr (默认 {}) | 否 | OTF 转置维度 | - |
| $broadcast | DenseI64ArrayAttr (默认 {}) | 否 | OTF 广播维度 | - |

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vadd | I8, I16, I32, F16, F32, I64 | OperElemTypeConstraints<[0, 1], [I8, I16, I32, F16, F32, I64]> |

#### 特殊特性

- **CommutativeOpTrait**：满足交换律，`a + b = b + a`
- **ImplByScalarOpInterface**：支持向量-标量操作
- **VectorizableOpInterface**：支持向量化
- **$src 类型为 AnyType**：允许标量输入（非 AnyShaped），支持 Vector-Scalar 模式

#### IR 示例

```mlir
%result = hivm.hir.vadd ins(%a, %b : tensor<?x5x10xf32>, tensor<?x5x10xf32>) outs(%dst : tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>

hivm.hir.vadd ins(%vec, %scalar : tensor<23x77xi32>, i32) outs(%dst : tensor<23x77xi32>) -> tensor<23x77xi32>

hivm.hir.vadd ins(%a, %b : memref<16xf16, #hivm.address_space<ub>>, memref<16xf16, #hivm.address_space<ub>>) outs(%c : memref<16xf16, #hivm.address_space<ub>>)
```

---

### hir.vsub — 逐元减法

#### TableGen 定义

```tablegen
def VSubOp
    : HIVM_ElementwiseBinaryOp<
          "vsub", [SameOperandsElementType, StaticMaxRankTrait<3>,
                   OperElemTypeConstraints<[0, 1], [I8, I16, I32, F16, F32, I64]>,
                   DeclareOpInterfaceMethods<ExtraBufferOpInterface,
                     ["getExtraBufferSize"]>,
                   DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
                   DeclareOpInterfaceMethods<VectorizableOpInterface>,
                   BroadcastableOTF]>
```

源码参考：[HIVMVectorOps.td#L596-L633](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L596-L633)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vsub | I8, I16, I32, F16, F32, I64 | OperElemTypeConstraints<[0, 1], [I8, I16, I32, F16, F32, I64]> |

注意：vsub 不具有 CommutativeOpTrait（减法不满足交换律）。

#### IR 示例

```mlir
%result = hivm.hir.vsub ins(%a, %b : tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

%result = hivm.hir.vsub ins(%a, %b : tensor<64x1xf32>, tensor<1x64xf32>) outs(%dst : tensor<64x64xf32>) broadcast = [0, 1] -> tensor<64x64xf32>
```

---

### hir.vmul — 逐元乘法

#### TableGen 定义

```tablegen
def VMulOp : HIVM_ElementwiseBinaryOp<"vmul",
    [SameOperandsElementType, StaticMaxRankTrait<3>,
     OperElemTypeConstraints<[0, 1], [I16, I32, F16, F32, I64]>,
     CommutativeOpTrait, VectorOnlyTrait<0>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
     DeclareOpInterfaceMethods<VectorizableOpInterface>,
     BroadcastableOTF
    ]>
```

源码参考：[HIVMVectorOps.td#L544-L576](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L544-L576)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vmul | I16, I32, F16, F32, I64 | OperElemTypeConstraints<[0, 1], [I16, I32, F16, F32, I64]> |

注意：vmul 不支持 I8 类型（与 vadd/vsub 不同）。

#### IR 示例

```mlir
%result = hivm.hir.vmul ins(%a, %b : tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
```

---

### hir.vdiv — 逐元除法

#### TableGen 定义

```tablegen
def VDivOp : HIVM_ElementwiseBinaryOp<
                 "vdiv", [SameOperandsElementType, StaticMaxRankTrait<3>,
                          OperElemTypeConstraints<[0, 1], [F16, F32, I16, I32, I64]>,
                          DeclareOpInterfaceMethods<ExtraBufferOpInterface,
                            ["getExtraBufferSize"]>,
                          DeclareOpInterfaceMethods<VectorizableOpInterface>,
                          BroadcastableOTF]> {
  let arguments = (ins Variadic<AnyType>:$src,
                       Variadic<AnyShaped>:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedAttr<BoolAttr, "true">:$isSigned,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast
  );
}
```

源码参考：[HIVMVectorOps.td#L635-L674](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L635-L674)

#### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| $src | Variadic\<AnyType\> | 是 | 两个输入操作数 | 元素类型 F16/F32/I16/I32/I64 |
| $dst | Variadic\<AnyShaped\> | 是 | 输出向量 | 与 src 相同元素类型 |
| $temp_buffer | Optional\<AnyMemRef\> | 否 | 临时缓冲区 | - |
| $isSigned | BoolAttr (默认 true) | 否 | 是否为有符号除法 | 仅对整数类型有效 |
| $transpose | DenseI64ArrayAttr (默认 {}) | 否 | OTF 转置维度 | - |
| $broadcast | DenseI64ArrayAttr (默认 {}) | 否 | OTF 广播维度 | - |

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vdiv | F16, F32, I16, I32, I64 | OperElemTypeConstraints<[0, 1], [F16, F32, I16, I32, I64]> |

注意：vdiv 不支持 I8 类型，且仅支持 Vector-Vector 操作（无 ImplByScalarOpInterface）。

#### IR 示例

```mlir
%result = hivm.hir.vdiv ins(%a, %b : tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

hivm.hir.vdiv ins(%a, %b : memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%dst : memref<5x?x10xf32>)
```

---

### hir.vmax — 逐元最大值

#### TableGen 定义

```tablegen
def VMaxOp : HIVM_ElementwiseBinaryOp<"vmax",
    [SameOperandsElementType, StaticMaxRankTrait<3>,
     OperElemTypeConstraints<[0, 1], [I16, I32, F16, F32, I64]>,
     CommutativeOpTrait, VectorOnlyTrait<0>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
     DeclareOpInterfaceMethods<VectorizableOpInterface>,
     BroadcastableOTF
    ]>
```

源码参考：[HIVMVectorOps.td#L676-L708](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L676-L708)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vmax | I16, I32, F16, F32, I64 | OperElemTypeConstraints<[0, 1], [I16, I32, F16, F32, I64]> |

#### IR 示例

```mlir
%result = hivm.hir.vmax ins(%vec, %scalar : tensor<23x77xi32>, i32) outs(%dst : tensor<23x77xi32>) -> tensor<23x77xi32>

hivm.hir.vmax ins(%a, %b : memref<5x?x10xf32>, memref<5x?x10xf32>) outs(%dst : memref<5x?x10xf32>)
```

---

### hir.vmin — 逐元最小值

#### TableGen 定义

```tablegen
def VMinOp : HIVM_ElementwiseBinaryOp<"vmin",
    [SameOperandsElementType, StaticMaxRankTrait<3>,
     OperElemTypeConstraints<[0, 1], [I16, I32, F16, F32, I64]>,
     CommutativeOpTrait, VectorOnlyTrait<0>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
     DeclareOpInterfaceMethods<VectorizableOpInterface>,
     BroadcastableOTF
    ]>
```

源码参考：[HIVMVectorOps.td#L710-L742](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L710-L742)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vmin | I16, I32, F16, F32, I64 | OperElemTypeConstraints<[0, 1], [I16, I32, F16, F32, I64]> |

#### IR 示例

```mlir
%result = hivm.hir.vmin ins(%vec, %scalar : tensor<23x77xi32>, i32) outs(%dst : tensor<23x77xi32>) -> tensor<23x77xi32>
```

---

### hir.vor — 逐元按位或

#### TableGen 定义

```tablegen
def VOrOp : HIVM_ElementwiseBinaryOp<"vor",
    [SameOperandsElementType, StaticMaxRankTrait<3>,
     OperElemTypeConstraints<[0, 1],
       [I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32]>,
     VectorOnlyTrait<0>, VectorOnlyTrait<1>, CommutativeOpTrait,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]>
```

源码参考：[HIVMVectorOps.td#L744-L775](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L744-L775)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vor | I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32 | OperElemTypeConstraints<[0, 1], [I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32]> |

注意：vor 两个输入均为 VectorOnly，仅支持 Vector-Vector 操作。

#### IR 示例

```mlir
%result = hivm.hir.vor ins(%a, %a : tensor<?x5x10xf32>, tensor<?x5x10xf32>) outs(%dst : tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>

hivm.hir.vor ins(%a, %b : memref<5x1x10xi32>, memref<5x1x10xi32>) outs(%dst : memref<5x?x10xi32>) broadcast = [1]
```

---

### hir.vand — 逐元按位与

#### TableGen 定义

```tablegen
def VAndOp : HIVM_ElementwiseBinaryOp<"vand",
    [SameOperandsElementType, StaticMaxRankTrait<3>,
     OperElemTypeConstraints<[0, 1],
       [I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32]>,
     VectorOnlyTrait<0>, VectorOnlyTrait<1>, CommutativeOpTrait,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]>
```

源码参考：[HIVMVectorOps.td#L777-L808](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L777-L808)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vand | I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32 | OperElemTypeConstraints<[0, 1], [I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32]> |

#### IR 示例

```mlir
%result = hivm.hir.vand ins(%a, %b : tensor<23x77xi32>, tensor<23x77xi32>) outs(%dst : tensor<23x77xi32>) -> tensor<23x77xi32>

hivm.hir.vand ins(%a, %b : memref<5x1x10xi32>, memref<5x1x10xi32>) outs(%dst : memref<5x?x10xi32>) broadcast = [1]
```

---

### hir.vxor — 逐元按位异或

#### TableGen 定义

```tablegen
def VXorOp : HIVM_ElementwiseBinaryOp<"vxor",
    [SameOperandsElementType, StaticMaxRankTrait<2>,
     OperElemTypeConstraints<[0, 1],
       [I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64]>,
     VectorOnlyTrait<0>, VectorOnlyTrait<1>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface, ["getExtraBufferSize"]>
    ]>
```

源码参考：[HIVMVectorOps.td#L810-L843](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L810-L843)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vxor | I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64 | OperElemTypeConstraints<[0, 1], [I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64]> |

注意：
- vxor 不支持浮点类型（与 vor/vand 不同）
- vxor 最大 Rank 为 2（与 vor/vand 的 3 不同）
- vxor 不具有 CommutativeOpTrait
- vxor 不支持 BroadcastableOTF

#### IR 示例

```mlir
hivm.hir.vxor ins(%a, %b : memref<5x?x10xi32>, memref<5x?x10xi32>) outs(%dst : memref<5x?x10xi32>)
```

---

### hir.vmod — 逐元取模（有符号）

#### TableGen 定义

```tablegen
def VModOp : HIVM_ElementwiseBinaryOp<"vmod",
    [SameOperandsElementType, StaticMaxRankTrait<1>,
     VectorOnlyTrait<0>,
     OperElemTypeConstraints<[0, 1],
       [I16, I32, I64]>
    ]>
```

源码参考：[HIVMVectorOps.td#L845-L857](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L845-L857)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vmod | I16, I32, I64 | OperElemTypeConstraints<[0, 1], [I16, I32, I64]> |

注意：vmod 最大 Rank 仅为 1，且使用默认的 ElementwiseNaryOp 参数（无 temp_buffer, 无 OTF 广播/转置）。

#### IR 示例

```mlir
%result = hivm.hir.vmod ins(%a, %b : tensor<32xi64>, i64) outs(%dst : tensor<32xi64>) -> tensor<32xi64>
```

降级到上游：`arith.remsi`

---

### hir.vmodui — 逐元取模（无符号）

#### TableGen 定义

```tablegen
def VModUIOp : HIVM_ElementwiseBinaryOp<"vmodui",
    [SameOperandsElementType, StaticMaxRankTrait<1>,
     VectorOnlyTrait<0>,
     OperElemTypeConstraints<[0, 1],
       [I16, I32, I64]>
    ]>
```

源码参考：[HIVMVectorOps.td#L859-L871](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L859-L871)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vmodui | I16, I32, I64 | OperElemTypeConstraints<[0, 1], [I16, I32, I64]> |

#### IR 示例

```mlir
%result = hivm.hir.vmodui ins(%a, %b : tensor<32xi64>, i64) outs(%dst : tensor<32xi64>) -> tensor<32xi64>
```

降级到上游：`arith.remui`

---

### hir.vpow — 逐元幂运算

#### TableGen 定义

```tablegen
def VPowOp : HIVM_ElementwiseBinaryOp<"vpow",
    [SameOperandsElementType, StaticMaxRankTrait<1>,
     OperElemTypeConstraints<[0, 1], [I32]>,
     VectorOnlyTrait<0>, VectorOnlyTrait<1>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface>
    ]>
```

源码参考：[HIVMVectorOps.td#L981-L1010](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L981-L1010)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vpow | I32 | OperElemTypeConstraints<[0, 1], [I32]> |

注意：vpow 仅支持 I32 类型，两个输入均为 VectorOnly。

#### IR 示例

```mlir
%result = hivm.hir.vpow ins(%base, %exp : tensor<32xi32>, tensor<32xi32>) outs(%dst : tensor<32xi32>) -> tensor<32xi32>
```

## 数据类型约束汇总

| 操作 | 支持的元素类型 | 最大 Rank | 交换律 | Vec-Scalar | OTF 广播 | Extra Buffer |
|------|--------------|-----------|--------|-----------|---------|-------------|
| vadd | I8, I16, I32, F16, F32, I64 | 3 | 是 | 是 | 是 | 是 |
| vsub | I8, I16, I32, F16, F32, I64 | 3 | 否 | 是 | 是 | 是 |
| vmul | I16, I32, F16, F32, I64 | 3 | 是 | 是 | 是 | 是 |
| vdiv | F16, F32, I16, I32, I64 | 3 | 否 | 否 | 是 | 是 |
| vmax | I16, I32, F16, F32, I64 | 3 | 是 | 是 | 是 | 是 |
| vmin | I16, I32, F16, F32, I64 | 3 | 是 | 是 | 是 | 是 |
| vor | I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32 | 3 | 是 | 否 | 是 | 是 |
| vand | I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32 | 3 | 是 | 否 | 是 | 是 |
| vxor | I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64 | 2 | 否 | 否 | 否 | 是 |
| vmod | I16, I32, I64 | 1 | 否 | 否 | 否 | 否 |
| vmodui | I16, I32, I64 | 1 | 否 | 否 | 否 | 否 |
| vpow | I32 | 1 | 否 | 否 | 否 | 是 |

## IR 层约束与验证

1. **元素类型一致性**：所有二元操作要求输入和输出具有相同的元素类型（`SameOperandsElementType`），vcmp 除外
2. **Rank 一致性**：输入和输出必须具有相同的 rank
3. **VectorOnly 约束**：标记为 VectorOnly 的操作数必须为向量类型
4. **ScalarOnly 约束**：标记为 ScalarOnly 的操作数必须为标量类型（如 vshl 的第二个操作数）
5. **vdiv 的 isSigned 属性**：对整数除法，isSigned 决定使用有符号还是无符号除法
6. **CommutativeOpTrait**：具有交换律的操作允许编译器交换输入操作数以优化性能

## 与其他 IR 操作的关系

| HIVM 操作 | 上游 linalg 降级 | 说明 |
|-----------|-----------------|------|
| vadd | linalg.add | - |
| vsub | linalg.sub | - |
| vmul | linalg.mul | - |
| vdiv | linalg.div | - |
| vmax | linalg.max | - |
| vmin | linalg.min | - |
| vor | linalg.map { arith.ori } | 浮点类型需要 bitcast |
| vand | linalg.map { arith.andi } | 浮点类型需要 bitcast |
| vxor | linalg.map { arith.xori } | - |
| vmod | arith.remsi | 有符号取模 |
| vmodui | arith.remui | 无符号取模 |

## 常见问题

**Q: vdiv 的 isSigned 默认值是什么？**
A: 默认为 `true`，即有符号除法。对于无符号整数除法，需要显式设置 `isSigned = false`。

**Q: 为什么 vor/vand 支持浮点类型而 vxor 不支持？**
A: vor/vand 对浮点类型执行的是位级别的或/与操作（先 bitcast 到整数类型，执行位运算，再 bitcast 回浮点），这在某些掩码操作中有用。vxor 的硬件实现不支持浮点位操作。

**Q: vmod 和 vmodui 的区别是什么？**
A: vmod 执行有符号取模（对应 C 语言的 `%` 运算符，降级为 `arith.remsi`），vmodui 执行无符号取模（对应 `arith.remui`）。

**Q: vpow 为什么只支持 I32？**
A: 这是硬件约束。vpow 的硬件实现仅针对 I32 类型设计，浮点幂运算需要通过其他方式实现。

## 相关文档

- Python API：docs_triton_ascend 中的算术运算符文档
- 源码参考：
  - [HIVMVectorOps.td - Binary Ops](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L506-L1010)
  - [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
