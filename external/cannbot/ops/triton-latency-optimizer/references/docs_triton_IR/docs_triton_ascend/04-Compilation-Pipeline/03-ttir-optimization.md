# TTIR 优化 Pass

## 概述

TTIR 优化阶段在 AST → TTIR 生成之后执行，对 Triton IR 进行一系列与硬件无关的通用优化。这些优化 Pass 由 [make_ttir()](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L72) 函数驱动，使用 MLIR 的 Pass Manager 框架按顺序执行。

TTIR 优化 Pass 与后端无关，所有 Triton 后端（CUDA、Ascend 等）共享相同的优化序列。

## 关键概念

| 概念 | 说明 |
|------|------|
| Pass Manager | MLIR 的 Pass 管理器，按顺序执行注册的 Pass |
| InlinerPass | 函数内联 Pass，将被调用函数体嵌入调用点 |
| CombinePass | 操作合并 Pass，将多个操作合并为更高效的形式 |
| CanonicalizerPass | 规范化 Pass，将 IR 转换为规范形式 |
| CSEPass | 公共子表达式消除 Pass |
| LICMPass | 循环不变量外提 Pass |
| LoopUnrollPass | 循环展开 Pass |
| ReorderBroadcastPass | 广播重排序 Pass |
| SymbolDCEPass | 死符号消除 Pass |

## make_ttir() 中的优化 Pass 序列

[make_ttir()](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L72) 函数定义了 TTIR 优化 Pass 的执行序列：

```python
def make_ttir(mod, metadata, opt):
    if "hash" not in metadata:
        metadata["hash"] = hashlib.sha256(f"{mod}-{metadata}".encode()).hexdigest()
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.ttir.add_combine(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_licm(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    pm.run(mod)
    if opt.debug:
        dump_manager = get_dump_manager(metadata["hash"])
        print(f"Dumping intermediate results to {dump_manager.cache_dir}")
        dump_manager.put(str(mod), "kernel.ttir.mlir", binary=False)
    return mod
```

### Pass 执行顺序

| 序号 | Pass | 来源 | 说明 |
|------|------|------|------|
| 1 | InlinerPass | `passes.common` | 函数内联 |
| 2 | CombinePass | `passes.ttir` | Triton 操作合并 |
| 3 | CanonicalizerPass | `passes.common` | IR 规范化 |
| 4 | ReorderBroadcastPass | `passes.ttir` | 广播操作重排序 |
| 5 | CSEPass | `passes.common` | 公共子表达式消除 |
| 6 | LICMPass | `passes.common` | 循环不变量外提 |
| 7 | SymbolDCEPass | `passes.common` | 死符号消除 |
| 8 | LoopUnrollPass | `passes.ttir` | 循环展开 |

## Pass 详解

### 1. InlinerPass

#### 功能

将 `@triton.jit` 装饰的非 kernel 函数内联到调用点，消除函数调用开销，并为后续优化提供更大的优化空间。

#### 输入/输出

| 项目 | 说明 |
|------|------|
| 输入 | 包含 `tt.call` 操作的 TTIR Module |
| 输出 | `tt.call` 被替换为函数体的 TTIR Module |

#### 适用场景

- Kernel 调用辅助函数时
- 多个 Kernel 共享公共逻辑时

#### IR 转换示例

```mlir
// Before: 函数调用
tt.func @helper(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = arith.mulf %arg0, %arg0 : tensor<1024xf32>
  tt.return %0 : tensor<1024xf32>
}
tt.func @kernel(%arg0: !tt.ptr<f32>) {
  %1 = tt.load %arg0 : tensor<1024xf32>
  %2 = tt.call @helper(%1) : (tensor<1024xf32>) -> tensor<1024xf32>
  tt.store %arg0, %2 : !tt.ptr<f32>
}

// After: 内联
tt.func @kernel(%arg0: !tt.ptr<f32>) {
  %1 = tt.load %arg0 : tensor<1024xf32>
  %2 = arith.mulf %1, %1 : tensor<1024xf32>
  tt.store %arg0, %2 : !tt.ptr<f32>
}
```

#### 注意事项

- 标记为 `noinline` 的函数不会被内联
- `noinline` 函数的参数必须是 scalar 或 constexpr

---

### 2. CombinePass

#### 功能

合并 Triton 特定的操作模式，将多个操作组合为更高效的单一操作。

#### 输入/输出

| 项目 | 说明 |
|------|------|
| 输入 | 包含可合并操作模式的 TTIR Module |
| 输出 | 合并后的 TTIR Module |

#### 适用场景

- 连续的类型转换操作
- 冗余的 broadcast + reshape 模式
- 可合并的算术操作链

#### IR 转换示例

```mlir
// Before: 冗余的类型转换链
%0 = arith.extf %x : tensor<1024xf16> to tensor<1024xf32>
%1 = arith.truncf %0 : tensor<1024xf32> to tensor<1024xf16>

// After: 消除冗余转换
// %x 直接使用
```

---

### 3. CanonicalizerPass

#### 功能

将 IR 转换为规范形式，包括常量折叠、死代码消除、操作简化等。这是 MLIR 的标准规范化 Pass。

#### 输入/输出

| 项目 | 说明 |
|------|------|
| 输入 | 任意 TTIR Module |
| 输出 | 规范化后的 TTIR Module |

#### 适用场景

- 所有编译场景（通用优化）
- 在其他 Pass 之后清理 IR

#### 规范化规则示例

| 模式 | 规范化结果 |
|------|-----------|
| `arith.addi %x, %c0` | `%x`（加零消除） |
| `arith.muli %x, %c1` | `%x`（乘一消除） |
| `arith.muli %x, %c0` | `%c0`（乘零替换） |
| 常量表达式 `arith.addi %c1, %c2` | 折叠为常量 |

---

### 4. ReorderBroadcastPass

#### 功能

重排广播操作的位置，将 broadcast 操作尽量移动到计算操作之前，减少广播后参与计算的数据量。

#### 输入/输出

| 项目 | 说明 |
|------|------|
| 输入 | 包含 `tt.broadcast` 操作的 TTIR Module |
| 输出 | 广播位置优化后的 TTIR Module |

#### 适用场景

- 标量与张量运算时，标量先广播再计算
- 多维广播操作

#### IR 转换示例

```mlir
// Before: 先广播再计算
%scalar = arith.constant 1.0 : f32
%broadcast_scalar = tt.splat %scalar : f32 -> tensor<1024xf32>
%load = tt.load %ptr : tensor<1024xf32>
%result = arith.addf %broadcast_scalar, %load : tensor<1024xf32>

// After: 计算后广播（如果可能，减少广播数据量）
// 具体变换取决于操作语义和广播维度
```

#### 优化原理

Broadcast 操作会扩展张量的维度，增加后续计算的数据量。将 broadcast 推迟到真正需要的位置，或者将标量操作直接应用于未广播的张量，可以减少计算量。

---

### 5. CSEPass (Common Subexpression Elimination)

#### 功能

消除公共子表达式，当多个操作计算相同的结果时，只保留一个操作，其他使用该结果的地方直接引用。

#### 输入/输出

| 项目 | 说明 |
|------|------|
| 输入 | 包含重复计算的 TTIR Module |
| 输出 | 消除重复计算后的 TTIR Module |

#### 适用场景

- 多个 load 相同地址
- 重复的算术计算
- 重复的类型转换

#### IR 转换示例

```mlir
// Before: 重复计算
%0 = arith.addf %a, %b : tensor<1024xf32>
%1 = arith.addf %a, %b : tensor<1024xf32>
%2 = arith.mulf %0, %c : tensor<1024xf32>
%3 = arith.mulf %1, %d : tensor<1024xf32>

// After: CSE 消除
%0 = arith.addf %a, %b : tensor<1024xf32>
%2 = arith.mulf %0, %c : tensor<1024xf32>
%3 = arith.mulf %0, %d : tensor<1024xf32>
```

---

### 6. LICMPass (Loop Invariant Code Motion)

#### 功能

将循环不变的计算移出循环体，避免在每次迭代中重复计算。

#### 输入/输出

| 项目 | 说明 |
|------|------|
| 输入 | 包含 `scf.for`/`scf.while` 的 TTIR Module |
| 输出 | 循环不变量外提后的 TTIR Module |

#### 适用场景

- 循环体内有不变的计算
- 常量加载在循环内
- 循环独立的地址计算

#### IR 转换示例

```mlir
// Before: 常量在循环内
scf.for %i = %lb to %ub step %step {
  %cst = arith.constant dense<1.0> : tensor<1024xf32>
  %load = tt.load %ptr[%i] : tensor<1024xf32>
  %result = arith.addf %load, %cst : tensor<1024xf32>
  tt.store %out[%i], %result
}

// After: 常量外提
%cst = arith.constant dense<1.0> : tensor<1024xf32>
scf.for %i = %lb to %ub step %step {
  %load = tt.load %ptr[%i] : tensor<1024xf32>
  %result = arith.addf %load, %cst : tensor<1024xf32>
  tt.store %out[%i], %result
}
```

---

### 7. SymbolDCEPass (Symbol Dead Code Elimination)

#### 功能

消除不可达的符号（函数、全局变量等），减少 Module 的大小。

#### 输入/输出

| 项目 | 说明 |
|------|------|
| 输入 | 包含未使用符号的 TTIR Module |
| 输出 | 清理后的 TTIR Module |

#### 适用场景

- 内联后遗留的未使用函数
- 死代码消除后遗留的符号

---

### 8. LoopUnrollPass

#### 功能

将循环展开，减少循环控制开销，并为后续优化（如指令调度）提供更大的优化空间。

#### 输入/输出

| 项目 | 说明 |
|------|------|
| 输入 | 包含 `scf.for` 的 TTIR Module |
| 输出 | 部分或完全展开的 TTIR Module |

#### 适用场景

- 小循环（迭代次数少且已知）
- 带有 `tt.loop_unroll_factor` 属性的循环

#### IR 转换示例

```mlir
// Before: 循环
scf.for %i = %c0 to %c4 step %c1 {
  %load = tt.load %ptr[%i] : f32
  tt.store %out[%i], %load
}

// After: 完全展开
%load0 = tt.load %ptr[%c0] : f32
tt.store %out[%c0], %load0
%load1 = tt.load %ptr[%c1] : f32
tt.store %out[%c1], %load1
%load2 = tt.load %ptr[%c2] : f32
tt.store %out[%c2], %load2
%load3 = tt.load %ptr[%c3] : f32
tt.store %out[%c3], %load3
```

#### 循环展开属性

循环展开行为可以通过 `for` 操作的属性控制：

| 属性 | 类型 | 说明 |
|------|------|------|
| `tt.loop_unroll_factor` | i32 | 循环展开因子 |
| `tt.num_stages` | i32 | 流水线阶段数 |
| `tt.disallow_acc_multi_buffer` | unit | 禁止累加器多缓冲 |
| `tt.flatten` | unit | 循环展平 |
| `tt.warp_specialize` | unit | Warp 特化 |
| `tt.disable_licm` | unit | 禁用 LICM |

## TTIR 优化前后对比示例

### 示例：向量加法 + 标量乘法

优化前（AST 直接生成的 TTIR）：

```mlir
tt.func @kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>,
                %arg3: i32, %arg4: f32) {
  %0 = tt.get_program_id x : i32
  %1 = arith.constant dense<1024> : tensor<1024xi32>
  %2 = arith.muli %0, %1 : i32
  %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %4 = arith.addi %2, %3 : tensor<1024xi32>
  %5 = arith.cmpi slt, %4, %arg3 : tensor<1024xi1>
  %6 = tt.addptr %arg0, %4 : !tt.ptr<f32>, tensor<1024xi32>
  %7 = tt.load %6, %5 : tensor<1024xf32>
  %8 = tt.addptr %arg1, %4 : !tt.ptr<f32>, tensor<1024xi32>
  %9 = tt.load %8, %5 : tensor<1024xf32>
  %10 = arith.addf %7, %9 : tensor<1024xf32>
  %11 = tt.splat %arg4 : tensor<1024xf32>
  %12 = arith.mulf %10, %11 : tensor<1024xf32>
  %13 = tt.addptr %arg2, %4 : !tt.ptr<f32>, tensor<1024xi32>
  tt.store %13, %12, %5
}
```

优化后（经过 make_ttir 优化）：

```mlir
tt.func @kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>,
                %arg3: i32, %arg4: f32) {
  %0 = tt.get_program_id x : i32
  %1 = arith.constant dense<1024> : tensor<1024xi32>
  %2 = arith.muli %0, %1 : i32
  %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %4 = arith.addi %2, %3 : tensor<1024xi32>
  %5 = arith.cmpi slt, %4, %arg3 : tensor<1024xi1>
  %6 = tt.addptr %arg0, %4 : !tt.ptr<f32>, tensor<1024xi32>
  %7 = tt.load %6, %5 : tensor<1024xf32>
  %8 = tt.addptr %arg1, %4 : !tt.ptr<f32>, tensor<1024xi32>
  %9 = tt.load %8, %5 : tensor<1024xf32>
  %10 = arith.addf %7, %9 : tensor<1024xf32>
  %11 = arith.mulf %10, %arg4 : tensor<1024xf32>
  %12 = tt.addptr %arg2, %4 : !tt.ptr<f32>, tensor<1024xi32>
  tt.store %12, %11, %5
}
```

主要优化点：
- `tt.splat %arg4` 被消除，标量直接参与运算（CombinePass/CanonicalizerPass）
- 常量折叠和规范化

## NPU 适配要点

1. **Pass 序列固定**：TTIR 优化 Pass 序列与 CUDA 后端共享，不包含 Ascend 特定优化
2. **Ascend 特定优化在后续阶段**：Ascend 特有的 Pass（如 TritonToStructured、TritonAffinityOpt 等）在 `ttir_to_linalg()` 阶段执行
3. **调试输出**：当 `opt.debug=True` 时，优化后的 TTIR 会被转储到 `kernel.ttir.mlir` 文件
4. **Pass Manager 调试**：`pm.enable_debug()` 启用 Pass Manager 的调试模式

## 常见问题

### Q: 为什么 TTIR 优化不包含 Ascend 特定优化？

TTIR 优化阶段是与后端无关的通用优化，所有 Triton 后端共享。Ascend 特定优化在 `ttir_to_linalg()` 阶段的 Ascend Passes 中执行。

### Q: 如何查看优化后的 TTIR？

设置 `TRITON_KERNEL_DUMP=1` 或在编译选项中设置 `debug=True`，优化后的 TTIR 会保存到缓存目录中的 `kernel.ttir.mlir` 文件。

### Q: LoopUnrollPass 会展开所有循环吗？

不会。LoopUnrollPass 只展开满足条件的循环（如迭代次数小且已知、带有 `tt.loop_unroll_factor` 属性的循环）。大循环不会被自动展开。

### Q: CSEPass 和 SymbolDCEPass 有何区别？

CSEPass 消除重复的计算操作（操作级别的去重），SymbolDCEPass 消除不可达的符号（函数/全局变量级别的清理）。两者互补。

## 相关文档

- [01-pipeline-overview.md](01-pipeline-overview.md) - 编译流水线总览
- [02-ttir-generation.md](02-ttir-generation.md) - AST → TTIR 生成
- [04-ascend-passes.md](04-ascend-passes.md) - Ascend 特有 Pass 详解
