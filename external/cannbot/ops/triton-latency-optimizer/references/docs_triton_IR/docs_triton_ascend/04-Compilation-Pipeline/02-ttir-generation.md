# AST → TTIR 生成

## 概述

AST → TTIR 生成是 Triton-Ascend 编译流水线的第一个阶段，负责将 Python 编写的 Triton Kernel（经过 `@triton.jit` 装饰的函数）转换为 Triton IR（TTIR）。该过程由 [code_generator.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/compiler/code_generator.py) 中的 `CodeGenerator` 类实现，通过遍历 Python AST 节点，逐一生成对应的 Triton Dialect MLIR 操作。

## 关键概念

| 概念 | 说明 |
|------|------|
| `CodeGenerator` | AST 访问器，将 Python AST 节点转换为 Triton IR 操作 |
| `ast_to_ttir()` | 顶层入口函数，创建 CodeGenerator 并驱动 AST 遍历 |
| `JITFunction` | Triton JIT 编译函数，包含原始函数、签名、常量等信息 |
| `constexpr` | 编译期常量，在 IR 生成阶段直接求值，不生成 IR 操作 |
| `gscope` | 全局作用域，包含 Kernel 函数外部的 Python 变量 |
| `lscope` | 局部作用域，包含当前函数内的变量定义 |
| `local_defs` | 局部定义映射，记录 SSA 名称到 tensor 的对应关系 |
| `builder` | IR 构建器，提供创建 MLIR 操作的接口 |
| `ascend_builder` | Ascend NPU 专用 IR 构建器，提供 Ascend 特定操作的创建接口 |
| `buffer_builder` | Buffer IR 构建器，提供 Buffer 相关操作的创建接口 |
| `WITH_DISPATCH` | `with` 语句处理器注册表，用于分发 Ascend 扩展的 `with` 语句 |

## code_generator.py 的工作原理

### 整体架构

```
Python Kernel (@triton.jit)
    │
    │  fn.parse() → Python AST
    │
    ▼
CodeGenerator (ast.NodeVisitor)
    │
    │  ├── builder (ir.builder): 标准 Triton IR 构建器
    │  ├── ascend_builder (ascend_ir.ascendnpu_ir_builder): Ascend NPU 构建器
    │  ├── buffer_builder (buffer_ir.buffer_builder): Buffer 构建器
    │  │
    │  │  setup_unified_builder() 将 ascend_builder 方法注入 builder
    │  │  setup_unified_builder_with_buffer_builder() 将 buffer_builder 方法注入 builder
    │  │
    │  ├── gscope: 全局作用域
    │  ├── lscope: 局部作用域
    │  ├── local_defs: SSA 定义
    │  ├── constants: 编译期常量
    │  └── scf_stack: 结构化控制流栈
    │
    ▼
TTIR Module (mlir.Module)
```

### 入口函数 ast_to_ttir()

[ast_to_ttir()](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/compiler/code_generator.py#L1354) 是 TTIR 生成的顶层入口：

```python
def ast_to_ttir(fn, specialization, context, options, codegen_fns, module_map, module=None):
    attrs = specialization.attrs
    constants = {cst_key(key): value for key, value in specialization.constants.items()}
    gscope = fn.__globals__.copy()
    function_name = fn.repr(specialization)
    # ... 构建 prototype、创建 CodeGenerator ...
    generator = CodeGenerator(context, prototype, gscope=gscope, constants=all_constants,
                              function_name=function_name, jit_fn=fn, attributes=fn_attrs,
                              is_kernel=True, file_name=file_name, begin_line=begin_line,
                              options=options, codegen_fns=codegen_fns, module_map=module_map,
                              module=module)
    generator.visit(fn.parse())
    ret = generator.module
    ret.context = context
    return ret
```

### CodeGenerator 初始化

[CodeGenerator.__init__()](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/compiler/code_generator.py#L210) 完成以下初始化：

1. **创建 IR 构建器**：根据 `compile_mode` 选择 `simt` 或 `simd` 模式
2. **设置统一构建器**：将 `ascend_builder` 和 `buffer_builder` 的方法注入标准 `builder`
3. **初始化作用域**：设置全局作用域 `gscope` 和局部作用域 `lscope`
4. **创建 Module**：通过 `builder.create_module()` 创建 MLIR Module

```python
def __init__(self, context, prototype, gscope, attributes, constants, ...):
    if hasattr(options, "force_simt_only") and options.force_simt_only:
        self.builder = ir.builder(context, compile_mode="simt")
    else:
        self.builder = ir.builder(context, compile_mode="simd")

    self.ascend_builder = ascend_ir.ascendnpu_ir_builder(context, getattr(options, "arch", ""))
    setup_unified_builder(self.builder, self.ascend_builder)
    self.buffer_builder = buffer_ir.buffer_builder(context)
    setup_unified_builder_with_buffer_builder(self.builder, self.buffer_builder)
```

## Python AST 到 Triton IR 的映射规则

### 语句映射

| Python AST 节点 | Triton IR 操作 | 说明 |
|-----------------|---------------|------|
| `FunctionDef` | `tt.func` | Kernel 函数定义 |
| `Assign` | `tt.store` / SSA 赋值 | 变量赋值 |
| `AnnAssign` | constexpr 或 SSA 赋值 | 带类型注解的赋值 |
| `AugAssign` | 展开为 `Assign` + `BinOp` | 增量赋值 |
| `Return` | `tt.return` | 函数返回 |
| `If` | `scf.if` / 条件编译 | 条件分支 |
| `For` | `scf.for` / `tt.static_range` | 循环 |
| `While` | `scf.while` | While 循环 |
| `With` | WITH_DISPATCH 分发 | With 语句 |
| `Assert` | `tt.assert` | 断言 |

### 表达式映射

| Python AST 节点 | Triton IR 操作 | 说明 |
|-----------------|---------------|------|
| `Call` | 对应 Triton builtin 函数 | 函数调用 |
| `BinOp` | `arith.addf`/`arith.mulf` 等 | 二元运算 |
| `Compare` | `arith.cmpi`/`arith.cmpf` | 比较运算 |
| `UnaryOp` | `arith.negf` 等 | 一元运算 |
| `Subscript` | `tt.load` / tensor 切片 | 下标访问 |
| `Attribute` | 属性访问 / `.T` 转置 | 属性访问 |
| `Constant` | `constexpr` | 常量 |
| `Name` | SSA 值查找 | 变量引用 |

### 二元运算映射

| Python 操作 | Triton IR 方法 | AST 操作符 |
|-------------|---------------|-----------|
| `+` | `__add__` | `ast.Add` |
| `-` | `__sub__` | `ast.Sub` |
| `*` | `__mul__` | `ast.Mult` |
| `/` | `__truediv__` | `ast.Div` |
| `//` | `__floordiv__` | `ast.FloorDiv` |
| `%` | `__mod__` | `ast.Mod` |
| `**` | `__pow__` | `ast.Pow` |
| `<<` | `__lshift__` | `ast.LShift` |
| `>>` | `__rshift__` | `ast.RShift` |
| `&` | `__and__` | `ast.BitAnd` |
| `\|` | `__or__` | `ast.BitOr` |
| `^` | `__xor__` | `ast.BitXor` |

## builtin 装饰器的处理

### builtin_namespace

CodeGenerator 定义了 [builtin_namespace](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/compiler/code_generator.py#L273)，包含 Kernel 中可用的 Python 内置函数：

```python
builtin_namespace: Dict[str, Any] = {
    _.__name__: _ for _ in (len, list, range, float, int, isinstance, getattr)
}
builtin_namespace.update((
    ('print', language.core.device_print),
    ('min', language.minimum),
    ('max', language.maximum),
))
```

### 静态实现函数

某些函数在编译期直接求值，不生成 IR 操作：

| 函数 | 处理方式 | 说明 |
|------|----------|------|
| `tl.static_assert` | `execute_static_assert` | 编译期断言 |
| `tl.static_print` | `static_executor(print)` | 编译期打印 |
| `int` | `static_executor(int)` | 编译期类型转换 |
| `len` | `static_executor(len)` | 编译期求长度 |
| `extension.int64` | `static_executor(extension.int64)` | 编译期 int64 转换 |

### JITFunction 调用

当调用 `@triton.jit` 装饰的函数时，[call_JitFunction()](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/compiler/code_generator.py#L1121) 会：

1. 解析函数参数，将 constexpr 参数分离
2. 生成函数名（含类型和常量的 mangle）
3. 如果函数未在 Module 中定义，递归创建 `CodeGenerator` 生成函数定义
4. 生成 `call` 操作调用函数

## constexpr 的处理

### constexpr 语义

`constexpr` 是 Triton 的编译期常量机制。标记为 `constexpr` 的参数在 IR 生成阶段直接求值为 Python 值，不生成任何 IR 操作。

### constexpr 的来源

1. **函数参数注解**：`x: tl.constexpr = 42`
2. **全局变量注解**：`GLOBAL: tl.constexpr = 42`
3. **显式构造**：`tl.constexpr(42)`

### constexpr 处理流程

```
Python 代码                    IR 生成行为
─────────────────────────────────────────────
x: tl.constexpr = 42    →     不生成 IR，x 直接为 Python int 42
y = x + 1               →     编译期求值，y = 43
z = tl.load(ptr)        →     生成 tt.load IR 操作
w = z * x               →     x 为 constexpr，生成 arith.mulf(z, 42)
```

### 全局 constexpr 访问

全局变量默认不允许在 Kernel 中访问，除非满足以下条件之一：

- 变量被注解为 `constexpr`
- 变量是模块导入
- 变量是 `JITFunction`
- 变量是 Triton builtin
- 变量是 `triton.dtype`
- 设置了 `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1`

## Ascend 扩展模块的注入

### extra.cann.extension

Triton-Ascend 通过 [triton.language.extra.cann.extension](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/compiler/code_generator.py#L11) 模块注入 Ascend 特定的语言扩展：

```python
import triton.language.extra.cann.extension as extension
```

### 扩展功能

| 扩展 | 说明 |
|------|------|
| `extension.parallel` | Ascend 并行迭代器，支持 `bind_sub_block` 等参数 |
| `extension.int64` | int64 类型转换 |
| `extension.is_builtin(fn)` | 判断函数是否为 Ascend builtin 扩展操作 |

### WITH_DISPATCH 注册

Ascend 扩展的 `with` 语句处理器通过 [WITH_DISPATCH](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/compiler/code_generator.py#L26) 注册：

```python
from triton.language.extra.cann.extension.dispatch import ASCEND_WITH_DISPATCH
WITH_DISPATCH.update(ASCEND_WITH_DISPATCH)
```

### Ascend Builder 注入

在 `visit_Call()` 中，当调用 Ascend builtin 扩展函数时，使用 `ascend_builder` 而非标准 `builder`：

```python
def visit_Call(self, node):
    fn = _unwrap_if_constexpr(self.visit(node.func))
    # ...
    if (hasattr(fn, '__self__') and _is_triton_value(fn.__self__)) or language.core.is_builtin(fn):
        _builder = self.ascend_builder if extension.is_builtin(fn) else self.builder
        # 使用 _builder 生成 IR
```

### extension.parallel 迭代器

`extension.parallel` 是 Ascend 特有的迭代器，在 `visit_For()` 中处理：

```python
if IteratorClass in [language.range, extension.parallel]:
    iterator = IteratorClass(*iter_args, **iter_kwargs)
    # ...
    if (IteratorClass is extension.parallel):
        bind_sub_block = iterator.bind_sub_block
        for_op.set_attr("hivm.parallel_loop", self.builder.get_unit_attr())
```

`extension.parallel` 支持的额外参数：

| 参数 | 说明 |
|------|------|
| `num_stages` | 流水线阶段数 |
| `loop_unroll_factor` | 循环展开因子 |
| `disallow_acc_multi_buffer` | 禁止累加器多缓冲 |
| `flatten` | 循环展平 |
| `warp_specialize` | Warp 特化 |
| `disable_licm` | 禁用循环不变量外提 |
| `bind_sub_block` | 绑定子块 |

## TTIR 示例

### 简单向量加法 Kernel

Python 代码：

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

生成的 TTIR：

```mlir
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                              %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                              %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                              %arg3: i32 {tt.divisibility = 16 : i32}) {
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
    %11 = tt.addptr %arg2, %4 : !tt.ptr<f32>, tensor<1024xi32>
    tt.store %11, %10, %5 : !tt.ptr<f32>
  }
}
```

### 包含 constexpr 的 Kernel

Python 代码：

```python
@triton.jit
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr = 1024):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
```

生成的 TTIR 中 `BLOCK_SIZE` 已被替换为常量值：

```mlir
module {
  tt.func public @kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %1 = tt.addptr %arg0, %0 : !tt.ptr<f32>, tensor<1024xi32>
    %2 = tt.load %1 : tensor<1024xf32>
  }
}
```

### 包含条件分支的 Kernel

Python 代码：

```python
@triton.jit
def kernel(x_ptr, flag, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    if flag:
        x = x * 2.0
    tl.store(x_ptr + offsets, x)
```

生成的 TTIR（使用 scf.if）：

```mlir
%cond = ... : i1
%result = scf.if %cond -> tensor<1024xf32> {
    %doubled = arith.mulf %x, %cst : tensor<1024xf32>
    scf.yield %doubled : tensor<1024xf32>
} else {
    scf.yield %x : tensor<1024xf32>
}
```

## NPU 适配要点

1. **compile_mode 传递**：`CodeGenerator` 根据 `force_simt_only` 选项选择 `simt` 或 `simd` 编译模式，影响 builder 的行为
2. **Ascend Builder 注入**：通过 `setup_unified_builder()` 将 Ascend 特定操作注入标准 builder，使 Kernel 代码可以透明地使用 Ascend 扩展
3. **extension.parallel**：Ascend 特有的并行迭代器，支持 `bind_sub_block` 等参数，在 `for` 循环中设置 `hivm.parallel_loop` 属性
4. **WITH_DISPATCH**：Ascend 扩展的 `with` 语句处理器，用于处理 `al.sync_block_*` 等同步操作
5. **buffer_builder**：Buffer IR 构建器，支持 `buffer_type` 等Ascend 特有类型

## 常见问题

### Q: 为什么全局变量不能在 Kernel 中访问？

Triton Kernel 在设备上执行，无法访问宿主机的 Python 全局变量。只有 `constexpr` 全局变量可以在编译期求值后嵌入 IR。如果确实需要访问非 constexpr 全局变量，可以设置 `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1`，但这不保证长期支持。

### Q: constexpr 和普通参数有何区别？

`constexpr` 参数在编译期求值，不生成 IR 操作。普通参数作为运行时值传递给 Kernel。`constexpr` 适用于 BLOCK_SIZE 等编译期已知的配置参数。

### Q: 如何添加 Ascend 特有的 IR 操作？

1. 在 `triton.language.extra.cann.extension` 中定义 Python 接口
2. 在 C++ 端实现对应的 Dialect 和 Operation
3. 在 `ascend_builder` 中注册操作创建方法
4. 通过 `setup_unified_builder()` 注入标准 builder

### Q: scf.if 和条件编译有何区别？

当 `if` 条件为 `constexpr` 或 Python 布尔值时，执行条件编译（只生成一个分支的 IR）。当条件为 Triton tensor 时，生成 `scf.if` 操作（两个分支都生成 IR）。

## 相关文档

- [01-pipeline-overview.md](01-pipeline-overview.md) - 编译流水线总览
- [03-ttir-optimization.md](03-ttir-optimization.md) - TTIR 优化 Pass 详解
- [04-ascend-passes.md](04-ascend-passes.md) - Ascend 特有 Pass 详解
