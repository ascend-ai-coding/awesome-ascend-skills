# PyPTO DFX 问题查找表

> **使用顺序**：先查 [经验表](experience-table.md)（已验证的修复配方），经验表未命中再查本表。
>
> 本表提供 **错误症状 → 官方文档/组件文档的具体章节** 的快速路由。
>
> **文档站 URL 前缀**：`https://pypto.gitcode.com/trouble_shooting/`

---

## 〇、关键字索引

> 按关键字首字母排序。每个关键字后标注目标文档，多个关键字可指向同一文档。

| 关键字 | 目标文档 | 章节 |
|---|---|---|
| `-> None` | `function.md` | INVALID_VAL |
| `16 element` / `32 element` / `alignment` / `对齐` | `pass.md` / `vector.md` / `matmul.md` | Tile Shape 超限 / 对齐约束 |
| `actualRawMagic` / `Shape size mismatch` / `Data size mismatch` | `machine.md` | encode 阶段 actualRawMagic 断言触发 |
| `aicore error` / `AIC ERROR` | `machine.md` | AIC ERROR |
| `Cannot convert symbols to int` | `function.md` | DYNAMIC_SHAPE_COMPUTE |
| `cce` / `.cce` 文件 | `machine.md` | AIC ERROR |
| `COMPILE_CODE_FAILED` / `F63001` / `maybe need a type` / `does not support the given target feature` | `codegen.md` | F63001（场景 1-4） |
| `core dump` / `aborted` / `segfault` / `segmentation fault` | `machine.md` | Host侧捕获异常 |
| `cube tile` / `set_cube_tile_shapes` / `matmul` / `matmul tile` | `matmul.md` | ERR_CONFIG_TILE |
| `DYNAMIC_SHAPE_COMPUTE` / `0x21009U` | `function.md` | DYNAMIC_SHAPE_COMPUTE |
| `OP_DEPENDENCY_CYCLE` / `0x2100AU` / `F2100A` / `cycle detected` / `拓扑排序失败` | `function.md` | OP_DEPENDENCY_CYCLE |
| `DYNAMIC` / `dynamic shape` / `动态 shape` | `function.md` | DYNAMIC_SHAPE_COMPUTE |
| `F0XXXX` | — | 外部写法问题（无组件文档） |
| `F21004` / `INVALID_VAL` / `0x21004U` | `function.md` | INVALID_VAL |
| `F40000`-`F40006` / `TENSOR_*` / `TENSOR_DYNAMIC_ATTR` | `pass.md` | Tensor相关错误 |
| `F41000`-`F41006` / `OP_*` | `pass.md` | Operation相关错误 |
| `F42000`-`F42006` / `FUNCTION_*` | `pass.md` | Function相关错误 |
| `F43000`-`F43007` / `GRAPH_*` | `pass.md` | Graph相关错误 |
| `F44000`-`F44003` / `CONFIG_*` | `pass.md` | Config相关错误 |
| `F62014` / `SYMBOL_NOT_FOUND` | `codegen.md` | F62014 |
| `F63001` / `COMPILE_CODE_FAILED` | `codegen.md` | F63001 |
| `F70006` / `HANDSHAKE_TIMEOUT` | `machine.md` | F70006 |
| `FC0000` / `FC0001` / `FC1000` / `FC1001` / `FC2000` / `FC2001` | `vector.md` | 对应错误码章节 |
| `FC3000`-`FC3002` / `FC4000`-`FC4002` / `FC5000`-`FC5002` | `matmul.md` | 对应错误码章节 |
| `FFFFFF` / `FFFFF` / `UNKNOWN` / `0x3FFFF` / `不透明错误` / `opaque error` | — | 先拉 slog（见 §四），再按错误码查表 |
| `from __future__ import annotations` / `string annotations` | `function.md` | INVALID_VAL (JIT 签名) |
| `INT32_MAX` / `INT32_MAX overflow` | `function.md` | DYNAMIC_SHAPE_COMPUTE |
| `Invalid value type` | `function.md` | INVALID_VAL |
| `kernel code` / `kernel 代码` / `TileOP` | `codegen.md` | 场景举例 |
| `L0A` / `L0B` / `L0C` / `L0 size exceeded` | `pass.md` | 常见前端问题修复建议 |
| `L1 size exceeded` | `pass.md` | 常见前端问题修复建议 |
| `launch aicpu failed` / `107003` | `machine.md` | AIC ERROR |
| `memory overlap` / `内存重叠` / `内存踩踏` | `machine.md` | 怀疑和MACHINE内存处理有关的精度问题 |
| `Non-tensor parameter` / `must not be a torch.Tensor` | `function.md` | INVALID_VAL (JIT 签名) |
| `Not concrete value` | `function.md` | INVALID_VAL |
| `OOM` / `Out of memory` / `rtMalloc failed` / `workspace 过大` | `machine.md` | Workspace 内存异常偏大 |
| `PASS` / `pass error` / `pass 错误` | `pass.md` | 前端用户通用排查方法 |
| `REGISTER_COPY` / `vec tile` / `vector tile` | `vector.md` | ERR_CONFIG_TILE |
| `set_vec_tile_shapes` / `tile shape not set` / `tile 未设置` | `vector.md` / `function.md` | ERR_CONFIG_TILE / INVALID_VAL |
| `shape mismatch` / `shape 不匹配` | `pass.md` | TENSOR_SHAPE_MISMATCH |
| `SIM` / `simulation` / `SIM mode` / `sim 模式` | `simulation.md` | 排查建议 |
| `stitch_function` / `stitch` | `machine.md` | AIC ERROR |
| `sym_*` / `use of undeclared identifier` / `undeclared` | `codegen.md` | 场景4: 变量未定义 |
| `symbolic` / `symbolic index` / `符号索引` / `符号值` | `function.md` | DYNAMIC_SHAPE_COMPUTE |
| `Tensor is not iterable` / `shape unpack` | `function.md` | INVALID_VAL (JIT 签名) |
| `Their size actually are` / `offsets mismatch` / `pypto.view` / `view 维度不匹配` | `function.md` | INVALID_VAL |
| `tile shape` / `tile shapes` / `tiling` / `tile 配置` | `pass.md` | 常见前端问题修复建议 |
| `transposeA` / `transposeB` / `转置` | `matmul.md` | ERR_PARAM_MISMATCH |
| `runtime_debug_mode` / `泳道图` / `profiling` / `dyn_topo` | `machine.md` | 泳道图相关问题指导 |
| `依赖异常` / `依赖边` / `dependency` | `machine.md` | 怀疑精度问题与运行时依赖异常有关 |
| `静默失败` / `结果全零` / `无错误信息` | — | 先拉 slog（见 §四），再按错误码查表 |
| `精度` / `precision`（属于 DFX 范围的） | `machine.md` | 怀疑和MACHINE内存处理有关的精度问题 |

> **使用方式**：搜索当前 error log 中的关键字，在上表中定位目标文档章节，然后到对应章节查看详细排查步骤。
> 关键字索引覆盖不全时，回退到下方「一、错误码路由」按错误码查找。

---

## 一、错误码 → 组件文档路由

| 错误码范围 | 归属组件 | 本地文档 | 官方文档 URL |
|---|---|---|---|
| `F0XXXX` | 外部写法问题 | — | — |
| `F1XXXX` | 框架内部公共问题 | — | — |
| `F2XXXX` - `F3XXXX` | **FUNCTION** | `function.md` | [`function.html`](https://pypto.gitcode.com/trouble_shooting/function.html) |
| `F4XXXX` - `F5XXXX` | **PASS** | `pass.md` | [`pass.html`](https://pypto.gitcode.com/trouble_shooting/pass.html) |
| `F6XXXX` | **CODEGEN** | `codegen.md` | [`codegen.html`](https://pypto.gitcode.com/trouble_shooting/codegen.html) |
| `F7XXXX` - `F8XXXX` | **MACHINE** | `machine.md` | [`machine.html`](https://pypto.gitcode.com/trouble_shooting/machine.html) |
| `F9XXXX` | **SIMULATION** | `simulation.md` | [`simulation.html`](https://pypto.gitcode.com/trouble_shooting/simulation.html) |
| `FAXXXX` | **DISTRIBUTED** | `distributed.md` | [`distributed.html`](https://pypto.gitcode.com/trouble_shooting/distributed.html) |
| `FBXXXX` | **VERIFY** | `verify.md` | [`verify.html`](https://pypto.gitcode.com/trouble_shooting/verify.html) |
| `FCXXXX` | **OPERATION**（总入口） | `operation.md` | — |
| `FC0XXX` - `FC2XXX` | VECTOR（OPERATION 子类） | `vector.md` | [`vector.html`](https://pypto.gitcode.com/trouble_shooting/vector.html) |
| `FC3XXX` - `FC5XXX` | MATMUL（OPERATION 子类） | `matmul.md` | [`matmul.html`](https://pypto.gitcode.com/trouble_shooting/matmul.html) |
| `FC6XXX` - `FC8XXX` | CONV（OPERATION 子类） | `conv.md` | [`conv.html`](https://pypto.gitcode.com/trouble_shooting/conv.html) |
| `FC9XXX` | 视图类 OP（OPERATION 子类） | `view_op.md` | [`view_op.html`](https://pypto.gitcode.com/trouble_shooting/view_op.html) |

---

## 二、运行时症状 → 文档章节

### MACHINE 组件 (machine.md / [machine.html](https://pypto.gitcode.com/trouble_shooting/machine.html))

| 症状 | 文档章节 |
|---|---|
| **aicore error** | [### AIC ERROR / The aicore execution is abnormal](https://pypto.gitcode.com/trouble_shooting/machine.html#aic-error-the-aicore-execution-is-abnormal) |
| 精度问题（怀疑内存相关） | [### 怀疑和MACHINE内存处理有关的精度问题](https://pypto.gitcode.com/trouble_shooting/machine.html#怀疑和machine内存处理有关的精度问题) |
| 精度问题（怀疑运行时依赖异常） | [### 怀疑精度问题与运行时依赖异常有关](https://pypto.gitcode.com/trouble_shooting/machine.html#怀疑精度问题与运行时依赖异常有关) |
| encode 阶段 actualRawMagic 断言 | [### encode 阶段 actualRawMagic 断言触发](https://pypto.gitcode.com/trouble_shooting/machine.html#encode-阶段-actualrawmagic-断言触发) |
| Workspace 内存异常偏大 / OOM | [### Workspace 内存异常偏大](https://pypto.gitcode.com/trouble_shooting/machine.html#workspace-内存异常偏大) |
| `F70006` / HANDSHAKE_TIMEOUT | [### F70006 HANDSHAKE_TIMEOUT](https://pypto.gitcode.com/trouble_shooting/machine.html#f70006-handshake-timeout) |
| Host 侧 segfault / 堆栈崩溃 | [### Host侧捕获异常打印汇编堆栈信息](https://pypto.gitcode.com/trouble_shooting/machine.html#host侧捕获异常打印汇编堆栈信息) |
| AiCore Print 调试 | [### AiCore Print 使用方法](https://pypto.gitcode.com/trouble_shooting/machine.html#aicore-print-使用方法) |
| 泳道图问题 | [### 泳道图相关问题指导](https://pypto.gitcode.com/trouble_shooting/machine.html#泳道图相关问题指导) |

### FUNCTION 组件 (function.md / [function.html](https://pypto.gitcode.com/trouble_shooting/function.html))

| 症状 | 文档章节 |
|---|---|
| **INVALID_VAL (0x21004U)** — 无效的值 | [### INVALID_VAL (0x21004U)](https://pypto.gitcode.com/trouble_shooting/function.html#invalid-val-0x21004u) |
| **INVALID_TYPE (0x21003U)** — 错误的类型 | [### INVALID_TYPE (0x21003U)](https://pypto.gitcode.com/trouble_shooting/function.html#invalid-type-0x21003u) |
| **DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED (0x21009U)** | [### DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED (0x21009U)](https://pypto.gitcode.com/trouble_shooting/function.html#dynamic-shape-compute-unsupported-0x21009u) |
| **INVALID_OPERATION (0x21002U)** | [### INVALID_OPERATION (0x21002U)](https://pypto.gitcode.com/trouble_shooting/function.html#invalid-operation-0x21002u) |
| **UNKNOWN (0x3FFFFU)** — 未知错误 | [### 未知错误码](https://pypto.gitcode.com/trouble_shooting/function.html#未知错误码) |
| **OUT_OF_RANGE (0x21006U)** | [### OUT_OF_RANGE (0x21006U)](https://pypto.gitcode.com/trouble_shooting/function.html#out-of-range-0x21006u) |
| **OP_DEPENDENCY_CYCLE (0x2100AU)** — 依赖环 / 拓扑排序失败 | [### OP_DEPENDENCY_CYCLE (0x2100AU)](https://pypto.gitcode.com/trouble_shooting/function.html#op-dependency-cycle-0x2100au) |
| EINTERNAL / INVALID_PTR / IS_EXIST / NOT_EXIST / BAD_FD / INVALID_FILE | 均见 `function.md` 对应章节 |

### PASS 组件 (pass.md / [pass.html](https://pypto.gitcode.com/trouble_shooting/pass.html))

| 症状 | 文档章节 |
|---|---|
| **Tile Shape 超限 / Shape 不匹配 / dtype 不支持** | [### 前端用户通用排查方法 → 常见前端问题修复建议](https://pypto.gitcode.com/trouble_shooting/pass.html#常见前端问题修复建议) |
| **F40000** TENSOR_NULL_POINTER | [#### F40000 TENSOR_NULL_POINTER](https://pypto.gitcode.com/trouble_shooting/pass.html#f40000-tensor-null-pointer) |
| **F40001** TENSOR_INVALID_MEMORY_TYPE | [#### F40001 TENSOR_INVALID_MEMORY_TYPE](https://pypto.gitcode.com/trouble_shooting/pass.html#f40001-tensor-invalid-memory-type) |
| **F40002** TENSOR_SUBGRAPH_BOUNDARY | [#### F40002 TENSOR_SUBGRAPH_BOUNDARY](https://pypto.gitcode.com/trouble_shooting/pass.html#f40002-tensor-subgraph-boundary) |
| **F40003** TENSOR_SHAPE_MISMATCH | [#### F40003 TENSOR_SHAPE_MISMATCH](https://pypto.gitcode.com/trouble_shooting/pass.html#f40003-tensor-shape-mismatch) |
| **F40004** TENSOR_UNSUPPORTED_DATATYPE | [#### F40004 TENSOR_UNSUPPORTED_DATATYPE](https://pypto.gitcode.com/trouble_shooting/pass.html#f40004-tensor-unsupported-datatype) |
| **F40005** TENSOR_MEMORY_ALLOCATION | [#### F40005 TENSOR_MEMORY_ALLOCATION](https://pypto.gitcode.com/trouble_shooting/pass.html#f40005-tensor-memory-allocation) |
| **F40006** TENSOR_DYNAMIC_ATTR | [#### F40006 TENSOR_DYNAMIC_ATTR](https://pypto.gitcode.com/trouble_shooting/pass.html#f40006-tensor-dynamic-attr) |
| **F41000** OP_INVALID_OPERAND_COUNT | `pass.md` § F41000 |
| **F41001** OP_NULL_POINTER | `pass.md` § F41001 |
| **F41002** OP_INVALID_OPCODE | `pass.md` § F41002 |
| **F41003** OP_PRODUCER_CONSUMER | `pass.md` § F41003 |
| **F41004** OP_SPECIAL_CONSTRAINT | `pass.md` § F41004 |
| **F41005** OP_NESTING_DEPTH | `pass.md` § F41005 |
| **F41006** OP_SEQUENCE_ERROR | `pass.md` § F41006 |
| **F42xxx** FUNCTION 图结构问题 | `pass.md` § Function相关错误 |
| **F43xxx** GRAPH 拓扑问题 | `pass.md` § Graph相关错误 |
| **F44xxx** CONFIG 配置问题 | `pass.md` § Config相关错误 |

### CODEGEN 组件 (codegen.md / [codegen.html](https://pypto.gitcode.com/trouble_shooting/codegen.html))

| 症状 | 文档章节 |
|---|---|
| Kernel 代码中 TileOP 参数不符合预期 | [#### 生成kernel代码中某个TileOP调用参数不符合预期](https://pypto.gitcode.com/trouble_shooting/codegen.html#生成kernel代码中某个tileop调用参数不符合预期) |
| **F62014** SYMBOL_NOT_FOUND | [#### 错误码 F62014：SYMBOL_NOT_FOUND](https://pypto.gitcode.com/trouble_shooting/codegen.html#错误码-f62014symbol-not-found) |
| **F63001** COMPILE_CODE_FAILED / 堆栈溢出 | [#### 错误码 F63001：COMPILE_CODE_FAILED](https://pypto.gitcode.com/trouble_shooting/codegen.html#错误码-f63001compile-code-failed) |
| **F63001** PTO 指令数据类型不匹配（`maybe need a type`） | [#### 场景2: PTO指令数据类型不匹配](https://pypto.gitcode.com/trouble_shooting/codegen.html#pto) |
| **F63001** 硬件平台不匹配（`does not support the given target feature`） | [#### 场景3: 生成PTO指令和二进制编译参数指定的硬件平台不匹配](https://pypto.gitcode.com/trouble_shooting/codegen.html#id6) |
| **F63001** 变量未定义（`sym_*` / `use of undeclared identifier`） | [#### 场景4: 变量未定义](https://pypto.gitcode.com/trouble_shooting/codegen.html#id7) |
| 二进制编译时长统计 | [#### 二进制编译时长统计](https://pypto.gitcode.com/trouble_shooting/codegen.html#id8) |

### OPERATION 子组件

| 症状 | 文档章节 |
|---|---|
| **FC0000** ERR_PARAM_INVALID（VECTOR） | [vector.md](https://pypto.gitcode.com/trouble_shooting/vector.html#fc0000-err-param-invalid) |
| **FC0001** ERR_PARAM_DTYPE_UNSUPPORTED（VECTOR） | [vector.md](https://pypto.gitcode.com/trouble_shooting/vector.html#fc0001-err-param-dtype-unsupported) |
| **FC1000** ERR_CONFIG_TILE（VECTOR） | [vector.md](https://pypto.gitcode.com/trouble_shooting/vector.html#fc1000-err-config-tile) |
| **FC1001** ERR_CONFIG_ALIGNMENT（VECTOR） | [vector.md](https://pypto.gitcode.com/trouble_shooting/vector.html#fc1001-err-config-alignment) |
| **FC3000** ERR_PARAM_INVALID（MATMUL） | [matmul.md](https://pypto.gitcode.com/trouble_shooting/matmul.html#fc3000-err-param-invalid) |
| **FC3001** ERR_PARAM_MISMATCH（MATMUL） | [matmul.md](https://pypto.gitcode.com/trouble_shooting/matmul.html#fc3001-err-param-mismatch) |
| **FC4000** ERR_CONFIG_TILE（MATMUL） | [matmul.md](https://pypto.gitcode.com/trouble_shooting/matmul.html#fc4000-err-config-tile) |
| **FC4001** ERR_CONFIG_ALIGNMENT（MATMUL） | [matmul.md](https://pypto.gitcode.com/trouble_shooting/matmul.html#fc4001-err-config-alignment) |

---

## 三、常见错误 → 修复速查表

> 以下为开发中最常遇到的错误及快速修复方案。标注 `→ doc` 的条目建议同时查阅上方对应文档。

| 错误信息 | 根因 | 修复 |
|---|---|---|
| `Non-tensor parameter 'x' must not be a torch.Tensor` | `from __future__ import annotations` 导致类型标注变字符串 | **删除**该 import |
| `Not concrete value` | Tile shapes 使用了符号值（symbolic） | 使用模块级常量：`TILE_M = 32`；→ `pass.md` |
| `Cannot convert symbols to int` | 直接对 tensor 做 `tensor[symbolic_idx]` 索引 | 改用 `pypto.view(tensor, shape, [idx, ...])` |
| `ValueError: Invalid value type` | `pypto.loop` bounds 使用了符号表达式 | 将 loop bounds 作为**具体 int 参数**传入 JIT |
| `Errcode: F21004! tile shape not set` | 第一个 op 之前未调用 `set_vec_tile_shapes` | 在 `@jit` 函数体**最顶部**调用；→ `function.md § INVALID_VAL` |
| `Their size actually are X and Y` (F21004) | `pypto.view` 的 shape 和 offsets 维度数不一致 | 确保 `len(shape) == len(offsets)`，必要时用 1 补齐 |
| AICore Error / `.cce` 文件路径 | Device 侧执行异常 | → `machine.md § AIC ERROR` |
| `Errcode: FFFFFF! launch aicpu failed: 107003` | NPU launch 失败 | → `machine.md`；检查 tile shapes 整除维度 + `torch.npu.set_device()` |
| 编译通过但结果全零/无错误信息 | 静默失败 | 拉取 slog：`ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1` |
| `L0A/L0B/L0C size exceeded` | Tile shape 超 L0 buffer | → `pass.md § 常见前端问题修复建议` |
| `L1 size exceeded` | Tile shape 超 L1 buffer | → `pass.md § 常见前端问题修复建议` |
| `F40005` TENSOR_MEMORY_ALLOCATION | Tensor 内存分配问题 | → `pass.md § F40005` |
| `F62014` SYMBOL_NOT_FOUND | Codegen 未定义变量 | → `codegen.md § F62014` |
| `F63001` COMPILE_CODE_FAILED | Kernel 代码编译失败 | → `codegen.md § F63001` |
| `F63001` + `maybe need a type` | PTO 指令数据类型不匹配 | → `codegen.md § 场景2`；检查前端 OP 参数或更换硬件支持的 dtype |
| `F63001` + `does not support the given target feature` | Vector/Cube 子图混合 | → `codegen.md § 场景3`；PASS 子图切分问题 |
| `F63001` + `use of undeclared identifier 'sym_*'` | 动态 Shape 变量未定义 | → `codegen.md § 场景4`；告知 PASS 子图 ID 分析变量缺失原因 |
| `F2100A` / `cycle detected` / 拓扑排序失败 | 算子依赖图存在环 | → `function.md § OP_DEPENDENCY_CYCLE`；用 `computation_graph_analyzer.py --detect-op-cycle` 定位环路径 |
| `F44003` 配置文件读取失败 | 配置项缺失或文件解析错误 | → `pass.md § F44003`；检查 `tile_fwk_config.json` 中配置项是否存在 |
| Workspace 内存异常偏大 / OOM | Workspace 分配过大 | → `machine.md § Workspace 内存异常偏大` |
| 精度问题（怀疑内存重叠） | 内存踩踏 | → `machine.md § 怀疑和MACHINE内存处理有关的精度问题` |
| 精度问题（怀疑依赖异常） | Stitch 依赖边丢失 | → `machine.md § 怀疑精度问题与运行时依赖异常有关` |
| Host 侧 segfault / core dump | 堆栈崩溃 | → `machine.md § Host侧捕获异常打印汇编堆栈信息` |
| SIM 模式精度 FAIL 但 NPU 正常 | SIM 模式不支持精度验证 | SIM 仅用于结构验证，**精度始终在 NPU 上验证**；→ `simulation.md` |
| `F70006` HANDSHAKE_TIMEOUT | 握手超时 | → `machine.md § F70006 HANDSHAKE_TIMEOUT` |
| 泳道图未生成 / 为空 | Profiling 配置问题 | → `machine.md § 泳道图相关问题指导` |

---
