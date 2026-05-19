# PyPTO 执行约束速查

## 1. 框架级约束

- PyTorch 集成支持单算子模式（eager）和图捕获模式（aclgraph）；`@pypto.frontend.jit` 默认按单算子模式执行。
- JIT kernel 不支持返回值；结果必须写回输出参数。
- 可用的输出写回方式包括 `out[:] = ...`、`out.move(...)`、`pypto.assemble(..., out)`；`out = ...` 只会绑定局部变量，不会修改出参。
- JIT 函数中的张量参数必须写成 `pypto.Tensor([...], dtype)` 类型注解。
- JIT 函数中张量参数在前，非张量参数在后。
- 动态轴必须在类型注解中标成 `pypto.DYNAMIC` 或 `pypto.DYN`。
- 标成 `pypto.DYNAMIC` 的轴变化时无需重编译；标成 `pypto.STATIC` 的轴变化会触发重编译。
- 固定整数轴只接受该固定大小；传入其他大小会报错（runtime_debug_mode=3 开启校验时）。
- `...` 表示剩余轴按静态轴处理。
- `pypto.tensor(...)` 创建的 Tensor 是未初始化随机值；使用前必须初始化。

## 2. 基础类型与前端基础设施

### `pypto.Element`

- 构造顺序固定为 `pypto.Element(dtype, value)`。
- 需要固定标量 dtype 时，显式使用 `Element`。

### `pypto.SymbolicScalar`

- 构造形式是 `pypto.SymbolicScalar(arg0=None, arg1=None)`。
- `arg0=int` 创建常量符号标量；`arg0=str` 创建具名符号标量；`arg0=SymbolicScalar` 复制已有符号标量；`arg1` 只在 `arg0` 是字符串时作为初始值生效。
- `SymbolicScalar` 用于动态 shape、运行时数值、算术/比较表达式。

### `pypto.from_torch`

- `from_torch(tensor, name="", dynamic_axis=None, tensor_format=None, dtype=None)` 把 `torch.Tensor` 转成 `pypto.Tensor`。
- 输入必须是 `torch.Tensor` 或其子类。
- 输入在指定内存格式下必须是连续的。
- `dynamic_axis` 用来把指定维度标记为动态轴。
- `dtype=None` 时按输入 tensor 的 dtype 推导；需要固定 PyPTO dtype 时显式传 `dtype`。
- 显式标记 `format` 时，传入的 torch tensor 必须与声明的 `format` 一致。

### Tile 配置

- `pypto.set_vec_tile_shapes(...)` 的每个维度都必须大于 0，参数个数最多 4 个。
- TileShape 维度数必须和相关输出维度匹配；否则会在扩图阶段直接报错。
- `pypto.set_cube_tile_shapes(...)` 是 `pypto.matmul` 的前置条件。

## 3. 控制流

### `pypto.loop`

- `pypto.loop` 返回的是符号索引，不是普通 Python 整数循环。
- 在 `pypto.loop` 中用 Python `print` 打印，看到的是构图阶段遍历到的路径，不是运行时真实循环次数。
- `pypto.loop` 默认会展开并分发到多核并行处理；循环迭代之间存在数据依赖时，必须设置 `submit_before_loop=True`。

### `pypto.loop_unroll`

- `pypto.loop_unroll` 返回 `(idx, unroll_factor)`。
- 只对最内层循环做 unroll。
- 循环体里包含 `pypto.cond` 时，增大 `unroll_list` 会显著增加编译路径数和编译时间。

### `pypto.cond`

- `pypto.cond` 必须和 Python `if/elif/else` 一起使用。
- 条件值应是 `SymInt` 或 `SymbolicScalar` 表达式。

### `pypto.is_loop_begin` / `pypto.is_loop_end`

- 这两个接口只接受 `pypto.loop(...)` 返回的循环索引。
- 不是循环索引时会抛 `ValueError`。

## 4. Operation 约束

### 4.1 初始化 / 创建类

- `zeros / ones`：调用前先设置 `set_vec_tile_shapes(...)`；文档覆盖的 dtype 是 `DT_FP32/DT_INT32/DT_INT16/DT_FP16/DT_BF16`。
- `full`：`fill_value` 与 `dtype` 必须一致；动态图尾块无法自动推导有效范围时必须显式传 `valid_shape`。
- `arange`：`step != 0`、`abs(step) > 1e-8`、`(end-start)/step > 0`；纯 int 输入受 int32 范围约束。

### 4.2 Eltwise 算术

- `add / sub / mul / div`：按 2-4 维、单轴广播、输入 dtype 一致来写。
- `add`：数字参数不做任意隐式转换，`other` 不支持 `nan/inf`。
- `sub`：标量路径文档化为 `float`，且不支持隐式转换。
- `mul`：dtype 为 `DT_FP16/DT_BF16/DT_INT16/DT_INT32/DT_FP32`。
- `div`：只支持 `DT_FP16/DT_BF16/DT_FP32`；`other` 不支持 `nan/inf`。
- `neg`：支持 `DT_FP32/DT_FP16/DT_BF16/DT_INT32/DT_INT16` 和 2-4 维；先配 `set_vec_tile_shapes(...)`。
- `abs`：支持 `DT_FP16/DT_BF16/DT_FP32` 和 2-4 维。

### 4.3 数学 / 激活

- `exp`：只支持 `DT_FP16/DT_BF16/DT_FP32` 和 2-4 维。
- `log`：支持 1-4 维和 `DT_FP32/DT_FP16/DT_BF16`；自己做输入域保护。
- `sqrt`：只支持 `DT_FP16/DT_BF16/DT_FP32`。
- `rsqrt`：负数输入返回 `NaN`，0 输入返回 `Inf`；先做数值保护。
- `sin / cos`：支持 `DT_FP32`。
- `sigmoid`：支持 `DT_FP32`。
- `relu`：只支持 `DT_FP16/DT_FP32/DT_BF16` 和 2-4 维；输入不支持 `nan/inf`。
- `lrelu`：`negative_slope` 必须是非负实数，且不能是 `nan/inf`。
- `prelu`：`weight` 必须是一维 Tensor，长度等于 `input` 的第二维；`input/weight` 不支持 `nan/inf`。

### 4.4 比较 / 选择 / 裁剪

- `eq / gt / maximum / minimum / where / clip`：按输入 dtype 一致和单轴广播来写。
- `eq`：返回 `DT_BOOL`；两侧 dtype 必须一致；只支持 2-4 维和一维广播。
- `gt`：返回 `DT_BOOL`；dtype 只覆盖 `DT_FP16/DT_BF16/DT_FP32`。
- `maximum / minimum`：至少一侧必须是 Tensor；若两侧都是 Tensor，只支持单轴广播。
- `where`：`condition` 必须是 `DT_BOOL` Tensor；`input/other` 只支持 `DT_FP32/DT_FP16/DT_BF16`。
- `clip`：`min/max` 类型必须一致；浮点路径定义了 `NaN/INF/-INF` 语义。

### 4.5 `min/max` 的正确用法

- Tensor 逐元素最大最小值：用 `pypto.maximum / pypto.minimum`。
- SymbolicScalar 最大最小值：用 `s.max(other)` / `s.min(other)`。
- `SymbolicScalar.max/min` 的 `other` 只支持 `SymbolicScalar | int`；两边都是具体值时返回具体常量，否则返回符号表达式。
- 宿主侧 Python 逻辑可以用原生 `min/max`；kernel 内不要拿 Python 原生 `min/max` 代替 `maximum/minimum` 或 `SymbolicScalar.min/max`。

### 4.6 归约 / 排序

- `sum / amax / amin / topk`：除了 `dim/keepdim`，还要检查 TileShape、尾轴对齐和 UB 限制。
- `sum`：只支持 `DT_FP32`；`keepdim=False` 后先重设 TileShape。
- `amax / amin`：只支持 `DT_FP16/DT_BF16/DT_FP32` 和 2-4 维；TileShape 受 `64KB`、尾轴 `32B` 对齐、次尾轴 `<=255` 约束。
- `topk`：只支持 `DT_FP32`，且只支持最后一个维度；要求 `k <= TileShape[-1]`，尾轴满足 `32B` 对齐和 `<22KB`。

### 4.7 矩阵 / 注意力相关

- `matmul`：显式给 `out_dtype`；调用前必须设置 `set_cube_tile_shapes(...)`；3D/4D 场景还要设置 `set_vec_tile_shapes(...)`。
- `softmax`：仅支持 `DT_FP32`。

### 4.8 Shape / 视图 / 拼接

- `reshape`：`inplace=True` 时，输入输出必须是当前 loop 的输入输出，且输出不能作为整个 Function 的输出。
- `transpose`：4D 只支持部分轴交换组合，5D 只支持 `(3,4)`。
- `view`：`offsets` 和 `valid_shape` 必须落在原 Tensor 的 shape 范围内。
- `view`：当有效 shape 依赖别的 Tensor 标识、框架无法自动推导时，必须显式传 `valid_shape`。
- `unsqueeze`：返回共享数据的 view；`dim` 必须满足 `[-input.dim-1, input.dim]`。
- `concat`：输入 tensor 数量要求 `2 <= len(tensors) <= 128`；除拼接轴外其余维度必须完全一致。
- `clone`：复制出的 Tensor 与输入保持同 shape、同 dtype。

### 4.9 索引 / 写回 / 填充

- `gather`：`index.dim` 必须等于 `input.dim`；被 gather 的 `dim` 轴不可切；`viewshape[dim] >= max(input.shape[dim], index.shape[dim])`。
- `index_select`：`index` 只支持 `DT_INT32/DT_INT64` 且 shape 只支持 1-2 维；被选维不可切。
- `scatter_update`：不支持 broadcast；`dim` 保持默认 `-2`。
- `assemble`：没有返回值，会直接修改 `out`；`offsets` 必须小于 `out.shape`。
- `assemble`：同一个 Tensor 在同一图里既被 `view` 读取、又被 `assemble` 写回，会形成图成环报错；同图内避免这种回环读写。
- `pad`：只支持 `constant` 模式；多维场景只支持右侧和底部填充；`pad_left/pad_top` 必须为 0；`value` 只支持 `-inf/inf/0.0`。

### 4.10 类型转换

- `cast`：显式暴露 `CastMode` 和 `SaturationMode`，不是简单的 `to(dtype)`。
- 浮点转整数时，`satmode=ON/OFF` 会直接改变溢出后的结果值。

## 5. 写代码前先检查这 8 件事

1. 需要建图执行的代码直接写成 `@pypto.frontend.jit` kernel；不要保留为普通 Python/Torch 逻辑。
2. 用 `[:]`、`move()` 或 `assemble()` 把结果明确写回出参；不要用 `out = ...` 代替写回。
3. 把所有动态轴显式标成 `pypto.DYNAMIC` 或 `pypto.DYN`。
4. 把依赖 Python 标量隐式 dtype 映射的写法改成显式 `Element` 或显式 dtype 转换。
5. 把多轴广播改写成文档支持的单轴广播或等价拆分写法。
6. 检查 TileShape 维度数、最后一维对齐和相关算子的 Tile 约束，再执行编译。
7. 无法自动推导动态 `view` 的 `valid_shape` 时，显式传入 `valid_shape`。
8. 避免同一 Tensor 在同一图里既被读取又被 `assemble` 回写。
