# PyPTO 执行约束速查

## 1. 框架级约束

- PyTorch 集成支持单算子模式（eager）和图捕获模式（aclgraph）；`@pypto.frontend.jit` 默认按单算子模式执行。
- JIT kernel 不支持返回值；结果必须写回输出参数。
- 可用的输出写回方式包括 `out[:] = ...`、`out.move(...)`、`pypto.assemble(..., out)`；`out = ...` 只会绑定局部变量，不会修改出参。
- JIT 函数中的张量参数必须写成 `pypto.Tensor([...], dtype)` 类型注解。
- JIT 函数中张量参数在前，非张量参数在后。
- 动态轴必须在类型注解中标成 `pypto.DYNAMIC` 或 `pypto.DYN`。**禁止使用 `pypto.Tensor()` / `pypto.Tensor([], dtype)` 等空注解**（门禁 OL31 会直接判 FAIL）。
- 标成 `pypto.DYNAMIC` 的轴变化时无需重编译；标成 `pypto.STATIC` 的轴变化会触发重编译。
- DESIGN.md 声明了动态轴时，JIT 函数内必须包含遍历动态轴的 `pypto.loop(...)` 调用，trip count 必须是 `tensor.shape[i]`、函数参数或其符号表达式；**禁止**用 `pypto.loop(1)` / 常量 trip count 的空循环冒充（门禁 OL43）。
- 固定整数轴只接受该固定大小；传入其他大小会报错（runtime_debug_mode=3 开启校验时）。
- `...` 表示剩余轴按静态轴处理。
- `pypto.tensor(...)` 创建的 Tensor 是未初始化随机值；使用前必须初始化。

## 2. 基础类型与前端基础设施

### `pypto.Element`

- 构造顺序固定为 `pypto.Element(dtype, value)`。
- 需要固定标量 dtype 时，显式使用 `Element`。
- `pypto.div`、`pypto.mul`、`pypto.add`、`pypto.sub` 等算术 API 不支持 `pypto.Element` 作为标量参数，传入会双重包装报错。这些 API 已内置 `float`/`int`→`Tensor` 类型转换，直接使用 Python 常量即可。

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
- `reshape`：含动态维度时框架无法自动推导有效形状，必须显式传 `valid_shape`。
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

## 5. 动态轴算子的实现模式（关键断点知识）

> **通用原则 —— "loop 切 tile，API 只吃静态"**
>
> 当算子需要动态轴，但所使用的 API 不支持接收含 `DYNAMIC` 维度的 tensor 时，统一使用以下策略，**不限于 matmul**，同样适用于任何在编译期需要 concrete shape 的计算 API。
> 
> **如何识别 API 是否支持动态 shape**：运行时报 `dim[i] = -1, must be > 0`、`Cannot convert symbols to int`、`Not concrete value` 等错误，或文档明确说明"shape 必须在编译期确定"，均表明该 API 不支持动态 shape。
>
> **处理步骤**：
> 1. **选合适的轴做动态轴**：优先选 batch / 序列长度等语义上天然变化的轴；所选轴**不能是 API 计算直接依赖的维度**（matmul 不选 K/N，归约不选归约 dim，view 的 shape 参数所对应的轴全不能选）。
> 2. **将所选轴标为 `pypto.DYNAMIC`，其余轴标为 `pypto.STATIC` 或常量整数。** 若有多个动态轴，对每个动态轴分别走 `pypto.loop` 嵌套处理。
> 3. **用 `pypto.loop` 沿动态轴迭代**，trip count 取自 `tensor.shape[i]` 或其符号表达式。
> 4. **循环体内 `pypto.view` 切出固定整数大小的 tile**（shape 参数必须全是 Python int）。
> 5. **所有受限 API 只操作静态 tile**，永远不让它们看到含 `DYNAMIC` 维度的 tensor。
> 6. **`pypto.assemble` 写回结果**，offset 可以是 SymbolicScalar，尾块用 `valid_shape` 标记有效范围。
>
> **多动态轴特例**：当算子有 2 个及以上动态轴（如 Batch + SeqLen）时，不能直接在高维 tensor 上调受限 API，必须采用 **"reshape 到 2D + 嵌套 loop + concrete tile"** 模式。

### 5.1 为什么 4D 多动态轴直接 matmul 会失败

```python
# ❌ 错误：B 和 S 都是 DYN，matmul 编译期需要 concrete shape
q: pypto.Tensor([pypto.DYN, N, pypto.DYN, D], pypto.DT_BF16)
scores = pypto.matmul(q, k, out_dtype=pypto.DT_FP32, b_trans=True)
# 报错: operand1 dim[0] = -1, must be > 0
```

PyPTO 的 matmul 在编译期需要所有维度的 concrete shape 来生成 tiling 代码。DYN 维度在编译期表现为 -1，硬件 matmul checker 直接拒绝。

### 5.2 pypto.view 的 shape 参数不接受 SymbolicScalar

```python
s = q.shape[2]  # SymbolicScalar
# ❌ 错误：shape 参数必须全部是 Python int
q_s = pypto.view(q, [1, N, s, D], [b_off, 0, 0, 0], valid_shape=...)
# 报错: View(): incompatible function arguments

# ❌ 错误：Python 切片内部也会对 SymbolicScalar 调 int() 转换
q_s = q[b_off:b_off + 1]
# 报错: Cannot convert symbols to int
```

`pypto.view` 三个参数的类型要求：
- `shape`: **必须全部是 Python int**，不接受 SymbolicScalar
- `offsets`: 接受 SymbolicScalar
- `valid_shape`: 接受 SymbolicScalar，用于尾块有效数据标记

Python `[]` 切片语法内部会对 index 做 `int()` 转换，因此也不能用 SymbolicScalar 做切片索引。

### 5.3 正确模式："2D reshape + 嵌套 loop + concrete tile"

参考 `models/glm_v4_5/glm_attention.py`：

```
4D [B, N, S, D]
      ↓ 在 Python wrapper 层做 reshape
2D [B*N*S, D]
      ↓ 进入 kernel
      ↓ pypto.loop(b) → pypto.loop(N) → pypto.loop(s_tiles)
      ↓ pypto.view([S_TILE, D], [symbolic_offset, 0], valid_shape=[actual_s, D])
2D tile [S_TILE, D]  ← shape 全是 concrete int，编译通过
      ↓ matmul / elementwise / sum（都是 2D 操作）
      ↓ pypto.assemble(result, [symbolic_offset, 0], output_2d)
```

关键点：
- **shape 全 concrete**：`[S_TILE, D]` 都是固定整数常量
- **动态性只进入 offset 和 loop bound**：offset 和 loop 的边界可以是 SymbolicScalar
- **valid_shape 处理尾块**：`actual_s = (s - s_idx * S_TILE).min(S_TILE)`
- **2D matmul**：`[S_TILE, D] × [D, S_TILE]` 编译期 shape 完全确定

### 5.4 循环内累加的标准模式

```python
acc = pypto.tensor([TILE, D], pypto.DT_FP32, "acc")
for idx in pypto.loop(n, name="LOOP", idx_name="idx", unroll_list=[4, 2, 1]):
    tile = compute_something(...)
    if pypto.is_loop_begin(idx):
        acc[:] = tile          # 首次迭代：初始化
    else:
        acc[:] = acc + tile    # 后续迭代：累加
    if pypto.is_loop_end(idx):
        result = pypto.cast(acc, pypto.DT_BF16)
        pypto.assemble(result, [offset, 0], output)  # 最后一次：写回
```

- `pypto.tensor()` 创建的是未初始化随机值，**必须在 is_loop_begin 中初始化**
- `unroll_list` 对内层循环使用，让编译器为不同迭代次数生成优化代码

### 5.5 梯度算子的两趟设计模式

当算子有多个输出需要在不同维度上累加时（如 FlashAttention backward 的 dQ/dK/dV），使用两趟分离计算：

```python
# 趟1: 计算 dQ（外层循环 S1 tiles，内层循环 S2 tiles 累加 dQ）
for s1_idx in pypto.loop(s_loop):
    dQ_acc = pypto.tensor(...)
    for s2_idx in pypto.loop(s_loop):
        dQ_acc += dS_ij @ K_j     # dQ 沿 S2 维度累加
    assemble(dQ_acc, ..., dq)

# 趟2: 计算 dK, dV（外层循环 S2 tiles，内层循环 S1 tiles 累加 dK/dV）
for s2_idx in pypto.loop(s_loop):
    dK_acc = pypto.tensor(...)
    dV_acc = pypto.tensor(...)
    for s1_idx in pypto.loop(s_loop):
        dK_acc += dS_ij^T @ Q_i   # dK 沿 S1 维度累加
        dV_acc += P_ij^T @ dY_i   # dV 沿 S1 维度累加
    assemble(dK_acc, ..., dk)
    assemble(dV_acc, ..., dv)
```

代价是中间结果（P、dS）重复计算一次，但避免了跨 loop 的读写依赖，不需要 `submit_before_loop=True`。

### 5.6 inplace=True 的限制

```python
# ❌ 错误：dq 是函数输出参数，不能用 inplace=True
dq_2d = pypto.reshape(dq, [bns, HEAD_DIM], inplace=True)
# 结果：静默错误，输出 NaN
```

约束：`inplace=True` 的输出不能是整个 Function 的输出参数。当输入 tensor 已经是目标形状时，直接赋值引用即可：

```python
# ✅ 正确：在 wrapper 层提前 reshape，kernel 直接使用
q_2d = q   # 已经是 2D，不需要 reshape
```

## 6. 写代码前先检查这 8 件事

1. 需要建图执行的代码直接写成 `@pypto.frontend.jit` kernel；不要保留为普通 Python/Torch 逻辑。
2. 用 `[:]`、`move()` 或 `assemble()` 把结果明确写回出参；不要用 `out = ...` 代替写回。
3. 把所有动态轴显式标成 `pypto.DYNAMIC` 或 `pypto.DYN`；禁止用 `pypto.Tensor()` 空注解。声明了动态轴的 kernel 必须含真实 `pypto.loop`（trip count 为符号表达式，不能是常量），不允许 `pypto.loop(1)` 这类空循环。
4. 把依赖 Python 标量隐式 dtype 映射的写法改成显式 `Element` 或显式 dtype 转换。
5. 把多轴广播改写成文档支持的单轴广播或等价拆分写法。
6. 检查 TileShape 维度数、最后一维对齐和相关算子的 Tile 约束，再执行编译。
7. 涉及动态轴的 `pypto.view` / `pypto.reshape` 必须显式传入 `valid_shape`。
8. 避免同一 Tensor 在同一图里既被读取又被 `assemble` 回写。
9. 多动态轴算子必须采用 "2D reshape + 嵌套 loop + concrete tile" 模式（见第 5 节），不要尝试在 4D DYN tensor 上直接 matmul。
10. `pypto.view` 的 `shape` 参数只接受 Python int；用 SymbolicScalar 做 offset 和 valid_shape，不要混入 shape。
11. 不要用 `inplace=True` reshape 在函数输出参数上；在 wrapper 层提前 reshape。
12. 每次 matmul / mul / cast / sum 前都必须设置 `set_vec_tile_shapes`（或 `set_cube_tile_shapes`），维度数匹配操作数。漏设会报 `tile shape not set` 或得到错误结果。
