# 外部写法问题经验

> 对应错误码范围：F0XXXX。这些错误源于用户侧 API 调用方式不当、JIT 语法限制、参数顺序错误等外部因素，而非 PyPTO 框架内部问题。

---

## 1. `from_torch()` 系列陷阱

| 错误写法 | 错误关键词 | 正确写法 |
|---------|----------|---------|
| `x_pto, _ = pypto.from_torch(x)` | `F00001 TypeError: Tensor is not iterable` | `x_pto = pypto.from_torch(x)`（单变量接收） |
| 将 `from_torch()` 返回值传给 `@jit` kernel | `RuntimeError: not a valid torch tensor type` | 直接传 `torch.Tensor`，decorator 自动转换 |
| `pypto.empty(M, D, ...)` | `AttributeError: module 'pypto' has no attribute 'empty'` | `torch.empty(M, D, dtype=..., device=device)` |

---

## 2. Element / 标量使用规则

### 2.1 `pypto.Element` 参数顺序：dtype 在前，value 在后

`Element.__init__` 仅接受 `int` 和 `float`，其他类型直接报错。

| 错误写法 | 错误关键词 | 正确写法 |
|---------|----------|---------|
| `pypto.Element(0.0, pypto.DT_FP32)` | `F00001 incompatible constructor arguments` | `pypto.Element(pypto.DT_FP32, 0.0)` |
| `pypto.full([1, D], pypto.DT_FP32, 0.0)` | `F00001 incompatible function arguments` | `pypto.full([1, D], 0.0, pypto.DT_FP32)`（shape, value, dtype） |
| `pypto.Element(pypto.DT_FP32, tensor_var)` | `F00002 Invalid data type for Element` | 删除 Element 包装，直接用 tensor 广播 |

### 2.2 `pypto.mul/add/sub/div` 第二参数：直接传 Python 标量，禁止显式 `pypto.Element`

`mul`、`div`、`add`、`sub` 内部会自动将 Python float/int 包装为 C++ Element。如果显式传入 `pypto.Element` 对象，会被再次包装，导致双重包装崩溃。

```python
# ❌ 显式 Element → 双重包装
pypto.mul(tensor, pypto.Element(pypto.DT_FP32, 32768.0))  # F00002

# ❌ 无意义操作
pypto.add(x, 0.0)  # F2FFFF

# ✅ 直接传 Python 标量
pypto.mul(tensor, 32768.0)
pypto.mul(square, 1.0 / Q_LORA_RANK)
```

> **例外**：`pypto.where` 的 isinstance 检查的是 C++ binding 层的 `pypto_impl.Element`，与 Python wrapper `pypto.Element` 不是同一类型。因此 `pypto.where` 也必须传 Python float，不能传 `pypto.Element` 对象。详见 `vector.md §1`。

---

## 3. `pypto.cond` 不接受任意比较表达式

`pypto.cond` 接受 `int` 或 `SymbolicScalar`，但 JIT 只特殊处理 `is_loop_begin`/`is_loop_end` 产生的条件和字面整数。传入任意比较表达式（如 `g_idx == 0`）会导致 `RecordIfBranch.__bool__()` 无法处理而报错。

| 写法 | 结果 |
|------|------|
| `if pypto.cond(pypto.is_loop_begin(bn)):` | 合法 |
| `if pypto.cond(pypto.is_loop_end(bn)):` | 合法 |
| `if pypto.cond(1):` | 合法（字面整数） |
| `if pypto.cond(g_idx == 0):` | `F00001 TypeError: RecordIfBranch` |
| `with pypto.cond(condition):` | `F00005 With is not supported` |

**解决方案**：非 loop 边界条件直接用 Python `if/else`（`@jit` 内核中 AST 解析器原生处理 Python `if`）。

---

## 4. JIT 语法限制

| 错误写法 | 错误关键词 | 正确写法 |
|---------|----------|---------|
| `@jit` kernel 内写 `return result` | `F00002 Return statements are not allowed` | 通过写入 output tensor 输出 |
| 传入 `np.int64(M)` 作为 API 参数 | `F00001` + `reshape() requires integer shape` | 用纯 Python `int(M)` |

---

## 5. `pypto.zeros` 的 dtype 必须用关键字参数

`pypto.zeros` 签名为 `def zeros(*size, dtype=None)`。`*size` 是可变位置参数，`dtype` 在其后，是 keyword-only 参数。`pypto.zeros([M, D], pypto.DT_FP32)` 中 `pypto.DT_FP32` 会被 `*size` 吞掉。

```python
# ❌ dtype 被 *size 捕获
pypto.zeros([M, D], pypto.DT_FP32)

# ✅ 方案 1：dtype 用关键字参数
pypto.zeros([M, D], dtype=pypto.DT_FP32)

# ✅ 方案 2（推荐）：统一用 pypto.full 代替
pypto.full([M, D], 0.0, pypto.DT_FP32)
```

---

## 6. `pypto.div` 第一个参数不能是 Python 浮点数

`pypto.div` 的第一个参数必须是 Tensor，不能是 Python float。

```python
# ❌ 第一参数是 Python float → F00001
pypto.div(448.0, amax_clamped)

# ✅ 用 pypto.full 创建常量 Tensor
scale_const = pypto.full(amax_clamped.shape, 448.0, pypto.DT_FP32)
pypto.div(scale_const, amax_clamped)
```

---

## 7. `pypto.div` 不支持 INT32 数据类型

`pypto.div` 仅支持 `DT_FP16`、`DT_BF16`、`DT_FP32`。

```python
# ❌ INT32 tensor 直接 div → FC0001
pypto.div(cur_seq_tensor, 128)

# ✅ 方案 A：cast 中转
block_fp32 = pypto.div(pypto.cast(cur_seq_tensor, pypto.DT_FP32), 128.0)
block_idx = pypto.cast(block_fp32, pypto.DT_INT32)

# ✅ 方案 B：SymbolicScalar 整除（非 tensor 操作）
block_count = (cur_seq + 127) // 128
```

---

## 8. `loop_unroll` 内 `.shape` 返回全量维度

在 `loop_unroll` 循环体内，对中间张量调用 `.shape` 返回的是全量 unroll 维度而非当前 tile 的局部维度。

**解决方案**：在函数顶部用输入 tensor 的 `.shape` 提取维度到 Python `int` 变量，后续全部使用这些变量，不在循环体内对中间结果调用 `.shape`。

---

## 9. kernel 参数 `pypto.DYNAMIC` 与固定值混用

kernel 参数类型注解中，部分参数的同一维度使用 `pypto.DYNAMIC`，其他参数使用固定值。JIT 编译期将 `DYNAMIC`（`-1`）与固定值视为不兼容，导致广播校验失败。

**解决方案**：统一所有 kernel 参数的同一维度——全部用固定值，或全部用 `pypto.DYNAMIC`。

---

## 10. F00002: `Not concrete value`

`pypto.DYNAMIC` 维度直接作为操作维度，或除 loop 变量外的其他动态维度无法被识别。

**解决方案**：常量维度直接写死；动态维度在 `pypto.loop` 内用 `view` 切出静态 shape tile 再计算。

---

## 11. JIT 内禁止局部变量类型注解

`x: pypto.Tensor = ...` 等局部变量类型注解会触发 `F0FFFF AnnAssign` / `F00003`。

**解决方案**：去掉类型注解。

---

## 12. `pypto.transpose` 仅接受 2 个维度参数

`pypto.transpose(x, dim0, dim1)` 仅交换两个维度。多维排列用 `pypto.permute(x, [0, 2, 1])`。

| 错误写法 | 错误关键词 | 正确写法 |
|---------|----------|---------|
| `pypto.transpose(x, [0,2,1])` | `F00001 TypeError: transpose dims` | `pypto.permute(x, [0, 2, 1])` |

---

## 13. `torch.library.impl` 不支持 `NPU` 分发键

**解决方案**：删除整个 `torch.library` 注册块（`Library`、`define`、`impl` 全部删除）。

---

## 14. `pypto.topk()` 不支持 `sorted` 关键字参数

从 `torch.topk(..., sorted=False)` 迁移时保留 `sorted` 参数会报错。

**解决方案**：移除 `sorted` 参数。

---

## 15. INT8 Tensor JIT 签名 format 必须为 ND

JIT 签名中 INT8 tensor 声明 `format=NZ`，但 wrapper 传入的是 ND 格式。`torch_npu.npu_format_cast` 对 INT8 tensor 为 no-op，无法运行时转换。

**解决方案**：将 JIT 签名中的 `format=NZ` 改为 `format=ND`。`pypto.matmul` 内部会处理 ND→NZ 格式转换。

---

## 16. cube tile 后必须重置 vec tile

`set_cube_tile_shapes()` 和 `set_vec_tile_shapes()` 是两套独立状态，互不覆盖。matmul 后直接做 vec 操作前，必须显式调用 `set_vec_tile_shapes()` 重置。

---

## 17. `pypto.rms_norm` 参数名是 `epsilon` 而非 `eps`

从 `torch.nn.RMSNorm(eps=...)` 迁移时使用 `eps=` 会报错。

```python
# ❌ torch 习惯写法
pypto.rms_norm(x, gamma, eps=1e-6)  # F00001

# ✅ PyPTO 正确写法
pypto.rms_norm(x, gamma, epsilon=1e-6)
```

---

## 18. `pypto.cast` 的 `satmode` 必须传枚举，不接受 Python bool

C++ binding 层 `satmode` 参数类型为 `SaturationMode` 枚举，pybind11 严格类型检查不接受 Python `bool`。

```python
# ❌ Python bool → F00001
pypto.cast(x, pypto.DT_INT8, satmode=True)

# ✅ SaturationMode 枚举
pypto.cast(x, pypto.DT_INT8, satmode=pypto.SaturationMode.ON)
```

---

## 19. vec tile 尾轴必须满足 32Byte 对齐

| dtype | 最后一维最小元素数 |
|-------|-------------------|
| FP32 (4B) | >= 8 |
| BF16/FP16 (2B) | >= 16 |

`set_vec_tile_shapes(4, 1)` → `FC1001 ERR_CONFIG_ALIGNMENT`。改为 `set_vec_tile_shapes(4, 8)` 等满足 32B 对齐的值。
