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

## 15. matmul 权重 TILEOP_NZ 入参须以 FRACTAL_NZ 传入（F00002）

JIT 签名声明 `format=pypto.TileOpFormat.TILEOP_NZ`，wrapper 却传入 ND 布局 tensor，报
`F00002: The format of input tensor ND does not match ... TileOpFormat.TILEOP_NZ`。

根因：`torch_npu.npu.config.allow_internal_format` 未置 True。开关关闭时，
`torch_npu.npu_format_cast(t, torch_npu.Format.FRACTAL_NZ)` 对 BF16/INT8/FP32 均静默返回 ND，权重未实际转换。

```python
import torch_npu
torch_npu.npu.config.allow_internal_format = True   # 无条件赋值；hasattr 在 torch_npu 2.12 误判 False
w_nz = torch_npu.npu_format_cast(w.contiguous(), torch_npu.Format.FRACTAL_NZ)
```

- `pypto.Tensor(shape, dtype, format=...)` 的 `format` 是入参格式契约，非可选优化；SPEC 声明权重 NZ，须同时落地「签名 TILEOP_NZ + host 转 FRACTAL_NZ」。
- 开关开启后，`npu_format_cast` 对 BF16/INT8/FP32 均真转 NZ（`get_npu_format` 返回 FRACTAL_NZ）。
- 仅当权重不满足 NZ 对齐（内轴 32B、外轴 16 元素整除）时，回退 ND 交由 `pypto.matmul` 内部转换。

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

---

## 20. 分支写法与语义域配对（`is_loop_begin/end` / 循环内分支）

PyPTO 前端有两个语义域，域判定**按装饰器，不按词法位置**：

- **PARSER 域**：带 `@pypto.frontend.jit` 或 `@pypto.frontend.function` 装饰器的函数体（AST 改写）。模块级 `@pypto.frontend.function` helper 会被 parser 内联，同属 PARSER 域——其内写 `pypto.cond` 同样报 F00001。
- **PLAIN 域**：未装饰的普通函数（trace 期按普通 Python eager 录制，parser 不碰）。

| 写法 \ 域 | PLAIN（未装饰 helper） | PARSER（@jit / @frontend.function） |
|---|---|---|
| `if pypto.cond(predicate):` | ✅ 推荐 | ❌ `F00001 TypeError: ... got RecordIfBranch` |
| 裸 `if pypto.is_loop_begin(idx):` | ❌ `F00002 Not concrete value` | ✅ 合法：parser 自动给 SymbolicScalar 条件包 cond |
| 裸 `if` + 比较表达式（`if s2_idx < trip:`） | ❌ 同上（`SymbolicScalar.__bool__`） | ✅ AST rewriter 改写 |
| @jit body 内嵌套 `def inner():` | — | ❌ `F00003 function nested is not allowed`（构造即非法） |

**推荐模式**：online-softmax 类循环放**未装饰 helper** + `pypto.cond(is_loop_begin/end)`；状态张量声明在循环外、全程 `buf[:]` 原地写。

**调试手段**：检查 trace 后分支是否录制，用 `pypto.pypto_impl.Dump()` 导出整个录制程序（`Function.dump()` 只看得到 entry 函数，循环体在隐藏 callee 函数里看不到）。

**关联条目**：`pypto.cond` 不接受任意比较表达式见 §3；嵌套动态 loop 仅最内层带 `unroll_list`。

---

## 21. JIT 编译卡死（jit_stall）/ 编译缓慢（slow_compile）排查

设备侧 JIT 编译卡死是确定性的——无进展时延长等待不会改变结果，**不要重跑全量 kernel**，按以下顺序排查：

**jit_stall（按顺序）：**

1. **分支写法与语义域配对** —— 按 §20 配对表检查 kernel。已知死锁模式：`@jit` body 内裸 `if pypto.is_loop_begin/end`，trace 期合法但可能导致设备侧编译卡死。
2. **检查 trace 后的分支结构** —— 用 `pypto.pypto_impl.Dump()` 导出整个录制程序（`Function.dump()` 只看 entry 函数，分支体不可见，见 §20）。

**slow_compile：** 检查 `unroll_list` 取值（多值列表导致编译路径爆炸）、循环嵌套深度、`ExpandFunction` 生成函数的数量与规模。
