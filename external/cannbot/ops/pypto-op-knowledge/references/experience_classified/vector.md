# VECTOR 组件经验

> 对应错误码范围：FC0-FC2XXX

---

## 1. A2A3 Cast 路径受限

A2A3 (Ascend910) 硬件仅支持固定的 dtype 转换路径，不在支持矩阵中的路径触发 `FC0001 ERR_PARAM_DTYPE_UNSUPPORTED` 或 `F63001 COMPILE_CODE_FAILED`。

### 支持矩阵（仅以下直转路径可用）

| 源类型 | 支持的目标类型 |
|--------|---------------|
| FP16 | FP32, INT32, INT16, INT8, UINT8, INT4 |
| BF16 | FP32, INT32 |
| FP32 | BF16, FP16, INT16, INT32, INT64 |
| INT32 | FP32, INT16, INT64, FP16 |
| INT16 | FP32, FP16 |
| INT64 | FP32, INT32 |
| UINT8 | FP16 |
| INT8 | FP16 |
| INT4 | FP16 |
| **BOOL** | **无（双向均不支持 cast）** |

### 常见不支持路径及替代方案

| 路径 | 替代方案 |
|------|----------|
| INT8 → FP32 | INT8 → FP16 → FP32 |
| INT8 → BF16 | INT8 → FP16 → FP32 → BF16 |
| INT8 → INT32 | 无直接替代 |
| INT8 → INT16 | INT8 → FP16 → FP32 → INT16 |
| BF16 ↔ FP16 | BF16 → FP32 → FP16 |
| FP32 → INT8 | FP32 → FP16 → INT8 |
| INT32 → INT8 | INT32 → FP16(ROUND) → INT8(TRUNC, satmode=ON) |
| UINT8 → FP32 | UINT8 → FP16 → FP32 |
| INT16 → INT32 | INT16 → FP16 → FP32（A2A3 不支持；A5 支持） |
| INT32 → UINT8 | INT32 → FP16 → UINT8（A2A3 不支持；A5 支持） |

### BOOL 替代方案

BOOL 双向均不支持 cast，必须用其他 API 替代：

```python
# FP32 → BOOL：用 pypto.ne 生成 BOOL mask
mask_bool = pypto.ne(mask_flat, 0.0)

# BOOL → FP32：用 pypto.where 条件选择（必须传 Python float，禁止传 pypto.Element 对象）
# ❌ pypto.Element 对象 → isinstance 不匹配 → 崩溃
mask_f32 = pypto.where(mask_bool, pypto.Element(pypto.DT_FP32, 1.0), pypto.Element(pypto.DT_FP32, 0.0))

# ✅ 直接传 Python float
mask_f32 = pypto.where(mask_bool, 1.0, 0.0)
```

> `pypto.where` 的 isinstance 检查的是 C++ binding 层的 `pypto_impl.Element`，与 Python wrapper `pypto._element.Element` 不是同一类型。传入 `pypto.Element(...)` 对象时 isinstance 为 False，落入 else 分支再次包装 → 崩溃。

---

## 2. Vec tile 32Byte 对齐

向量操作 tile 最后一维字节数必须能被 32 整除。

| dtype | 最后一维最小元素数 |
|-------|-------------------|
| FP32 (4B) | >= 8 |
| BF16/FP16 (2B) | >= 16 |

```python
# ❌ 4×4=16B 不满足 32B 对齐 → FC1001
pypto.set_vec_tile_shapes(4, 4)

# ✅ 4×8=32B
pypto.set_vec_tile_shapes(4, 8)
```

---

## 3. `pypto.gather` 替代方案

gather 输入 tensor 全量进 UB（`F40005`）或在动态维度上（`F21009`）时，改用手动 loop + view + assemble：

```python
sel_buf = pypto.Tensor([topk, D], pypto.DT_BF16, name="sel_buf")
for idx in pypto.loop(topk, idx_name="i"):
    row = pypto.view(k, [1, D], [query_inds[idx], 0])
    pypto.assemble(row, [idx, 0], sel_buf)
```

---

## 4. transpose 把**真实为 1** 的数据维搬到尾轴 → 32B 对齐失败，且无法用 tile 补齐

**适用条件**：实现里有 transpose/permute，且某个维度的**真实长度可能为 1**（不是 tile 切出来的 1）。

**规则**：尾轴长度 × dtype 字节数必须满足 32B 对齐。当 transpose 把一个长度为 1 的**数据维**
换到尾轴时，尾轴只有 1 个元素（FP32 = 4B），对齐必然失败——**而且补不了**：
tile 上的 1 可以 padding，**数据本身的 1 不能**，padding 会改变语义。

**解决方案是消掉 transpose，不是调 tile**：为该退化情形改走低一维的实现路径
（把 `[t, 1, D]` 当作 `[t, D]` 直接处理），从根本上不产生这次换轴。

**推广**：任何"某维退化为 1"的分支都值得单独想一遍——它同时会让
tile 切分、对齐、广播语义与一般情形不同。

**实例**：位置编码 3D 实现在 `nh=1` 时，interleave 阶段的
`transpose(x_rotate, 1, 2)` 把 `[t, 1, 64]` 变成 `[t, 64, 1]`，尾轴为 1 触发对齐失败；
改用 2D 路径 `[t, 64]` 直接操作即可。
