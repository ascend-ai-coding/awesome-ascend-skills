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

## 4. RoPE 3D 在 nh=1 时尾轴被压到 1，触发 32B 对齐失败

nh=1 时使用 `rope_3d()`，其 interleave 阶段 `pypto.transpose(x_rotate, 1, 2)` 将 `[t, 1, 64]` 转为 `[t, 64, 1]`。输出尾轴为 1，FP32 占 4B，不满足 32B 对齐，且 1 是数据维度的真实值，无法通过调整 tile 补齐。

**解决方案**：nh=1 时改用 2D RoPE，输入 `[t, 64]` 直接操作，无 transpose。
