# 视图类OP 经验

> 对应错误码范围：FC9XXX

---

## 1. `pypto.view()` 维度/参数错误

### 1.1 `len(shape) != len(offsets)`

**触发场景**：从 3D tensor 取 2D view，shape 2 元素但 offsets 3 元素。

**错误关键词**：`F00003 → F21004 INVALID_VAL — Their size actually are 3 and 2`

**解决方案**：先 reshape 到 2D 再 view；或 3D→3D view 再 inplace reshape 到 2D。

### 1.2 传 4 个位置参数

**触发场景**：`view()` 只接受 3 个位置参数（tensor, shape, offsets）。

**错误关键词**：`view() takes from 1 to 3 positional arguments but 4 were given`

**解决方案**：额外参数通过 `valid_shape=` 关键字传入。

---

## 2. 3D↔2D view 的确定性模式

**触发场景**：`pypto.view(3D, [2D], [3D])` 内部保留 3D 底层描述符，下游 2D 操作仍将结果视为 3D。

**错误关键词**：`FC1001 → F21003 → F21004 INVALID_VAL — lhs.size():3, rhs.size():2`（常伴随 tile 对齐和 pass_options 错误）

**方案 1（推荐）**：wrapper 层 torch reshape 3D→2D，kernel 内纯 2D→2D view：
```python
x_2d = x.reshape(total, hidden_dim)  # wrapper 层 torch reshape
kernel(x_2d, ...)
# kernel 内：tile = pypto.view(x_2d, [tile_size, D], [offset, 0])
```

**方案 2**：3D→3D view 再 inplace reshape 到 2D：
```python
tile_3d = pypto.view(x, [1, tile_size, D], [b_ofs, h_ofs, 0])
tile_2d = pypto.reshape(tile_3d, [tile_size, D], inplace=True)
```

---

## 3. `pypto.view` offset 超出当前 vec_tile_shapes 覆盖范围

**触发场景**：vec_tile_shapes 声明了 tile 维度（如 `(1, 16, 64)`），但 `pypto.view` 的 offset 超出了该 tile 覆盖 range。例如 tile 覆盖 dim 0–63，但 offset `[0, 0, rope_dim=64]` 从 dim 64 开始读取——tile 内无此数据。

**错误关键词**：`PRECISION_FAIL` — 大面积 OOT（~75% mismatch），无明显编译/运行时错误

**解决方案**：收窄 tile 后，确保所有 view offset 在 tile 维度范围内。若需要跨 tile 边界读取，要么放宽 tile 覆盖更宽的范围，要么在切换到 RoPE 子计算前先 view 出所需数据（view = 对原始 tensor 的引用，offset 基于原始 tensor，不受 tile 限制——但 vec tile shapes 必须能容纳 view 后的 tensor 完整维度）。

```python
# ❌ tile (1, 16, 64) 覆盖 dim 0-63，但 offset [0, 0, 64] 读 dim 64-127
pypto.set_vec_tile_shapes(1, 16, 64)   # 收窄 tile 给 RoPE
x_rope = pypto.view(x, [t_tile, n_heads, rope_dim], [t_idx, h_idx, 64])
# → view offset=64 在 tile 范围外，读到随机数据

# ✅ 方案 A：放宽 tile 覆盖全部 dim
pypto.set_vec_tile_shapes(1, 16, 128)  # 覆盖全 dim
x_rope = pypto.view(x, [t_tile, n_heads, rope_dim], [t_idx, h_idx, 64])

# ✅ 方案 B：view 出完整 dim，再 narrow
pypto.set_vec_tile_shapes(1, 16, 128)
x_full = pypto.view(x, [t_tile, n_heads, 128], [t_idx, h_idx, 0])
x_rope = pypto.view(x_full, [t_tile, n_heads, rope_dim], [0, 0, 64])
```

---

## 4. tail tile：assemble 写入行数超过输出 buffer

**触发场景**：动态 M 的 loop 中，最后一个 tile 的实际行数 `rem` < 静态 tile 行数 `BM`（如 M=1, BM=16）。`pypto.assemble` 写入 BM 行到只分配了 `rem` 行的输出 buffer → 溢出。

**错误关键词**：无显式报错（静默内存越界）；或 `F21004 INVALID_VAL` / 下游精度异常

**解决方案**：输出 buffer 分配时向上取整到 BM 的倍数，`rem = min(M - ofs, BM)` 控制 assemble 写入量：
```python
# ❌ M=1 时 buffer 只有 1 行，assemble 写 16 行
buf = torch.empty(actual_M, D, ...)
# → assemble writes BM rows to 1-row buffer → overflow

# ✅ buffer 按 BM 整倍数分配
buf = torch.empty(((M + BM - 1) // BM) * BM, D, ...)
rem = (M - batch_ofs).min(BM)
pypto.assemble(tile, [batch_ofs, 0], buf, valid_shape=[rem, D])
```

