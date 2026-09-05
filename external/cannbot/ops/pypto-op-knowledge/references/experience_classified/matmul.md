# MATMUL 组件经验

> 对应错误码范围：FC3-FC5XXX

---

## 1. Cube Tile 对齐规则

**触发场景**：

| 触发条件 | 错误关键词 |
|---------|----------|
| kL0/nL0 不能整除 16（mL0 无此约束） | `FC4001 ERR_CONFIG_ALIGNMENT: kL0(X) and nL0(Y) must be aligned to 16 elements` |
| L0 > L1 / L1%L0 != 0 | `FC4000 ERR_CONFIG_TILE: Invalid L1/L0 relation` / `require L0 <= L1 && L1 % L0 == 0` |
| cube tile 乘积超 L0B 64KB | `F40005 TENSOR_MEMORY_ALLOCATION` |

**解决方案**：
```python
pypto.set_cube_tile_shapes([16, 16], [256, 512], [16, 16])  # L0≤L1, L1%L0==0
```

> matmul 本身仅用 cube tile，但 matmul 后续的 vec 操作需要匹配 matmul 输出维度的 tile shape。

---

## 2. Cast Dtype 一致性

**触发场景**：`pypto.matmul(a, b, out_dtype)` 中 a 和 b dtype 不同。

**错误关键词**：`F0FFFF Non-FP8 inputs require identical dtypes`

**解决方案**：
```python
x_fp32 = pypto.cast(x_bf16, pypto.DT_FP32)
k = pypto.matmul(x_fp32, w_fp32, pypto.DT_FP32)
```

---

## 3. Golden 函数中 matmul 必须用 FP32 输入

**触发场景**：golden 函数使用 `torch.matmul(a_bf16, b_bf16).float()` 与 kernel 的 `pypto.matmul(bf16, bf16, DT_FP32)` 对比，精度校验失败。max_abs_diff 随 K 维度线性增长（K=4096 时达 ~0.96），影响 8+ 算子。

**错误关键词**：`PRECISION_FAIL` — max_abs_diff 随 K 线性增长（K=1536 时达 0.498）

**根因**：`torch_npu` 在 NPU 上执行 `torch.matmul(bf16, bf16)` 时的累加实现/tiling 策略与 `pypto.matmul(bf16, bf16, DT_FP32)` 的 Cube FP32 累加路径不一致（并非简单的 "BF16 累加 vs FP32 累加"，而是两套 matmul 实现的数值行为差异）。`.float()` 只是把已截断的 bf16 结果转 FP32 表示，无法恢复累加过程中丢失的精度。

**实测数据**：
- `torch.matmul(bf16, bf16).float()` vs `pypto.matmul(bf16, bf16, DT_FP32)` → max_abs_diff=0.498（K=1536）
- `torch.matmul(a.float(), b.float())` vs `pypto.matmul(bf16, bf16, DT_FP32)` → max_abs_diff=0.000107（几乎一致）

**解决方案**：golden 函数中 matmul 输入必须 `.float()` 后再计算，确保 FP32 累加与 `pypto.matmul` 对齐。

```python
# ❌ torch_npu NPU matmul 累加路径与 pypto Cube 不一致，精度已丢
golden = torch.matmul(a_bf16, b_bf16).float()

# ✅ FP32 输入，与 pypto.matmul Cube FP32 累加路径对齐
golden = torch.matmul(a_bf16.float(), b_bf16.float())
```

---

## 4. n_tiles 边界精度问题

**触发场景**：cube tile 的 `n_tiles` 不能整除输出列数，最后一段 tile 边界累积顺序不同于 golden。

**错误关键词**：`PRECISION_FAIL` — `0.195% mismatch (max diff 20.86)` / `0.68% mismatch (max diff 277.45)` / `tile boundary issue at the last nL1 iteration`（最后一行/列，cols 6496+ / 7056+）

**解决方案**：`n_tiles` 设为输出列数的干净约数：
```python
# N=7168: n_tiles=[112,224] → 边界偏差
n_tiles = [128, 256]    # 7168/256=28，整除
```

---

## 5. K=1 matmul 与 cube tile 存在根本性不兼容

**触发场景**：梯度计算或稀疏 attention 中，存在 K 维度为 1 的 matmul
（如 `P^T @ dO` 在 sparse_attention_grad_tnd 中 K=1）。cube tile 要求
kL0 整除 16，当 K=1 时：
- `cube_tile=[16,16]` → kL0=16 超出矩阵实际 K 维度 → CCU crash
- `cube_tile=[1,1]` → kL0=1 不满足 16 对齐 → FC4001

**错误关键词**：`aicore: CCU crash` / `FC4001 ERR_CONFIG_ALIGNMENT` / K=1 matmul 无合法 cube tile 配置

**解决方案**：用 vec 级逐元素操作（mul+sum）替代 matmul，避免 cube tile 约束：

```python
# ❌ K=1 时不可用
result = pypto.matmul(A, B, out_dtype)

# ✅ 改用 vec 操作
product = pypto.mul(A, B)
result = pypto.sum(product, dim=1)
```

若算子中有多个 matmul 且仅个别 K=1，可对 K=1 的 matmul 单独使用 vec 路径，
其余正常的 matmul 仍用 cube tile。

| 触发条件 | 错误关键词 |
|---------|----------|
| matmul 的 K 维度 = 1 | `aicore: CCU crash` / `FC4001` / `K=1 matmul incompatible` |
| gradient matmul 或 sparse mask 导致窄维度 | 同上 + `cube tile [16,16] causes crash, [1,1] violates alignment` |
