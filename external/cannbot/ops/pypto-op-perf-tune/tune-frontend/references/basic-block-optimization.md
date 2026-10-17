#### 基本块优化

对算子中所有 operation 逐个审查 shape 与基本块（TileShape）设置，识别维度不匹配、设置不合理等问题，产出审查表格后再进行优化。

##### 1. 逐 Operation Shape 与基本块审查

**分析方法**：
1. 逐行阅读算子 kernel 代码，记录每个 operation（matmul、cast、reshape、view、concat、add、mul、sum 等）
2. 统计每个 operation 的：输入 shape、输出 shape、TileShape 设置（cube_tile / vec_tile）
3. 检查每个 TileShape 是否与实际 shape 匹配、是否合理
4. 汇总问题清单，按优先级排序

**审查表格模板**：

| # | Operation | 输入 Shape | 输出 Shape | TileShape 设置 | 合理? | 问题与建议 |
|---|-----------|-----------|-----------|---------------|------|-----------|
| 1 | `matmul(A, B, b_trans=True)` | A:`[1,4096]` B:`[6144,4096]` → `[1,6144]` FP32 | `cube([16,16],[64,256],[256,256])` | ⚠️ | M=1, K 轴未驻留，建议 `[16,16],[128,4096,256],[256,256]` |

**关键检查项**（对每个 operation 逐一检查）：
1. **TileShape 维度与 Shape 维度是否匹配**：vec_tile_shapes 的每个维度不应超过对应 tensor 维度
2. **Cube TileShape 对齐约束**：L0 各维度必须 16 元素对齐（BF16/FP16 场景）
3. **L1 是否超过实际轴长**：mL1/kL1/nL1 超过对应维度实际大小是无意义的
4. **大轴的 L1 是否过小**：导致过多的重复载入次数
5. **Reduce 轴是否被不必要地切分**：归约计算尽量不对归约轴切分
6. **多个 matmul 是否共用同一 TileShape**：不同 shape 的 matmul 必须独立配置
7. **reshape/view 前后 vec_tile 是否匹配对应阶段的 tensor shape**：reshape 前按源 shape 设，reshape 后按目标 shape 重设；特别关注 reshape 后紧跟 assemble 的场景

##### 2. Cube TileShape 设置规范

**函数原型**：
```python
pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kL1], [nL0, nL1])
# 高级用法：A/B 矩阵独立设置 K 轴切分
pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kAL1, kBL1], [nL0, nL1])
# K 轴分核
pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kL1], [nL0, nL1], enable_split_k=True)
```

**参数说明**：
- `mL0/mL1`：M 维度在 L0/L1 上的切分大小
- `kL0/kL1`：K 维度在 L0/L1 上的切分大小；三维 `[kL0, kAL1, kBL1]` 可分别设置 A/B 矩阵 K 轴切分
- `nL0/nL1`：N 维度在 L0/L1 上的切分大小
- `enable_split_k`：是否启用 K 轴分核，让不同核并行计算 K 轴的不同分块（默认不启用）

**对齐约束**（BF16/FP16 场景）：
- L0 各维度必须 16 元素对齐（即 L0_M, L0_K, L0_N 均为 16 的倍数）
- kL0, kL1, nL0, nL1 需满足 32 字节对齐
- `L0 <= L1` 且 `L1 % L0 == 0`

**训练场景（M/N 较大）推荐初始配置**（A2/A3 平台）：
```python
pypto.set_cube_tile_shapes([128, 128], [64, 256], [256, 256])
pypto.set_cube_tile_shapes([256, 256], [64, 256], [128, 128])
pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128])
```

优点：在满足 L0 Buffer 约束下达到较大算数强度，可开启 Double Buffer。

**推理/Decode 场景（M=1 或 M 较小）**：

M 轴较小时无法通过 M/N 切分提高算数强度，优化思路是**减少重复载入**：
- 使用 K 轴三维配置 `[kL0, kAL1, kBL1]`，让 A 矩阵一次搬入 L1 驻留
- 尽量增大 nL1，减少 N 轴切分次数

```python
# Decode 场景 matmul 示例：A=[1,K] × B^T=[N,K]
pypto.set_cube_tile_shapes([16, 16], [16, K, 256], [256, 256])
# A 矩阵 K 轴整体驻留 L1(kAL1=K)，B 矩阵 K 轴按 256 分批载入(kBL1=256)
```

**配置要点**：
1. **L1 不超过实际轴长**：mL1/kL1/nL1 不应超过对应维度实际大小，超过无意义且浪费 L1 空间
2. **小轴不切**：某轴本身较小（如 K=128），L0=L1=min(实际值, 最大对齐值)，不切分
3. **大轴大 tile**：某轴较大（如 N=2048 或 K=4096），用较大 L1 减少切分次数

**⚠️ 独立设置原则**：同一算子中多个不同 shape 的 matmul，**必须在每个 matmul 前分别调用 `set_cube_tile_shapes`**，不要用统一值。

**🔥 案例**：[多 Matmul 独立 TileShape 优化](../cases/per-matmul-tile-shapes.md)（3 个 matmul 独立设 tile，-46.1%）

##### 3. Vector TileShape 设置规范

**配置原则**：
1. 满足特定 Operation 对 TileShape 的规格约束
2. 保证 Operation 的输入与输出 Tensor 可以在 UB 中分配内存
3. TileShape 不能过大也不能过小（数据块大小在 16 到 64KB 之间）
4. **优先用满尾轴**，即尾轴 TileShape 设为与实际 Shape 尾轴相同；尾轴过大必须切分时，按 **512B 对齐**切分
5. 归约类计算尽可能不要在归约轴上进行切分

```python
pypto.set_vec_tile_shapes(64, 512)
```

**⚠️ 维度匹配检查**：vec_tile_shapes 每个维度的值不应超过对应 tensor 的实际维度大小。例如 tensor shape 为 `[1, 1024]`，不应设置 `vec(8, 128)`（第 1 维 8 > 实际 1）。

**归约轴切分问题**：
- ❌ 对 reduce 轴切分：多个子图的输出需要在同一个子图进行 reduce 操作，产生 GM 搬运和调度开销
- ✅ 不对 reduce 轴切分：上下游子图合并，没有额外开销

**⚠️ reshape 前后的 vec_tile_shapes 设置规则**：

`reshape` / `view` 操作会改变 tensor 的 shape，但 vec_tile_shapes 需要与当前操作的实际 tensor shape 匹配。因此必须遵循：

1. **reshape 前**：vec_tile_shapes 按源 tensor（reshape 前）的 shape 设置
2. **reshape 后**：在操作 reshape 后的 tensor 之前，重新设置 vec_tile_shapes 按目标 tensor（reshape 后）的 shape

```python
# ✅ 正确：reshape 前按源 shape 设，reshape 后按目标 shape 重新设
pypto.set_vec_tile_shapes(8, 128)          # 源 tensor [8, 128]
k_embed_3d = pypto.reshape(k_embed, [8, 1, 128])
pypto.set_vec_tile_shapes(8, 1, 128)      # 目标 tensor [8, 1, 128]
pypto.assemble(k_embed_3d, [0, pos, 0], cache)

# ❌ 错误：reshape 前就按目标 shape 设了 vec_tile
pypto.set_vec_tile_shapes(8, 1, 128)      # ❌ 此时 tensor 还是 [8, 128]
k_embed_3d = pypto.reshape(k_embed, [8, 1, 128])
pypto.assemble(k_embed_3d, [0, pos, 0], cache)

# ❌ 错误：reshape 后未重新设 vec_tile，沿用旧的设置
v_cur = pypto.reshape(v_bf16, [8, 1, 128])  # 源 [1, 1024] → 目标 [8, 1, 128]
# ❌ 缺少 set_vec_tile_shapes(8, 1, 128)
pypto.assemble(v_cur, [0, pos, 0], cache)
```

**常见遗漏场景**：
- reshape 后紧跟 `assemble`：assemble 消费的是目标 shape，必须在 reshape 后、assemble 前设置匹配目标 shape 的 vec_tile
- reshape 后紧跟 `matmul`：matmul 由 cube_tile 控制，vec_tile 影响较小，但仍建议按目标 shape 设置
- 多个连续 reshape：每次 reshape 后都需确认 vec_tile 是否匹配

**冗余配置检查**：
检查是否存在连续多次 `set_vec_tile_shapes` 调用且参数相同的情况（常见于 copy-paste 残留），合并为一次调用；同时确认每个 vec_tile_shapes 是否与当前 tensor shape 匹配，不匹配的及时修正。此检查应在阶段 A4（基本块审查表）的"冗余操作一并检查"中完成。

##### 4. 常见基本块问题检查清单

| # | 检查项 | 症状 | 修复方法 |
|---|--------|------|---------|
| 1 | vec_tile 维度超过 tensor 实际维度 | 运行时异常或性能异常 | vec_tile 每维 ≤ 对应 tensor 维度 |
| 2 | cube L1 超过实际轴长 | L1 空间浪费，其他轴可用空间减少 | L1 设为 min(推荐值, 实际轴长)，并满足对齐 |
| 3 | K 轴 L1 过小导致重复载入 | GM 搬运开销大（尤其 K≥4096 时） | 增大 kL1，或用 `[kL0, kAL1, kBL1]` 独立配置 |
| 4 | N 轴 L1 过小导致切分次数多 | 任务数过多，调度开销大 | 增大 nL1 减少切分 |
| 5 | 多个 matmul 共用同一 TileShape | 不同 shape 的 matmul 用统一 tile 导致次优 | 每个 matmul 前独立设置 |
| 6 | reduce 轴被切分 | 额外 GM 搬运和调度开销 | 归约轴 tile 设为全长 |
| 7 | Decode M=1 未利用 K 轴驻留 | A 矩阵重复载入 | 使用 `[kL0, kAL1, kBL1]` 让 A 矩阵驻留 L1 |
| 8 | reshape 前按目标 shape 设 vec_tile | reshape 操作使用了不匹配的 tile 配置 | reshape 前按源 tensor shape 设，reshape 后按目标 shape 重设 |
| 9 | reshape 后 assemble 前缺少 vec_tile | assemble 使用了不匹配的 tile 配置 | reshape 后、assemble 前显式设置匹配目标 shape 的 vec_tile |
