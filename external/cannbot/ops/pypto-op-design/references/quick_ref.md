# PyPTO 算子设计速查

> 本文档聚焦**约束和决策点**，用于设计阶段的快速检查。
> API 用法细节请查阅 `pypto/docs/zh/` 目录。

---

## 1. API 精度约束

| API | dtype 限制 | 不满足时的后果 |
|-----|-----------|---------------|
| `pypto.sum` | 仅支持 FP32 输入 | 运行时错误 |
| `pypto.matmul` | 两个操作数 dtype 必须一致 | 编译失败 |
| `pypto.amax` | FP16, BF16, FP32 | BF16 精度可能不足 |
| `pypto.exp` | FP16, BF16, FP32 | 不支持整型 |

**设计影响**：若计算链中包含 `sum`，必须在 `sum` 前插入 `cast(x, DT_FP32)`，并规划 cast 回目标 dtype 的位置。

---

## 2. Tiling 约束

### 硬约束

| 约束 | 说明 |
|------|------|
| 尾轴 32B 对齐 | FP32: 8 元素, BF16/FP16: 16 元素, 当尾轴实际长度小于32B时可以按32B配置 |
| TileShape 维度数 = 输出 tensor 维度数 | 最多 4 维 |
| TileSize × 驻留 tile 数 × dtype_bytes ≤ UB 容量 | 超出导致 spill |
| (TensorShape / TileShape) × (1 + input_count) ≤ 18000 | 超出导致编译失败 |
| cube 配置独立于 vec | matmul 前必须单独调用 set_cube_tile_shapes |

### Tiling 推导步骤

1. 确定尾轴对齐值：`align = 32 / dtype_bytes`（fp16/bf16=16, fp32=8）
2. 尾轴 tile = min(尾轴长度, 最大对齐倍数 ≤ UB 容量允许范围)
3. 其余维度从高维开始，在 UB 容量内尽量取大值
4. 验证展开约束 `(shape / tile) × tensor_count ≤ 18000`
5. 如果有 matmul，额外推导 cube tile：`[M, K], [K, N], [M, N]`

### 2.4 动态轴模式

**触发场景**：有Agent开发的PyPTO算子必须支持动态轴，但部分计算 API在编译期需要 concrete shape，不接受含 `DYNAMIC` 维度的 tensor（报错特征：`dim = -1`、`Cannot convert symbols to int`、`has invalid shape value: -1`）。

**通用策略 —— "loop 切 tile，API 只吃静态"**：
1. **选合适的轴做动态轴**：优先 batch / 序列长度等语义上天然变化的轴；所选轴**不能是 API 直接计算依赖的维度**（matmul 不选 K/N，归约不选归约 dim）。
2. **所选轴标 `DYNAMIC`，其余标 `STATIC`**；多个动态轴对每个分别走 `pypto.loop` 嵌套处理。
3. **`pypto.loop` 沿动态轴迭代**，trip count 取 `tensor.shape[i]` 或符号表达式。
4. **`pypto.view` 切出固定整数 tile**（shape 参数全是 Python int）。
5. **受限 API 只操作静态 tile**，永不看到动态维度。
6. **`pypto.assemble` 写回**，offset 可用 SymbolicScalar，尾块用 `valid_shape`。

**多动态轴特例**（Batch + SeqLen 同时动态，不能在高维 tensor 上直接调受限 API）：

1. **wrapper 层 reshape 到 2D**：`[B, N, S, D]` → `[B*N*S, D]`
2. **嵌套 loop 拆解各维度**：`loop(B) → loop(N) → loop(S // S_TILE)`
3. **view 使用 concrete tile shapes**：`pypto.view(t, [S_TILE, D], [offset, 0], valid_shape=[actual_s, D])`
4. **API 操作静态 tile**：`[S_TILE, D]`，编译期 shape 完全确定
5. **assemble 写回**：`pypto.assemble(result, [offset, 0], output_2d)`

**参考实现**：`models/glm_v4_5/glm_attention.py`

**关键约束**：
- `pypto.view` 的 `shape` 参数**只接受 Python int**，SymbolicScalar 只能用在 `offsets` 和 `valid_shape` 中
- Python 切片 `tensor[sym:sym+1]` 同样不能用 SymbolicScalar 索引
- `inplace=True` reshape 不能用在 kernel 输出参数上（产生静默 NaN）

**循环内累加标准模式**：

```python
acc = pypto.tensor([TILE, D], pypto.DT_FP32, "acc")
for idx in pypto.loop(n, name="LOOP", idx_name="idx", unroll_list=[4, 2, 1]):
    tile = compute(...)
    if pypto.is_loop_begin(idx):
        acc[:] = tile          # 首次：初始化
    else:
        acc[:] = acc + tile    # 后续：累加
    if pypto.is_loop_end(idx):
        pypto.assemble(pypto.cast(acc, pypto.DT_BF16), [offset, 0], output)
```

**梯度算子两趟模式**：当多个输出在不同维度累加时（如 dQ 沿 S2、dK/dV 沿 S1），用两趟分别计算，避免跨 loop 依赖。

---

## 3. Loop 决策树

```
动态轴？
├── 否 → 不需要 loop，compiler 自动 tile
└── 是 → 需要 pypto.loop
         ├── 跨迭代有依赖？ → submit_before_loop=True
         ├── 轴范围大且变化？ → 使用 loop_unroll + unroll_list
         └── 尾块不对齐？ → view 中使用 valid_shape
```

### Loop 关键约束

| 约束 | 说明 |
|------|------|
| 静态轴用 Python for | 避免不必要的 loop 编译开销 |
| 动态轴用 pypto.loop | 编译器需要知道这是运行时循环 |
| loop 索引是符号值 | 不能用于 Python list 索引或 if 判断 |
| 跨迭代依赖 → submit_before_loop=True | 否则迭代间数据不可见 |
| unroll_list 必须覆盖所有可能迭代数 | 否则运行时 assert |

---

## 4. 数据搬运约束

| 操作 | 用途 | 关键约束 |
|------|------|----------|
| `view(tensor, shape, offset)` | 只读子视图 | 不能对同一 tensor 同时 view 和 assemble |
| `assemble(result, offset, output)` | 写入子区域 | 与 view 不能作用于同一 tensor |
| `output[:] = result` | 整体写回 | 最简单，shape 必须匹配 |
| `output[a:b] = result` | 切片写回 | 要求 result shape 与切片 shape 一致 |

**DAG 无环约束**：同一个 tensor 不能在同一个 JIT 图中既被 view 读取又被 assemble 写入，否则形成环路导致编译失败。

---

## 5. 约束冲突速查表

| 冲突场景 | 表现 | 解法 |
|----------|------|------|
| sum 要求 FP32 + 输入为 BF16 | dtype 不匹配 | sum 前 cast，sum 后 cast 回 |
| matmul 两侧 dtype 不一致 | 编译失败 | 统一 cast 到相同 dtype |
| 动态轴参与受限 API（matmul/sum/amax/mean/prod 等） | `dim = -1`、`Cannot convert symbols to int` | 选语义合适的轴做 DYNAMIC + loop；受限 API 只操作静态 tile（见 §2.4）|
| view + assemble 同一 tensor | DAG 环路 | 使用两个独立 tensor 或拆分 JIT 函数 |
| TileShape 过小 | 表达式膨胀 > 编译超时 | 增大 tile，减少循环次数 |
| TileShape 过大 | UB/L1 溢出 | 缩小 tile，增加循环次数 |
| Cube + Vec 混合计算 | 不同阶段需不同 tiling | 在计算阶段间切换 tile 配置 |

---

## 6. 标准精度路由模式

```text
输入 BF16/FP16
  → cast to FP32（在 sum/reduce 等精度敏感操作前）
  → FP32 计算链
  → cast 回原 dtype（在输出前）
```

决策要点：
- `pypto.sum` 强制要求 FP32 输入
- 累加器（跨 loop 迭代的状态 tensor）应使用 FP32
- matmul 可通过 `out_dtype=pypto.DT_FP32` 直接输出 FP32

---

## 7. 搜索优先级

当设计阶段需要查证 API 能力时：
1. `pypto/docs/zh/api/` — API 签名和约束（权威）
2. `pypto/docs/zh/tutorials/` — 用法示例和常见模式
3. `models/` — 真实算子实现参考
