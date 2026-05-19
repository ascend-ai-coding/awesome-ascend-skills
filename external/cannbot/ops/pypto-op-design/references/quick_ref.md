# PyPTO 开发速查

> 详细信息通过搜索 docs/ 获取，本文件仅提供核心原则和约束。

---

## 1. Tiling 原则

### 1.1 算子类型判断

```
含 matmul/@ → Cube → set_cube_tile_shapes
仅逐元素/归约 → Vector → set_vec_tile_shapes
混合 → 两者都需要
```

### 1.2 HARD 约束

| 规则 | 说明 | 证据 |
|------|------|------|
| Vector TileShape | 每维 > 0，最多 4 维 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| Cube 必须设置 | matmul 前必须调用 set_cube_tile_shapes | `docs/api/config/pypto-set_cube_tile_shapes.md` |
| 尾轴 32B 对齐 | 尾轴需满足对齐要求，非尾轴无对齐要求 | `docs/tutorials/development/tiling.md` |

### 1.3 TileShape 设计要点

| 要点 | 说明 | 原因 |
|------|------|------|
| 尾轴 32B 对齐 | fp16/bf16 尾轴需为 16 的倍数，fp32 需为 8 的倍数 | 硬件 DMA 搬运单位为 32B，不对齐会导致编译失败或性能劣化 |
| TileShape 维度数 ≤ 输入维度数 | `set_vec_tile_shapes` 最多接受 4 维 | 超维数会触发编译错误 |
| 尾轴尽量取满 | 在 L0 容量允许范围内，尾轴取到对齐后最大值 | 尾轴越大，向量化效率越高，减少循环次数 |
| L1/L0 容量约束 | 所有同时驻留的 Tile 总大小不超过对应 buffer 容量 | 超容量触发 spill，严重劣化性能 |

---

## 2. Loop 原则

### 2.1 是否需要 Loop

按以下顺序逐条检查，命中即确定：

| # | 检查条件 | 结论 | Loop 类型 | 原因 |
|---|----------|------|-----------|------|
| 1 | 存在动态轴（运行时才知道大小） | 需要 Loop | `pypto.loop` | 编译期无法展开，必须用运行时循环遍历动态维度 |
| 2 | 多步骤分块计算（如 FlashAttention 分块累加） | 需要 Loop | `pypto.loop` 或 Python for | 数据量超单次 Tile 容量，需分块迭代并维护中间状态 |
| 3 | 动态轴范围跨度大（如 1~64k） | 需要 Loop | `pypto.loop_unroll` | 大范围动态轴用 `loop_unroll` 可在编译期生成多版本代码，兼顾灵活性与性能 |
| 4 | 所有轴编译期已知 & 单次 Tile 可处理 | **不需要** Loop | — | 编译器自动处理，无需手动循环 |

**默认推荐**：简单逐元素算子（如 sinh、relu、add）通常命中条件 4，不需要 Loop。

### 2.2 HARD 约束

| 规则 | 说明 | 证据 |
|------|------|------|
| 静态轴优先 Python for | pypto.loop 将静态轴展开增加编译复杂度 | `docs/tutorials/debug/performance.md` |
| 动态轴使用 pypto.loop | 并补齐边界控制 | `docs/tutorials/development/loops.md` |

### 2.3 标准写法

```python
# 静态轴 — Python for
for i in range(num_heads):
    head_i = pypto.view(x, [seq_len, head_dim], [0, i * head_dim])

# 动态轴 — pypto.loop
for i in pypto.loop(batch_size, name="LOOP_BATCH"):
    x_i = pypto.view(x, [seq_len, hidden], [i * seq_len, 0])
```

---

## 3. Runtime 硬约束

| 规则 | 说明 | 证据 |
|------|------|------|
| run_mode | 0=NPU，1=模拟器 | `docs/api/pypto-frontend-jit.md` |
| NPU 需 CANN | run_mode=0 时需 source CANN 环境 | `docs/install/prepare_environment.md` |

### 示例

```python
@pypto.frontend.jit(
    runtime_options={"run_mode": pypto.RunMode.NPU}
)
def kernel(...):
    ...
```

---

## 4. from_torch 约束

- **dtype**: FP16/BF16/FP32/INT8-64/BOOL
- **contiguous**: 必须连续（is_contiguous() == True）
- **证据**: `docs/api/others/pypto-from_torch.md`

---

## 5. 搜索优先级

```
docs/（官方文档）→ 最高优先级
models/（生产代码）→ 次优先级
examples/（示例）→ 参考优先级
```
