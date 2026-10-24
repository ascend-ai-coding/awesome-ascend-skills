### TileShape 深度调优

#### 1. Matmul TileShape 深度调优

Matmul 的 TileShape 深度调优需要根据 M、N、K 的实际大小和硬件规格综合选择策略。核心目标是：保证分满核的前提下，最大化算数强度 / L2 命中率，最小化重复载入。

**调优决策树**：

```

M、N 是否足够大（可同时分满 M、N 轴）？
├─ 是（训练场景，如 M=N=6144）
│   ├─ 目标 → Compute Bound（算数强度最大化）
│   ├─ TileShape → mL1=nL1=128~256，如 [128,128],[64,256],[256,256]
│   └─ 额外优化 → 分核布局调优提升 L2 命中率（→ §6）
│
├─ M 或 N 中有一维较小（推理/Decode 场景）
│   ├─ 目标 → Memory Bound（减少重复载入）
│   ├─ 策略 → 小维不切分（mL1=M 或 nL1=N），kAL1/kBL1 独立配置
│   └─ 示例 → [96,96],[64,1536,256],[128,128]（A 矩阵驻留 L1）
│
└─ M、N 均小，K 极大
    ├─ 目标 → 增加并行度
    ├─ 策略 → K 轴分核（enable_split_k=True）
    └─ 示例 → [128,128],[64,256],[128,128], enable_split_k=True

```

##### 1.1 减少重复载入

当 M、N 中有某个维度相对较小时，Matmul 切分前的算数强度较小，一般只能达到 Memory Bound。此时优化核心是：保证分满核（核使用率 ≥ 80%）的前提下，最小化 MTE2 重复载入量。

**方式一：增大 L1 减少切分次数**

```python
pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 256])

```

每一个 list 中前面的代表 L0 的切块大小，后面的代表 L1 的切块大小。L1 设大可以减少切分次数，从而减少重复载入。

**方式二：K 轴独立配置，A 矩阵驻留 L1（推荐）**

```python
pypto.set_cube_tile_shapes([mL0, mL1], [kL0, kAL1, kBL1], [nL0, nL1])

```

- `kAL1 = K`：A 矩阵的 K 轴整体搬入 L1 并驻留，消除 A 的重复搬运
- `kBL1 = 256`：B 矩阵的 K 轴按 256 分批载入

示例（M=96, K=1536, N=3072, FP16）：

```python
pypto.set_cube_tile_shapes([96, 96], [64, 1536, 256], [128, 128])

```

**理论依据**：MTE2 总载入量 = M·K·N/nL1·aByte + K·N·M/mL1·bByte。当 A 矩阵驻留 L1 时，载入量降为 M·K·aByte + K·N·bByte，消除 A 矩阵的重复搬运。

##### 1.2 K 轴分核

当 M、N 较小而 K 轴极大时，仅在 M、N 轴做分核可能无法用满核。此时将 K 轴切分到多核并行计算部分和再累加。提供两种方式：

**方式一：自动切 K（`enable_split_k=True`）**

框架自动将 K 轴切分到多核并行，使用简单，但部分和累加顺序不定导致**非确定性计算**，逐次运行结果可能存在微小差异。

```python
pypto.set_cube_tile_shapes([128, 128], [64, 256], [128, 128],
    enable_split_k=True)

```

**约束**：
- 仅支持 2 维矩阵（3 维/4 维不支持）
- 仅支持 out_dtype 为 DT_FP32 或 DT_INT32
- 不支持叠加 Bias、FixPipe 反量化

**方式二：手动切 K（前端 loop 实现）**

在前端手动编写 K 轴循环，显式控制分块加载和部分和累加，累加顺序确定，**不存在确定性计算问题**。

```python
c = pypto.Tensor([M, N], pypto.DT_FP32)
c.fill(0.0)
for kIdx in pypto.loop(K // kL1, name="LOOP_K", idx_name="kIdx"):
    a_block = pypto.view(a, [M, kL1], [0, kIdx * kL1])
    b_block = pypto.view(b, [kL1, N], [kIdx * kL1, 0])
    pypto.set_cube_tile_shapes([mL1, mL1], [64, kL1], [nL1, nL1])
    c_partial = pypto.matmul(a_block, b_block, out_dtype=pypto.DT_FP32)
    c = c + c_partial

```

**选择建议**：精度敏感场景优先使用方式二（手动切 K）；追求简便且接受微小精度波动的场景使用方式一（`enable_split_k=True`）。

##### 1.3 TileShape 选择策略速查

| Shape 特征 | 典型场景 | TileShape 策略 | 收益目标 |
|-----------|---------|---------------|---------|
| M,N,K 均大（>4096） | 大 Batch 训练 | [128,128],[64,256],[256,256] + 分核布局优化（见 §6） | Compute Bound + L2 调优 |
| M=N=6144 | 大 Shape Matmul | mL1=128,nL1=256,mDim=6,nDim=4 | 2.1ms→1.6ms（+31%） |
| M<256，N,K 大 | Decode/推理 | mL1=M, kAL1=K, kBL1=256, nL1≥128 | 消除 A 矩阵重复载入 |
| M=96,K=1536,N=3072 | 典型推理 | [96,96],[64,1536,256],[128,128] | A 矩阵驻留 L1 |
| M,N 小，K>4096 | 特殊 Shape | enable_split_k=True | K 轴多核并行 |
| M 极大，N 极小 | 小 Shape Matmul | 优先走 I-1（Vector 预处理），再按上述策略 | 见核内调优 |

#### 2. Vector TileShape 深度调优

**原则 1**：下游 Vector Operation 的 TileShape 尽可能使用上游 Operation 的输出 TileShape

```python
# 上下游 TileShape 对齐时，可以合并在一个子图
# Transpose TileShape: (64, 128)
# Add TileShape 应优先选择: (128, 64)

```

**原则 2**：根据泳道图上的子图大小和并行度调整

- 并行核数较少（<一半 Vector 核）：减小 TileShape
- 子图耗时短、调度开销占比较高：增大 TileShape

**原则 3**：调整相邻 Cube 和 Vector Operation 的 TileShape，使依赖更简单
