# PyPTO 算子设计速查

> 本文档聚焦**约束和决策点**，用于设计阶段的快速检查。
> API 用法细节请查阅 `docs/` 目录。

---

## 0. 术语澄清：tile shape vs view shape

PyPTO 里有两个独立但常被混淆的概念，进入 §2 Tiling 之前先钉清：

| 概念 | 主体 | 形态 | 决定的事 |
|---|---|---|---|
| **view shape** | 用户写 `pypto.view(t, view_shape, offsets)` 第二参数 | concrete Python int list | 每次 loop 迭代处理的子张量大小；用户级数据切片 |
| **tile shape** | 用户写 `pypto.set_vec_tile_shapes(...)` / `pypto.set_cube_tile_shapes(...)` | concrete Python int(s) | pass 阶段每个 operation 拆出的 tile operation 颗粒 |

**Subgraph 是 tile shape 决定的，不是反过来**。pass 阶段流程：

1. 用户写完整个 pypto kernel
2. 框架根据 tile shape 把每个 operation 拆成多条 tile operation（每条处理一个 tile 大小的数据）
3. 框架尝试把多条 tile operation 组合成 **subgraph**：`GM → UB/L1 → 计算 → UB/L1 → GM` 一个完整往返；组合的硬约束是 subgraph 总 UB 占用 ≤ UB 容量
4. 不同 subgraph 实例并行分发到不同核

**关键**：在写 kernel 时无法预知一个 subgraph 包含多少 operation。因此 UB 估算只能按**单个 operation 的 UB 占用**估，让 tile 足够小，框架自然能把多条 tile op 组进同一个 subgraph。

**UB 估算公式（per operation）**：

```
tile_size × dtype_bytes × tensor_count ≤ UB 容量
```

`tensor_count` 取决于 operation 类型：

| Op 类型 | tensor_count | 说明 |
|---|---|---|
| unary | 2 | 1 输入 + 1 输出 |
| binary | 3 | 2 输入 + 1 输出 |
| reduce / expand | 4（保守值） | 涉及形状变换，活 buffer 较多 |

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

### 硬约束 (PyPTO syntax — 违反则 compile/runtime fail)

| 约束 | 说明 |
|------|------|
| 尾轴 32B 对齐 | FP32: 8 元素, BF16/FP16: 16 元素, 当尾轴实际长度小于32B时可以按32B配置 |
| TileShape 维度数 = 输出 tensor 维度数 | 最多 4 维 |
| 单 op UB：`TileSize × dtype_bytes × tensor_count ≤ UB 容量`（unary=2, binary=3, reduce/expand 保守 4） | 超出导致 spill |
| (TensorShape / TileShape) × (1 + input_count) ≤ 18000 | 超出导致编译失败 |
| cube 配置独立于 vec | matmul 前必须单独调用 set_cube_tile_shapes |
| `matmul` ND 输入最后一维 ≤ 65535 | A、B 两个输入都会检查；检查的是传入 `pypto.matmul` 时的实际 shape 最后一维（`shape[-1]`），不是公式里的 K。若 K 很大，先通过 A/B 布局、`pypto.view` shape 或 `a_trans` / `b_trans`，避免 K 落在任一输入的最后一维上 |
| **cube tile dim 关系**: `mL0 ≤ mL1`, `kL0 ≤ kL1`, `nL0 ≤ nL1` | 违反则编译失败 |
| **cube tile 16 元素 alignment** (BF16/FP16 场合) | 违反则编译失败 |
| **broadcast 1-轴 rule** | PyTorch 中允许多轴同时 expand，但 PyPTO 仅支持单轴 |
| **vec tile per-stage rule** | 若有多个不同 shape class 的 vec op，需在每个 stage 前**重新 set** `set_vec_tile_shapes`. 单一 tile 仅适用于 shape class 仅 1 种的情况. (与 cube tile 的 per-matmul 规则对称) |
| Stage 7 前 Tile shape 基线 | `set_cube_tile_shapes([128, 128], [128, 128], [128, 128])`；vec tile 按常规默认规则 |

### Stage 7 前 Tile shape 基线

Stage 7 前不做 cube 性能调优型分支，vec tile 保持常规默认策略：

| 类型 | 基线 | 说明 |
|---|---|---|
| Cube / matmul | `[128, 128], [128, 128], [128, 128]` | 统一首跑基线，避免 16/32 等过小 tile 造成编译展开爆炸 |
| Vector | 每轴通常在 `[16, 64]` | 首选偏小值（16/32）；按 alignment、UB 与 rank 约束调整 |
| 例外 | 最近的合法稳定值 | 仅当 API/shape 硬约束不允许默认值时使用，并在 DESIGN.md §3.2.2 写明原因 |

### Tiling 推导步骤（Stage 7 前）

1. 判断计算类型，并按 Tile shape 基线选择配置；混合算子在对应操作前切换配置。
2. Cube / matmul 统一使用 128 基线，不按训练、decode、核利用率做性能化分支。
3. Vec tile 每轴通常取 `[16, 64]` 范围内的稳定值；首选偏小值，必要时按 API/shape/UB 约束调整并记录原因。
4. 验证 per-op UB/L1 占用、尾块 `valid_shape` 和展开约束 `(shape / tile) × tensor_count ≤ 18000`。
5. 只有 API/shape 硬约束不允许默认值时，才采用最近的合法值并记录原因；性能调优留到 Stage 7。

### Tile shape anti-patterns (避免)

| Anti-pattern | 为什么坏 | 正确做法 |
|--------------|---------|---------|
| **尾轴 tile 不是 alignment 倍数** (e.g. FP32 last dim = 7 / 10 / 17) | 编译报错或运行时性能极差 | round up 到对齐倍数 (FP32: 8/16/24/32...; BF16: 16/32/48/64...) |
| **vec tile 单轴 > 64** | architect 阶段不要做性能优化；大 tile 单 op UB 占用大，subgraph 易撑爆 UB | 默认 [16, 64] 范围；Stage 7 optimizer 在 profiling 后再上调（cube tile 不在此条约束内，Stage 7 前按 Tile shape 基线） |
| **vec tile 全部按 tensor.shape 取（单 op UB 占用 = 整 tensor 字节数）** | 单 op UB 占用 = 整 tensor 字节数，撑爆 UB | vec tile 各轴 ∈ [16, 64]；single-op UB 自然 fit |
| **多个 shape class 的 vec op 用单一 tile 处理** (e.g. `(1,1,BT,K)` setting 同时处理 `[1,1,BT,BT]` 和 `[1,1,K,V]` 的 op) | 大部分 op 的 partitioning 错位，UB / 并行度恶化 | 按 shape class per-stage 重新 set `set_vec_tile_shapes` |

### 设计参考: Vec tile 推荐 (architect 默认值，按 op 类型分类)

| Op 类型 | 推荐 vec tile | 注意 |
|---|---|---|
| Pointwise (4D, 单 tensor) | `(1, 16, 16, 64)` 或 `(1, 16, 32, 32)` | 每轴 ∈ [16, 64]；batch 维度因 view 切分为 1 |
| Pointwise (2D, 单 tensor) | `(32, 32)` 或 `(16, 64)` | 每轴 ∈ [16, 64] |
| Reduction (sum/max along last) | `(16, 32)` 或 `(16, 64)` | reduce 轴用 tile 切分，编译器自动累加 |
| Broadcast multiply | broadcast 轴用 size-1 明示 (e.g. `(1, 1, 16, 1)` for K-side broadcast) | broadcast 轴用 size-1 明示，禁止隐式 expand |
| Mixed cube + vec | 两者都 set (cube 在 matmul 前统一 128；vec 在 vec op 前按 [16, 64] 范围) | 应用 per-stage tile 作用域，性能调优留到 Stage 7 |

### 2.4 动态轴模式

**触发场景**：有Agent开发的PyPTO算子必须支持动态轴，但部分计算 API在编译期需要 concrete shape，不接受含 `DYNAMIC` 维度的 tensor（报错特征：`dim = -1`、`Cannot convert symbols to int`、`has invalid shape value: -1`）。

**Production kernel 必须同时具备 4 要素**（缺一就 production unusable，参见 `pypto-general-debug/references/jit-signature.md` §9.13）：

| # | 要素 | 示例 |
|---|------|------|
| 1 | Annotation 标 `pypto.DYNAMIC` literal | `pypto.Tensor([pypto.DYNAMIC, 16, 64], dtype)` |
| 2 | Tile config | `pypto.set_vec_tile_shapes(1, 16, 64)` |
| 3 | 沿动态轴 loop | `for b in pypto.loop(B, name=..., unroll_list=[1])`（Stage 6 之前单一值，默认 `[1]`） |
| 4 | loop 内 `pypto.view` 切 concrete tile（**单次 view 4 KB 量级即可，不要 view 大块**） | `pypto.view(x, [1, 16, 64], [b, 0, 0])` |

**反模式（必须避免）**：
- ❌ 空 `pypto.Tensor()` / `pypto.Tensor([])` 注解 → 看似能编译通过但 production crash
- ❌ DYNAMIC tensor 直接喂 compute API（无 `pypto.view`）→ workspace estimator INT32 overflow

**通用策略 —— "loop 切 tile，API 只吃静态"**：
1. **选合适的轴做动态轴**：优先 batch / 序列长度等语义上天然变化的轴；所选轴**不能是 API 直接计算依赖的维度**（matmul 不选 K/N，归约不选归约 dim）。
2. **所选轴标 `DYNAMIC`，其余标 `STATIC`**；多个动态轴对每个分别走 `pypto.loop` 嵌套处理。
3. **`pypto.loop` 沿动态轴迭代**，trip count 取 `tensor.shape[i]` 或符号表达式。
4. **`pypto.view` 切出固定整数 tile**（shape 参数全是 Python int）。
5. **受限 API 只操作静态 tile**，永不看到动态维度。
6. **`pypto.assemble` 写回**，offset 可用 SymbolicScalar。

**多动态轴特例**（Batch + SeqLen 同时动态，不能在高维 tensor 上直接调受限 API）：

1. **wrapper 层 reshape 到 2D**：`[B, N, S, D]` → `[B*N*S, D]`
2. **嵌套 loop 拆解各维度**：`loop(B) → loop(N) → loop(S // S_TILE)`
3. **view 使用 concrete tile shapes**：`pypto.view(t, [S_TILE, D], [offset, 0], valid_shape=[actual_s, D])`
4. **API 操作静态 tile**：`[S_TILE, D]`，编译期 shape 完全确定
5. **assemble 写回**：`pypto.assemble(result, [offset, 0], output_2d)`

**参考实现**：用 skill `pypto-docs-search` 搜索 attention 类算子范本（如 glm_attention 之类）

**关键约束**：
- `pypto.view` 的 `shape` 参数**只接受 Python int**，SymbolicScalar 只能用在 `offsets` 和 `valid_shape` 中
- Python 切片 `tensor[sym:sym+1]` 同样不能用 SymbolicScalar 索引
- `inplace=True` reshape 不能用在 kernel 输出参数上（产生静默 NaN）

**循环内累加标准模式**：

```python
acc = pypto.tensor([TILE, D], pypto.DT_FP32, "acc")
for idx in pypto.loop(n, name="LOOP", idx_name="idx", unroll_list=[1]):  # Stage 6 之前单一值
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
         ├── 轴范围大且变化？ → 在**最内层** pypto.loop 用 loop_unroll + unroll_list（外层 loop 禁止加 unroll_list）；**Stage 6 之前 unroll_list 只写单一值（默认 `[1]`），多值调优留到 Stage 7**
         └── 尾块不对齐？ → view 中使用 valid_shape
```

### Loop 关键约束

| 约束 | 说明 |
|------|------|
| 静态轴用 Python for | 避免不必要的 loop 编译开销 |
| 动态轴用 pypto.loop | 编译器需要知道这是运行时循环 |
| loop 索引是符号值 | 不能用于 Python list 索引或 if 判断 |
| 跨迭代依赖 → submit_before_loop=True | 否则迭代间数据不可见 |
| unroll_list 仅放在**最内层 pypto.loop** | 外层 loop 加 unroll_list 会触发编译路径爆炸或寄存器拷贝 pass 引起的精度异常 |
| unroll_list 在 **Stage 6 之前只含单一值**（默认 `[1]`） | 多值会触发编译路径爆炸、拖慢编译并使开发超时；多值展开调优仅允许在 Stage 7 optimization。有依据时可用其它单一值并在 §4 记录理由 |
| unroll_list 必须覆盖所有可能迭代数 | 否则运行时 assert（此约束在 Stage 7 多值调优时才需考虑；Stage 6 之前固定单值 `[1]` 不受影响） |

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
| TileShape 过小 | 表达式膨胀 > 编译超时 | Stage 7 前回到 Tile shape 基线；不要用 16/32 等过小 cube tile 首跑 |
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

当设计阶段需要查证 API 能力时，用 `pypto-docs-search` 按以下优先级搜索：
1. API 文档（`docs/api/`）— API 签名和约束（权威）
2. 教程（`docs/tutorials/`）— 用法示例和常见模式
3. 算子参考实现— 真实算子实现参考
