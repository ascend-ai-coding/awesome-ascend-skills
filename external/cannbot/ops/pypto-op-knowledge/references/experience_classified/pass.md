# PASS 组件经验

> 对应错误码范围：F4-F5XXXX

---

## 1. F40005 UB/L0B/L1 溢出 — 诊断与修复

**触发场景**：UB（196608 bytes）/ L0B（65536 bytes）/ L1（524288 bytes）内存超限。

> **内存阈值**：UB=192KB, L0A=64KB, L0B=64KB, L0C=128KB, L1=512KB。

**错误关键词**：
- `F40005 TENSOR_MEMORY_ALLOCATION: Alloc tensor [xxx] size [xxx] exceeds MEM_UB size [196608]`
- `F40005 TENSOR_MEMORY_ALLOCATION: Alloc tensor [xxx] size [xxx] exceeds MEM_L0B size [65536]`
- `F40005 TENSOR_MEMORY_ALLOCATION: tensor [xxx] size [xxx] exceeds MEM_L1 [524288]`

---

### 诊断三步法

**Step 1 — 区分内存池**：看报错是 `MEM_UB`（UB 溢出，192KB）还是 `MEM_L0B`（L0B 溢出，64KB）。

**Step 2 — 反算溢出 tensor**：

从报错 size 反推 tensor shape。公式：`报错字节数 ÷ dtype_bytes = 总元素数 → 分解出 shape`。

常见 dtype 字节数：FP32 = 4B，BF16/FP16 = 2B，INT8 = 1B。

```
示例 A（UB 溢出）:
  报错: Alloc tensor [26319] size [262144] exceeds MEM_UB size [196608]
  反算: 262144 ÷ 2(BF16) = 131072 = 128 × 1024 = 128 × M × 512 × 2
        → 溢出 tensor = (128, M, 512) BF16，来自 W_UK matmul 的中间结果
        → 128×512×2 = 128KB/head，prefill M≥2 时 256KB > 192KB

示例 B（L0B 溢出）:
  报错: Alloc tensor [26319] size [131072] exceeds MEM_L0B size [65536]
  反算: 131072 ÷ 4(FP32) = 32768 = 256 × 128
        → 溢出 tile = K_L0=256, N_L0=128，256×128×4=128KB > 64KB L0B
```

**Step 3 — 判断单 op vs 多 op**：

| 特征 | 类型 | 有效修复 |
|------|------|----------|
| 反算的 shape 直接对应到某个 matmul/view 的输出 | 单 op 中间结果溢出 | **必须改计算结构**（方向 4/5/6），stitch/unroll/tile 均无效 |
| 反算的 shape 是一组操作的累加，减小 unroll 后溢出消失 | 多 op 累加溢出 | 可调 unroll_list / vec tile（方向 1/2） |

> **关键认知**：stitch 和 unroll 拆的是 kernel **函数**的调度粒度，拆不了单次 matmul **内部**的中间 buffer。物理上限面前，参数调优是徒劳的。

---

### 修复方向

#### 方向 1：vec_tile_shapes 乘积超 UB

**触发条件**：`set_vec_tile_shapes(d1, d2, ...)` 所有维度乘积 × dtype_bytes > 196608。

```python
# ❌ 128×512×4(FP32) = 262144 > 196608 → 溢出
pypto.set_vec_tile_shapes(128, 512)
o = pypto.cast(x, pypto.DT_FP32)

# ✅ 32×512×4 = 65536 < 196608
pypto.set_vec_tile_shapes(32, 512)
o = pypto.cast(x, pypto.DT_FP32)
```

---

#### 方向 2：unroll_list 过大（仅多 op 累加有效）

**触发条件**：编译器为 unroll_list 每层分配完整 UB 缓冲区，多层叠加后总需求超 192KB。**⚠️ 仅当溢出是多个 op 累积的结果时有效**。若反算显示溢出来自单个 matmul 的中间结果，此方向无效，需用方向 4。

```python
# ❌ unroll 层数过多，多 op UB 需求叠加超限
unroll_list = [128, 64, 32, 16, 8, 4, 2, 1]

# ✅ 只保留必要层数
unroll_list = [4, 2, 1]
```

---

#### 方向 3：维度不对齐 → 正确设置 kL0/kL1 或 Padding

**触发条件**：cube tile 的 kL0/kL1 设置不满足约束（`kL0 % 16 == 0`, `kL1 % kL0 == 0`, `kL0 ≤ kL1`），报 `FC4000`/`FC4001`。

**常见错误**：将 `K/kL0` 的商误当 kL1。如 K=576, kL0=16 时，`576/16=36`，误设 `kL1=36`（36 % 16 ≠ 0 → FC4001）。正确做法是 `kL1=K=576`（576 % 16 = 0 ✓）。

**错误关键词**：`FC4001 ERR_CONFIG_ALIGNMENT` → `FC4000 ERR_CONFIG_TILE` → `F40005`

```python
# ❌ 误将 K/kL0=36 当 kL1 → 36 % 16 ≠ 0
pypto.set_cube_tile_shapes([16, 16], [16, 36], [16, 16])  # FC4001

# ❌ kL0 > kL1 → 违反 kL0 ≤ kL1
pypto.set_cube_tile_shapes([16, 16], [576, 512], [16, 16])  # FC4000

# ✅ kL0=16, kL1=K=576（576 % 16 = 0, 16 ≤ 576）
pypto.set_cube_tile_shapes([16, 16], [16, 576], [16, 16])
```

**仅当 K 确实无合法 tile 时才 padding**：如 K 为质数且 < 16，无法找到满足 `kL0 % 16 == 0` 且 `kL0 ≤ K` 的值，此时需 pad K 到 16 的倍数。

---

#### 方向 4：单 matmul 中间结果超 UB → Head 分组

**触发条件**：单次 matmul 的中间 tensor 乘积直接超 192KB。最常见：W_UK matmul 输出 `(128, M, 512) BF16`，每个 token 维度 M 增加 1 就多 128KB（128×512×2B）。decode 场景 M=1 时 128KB < 192KB 安全；prefill 场景 M≥2 时 256KB > 192KB 溢出。

**关键认知**：stitch 能拆 kernel 但不能拆单次 matmul 内部。`stitch_function_max_num`、`unroll_list`、`set_cube_tile_shapes` 对此均无效。必须改变计算分解方式。

```python
# ❌ 直接对 128 head 整体 matmul → M≥2 时中间结果 (128, M, 512) BF16 > 192KB
q_nope_hd = pypto.reshape(q_nope, [M, 128, 512])
w_uk_3d = pypto.reshape(w_uk, [128, 512, 512])
result = pypto.matmul(q_nope_hd, w_uk_3d, pypto.DT_FP32)
# M=2: (128, 2, 512) BF16 = 128×2×512×2 = 256KB > 192KB ← 溢出

# ✅ 128 head → 8 组 × 16 head，每组算完 assemble 到 DDR 释放 UB
N_HEADS_PER_GROUP = 16
N_GROUPS = 128 // N_HEADS_PER_GROUP
result_buf = pypto.tensor([M, 128, 512], pypto.DT_FP32, name="result_buf")
for g in range(N_GROUPS):
    head_start = g * N_HEADS_PER_GROUP
    q_group = pypto.view(q_nope_hd, [M, N_HEADS_PER_GROUP, 512], [0, head_start, 0])
    w_group = pypto.view(w_uk_3d, [N_HEADS_PER_GROUP, 512, 512], [head_start, 0, 0])
    g_result = pypto.matmul(q_group, w_group, pypto.DT_FP32)
    pypto.assemble(g_result, [0, head_start, 0], result_buf)
# 每组 (16, M, 512) BF16，M=8 时 16×8×512×2 = 128KB < 192KB
```

**分组数计算**：`N_PER_GROUP = 192KB ÷ (M_max × dim × sizeof)` 向下取 16 的倍数。例：M_max=8, dim=512, BF16 → 196608÷(8×512×2) = 24 → 取 16。

---

#### 方向 5：FP32 替代 INT8 量化路径 → UB 放大 4×

**触发条件**：golden 走 INT8 dequant pipeline（中间 buffer 1B/elem），实现时手动改为 FP32 量化（4B/elem）。同一 tensor 的 UB 需求翻 4 倍。

```python
# ❌ 手动 FP32 per-token quant，中间 buffer UB 需求 4×
# (128, M, 512) FP32 = 128×M×512×4 = M×256KB，M=2 → 512KB > 192KB
q_fp32 = pypto.cast(q_int8, pypto.DT_FP32)    # ← UB 放大 4×
w_fp32 = pypto.cast(w_int8, pypto.DT_FP32)
result = pypto.matmul(q_fp32, w_fp32, pypto.DT_FP32)

# ✅ 保持 INT8 dequant pipeline（与 golden 一致）
result_i32 = pypto.matmul(q_int8, w_int8, pypto.DT_INT32)
result = pypto.cast(result_i32, pypto.DT_FP16)   # dequant
```

---

#### 方向 6：Cube tile K_ext 一级过大 → L0B 溢出

**触发条件**：`set_cube_tile_shapes` 的 K_ext 一级吃下整个 K 维度，`K_L0 × N_L0 × sizeof > 65536`。如 H=2048 时 K_ext=[256, 1024]，256×1024×4=1024KB > 64KB。

```python
# ❌ H=2048 时 K_ext 未拆分，一级 256×1024×4 = 1024KB > 64KB
pypto.set_cube_tile_shapes([16, 16], [256, 1024], [16, 16])
# → F40005 TENSOR_MEMORY_ALLOCATION: exceeds MEM_L0B size [65536]

# ✅ K_ext 多级拆分，每级都在 L0B 内
# 参考 golden: H=7168 用 [256, 256], H=2048 用 [256, 256, 256, 256]
pypto.set_cube_tile_shapes([16, 16], [256, 256], [16, 16])
# 通过多级 cube tile 自然覆盖全部 K 维度
```

---

#### 方向 7：Cube tile K 维 L1 过大 → L1 溢出

**触发条件**：`set_cube_tile_shapes` 的 K 维 L1 设为完整 K 维度（如 K=7168），编译器为 matmul 分配的中间 buffer 超出 L1 预算（512KB）。

**错误关键词**：`F40005 TENSOR_MEMORY_ALLOCATION: tensor [xxx] size [xxx] exceeds MEM_L1 [524288]`

**与方向 6 的区分**：方向 6 是 L0B 溢出（64KB），本方向是 L1 溢出（512KB）。两者根因相同（K-tile 过大），但溢出层级不同。方向 6 的 K_ext 多级拆分也可解决 L1 溢出，但本方向提供更直接的 K_TILE 缩小方案。

```python
# ❌ K=7168 时 K-tile L1 设为完整 K 维度
# B tile = 16×7168×2B(BF16) = 224KB，但编译器中间 buffer 1.75MB > 512KB L1
pypto.set_cube_tile_shapes([16, 16], [16, 7168], [128, 128])
# → F40005 TENSOR_MEMORY_ALLOCATION: exceeds MEM_L1 [524288]

# ✅ K-tile L1 缩小至 256，框架自动多级 K-tile 迭代覆盖完整 K=7168
# 7168 / 256 = 28 次迭代（整除）
pypto.set_cube_tile_shapes([16, 16], [16, 256], [128, 128])
# B tile = 16×256×2B = 8KB，中间 buffer 在 L1 预算内
```

**K_TILE 安全阈值**：

| K 维度范围 | 推荐 K_TILE | 说明 |
|-----------|------------|------|
| K ≤ 1024 | K_TILE = K | 无需拆分 |
| 1024 < K ≤ 4096 | K_TILE ≤ 512 | 确保 B tile 在 L1 预算内 |
| K > 4096 | K_TILE ≤ 256 | 256 是安全默认值（参考多个已有实现） |

**约束检查**：K_TILE 必须满足 `K_TILE % 16 == 0` 且 `K % K_TILE == 0`（整除）。

| 触发条件 | 错误关键词 |
|---------|----------|
| vec_tile_shapes 乘积超 UB（192KB） | `F40005 TENSOR_MEMORY_ALLOCATION: exceeds MEM_UB` |
| 单 matmul 中间结果超 UB | 同上（方向 4：head 分组） |
| cube tile K_ext 一级超 L0B（64KB） | 同上（方向 6：K_ext 多级拆分） |
| K-tile L1 过大超 L1（512KB） | `F40005: exceeds MEM_L1`（方向 7） |
| FP32 替代 INT8 量化路径 → UB 放大 4× | `F40005: exceeds MEM_UB`（方向 5） |

---

## 2. 编译超时 / 卡死 — 诊断与修复

**触发场景**：JIT 编译在 Pass_27（SubgraphToFunction）、Pass_31（OoOSchedule）或 CODEGEN（make/bisheng）阶段超时（>300s）或挂起（永不完成），进程被 timeout kill（exit 124）或 compile monitor 报 F0F619。

**错误关键词**：
- `Pass_27_SubgraphToFunction` 后无输出（静默挂起）
- `Pass_31_OoOSchedule` 耗时 >600s 或进程被 timeout kill（exit 124）
- `[Compiler Monitor]` Stage Pass 单函数 >300s
- `F0F619 COMPILE_CODE_FAILED` + `make: *** [UnrollXX...o] Terminated` + `ret = 15`（unroll 导致 CODEGEN 阶段超时被 kill）

---

### 诊断两步法

**Step 1 — 确定卡在哪个阶段**：

| 症状 | 阶段 | 见方向 |
|------|------|--------|
| `make: *** Terminated` + make 目标含 `UnrollXX` + `ret = 15` | CODEGEN（bisheng 编译器） | 方向 2（表现 B） |
| `Pass_27_SubgraphToFunction/` 目录最后写入，进程无输出 | Pass_27 挂起 | 方向 2（表现 A） |
| `[Compiler Monitor]` Stage Pass 单函数 >300s | Pass 阶段编译过慢 | 方向 2（表现 C） |
| 进程到 `timeout` 无错误码，无 stage 信息 | Pass_31 超时 | 方向 1/3 |

**Step 2 — 对比 golden**：

golden 真实算子实现版（用 `pypto-docs-search` 搜索算子参考实现）通常使用：递进 unroll（`[N, N/2, ..., 1]`）、batch matmul（3D cube tensor）、大 vec tile（`(16, 512)` 等）、默认 `stitch_function_max_num`。任何偏离这些的配置都可能触发编译超时。

---

### 修复方向

#### 方向 1：Python for-loop IR 爆炸 → Pass_31 超时

**触发条件**：JIT kernel 内使用 Python `for h in range(N)` 而非 `pypto.loop`。每个 iteration 生成一份**完整的独立 IR 副本**（含 matmul + quantize + hadamard 全套），Pass_31 需要同时调度 N 份几乎相同的子图。编译时间不是 N 倍而是 ~N× 指数级。

```python
# ❌ Python for-loop — 128 head 各生成独立 IR 副本
# Pass_31 需调度 128× IR → >600s 超时
for h in range(128):
    q_head = pypto.view(q_nope_hd_bt, [M, 1, 512], [0, h, 0])
    q_head_2d = pypto.reshape(q_head, [M, 512])
    w_uk_h = pypto.view(w_uk_tile, [1, 512, 512], [h, 0, 0])
    w_uk_h_2d = pypto.reshape(w_uk_h, [512, 512])
    q_nope_h = pypto.matmul(q_head_2d, w_uk_h_2d, pypto.DT_FP32)
    # + quantize + hadamard per head

# ✅ batch matmul — 3D cube tensor 一次性处理全部 head
# golden 版正确做法
q_nope_3d = pypto.reshape(q_nope, [M, 128, 512])
w_uk_3d = pypto.reshape(w_uk, [128, 512, 512])
result = pypto.matmul(q_nope_3d, w_uk_3d, pypto.DT_FP32)
# 单次 matmul，单份 IR，Pass_31 无需处理 128 份子图
```

---

#### 方向 2：unroll_list → 编译超时/挂起

**根因**：`unroll_list` 控制 loop body 的展开层数和粒度，展开越激进，编译器需要生成的代码量越大。三种表现，一种根因。

| 表现 | 症状 | 机制 | 修复 |
|------|------|------|------|
| A：Pass_27 挂起 | `unroll_list=[128,1]` 单级跳跃，Pass_27 一次性处理 128× IR → ~8× 于递进级联，>600s | 编译器在 SubgraphToFunction 中一次性展开全部层级 | 递进级联：`[128,64,32,16,8,4,2,1]` |
| B：CODEGEN 超时 | make 目标 `TENSOR_M_Unroll16_PATH0` >900s，F0F619 ret=15 | unroll 生成展开函数（含完整 loop body），body 内 op 多时 bisheng 编译超时 | 移除 `unroll_list`，用普通 `pypto.loop` + `valid_shape` |
| C：Pass 膨胀 | `unroll_list=[4,2]` + `vec_tile(1,512)`，Compiler Monitor 单函数 394s | unroll 多层 + tile 过小 → tile 数量海量 → Pass 编译指数增长 | 减少 unroll 层 + 增大 vec tile 第一维 |

```python
# 表现 A：单级跳跃 → Pass_27 挂起
# ❌ unroll_list=[128, 1]
for bo in pypto.loop(t, name="ML", unroll_list=[128, 1]):
    # 4 matmul + 2 RMSNorm + 2 RoPE

# ✅ 递进级联 — golden 版
for bo in pypto.loop(t, name="ML", unroll_list=[128, 64, 32, 16, 8, 4, 2, 1]):


# 表现 B：unroll 生成函数 bisheng 编译超时
# ❌ unroll_list=[16]，body 含 3 matmul + 30 vec + 2 INT8 链
# make 目标: TENSOR_M_Unroll16_PATH0_hiddenfunc0 → >900s
for m in pypto.loop(B, name="M", unroll_list=[16]):
    # 大量 matmul + vec ops + INT8 量化

# ✅ 移除 unroll
for m in pypto.loop(B, name="M"):
    # 普通 dynamic loop，valid_shape 处理 tail


# 表现 C：unroll + 小 tile → Pass 膨胀
# ❌ unroll=[4,2] + vec_tile(1,512)，单函数 128 ops → Pass 394s
for q_idx in pypto.loop(b_s1, name="LOOP_query", unroll_list=[4, 2]):
    pypto.set_vec_tile_shapes(1, 512)

# ✅ unroll=[2,1] + vec_tile(16,512)，编译 34s — 参考 golden
for q_idx in pypto.loop(b_s1, name="LOOP_query", unroll_list=[2, 1]):
    pypto.set_vec_tile_shapes(16, 512)
```

> **原则**：unroll 层数越多、单层值越大、vec tile 第一维越接近 1，编译越慢。对比 golden：递进级联、大 vec tile、单层或低层 unroll。

**注意**：移除 unroll 后精度可能变化，需验证；功能上通过 `valid_shape` 参数等效处理 tail。

| 触发条件 | 错误关键词 |
|---------|----------|
| Python `for` 展开 N 份 IR → Pass_31 超时 | `exit 124 timeout` / `Pass_31` >600s |
| `unroll_list` 单级跳跃 → Pass_27 挂起 | `Pass_27_SubgraphToFunction` 后无输出 |
| `unroll_list` 过大 → CODEGEN 超时 | `F0F619` + `make: *** Terminated` + `ret = 15` |
| 大张量 + 小 tile → tile 数量爆炸 | 编译挂起，无错误码 |

---

#### 方向 3：stitch_function_max_num 过大 → 编译器调度压力

**触发条件**：`runtime_options={"stitch_function_max_num": 128}` — 强制 stitch 调度器同时绑定所有子图。配合 unroll 或 for-loop 时，编译器需并行调度大量子图，增加调度和内存开销。与方向 1/2 叠加时加剧超时。

```python
# ❌ stitch_function_max_num=128 + Python for-loop 128 head
kernel(pass_options={...},
       runtime_options={"stitch_function_max_num": 128})

# ✅ 缩小至 8（框架默认 128，过大时加剧编译超时）
kernel(pass_options={...},
       runtime_options={"stitch_function_max_num": 8})
```

---

#### 方向 4：大张量逐元素运算 + 小 tile shape → tile 数量爆炸

**触发条件**：大张量（>100K 元素）的逐元素运算（如 `pypto.mul`）配合第一维为 1 的 vec tile shape（如 `(1, 64)`），产生海量 tile（如 [1024,3072] × tile (1,64) = 49,152 tiles），编译器代码生成阶段挂起，无错误码输出。

**与方向 2 的区分**：方向 2 是 unroll/IR 展开导致代码量爆炸；本方向是 tile 切分粒度过细导致 tile 数量爆炸。两者机制不同但症状相似（编译挂起/超时）。

```python
# ❌ 大张量 × 小 tile → 49152 tiles → 编译挂起
pypto.set_vec_tile_shapes(1, 64)
w_scaled = pypto.mul(w_qb, w_qb_scale)  # w_qb [1024,3072], w_qb_scale [1,3072]

# ✅ 将大张量乘法移至 host wrapper 预计算
def wrapper(w_qb_int8, w_qb_scale, x):
    w_qb_fp32 = w_qb_int8.to(torch.float16).to(torch.float32)
    w_scaled = w_qb_fp32 * w_qb_scale  # host 侧一次性
    return kernel(w_scaled, x)
```

| 触发条件 | 错误关键词 |
|---------|----------|
| 大张量逐元素运算 + tile 第一维 = 1 | 编译挂起，无错误码，timeout |

---

## 3. Cube Tile 与 Vec Tile 对齐规则

> 详见 [matmul.md §1](matmul.md) 和 [vector.md §2](vector.md)。vec tile 维度数必须匹配操作输入 tensor 的维度数，详见 [function.md §10](function.md)。

---

## 4. stitch_function_max_num 过大导致 NPU OOM

**根因**：workspace 总预算与 `stitch_function_max_num × MAX_UNROLL_TIMES` 线性缩放。

**触发场景**：`runtime_options={"stitch_function_max_num": 128}` — workspace 总预算随 `128 × MAX_UNROLL_TIMES` 线性放大，对含大中间 tensor 的算子可达数 GiB，超出设备内存。

**错误关键词**：`RuntimeError: NPU out of memory. Tried to allocate XXX GiB`

**解决方案**：按比例缩小：
```python
runtime_options={"stitch_function_max_num": 8}
```

---

## 5. 大静态权重在 loop_unroll 内 cast 导致 IR 爆炸 / 编译超时

**触发场景**：大静态权重 tensor（如 7168×1536 INT8，~11M elements）的 cast 链
（INT8→FP16→FP32）位于 `pypto.loop_unroll` 循环体内。Pass optimizer 尝试按每
循环迭代展开/内联这段 cast 链 → IR 节点数爆炸 → Pass_04→Pass_05 无限 hang。

**错误关键词**：`exit 124 timeout` / `Pass_04_ExpandFunction → Pass_05_MergeViewAssemble`

**解决方案**：将大静态权重的 cast 链从 JIT kernel 循环内移至 host wrapper：
```python
# ❌ loop_unroll 内包含大权重 cast（1536× 展开）
@pypto.frontend.jit
def kernel(w_dq_int8, x):
    for b in pypto.loop_unroll(0, M, TILE, unroll_list=[...]):
        w_dq = pypto.cast(w_dq_int8, pypto.DT_FP32)  # 11M elems per iteration!
        result = pypto.matmul(x, w_dq, ...)

# ✅ host wrapper 一次性 cast，kernel 签名改为 FP32
def wrapper(w_dq_int8, x):
    w_dq = w_dq_int8.to(torch.float16).to(torch.float32)  # host 侧一次性
    return kernel(w_dq, x)  # JIT 入参已是 FP32
```

| 触发条件 | 错误关键词 |
|---------|----------|
| 大静态权重(>1M elems) + loop_unroll + 多步 cast | `exit 124 timeout` + `Pass_04` → `Pass_05` hang |
| INT8 量化权重在 JIT 内展开 cast | 同上 |
