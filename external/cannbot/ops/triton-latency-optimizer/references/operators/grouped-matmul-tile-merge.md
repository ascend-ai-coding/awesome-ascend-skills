# Grouped MatMul（MoE 专家分组矩阵乘）Tile 合并优化经验

**以 GroupedMatmulSwigluQuantV2 为典型案例，提炼可复用于同类 grouped matmul / MoE expert GEMM 算子的设计约束与优化技巧。**

**算子类别**: grouped matmul + activation + quant 融合（`npu_grouped_matmul_swiglu_quant_v2` 同类；MoE expert GEMM、grouped GEMM + SwiGLU/GELU + per-token quant 等）
**典型特征**: `x [M,K] int8` 按 `group_list`（cumsum 或 count 模式）切分为 E 个 expert 行区间，各 expert 配 `weight [E,K,N] int8` + per-channel scale；输出 SwiGLU 激活后 int8 量化结果
**性能基准**: GroupedMatmulSwigluQuantV2 几何平均 **0.7738x** vs torch_npu（50/50 通过；19 case ≥1.0x，最高 1.68x）；优化过程 implementation 平均延迟 **0.4129 → 0.0459 ms（约 9x）**

---

## Layer 1: 设计约束（Agent 必须遵守）

### L1.1 行 tile 必须按 expert 边界对齐，禁止跨边界
- **必须**把 M 维按 group_list 切分为 per-expert 行区间 `[start_e, end_e)`，BLOCK_M tile 只在区间内切分，`mask_m = offs_m < end_e`。
- **禁止**按全局 M 均匀切 tile——跨边界的 tile 内不同行属于不同 expert，权重选择错误且引入写竞争。
- **Why**: grouped matmul 每行按边界映射唯一 expert；tile 越界行若不被 mask，会用错误 expert 的 weight 计算并以 row_max/workspace 污染下一 expert 的行。
- **How to apply**: kernel 内标量 E 循环累计边界与 tile 数，`tl.where(hit, ...)` 选出当前 m_tile 的 expert_id / row_base / end_sel / m_local（E ≤ 8，开销可忽略；E 为运行时值时循环不展开，不受 checklist 规范 7 约束）。

### L1.2 禁止 host 侧读取 group_list（D2H 同步禁令）
- **必须**只在 kernel 内消费 group_list（`tl.load(offsets_ptr + e)`）；host 侧 tile 规划只允许用 M/K/N/E 等 shape 元数据。
- **禁止** `group_list.item()` / `torch.nonzero` 等把边界读回 host 来决定 grid 或 BLOCK_M。
- **Why**: 违反"张量数值必须在 kernel 内消费"约束；且每次 forward 引入 D2H 同步，小 case 直接触顶延迟。
- **How to apply**: grid 用上界估计 `(cdiv(M, BLOCK_M) + E) * num_n_tiles`（per-expert 取整至多多 E-1 个 tile），多余 program 由空 stride 循环自然跳过。

### L1.3 含 tl.dot 的 kernel grid ≤ num_aicore，纯 vector kernel grid ≤ num_vectorcore
- **必须**对 mix kernel（Cube+Vector 混合，含 `tl.dot`）用 `props.get("num_aicore")`（910b1=20）；纯向量 kernel（量化/scatter 等）用 `props.get("num_vectorcore")`（910b1=40）。
- **Why**: checklist 规范 5；mix kernel 调度在 AI core 上，超 cube 核数的 grid 只会串行排队。
- **How to apply**: `triton.runtime.driver.active.utils.get_device_properties(dev_idx)`，结果按 device index 缓存到 `ModelNew._core_cache`（每 forward 查询有 host 开销）。

### L1.4 dot 的 M 维必须 ≥16（行循环维度并入 dot M 维，优化点 #22）
- **必须**把按行/按 token 的外层循环并入 `tl.dot` 的 M 维：BLOCK_M ∈ {16,32,64}，自适应选取（见 L2.3），用连续 `[BLOCK_M, BLOCK_N]` 单 tile。
- **禁止** BLOCK_M=1 逐行处理——每个 dot 付满额 issue + cube↔vector 同步开销，且同一 expert 的 weight 被每个行 program 整份重复加载（流量放大 M 倍）。
- **Why**: Ascend cube 微块 16×16；M=1 时微块 15/16 空转，latency-bound。实测 BLOCK_M 1→16 提升 7 倍（0.079x→0.554x）。
- **注意**: BLOCK_M 上限受 UB 锁死（192KB）：acc 2×BM×BN×4B 主导，BM=64/BN=128/BK=256 时峰值 ~180KB 已是边界，BM=128 必溢出。

### L1.5 禁止 atomic_max 归约 row_max，必须 per-tile partial + 二阶段归约
- **必须**在 n_tile 并行时把行最大值写为 partial 缓冲 `[num_n_tiles, M]`，由后续 kernel（或消费方）标量循环归约。
- **禁止** `tl.atomic_max` 做 row_max 归约。
- **Why**: atomic 会禁用 auto-blockify；且 partial 归约是顺序无关的 max，数值与单遍计算 bit-exact，不引入精度风险。
- **How to apply**: `tl.store(part_ptr + n_tile * M + offs_m, tile_max, mask=mask_m)`；Pass2 内 `for t in range(NUM_N_TILES): max_vec = tl.maximum(max_vec, load(part + t*M + offs_m))`。

### L1.6 fp16 预缩放精度路径的 BLOCK_K 被 UB 锁死在 256
- **必须**对"fp16 预缩放后 dot"的数值对齐路径（如 dequant_mode=1 需匹配 NPU fp16 累加行为）保持 BLOCK_K ≤ 256。
- **Why**: 该路径需物化 fp16 权重 tile（BK×BN×2B）；BK=512/BN=128 时仅 w_fp16 就 128KB，加 x_fp16 与 fp32 acc 必超 192KB UB。int8 dot 路径（后置 scale）无此副本，BK 可到 512（dot 数减半）。
- **How to apply**: host 按 `dequant_mode`（python 标量参数，非 tensor 数据）分档 BLOCK_K；此类路径的剩余差距属结构性，不要反复尝试。

### L1.7 forward() 内禁止 Python while 循环
- **必须**用链式三元表达式实现自适应参数选择。
- **Why**: `validate_triton_impl.py` 将 forward() 中 while 判为 Type-3 PyTorch 退化（"核心计算必须在 kernel 内"规则的保守外延），直接挡下 AST 预检查。
- **How to apply**: `BM = 64 if (M//64)*nt >= cores else 32 if (M//32)*nt >= cores else ... else 1`。

### L1.8 多 case 任务的 verify 目录三件套
- **必须**保证 verify_dir 内 `{op}_torch.py`、`{op}_triton_<impl>.py`、`{op}.json` 三者同名共存。
- **Why**: `get_input_groups()` 按 `__file__` 同目录读同名 .json；缺 .json 报 FileNotFoundError，torch 模块缺 `_torch` 后缀报 ModuleNotFoundError。
- **How to apply**: 复制任务 .py 时同时复制 .json，并给参考副本加 `_torch` 后缀。

---

## Layer 2: 算法骨架（Agent 可参考架构）

### L2.1 扁平 (m_tile, n_tile) tile 索引空间（Pass1 主骨架）

```python
# host: grid = (min((cdiv(M, BM) + E) * num_n_tiles, num_aicore),)
# kernel:
# 第一遍标量 E 循环：累计 per-expert tile 总数 -> total_m_tiles（作 stride 循环上界）
for flat in range(pid, total_m_tiles * NUM_N_TILES, NUM_CORES):
    m_tile = flat // NUM_N_TILES
    n_tile = flat - m_tile * NUM_N_TILES      # 禁用 %（checklist 规范 4）
    # 第二遍标量 E 循环：tl.where(hit,...) 定位 expert_id / row_base / end_sel / m_local
    offs_m = row_base + m_local * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < end_sel                 # 同时保证 < M
    x_s = tl.load(x_scale_ptr + offs_m, mask=mask_m, other=1.0)
    w_base = w_ptr + expert_id * stride_we
    acc_left = tl.zeros((BLOCK_M, BLOCK_N), tl.float32); acc_right = ...
    for k0 in range(0, K, BLOCK_K):           # 左右半各一次 dot（SwiGLU 双分支）
        ...tl.dot(x_tile, w_half)...
    gate = acc_left / (1.0 + tl.exp(-acc_left))   # 手写 sigmoid（禁 tl.sigmoid）
    swiglu = gate * acc_right
    tl.store(workspace_ptr + offs_m[:, None] * HALF_N + offs_n_out[None, :], swiglu, mask=...)
    tl.store(part_max_ptr + n_tile * M + offs_m, tl.max(tl.abs(swiglu), axis=1), mask=mask_m)
```

要点：
- 每个 program 处理**同一 expert 内连续 BLOCK_M 行 × 单个 n_tile**，权重加载被 BLOCK_M 行摊销。
- 两遍 E 循环均为运行时界标量循环（不展开），累计器写法合法（checklist 规范 7 仅约束 constexpr 展开循环）。

### L2.2 Pass2（量化）骨架：partial max 归约 + per-token 量化

```python
for block_idx in range(pid, num_row_blocks, NUM_CORES):
    offs_m = block_idx * BLOCK_M + tl.arange(0, BLOCK_M); mask_m = offs_m < M
    max_vec = tl.zeros((BLOCK_M,), tl.float32)
    for t in range(0, NUM_N_TILES):           # 归约 partial max -> per-row max
        max_vec = tl.maximum(max_vec, tl.load(part_max_ptr + t * M + offs_m, mask=mask_m, other=0.0))
    scale_vec = max_vec / 127.0
    inv_scale_vec = 127.0 / tl.where(max_vec > 0.0, max_vec, 1.0)
    for n_tile in range(0, num_n_tiles):
        q = swiglu * inv_scale_vec[:, None]
        q_i32 = tl.where(q >= 0, q + 0.5, q - 0.5).to(tl.int32)   # round-half-away-from-zero
        q_i32 = tl.minimum(tl.maximum(q_i32, -128), 127).to(tl.int8)
```

### L2.3 BLOCK_M 自适应（host 侧，仅用 shape 元数据）

```python
BLOCK_M_PASS1 = (
    64 if (M // 64) * num_n_tiles >= num_aicore else
    32 if (M // 32) * num_n_tiles >= num_aicore else
    16 if (M // 16) * num_n_tiles >= num_aicore else
    8 if (M // 8) * num_n_tiles >= num_aicore else
    4 if (M // 4) * num_n_tiles >= num_aicore else
    2 if (M // 2) * num_n_tiles >= num_aicore else
    1
)
```

判据：取最大的 BM 使粗估 tile 总数 `(M // BM) * num_n_tiles` 仍能填满 aicore——大 shape 拿满权重摊销，小 shape 保住并行度。
BLOCK_K 分档：`512 if (dequant 走 int8 dot 且 K % 512 == 0) else 256`。

---

## Layer 3: 关键技巧（Agent 可参考但不可复制代码结构）

### L3.1 expert 定位的 found-flag 写法（kernel 内标量 E 循环）
```python
hit = (found == 0) & (m_tile >= cum_tiles) & (m_tile < cum_tiles + tiles_e)
expert_id = tl.where(hit, e, expert_id); row_base = tl.where(hit, boundary, row_base)
end_sel = tl.where(hit, end_e, end_sel);  m_local = tl.where(hit, m_tile - cum_tiles, m_local)
found = tl.where(hit, 1, found)           # 禁 break/continue，用 flag 收敛
```

### L3.2 count 模式 group_list 的边界累计
`group_list_type=1` 时边界需顺序累加：`end_e = boundary + load(offsets+e)`；cumsum 模式直接 `end_e = load(offsets+e)`，`start_e` 取上一轮 boundary。两种模式统一在同一个 E 循环内处理，`group_list_type` 作 constexpr 分支。

### L3.3 masked 行的安全处理
越界行（offs_m ≥ end_sel）x 加载 `other=0`、x_scale `other=1.0`，garbage 计算结果靠 store mask 屏蔽；workspace 与 partial max 必须**都**用 `mask_m` 屏蔽，否则会以错误 expert 的垃圾值污染下一 expert 的行。

### L3.4 冒烟测试先行（省一轮全量验证）
改动 tiling 参数后，先用 5 个代表 case（最大 K 的 mode-0 / mode-1、count 模式、最小 M、E 最大的 case）直接实例化对比 int8 maxdiff ≤ 1，再进全量 verify——粗略 host 计时不可信（被 launch 开销淹没），只验正确性与可编译性。

---

## Layer 4: 典型坑表（Agent 应避免）

| 坑 | 现象 | 修复 |
|---|------|------|
| BLOCK_M=1 行循环 | 整体 0.079x，latency-bound | 行维并入 dot M 维（L1.4），16→32→64 递进实测 |
| tile 跨 expert 边界 | 精度错 + 写竞争 | per-expert 对齐切分 + `mask = offs_m < end_e`（L1.1） |
| atomic_max 归约 row_max | auto-blockify 被禁用 | per-n_tile partial + Pass2 归约（L1.5） |
| host 读 group_list 定 grid | D2H 同步、违反约束 | grid 上界估计 `(cdiv(M,BM)+E)*nt`（L1.2） |
| mix kernel grid 用 vectorcore 数 | 超 cube 核数排队 | dot kernel 用 num_aicore（L1.3） |
| fp16 预缩放路径硬上 BK=512 | UB 溢出/编译失败 | 该路径锁死 BK=256（L1.6） |
| forward 内 while 自适应 | AST 校验 Type-3 拦截 | 链式三元表达式（L1.7） |
| kernel 内用 `%` 求 n_tile | checklist 规范 4 违规 | `flat - m_tile * NUM_N_TILES` |
| verify_dir 缺 .json / 缺 _torch 后缀 | FileNotFoundError / ModuleNotFoundError | 三件套齐备（L1.8） |
| BM=128 继续放大 | acc 2×128×128×4=128KB 必溢出 | BM 上限 64（UB 核算，L1.4 注意） |
| 粗略 python 计时判断优化效果 | 结论被 host 开销淹没 | 只信 benchmark.py 的 profiler 数据 |

---

## 优化路径实测（GroupedMatmulSwigluQuantV2, ascend910b1, 50 case）

| 版本 | 改动 | 几何平均 vs torch_npu |
|------|------|---------------------|
| 初始 | BLOCK_M=1 行循环 + 整份 weight 重复加载 | 0.0793x |
| opt_iter_1 | expert 对齐 tile BM=16 + 扁平 (m,n) 并行 + partial max + grid 修正 | 0.5539x |
| opt_iter_2 | BM 16→32 | 0.6600x |
| opt_iter_3 | BM 32→64 | 0.6889x |
| opt_iter_4 | int8 dot 路径 BK 256→512 | **0.7738x** |

**收益递减判据（何时停）**：BM 轴 +599%→+19%→+4.4%，BK 轴 +12%；当某轴收益 <5% 或被 UB 硬锁（L1.6）时停止，剩余差距标注结构性来源。

## 与 latency-optimizer 优化点的对应关系

| 本类算子高频优化点 | 对应序号 | 说明 |
|-----------------|---------|------|
| 行循环维度并入 dot M 维 | **22** | 核心收益来源（本经验主体） |
| BLOCK_K/BLOCK_M tile 调参 | 2 / 13 | BK 按 dequant 路径分档、BM 自适应 |
| Grid 分核与多路径 | 3 / 12 | mix≤aicore、小大 shape 自适应 BM |
| Kernel 分裂（mode 分档） | 18 | dequant_mode 决定 BK 上限，本质是按数值路径分裂 |
| CV 融合 | 29 | dot + SwiGLU + quant 的 scope 结构（本经验 Pass1/Pass2 两段式为已验证形态） |

## 复用到其他 grouped matmul / MoE 算子

1. 去掉 SwiGLU/quant 尾巴即为纯 grouped GEMM：保留 L1.1-L1.4 + L2.1，Pass2 换成对应 epilogue。
2. group_list 换成 `expert_indices`（每行显式 expert id）时：L1.1 退化为"按 expert 排序后分段"或 device 侧前缀扫描，partial-max 技巧照用。
3. 权重共享场景（E=1）可直接砍掉 E 循环，BLOCK_M 上限由 UB 核算重新决定。
