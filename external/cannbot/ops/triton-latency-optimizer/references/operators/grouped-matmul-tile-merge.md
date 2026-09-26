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
- **How to apply**: E 很小时可用 kernel 内标量 E 循环累计边界与 tile 数；当扫描链被每个 M/N tile 重复执行时，即使 E=8 也不能直接假定开销可忽略。必须同时测试“直接 expert×N-tile 映射”，每个任务只读取本 expert 的起止边界并在 expert 内遍历 M tile。两条路径按 E、M 和平均 expert 行数做 host shape 特化。

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

### L1.9 K 对齐与 BLOCK_K 选择必须解耦
- **必须**先明确逻辑 K 与物理 padding K；Cube 微块要求按 16 对齐，但 `BLOCK_K` 是 tiling 参数，不要求整除逻辑 K。
- **禁止**使用 `256 if K % 256 == 0 else ... else 16` 一类“最大整除因子”策略。它会让 `K=4112` 退化为 `BLOCK_K=16`，产生 257 次 K 循环；`BLOCK_K=256` 只需 17 次。
- **How to apply**: `K_ALIGNED = cdiv(K, 16) * 16` 仅用于 host 选择 tile；kernel 尾块仍以 `offs_k < K` mask，masked load 的 `other=0`。只有输入物理存储明确带零 padding 时才能用物理 K 作边界。
- **UB 边界**: int8 dot + 后置 scale 路径优先实测 256/512；需要物化 fp16 tile 的预缩放路径仍遵守 L1.6 的 `BLOCK_K <= 256`。

### L1.10 BLOCK_N 必须同时平衡 dot 开销与 Cube 占用率
- **必须**允许 N 尾块 mask，不得因 `N % BLOCK_N != 0` 自动退回较小 tile；同时检查每段行数、N tile 数、静态任务数和 UB 活跃集。
- **必须**把“填满 Cube 核”和“减少低利用率 dot 次数”作为两个候选方向。任务数明显不足时通常减小 BLOCK_N；每段只有少量 Cube M 微块、且宽 tile 后任务数仍与 Cube 核数接近时，必须额外测试更宽 BLOCK_N。
- **Why**: 填满 Cube 核只是启发式，不是硬门槛。行数少时，dot 固定 issue/同步开销可能占主导，减少 N tile 数的收益可超过轻微欠占用；但宽 tile 使任务数大幅低于 Cube 核数时，欠并行会反过来主导。
- **How to apply**: 先按任务数选择 occupancy 候选；再计算宽 tile 的任务数和 N tile 缩减幅度。若平均段行数仅覆盖少量 Cube M 微块、宽 tile 仍保留接近 Cube 核数的任务，并能明显减少 N tile，则把宽 tile 加入 benchmark 候选集。候选必须满足 `acc + x + weight + 临时量` 容量约束，并与原分派逐 shape 比较；未稳定获益则回退。
- **禁止**只优化 tile 数或只优化占用率，也禁止把少量已验证 shape 固化为 E/M/N 的绝对阈值，或把一种 expert 分布的结论直接外推到另一种分布。

### L1.11 完整 workspace + Vector two-pass 只能作为跨 N 逐行归约 epilogue 的条件分支
- **必须**保留原 partial-max 骨架作为基线路径。只有 epilogue 需要跨完整 N 维计算逐行统计量、完整中间结果 workspace 可承受，并且同 shape 实测 two-pass 稳定获益时，才能增加完整 workspace + Vector two-pass 分支。
- **禁止**把该分支写成所有 grouped matmul 的强制骨架，也禁止因某一类 shape 获益而删除或覆盖原 partial-max 路径。
- **Why**: partial-max 路径只物化每个 N tile 的行统计量，额外空间小，适合 workspace 紧张或大 M×N 场景；完整 workspace 能让 Vector program 独占完整行并消除跨 N tile 的 partial 归约，但会增加中间结果写回、再次读取和 epilogue 重算，收益取决于 shape 与归约开销占比。
- **How to apply**: host 先用 M/N/E、dtype 与可用 workspace 估算完整中间结果大小；候选分支由 Cube 写完整中间结果，Vector 第一遍跨 N 计算 epilogue 和逐行统计量，第二遍重算 epilogue 并按统计量写最终输出。两条路径逐 shape 使用相同精度口径和设备侧 benchmark 比较；只有稳定获益的 shape 才按 shape 元数据分派到 two-pass，其余 shape 回退 partial-max 基线。

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

#### 大 E/重复扫描场景的直接 expert 映射候选

当每个输出 tile 扫描全部 E 个 expert，且耗时随 E 或 tile 数明显增长时，使用一维
持久化 grid 遍历 `E * NUM_N_TILES`。`expert = task // NUM_N_TILES`，任务只读取
`group_list[expert-1:expert+1]`，再在 `[row_lo,row_hi)` 内遍历 BLOCK_M。
该路径消除每 tile 的 E 次边界加载与 `tl.where` 选择链。必须与前缀扫描路径逐 shape
实测，不能假定任一路径恒优；host 只允许使用 shape 元数据做分派。

对直接映射路径，`M // E` 可作为平均 expert 行数的静态近似，用来选择 BLOCK_M。
本次 A8W8 模型验证中，`M=128, E=16/8` 分别选择 BLOCK_M=16/32；相对前缀扫描路径，
Triton 设备耗时分别下降约 32%/29%，说明“小 E 扫描开销可忽略”不是可靠规则。

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

判据：取最大的 BM 使粗估 tile 总数 `(M // BM) * num_n_tiles` 仍能填满 aicore——大 shape 拿满权重摊销，小 shape 保住并行度。若 `M` 略大于一个 BM（如 M=37/BM=32），还需实测上调一档以消除第二个低覆盖率 M tile。

### L2.4 BLOCK_K 分档不得依赖整除性

以下代码仅作为**缺少已验证 BLOCK_K 分派时的保守初始候选**：适用于逻辑 K 按 16 对齐、kernel 能用逻辑 K mask 处理尾块，且 64/128/256 候选均满足当前数据路径 UB 约束的场景。若原实现已有 shape 特化或 autotune 结果，应保留原策略，只补充非整除候选并逐 shape 实测，不得用本示例覆盖。已明确为 int8 dot + 后置 scale 的路径可继续测试 512；需要物化 fp16 tile 的预缩放路径仍遵守 L1.6 的 `BLOCK_K <= 256`。

```python
K_ALIGNED = triton.cdiv(K, 16) * 16
BLOCK_K_CANDIDATE = 256 if K_ALIGNED >= 256 else 128 if K_ALIGNED >= 128 else 64
# kernel: mask_k = offs_k < K, masked load other=0
# 仅在逐 shape benchmark 稳定获益后采用；否则保留原 BLOCK_K 分派
```

### L2.5 BLOCK_N 基线分派与宽 tile 补充候选

以下代码保留**直接 segment/expert × N-tile 映射的原有基线分派**：tiny segment 在任务数足以维持 Cube 占用率时使用宽 tile，其余 shape 从大到小选择能填满 Cube 核的 tile。若采用 `(M-tile, N-tile)` 扁平映射，必须改用实际 M tile 数计算任务数，不能套用 `E * ceil(N / BLOCK_N)`。在基线之外，仅当平均段行数少、宽 tile 后任务数没有严重不足且 N tile 数明显下降时，才把宽 tile 作为补充候选；这些派生特征用于触发实测，不应展开成固定 E/M/N 阈值。

```python
AVG_ROWS = M // E
BLOCK_N_BASELINE = (
    512
    if AVG_ROWS < 16 and E * triton.cdiv(N, 512) >= cube_cores
    else 256 if E * triton.cdiv(N, 256) >= cube_cores else
    128 if E * triton.cdiv(N, 128) >= cube_cores else
    64 if E * triton.cdiv(N, 64) >= cube_cores else 32
)

BLOCK_N_WIDE_CANDIDATE = 512
WIDE_TASKS = E * triton.cdiv(N, BLOCK_N_WIDE_CANDIDATE)
WIDE_NUM_N_TILES = triton.cdiv(N, BLOCK_N_WIDE_CANDIDATE)
BASE_NUM_N_TILES = triton.cdiv(N, BLOCK_N_BASELINE)
# 结合 AVG_ROWS、WIDE_TASKS/cube_cores 和 N tile 缩减幅度决定是否追加宽 tile；
# 只有逐 shape benchmark 稳定获益才建立新分支，否则继续使用 BLOCK_N_BASELINE。
```

### L2.6 跨 N 逐行归约 epilogue 的双骨架分派

1. **默认骨架**：沿用 L2.1/L2.2 的 per-N-tile partial-max + 二阶段归约，不改变已验证实现。
2. **候选骨架**：Cube 将完整中间结果写入 `[M, N]`（或等价的左右分支布局）workspace；Vector program 负责完整行块，第一遍遍历 N 计算激活及逐行统计量，第二遍重新计算激活并完成量化/写回。
3. **容量门槛**：按实际中间 dtype 和分支数计算 workspace 字节数，同时核算单个 Vector program 的行块及临时量；超过预算时不得进入候选骨架。
4. **收益门槛**：候选骨架必须与 partial-max 基线逐 shape 实测。只有两遍读取和重算的成本低于 partial 写回、读取及归约成本时才建立 shape 分支；否则保留默认骨架。

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
| BLOCK_K 按 K 的最大整除因子选择 | `K=4096+16` 落到 BK=16，K 循环暴增 | K 仅按 16 对齐；BK 按 UB/循环次数选，尾块 mask 补零 |
| N 非整除就回退小 BLOCK_N | N tile 数和 dot 次数无谓翻倍 | 保持大 tile，尾块用 `mask_n` |
| 只追求大 BLOCK_N | 小 E、窄 N 时 Cube 严重欠并行 | 用 `E*ceil(N/BN)` 检查任务数，不足则减小 BN |
| tiny expert 仍固定 BN=128/256 | 大量 M<16 的 dot 固定开销占主导 | UB 允许时实测 BN=512，按平均 expert 行数分派 |

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

### A8W8 GMM + SwiGLU + Quant 补充实测（Ascend 910B，12 case）

| 版本 | 核心变化 | Triton 自身几何平均提速 | CANN/Triton 几何比 | ≥0.48 case |
|------|----------|-------------------------|--------------------|------------|
| 旧版 | BK 取 K 的最大整除因子；N 非整除回退 BN=128 | 1.000x | 0.2456 | 4/12 |
| 新版 | K 16 对齐后独立选 BK；N occupancy 分派；tiny-expert BN=512；BM 阈值特化 | **1.9625x** | **0.4820** | **7/12** |

关键单点：`K=4112` 从 BK=16 改为 BK=256 后，典型用例减少 240 次 K 循环；窄 N/大 K 用例通过 N occupancy 分派避免只启动 3 个 Cube 任务；多 expert、平均约 1-2 行的用例用 BN=512 摊薄低 M dot 固定开销。以上数据均来自 5 次预热、20 次采样，且 12/12 精度通过。

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
