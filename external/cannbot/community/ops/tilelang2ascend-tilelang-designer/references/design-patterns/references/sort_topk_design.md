# 排序 / TopK / 采样类算子设计要点（Design 视角）

> 本文件含**设计阶段决策** + **值域二分的 TileLang API 速查与伪代码模板**
> （均属 TileLang DSL 设计表达）。AscendC 实现细节
> （分段 sort 的 Sort/MrgSort 用法、GatherMask 约束、二分 PAD range 等）见
> translator references/ascendc_sort_topk_patterns.md。

## 适用

`sort / topk / top_k_top_p / sampling` 等**找第 k 大 / 排序列选 / 采样过滤**类算子，
即 forward() 中需要对最后一维（或某维）做全量排序、取前 k、或按概率阈值过滤的算子。

## 设计决策 1：TopK 结构路由 —— 值域二分 vs 分段 sort（核心）

「找第 k 大」有两种结构，成本模型完全不同，必须按规模先定走哪条：

| 结构 | 成本模型 | 优势条件 | 劣势条件 |
|---|---|---|---|
| **值域二分**（不排序，每遍 `count(x >= mid)`） | O(迭代次数 × N)，带宽受限 | N 较小、需全局统计量（min/max/count/sum）、UB 装不下整行无法整行 sort | N 大时多遍全量扫描是硬伤 |
| **分段 sort + 归并取 top-k** | O(N) 搬移 + 硬件 sort/归并 | N 大，单遍优于多遍；需精确切分 tie 时天然位置阈值 | 固定开销 + 归并成本，小 N 不划算 |

### 1.0 🛑 路由前强制分析（先否决、再估算，缺一不可选二分）

**第一步：输出形态否决（一票否决）**——先回答"算子输出的是什么"：
- 输出**有序值或有序索引**（argsort/topk 取下标、后续 gather/scatter 按序消费，
  如 MoE gating 的 `expert_idx`）→ 值域二分**不适用**：它只产阈值/掩码、不产顺序，
  排序步骤无法省掉 → 直接走排序路径（整行 sort 或分段 sort + 归并）
- 输出**掩码/阈值过滤**（保留 `x >= kth`、置 -inf、概率截断）→ 二分候选，继续第二步

**第二步：规模结构否决**——按"行/组"的实际键长 G 判：
- 分组小 N（每组 ≤1024，无论组数多少）→ 每组硬件 sort 近乎免费，二分的
  O(iter·G) 多遍扫描纯亏（iter 通常 20~30）→ 排序路径
- 单行 N ≤ 交叉点（≈8192，910B3 实测）→ 二分候选；N > 交叉点 → 分段 sort
- 反例（MoeGatingTopKV2）：输出为 argsort 有序 top-k 索引（第一步即否决）；
  组内 32 元素求 top2、全行 256（第二步两条全中）→ 排序路径

**第三步：复杂度 + 硬件效率强制估算（写入设计文档，禁止凭"感觉更快"选路）**：
- 二分总成本 ≈ `iter × N × (比较+选择+归约 3 条向量 op)` 的指令数与扫描流量，
  `iter ≥ log2(含 padding 的 range / gap)`（通常 20~30 遍全扫）
- 排序总成本 ≈ 单遍 O(N) 搬移 + 硬件 sort/归并固定开销（建表/launch/归并）
- k 极小（k≤2）时的第三选项「k 遍 reduce-max」也必须同样估算——**不许默认比
  sort 快**（反例：MoeGatingTopKV2 组内 32 元素 top2，reduce-max 并不比直接 sort 快）
- 全局统计量（min/max/sum/softmax 分母）可复用时，二分有效成本降一档，计入估算
- 仅当估算显示二分总成本明确 < 排序总成本时才允许选二分；估算写进 design 文档备查

**实测判据（910B3）**：交叉点 ≈ N=8192（单行、掩码输出场景）。小/中 N（≤8192）值域二分快官方 1.7~3x；
大 N（>8192）二分慢 2.5~3.5x。因此采用**混合策略**：按 N 阈值分野，两条路径在
block/tile 设计早期完全独立（buffer 分派互不污染），避免一种路径污染另一种。

**🛑 二分法的严格前置（不满足则一律走分段 sort）**：值域二分每遍 `count(x >= mid)` 的
计数必须能用矢量 API 加速——即分批「`compare_scalar` + `select`（产 0/1）+ `reduce_sum`
计数」，**不能是标量逐元素循环**。标量循环会让"多遍扫描"的优势荡然无存，此时直接 sort
反而更优。只有矢量计数成立，下面 4 条优越性条件才生效。

**二分法的优越性条件（普适归纳，遇到"找第 k 大 / 阈值过滤"先核对）**：
1. **N 较小**：单行多次全量扫描的总带宽仍小于 sort 的固定开销 + O(N log N) 归并；
2. **需要全局统计量**：算子里本来就要 min/max/sum/count（如 softmax 分母、TopP 累加），
   二分顺手复用这些统计，不额外排序；
3. **UB 装不下整行**：整行装不进 UB 无法一次 sort 时，二分是 O(iter·N) 的免排序替代，
   不需要把 sort buffer 搬去 GM（sort 只在 UB 执行，GM 化绕不过限制）；
4. **tie 语义天然值阈值**：`mask < kth` 语义下二分按"值"切，天然保留所有 tie（见决策 2）。

**分段 sort 的优越性条件**：N 大（单遍 > 多遍）；或需要精确切分 tie 边界时
（位置阈值可精确控制保留数量）。

> 关键：二分法的迭代次数决定性能档位，迭代次数下界要按「**含 padding 的实际 range**」
> 算（`n > log2(range/gap)`），不能只看真实数据 range——padding 撑大会把精度档位拉低
> 导致误收集。用 eps 提前收敛替代固定次数，对 range 变化更健壮。

## 设计决策 2：TopK 的 tie 语义（位置阈值 vs 值阈值）

参考语义（`sort(ascending, stable)` 后 `mask value < kth_value`）等价于**保留所有
`value >= kth_value`**（值阈值），即并列在第 k 大边界的元素**全部保留**，保留数可 > k。

- **值域二分**天然是"值阈值"（按 `x >= mid` 计数收敛到 kth_value），直接对齐参考语义；
- **分段 sort** 天然是"位置阈值"（恰好取 k 个，并列按索引序切），与参考语义不一致——
  需要「sort 只求 kth_value → 再收集所有 `>= kth_value`」两步才能对齐（见 translator
  ascendc_sort_topk_patterns.md）。

设计时必须先确认参考实现的 tie 语义是"值阈值"还是"位置阈值"，再选结构。

## 设计决策 3：TopP 累积方向等价

TopP 的「升序 cumsum，mask `cumsum <= 1-p`」与「降序 cumsum，keep `cumsum <= p`」是
等价的 keep 集（精确算术下）。设计时按参考实现的写法选一边，不必两边都做。

## 设计决策 4：采样类输入约束（p / k）

采样过滤类算子输入有强约束，设计期就要定好并同步给 workload/评测：
- `p`（top-p 阈值）值域 **[0, 1]**：若 workload 生成时 p 无 range（默认 randn 标准正态），
  约 40~70% 的 p 超 [0,1]，官方算子在 p>1 时有 bug，评测无法对齐。**必须给 p 加
  `range: [0,1]`**（生成 uniform）。
- `k`（top-k 阈值）值域 **[1, min(N, 1024)]**：k>N 或 k>1024 会导致官方算子 aicore
  exception，须裁剪。

## 设计决策 5：核数分档（小 B）

`blockDim = B`（每行一核）时，小 B（如 1~2 行）只启用 1~2 核，二分/排序的固定开销主导，
性能档位差。小 B 小 N 场景的多核切分收益受限（行内切 N 需要全局统计归约），
留待大方向落地后再评估。

## 设计决策 6：dtype 对路径的影响

- fp16/bf16 的 tie 密集（尾数位少），tie 语义对齐更重要——优先走"值阈值 + 收集所有 tie"；
- fp32 tie 几乎不存在，值阈值与位置阈值等价；
- 计算统一升 fp32 做累加/softmax（精度保证）。

---

## 值域二分的 TileLang API 速查与伪代码模板

> 值域二分不排序，全用「分块比较 + 归约计数」表达，对应的 TileLang API 是一组
> 小而固定的集合。设计阶段可直接照下表选 API、照伪代码搭骨架，不必再逐 API 摸索。

### API 速查表（值域二分常用）

| 用途 | TileLang API | 关键点 |
| --- | --- | --- |
| kernel 入口 | `T.prim_func` + `T.func_attr({"enable_auto_sync": True})` + `T.Kernel(B, is_npu=True) as (bid, vid)` | 每行一核；`bid` 即行号 |
| 编译入口 | `tilelang.jit(out_idx=[-1], pass_configs={...})` | pass_configs 用 `TL_ASCEND_AUTO_SYNC` + `TL_ASCEND_MEMORY_PLANNING` + `TL_ASCEND_AUTO_CV_COMBINE` |
| UB 分配 | `T.alloc_ub((shape,), dtype)` | 分「vector buffer（block_N）」与「scalar buffer（SCALAR_SIZE=64，256B 对齐）」 |
| 归约 workspace | `T.alloc_ub((2*block_N,), "uint8")` | `reduce_*` 的 work buffer |
| 数据搬运 | `T.copy(gm_slice, ub)` / `T.copy(ub, gm_slice)` | 2D 切片 `logits[bid:bid+1, nb*block_N:(nb+1)*block_N]` |
| 归约到标量 | `T.reduce_sum(src, dst, tmp, dim=-1)` / `T.reduce_min` / `T.reduce_max` | dst 是标量 buffer，结果读 `dst[0]` |
| 元素级算数 | `T.tile.add / sub / mul / div / min / max` | 支持 buffer-buffer 与 buffer-标量 |
| 比较（CompareScalar 等价） | `T.tile.compare(mask, src, scalar, "GE"/"LE"/"GT"/"EQ")` | 产生 0/1 mask；`scalar` 是**标量 buffer 元素**（`mid_ub[0]`） |
| 选择（Select 等价） | `T.tile.select(dst, mask, x, y, "VSEL_TENSOR_SCALAR_MODE"/"VSEL_TENSOR_TENSOR_MODE")` | 按 mask 取 x/y |
| 超越函数 | `T.tile.exp(dst, src)` | exp（softmax 用） |
| 填充/常量 | `T.tile.fill(buf, scalar)` | 填 0 / 1 / 端点值 |
| 类型转换 | `T.tile.cast(dst, src, "CAST_NONE", n)` | int32→fp32 等 |
| 标量读 | `buf[0]` | 标量 buffer 元素索引 |
| 循环 | `T.serial(n)` | 块遍历 `n_blocks`、二分迭代 `BS_ITERS` |
| 同步 | `T.barrier_all()` | 每步后屏障（AUTO_SYNC 下多数可省，先保留跑通） |
| 常量/无穷 | `T.float32(x)` / `-T.infinity(dtype)` | 标量常量 / 真 -inf 掩码 |

### 伪代码模板（值域二分 TopK+TopP，单行）

```
# 每行一个 block；N 按 block_N 分块 stream 进 UB
BS_ITERS = 26        # 二分迭代，下界 = log2(含 padding 的 range / gap)
SCALAR_SIZE = 64     # 标量 buffer 256B 对齐（64 × 4B）

main(logits[B,N] fp32, p[B] fp32, k[B] int32, output[B,N] fp32):
  # --- UB 分配 ---
  a_ub      = alloc_ub(block_N)                 # 流式块（vector）
  mask/ones/one_zero/exp/out 等 = alloc_ub(block_N)
  lo/hi/mid/lo2/hi2/mid2/max/count_ge... = alloc_ub(SCALAR_SIZE)  # 标量(256B 对齐)
  k_ub/p_ub/one_minus_p/total_sum/count = alloc_ub(1)
  tmp = alloc_ub(2*block_N, uint8)              # 归约 workspace

  # 0) 装载 k→int→fp32；p→fp32；one_minus_p = 1 - p

  # 1) FindMinMax：分块归约累积 lo/hi
  lo=+inf, hi=-inf
  for nb in serial(n_blocks):
    copy(logits[bid, nb*block_N:(nb+1)*block_N] -> a_ub)
    reduce_min(a_ub -> count, tmp);  lo = min(lo, count[0])
    reduce_max(a_ub -> count, tmp);  hi = max(hi, count[0])
  max = hi

  # 2) TopK 二分：数 count(x >= mid)，收敛 lo ≈ kth_value
  for it in serial(BS_ITERS):
    mid = (lo + hi) * 0.5
    total = 0
    for nb in serial(n_blocks):
      copy(块 -> a_ub)
      compare(mask, a_ub, mid[0], "GE")          # a >= mid
      select(one_zero, mask, ones, 0.0, SCALAR_MODE)  # 0/1
      reduce_sum(one_zero -> count, tmp);  total += count[0]
    # 收敛更新：if/else 是编译期分支 → compare+select+算术
    ge = select(total >= k ? 1 : 0)               # 用 compare+select 表达
    lo = lo + ge*(mid - lo);  hi = hi + (1-ge)*(mid - hi)

  # 3) ComputeSumExp：Σ exp(x - max)（masked 元素 exp 记 0），预计算一次
  total_sum = 0
  for nb in serial(n_blocks):
    copy(块 -> a_ub)
    masked = select(a_ub >= lo[0] ? a_ub : -inf)   # TopK mask（真 -inf）
    exp_ub = exp(masked - max[0])
    reduce_sum(exp_ub -> count, tmp);  total_sum += count[0]

  # 4) TopP 二分（exp 域）：数 Σ exp for x <= mid2，收敛 lo2 ≈ 阈值
  lo2=0, hi2=1
  for it in serial(BS_ITERS):
    mid2 = (lo2 + hi2) * 0.5
    total = 0
    for nb in serial(n_blocks):
      copy(块 -> a_ub)
      masked = select(a_ub >= lo[0] ? a_ub : -inf)
      exp_ub = exp(masked - max[0])
      zeroed = select(exp_ub <= mid2[0] ? exp_ub : 0.0)  # 阈值切
      reduce_sum(zeroed -> count, tmp);  total += count[0]
    S = total / total_sum
    le = select(S <= one_minus_p[0] ? 1 : 0)
    lo2 = lo2 + le*(mid2-lo2);  hi2 = hi2 + (1-le)*(mid2-hi2)

  # 5) ApplyMasks 写回
  for nb in serial(n_blocks):
    copy(块 -> a_ub)
    out = select(a_ub >= lo[0]  ? a_ub : -inf)   # TopK
    out = select(a_ub >  lo2[0] ? out  : -inf)   # TopP
    out = select(a_ub >= max[0] ? a_ub : out)    # 保 max（TENSOR_MODE）
    copy(out -> output[bid, nb*block_N:(nb+1)*block_N])
```

### TileLang 特有坑（值域二分）

| 坑 | 现象 | 解法 |
| --- | --- | --- |
| **标量 buffer 未 256B 对齐** | `compare`/`select` 的标量操作数编译报错 | 标量 buffer 用 `SCALAR_SIZE=64`（64×4B=256B） |
| **if/else 是编译期分支** | 运行时条件（count>=k）无法正确更新 lo/hi | 用 `compare` + `select` + 算术：`lo += ge*(mid-lo)` |
| **-inf 端点导致二分卡死** | 二分不收敛 | 端点用有限值（-1e6/-1e38），不用 -inf（`-inf>=-inf` 恒真） |
| **掩码值非真 -inf** | 输出 0 或有限近似，与参考不匹配 | 用 `-T.infinity(dtype)`（IEEE 位模式） |
| **exp 在 TopP 二分里重算** | 超越函数 40×N 次，V 管减负失败 | exp 与阈值无关，预计算一次 / 搬到 exp 域二分 |
