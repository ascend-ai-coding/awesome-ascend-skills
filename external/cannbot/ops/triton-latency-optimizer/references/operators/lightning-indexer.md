# LightningIndexer 优化经验（拆分路径 + profile-driven tile 放大 + program 开销削减）

**以 LightningIndexer（topk-select 稀疏注意力索引器）大 shape 拆分路径为典型案例，提炼"先归因、再探查、后放大"的 tile 优化方法与 finalize 类小 kernel 的 program 合并技巧。**

**算子类别**: `topk-select`（QK^T 打分 + relu + 权重头归约 + 行内降序 top-K 选位，输出 int32 索引 + 可选 fp32 分数）
**典型特征**: 满足形状感知分派条件（`rows >= 256 且 rows*S2P >= 1M`）的 shape 走拆分路径——`qk_kernel`（QK^T + fp16 量化 + relu，产出 y16 中间缓冲）与 `hsum_kernel`（fp32 块对角头归约 + causal mask，产出 scores）；其余保留单融合核；选位由 `.sort()` 稳定排序完成（环境约束：`torch.topk`/`npu_sort_v2` 均被 validator 拦截，`.sort()` 是唯一放行原语，详见 transformer-inference.md §6 L1.4）
**性能基准**: hsum_kernel BLOCK_J 128→256（opt_16）单 kernel **-30%**（B2,2048,16384：7365→5169µs）；finalize 4 行合并（opt_17）单 kernel **-43~47%**（case16 2399→1267µs）；形状感知分派阈值 A/B 定档（opt_18）使弱 shape 0.367→0.48、任务 8-shape 5.63→5.75×；16-shape（work 0.1M~268M）vs ops-transformer AscendC 核级几何平均 **0.362 → 0.433 → 0.516 → 0.526（opt_20，全部 16 shape 严格 >0.41）**；输出与 torch 参考位级一致（大 shape 复验索引 |diff|>1 = 0/8.4M 元素）

---

## Layer 1: 设计约束（Agent 必须遵守）

### L1.1 profile 归因必须先修正同名 kernel 混淆，否则优化对象会搞反

- **必须** 用每个 kernel 的**完整唯一名**做耗时归因；对多 kernel 流水逐 kernel 单独采集核对。
- **禁止** 依赖 profiler 输出里的截断别名判断"哪个 kernel 是大头"。
- **Why（实测教训）**：本算子拆分为 `qk_kernel` 与 `hsum_kernel` 后，profiler 的 operator 汇总把两个名字都截断显示成 `kernel`，导致把 hsum 的 7.4ms 错记在 qk 头上（真实构成 qk 2.2ms / hsum 7.4ms），优化对象完全颠倒。名字相同不代表同一 kernel，归因错误会让后续所有变体探查打在错误的目标上。
- **How to apply**: 解析 `kernel_details.csv` 时保留 `Name` 完整字段；或对目标 kernel 单独写最小复现脚本隔离测量。

### L1.2 变体探查必须矩阵式单变量，并用消融区分边际成本

- **必须** 对目标 kernel 单独跑变体矩阵（R × BLOCK_J × multibuffer 等，每次只变一个维度），并用消融变体（去掉某个 dot / 去掉转换链 / 去掉 mask）区分各环节的边际成本。
- **Why**: 单看总耗时无法判断该动哪里。本算子消融实测：访存地板（k 载入 + scores 写回 + mask）仅占 6%，dot + 逐元素流水占 94%——据此把全部精力投向 tile/流水结构而不是访存。
- **How to apply**: 见 Layer 2 骨架；消融变体只用于计时，数值可错，但必须保证与原 kernel 的访存量近似（否则比较失真）。

### L1.3 tile 可行性必须实测，禁止凭朴素 UB 估算否决

- **必须** 用编译/运行实测 BLOCK 阶梯；被朴素 UB 估算否决的配置要实际编译一次再下结论。
- **Why**: UB 占用的估算存在两类截然不同的行为：
  - **"载入 → 转换 → 送入 dot"链**（hsum 的 `y16 载入 → to fp32 → tl.dot`）：编译器会**分段搬运、边转换边送入 dot**，不需要在 UB 中一次性放下完整 fp32 副本——朴素估算（216KB）超过 192KB 上限，实测却可行（-30%）；
  - **"dot 输出 → fixpipe → 后处理"链**（qk 的 dot 输出经 fixpipe 送 UB 再做 fp16 量化）：输出必须**一次性整块搬进 UB**，没有分段搬运的余地——R=16 时实测 320KB 确实溢出，估算是对的。
  - 两种链的 UB 行为不同，只有实测能区分，估算无法替代。
- **How to apply**: 直接改 BLOCK_J 编译跑一次；失败时读编译报错的 `requires X bits while Y bits available`，精确知道差多少、决定是降 tile 还是砍缓冲。

### L1.4 tile 放大必须论证位级一致不变性，只放大安全的维度

- **必须** 放大 tile 前先论证输出不变：本算子输出是逐位比对的索引，任何结构变化都要确认分数计算链不变。
- **Why**: 只放大 **N 维（BLOCK_J）** 不改变 dot 的 K 维归约序（本算子头归约是 K=R*N 的块对角 dot，K 归约决定分数位级）→ 结果位级不变；而放大 **M 维（R）** 会改变块对角 dot 的 K=R*N 长度，K 归约随之变化，必须逐档实测位级一致（K=8/16/32/64/128 实测均一致，但每一档都要验）。
- **How to apply**: 结构改动后做大 shape 位级复验（与 torch 参考逐元素比对索引 |diff|<=1、分数按 fp32 容差），而不只依赖小 shape 的 verify（小 shape 可能走的是另一条分派路径）。

---

### L1.5 finalize 类小 kernel 的行合并必须全位置写回，禁止 masked-store

- **必须** 行合并（每 program 处理 R 行）时用 `tl.where` 把有效值与默认值（-1/-inf）**一起写满全部位置**。
- **禁止** 用 `mask=valid` 只写有效位置：输出是 `torch.empty` 分配，未写的位置会残留脏值——
  本算子实测该写法导致 3/8 case 失败（输出出现未初始化的垃圾 int32）。
- **Why**：逐行单 program 时 small shape 的 finalize 占整算子 5% 以上（大行数时 2.4ms），
  program 数削减有实利；但"只写有效位"看似省指令，实则会破坏"输出全位置有定义"的隐含契约，
  除非 host 侧先 `torch.full` 预填默认值（实测预填+masked-store 的收益为负，不如 where 写回）。
- **How to apply**: 见 L3.4；R=4 是 UB 边界（R=8 时 `[8, 2048]` 的 int64 载入组合需约 320KB > 192KB，
  编译报错 `requires 2621696 bits`）。

## Layer 2: 算法骨架（Agent 可参考架构）

### L1.6 双路径分派阈值必须同会话 A/B 实测定档，且规则需形状感知

- **必须** 对"融合 vs 拆分"做同会话 A/B 交叉点扫描（同一进程、同一 profiler 设置），而不是拍一个 work 阈值。
- **必须** 在 work 之外纳入行数维度：本项目实测同一 work（1.05M）下，`rows=512` 拆分优 +31%，
  而 `rows=128` 的长 KV 形态融合优 -13%（拆分后 qk/hsum 的 program 数太少、串行 tile 过长）。
  最终定档规则：`rows >= 256 且 rows * S2P >= 1M` 走拆分。
- **Why**：拆分收益 = 微 dot 合并省下的同步税 − 额外一次调度与 `[rows*N, S2P]` 中间缓冲的代价；
  前者随 tile 数增长、后者是固定税，交叉点必须实测；只按 work 拍阈值会在低行数长 KV 形态上误伤。
- **How to apply**: A/B 脚本对候选 shape 各测融合与拆分（同会话各 warmup 3 + repeats 6+），
  取交叉点；实测数据见"性能基准"表 opt_18 行。

### L2.1 profile 变体探查流程

```python
# 1) 归因修正: 逐 kernel 单独采集, 保留完整 Name
path = BM.run_profiler_with_config(fn, warmup=3, repeats=6, profile_name="probe")
ops, total_ms = BM.parse_operator_latency(path, active_count=6)
for name, v in ops.items():            # 必须看完整 name, 勿用截断别名
    print(name, v["avg_us"])

# 2) 变体矩阵 (单变量, 每次只改一个维度)
for R in (8, 16):
    for BJ in (128, 256):
        for MB in (True, False):
            t = kernel_us(lambda: kernel[(cdiv(rows, R),)](
                ..., BLOCK_J=BJ, R=R, multibuffer=MB))

# 3) 消融 (边际成本, 只用于计时)
t_full = kernel_us(full_kernel)          # 完整
t_no_hs = kernel_us(kernel_no_headsum)   # 去头归约 dot
t_no_cv = kernel_us(kernel_no_cast_relu) # 去 fp16 量化+relu 链
t_mem   = kernel_us(kernel_load_store_only)  # 仅载入+mask+写回 (访存地板)
```

### L2.2 hsum_kernel 结构（本技巧的落地形态）

```python
# 每 program 处理 R=16 行 (S1 % 16 == 0 时), 行内按 BLOCK_J=256 分块
pid = tl.program_id(0)
row_vec = pid * R + tl.arange(0, R)
fr = pid * (R * N) + tl.arange(0, R * N)          # 连续 flat 行 (避开 tl.reshape 缺陷)
m1 = tl.where((cc // N)[None, :] == rr[:, None], w_all[None, :], 0.0)  # 块对角, 外提
for j0 in range(0, S2P, BLOCK_J):
    y16 = tl.load(y_ptr + fr[:, None] * S2P + jn[None, :], ...)   # [R*N, BJ] f16
    acc = tl.dot(m1, y16.to(tl.float32))                          # 转换可分段搬运
    cond = cond_row | (jf[None, :] >= akf) | ...                  # causal mask
    tl.store(s_ptr + row_vec[:, None] * S2P + jn[None, :], tl.where(cond, -inf, acc), ...)
```

---

## Layer 3: 关键技巧

### L3.1 hsum_kernel 的 R=16 + BLOCK_J=256 配置（实测最优，-30%）

```python
# 变体实测 (B2,2048,16384, 单 kernel):
#   R16/BJ128 (原配置)   7365 us
#   R8 /BJ256           6318 us   <- M=8 半填 cube 微块, 次优
#   R16/BJ256           5169 us   <- 采纳: M=16 填满 + tile 减半
```

**可替代方向**: 若 golden 允许近似（不必位级一致），头归约可换 fp16 键 dot 进一步提速；
但在位级一致约束下，fp32 键 + K=R*N 块对角是唯一解，本配置即该约束内的最优形态。

### L3.2 "分段搬运"的触发判据（泛化到同类 kernel 的 tile 决策）

- **适用（朴素 UB 估算偏保守，可尝试更大 tile）**：数据从 GM 载入 → 做 dtype 转换/逐元素变换 → 送入 `tl.dot` 的链——编译器可分段搬运，转换结果不必整块驻留 UB。
- **不适用（朴素估算即真实上限）**：`tl.dot` 输出 → 再做逐元素后处理的链——dot 结果经 fixpipe 必须一次性整块进 UB，没有分段余地。
- **判据口诀**: 先载入后 dot 的转换链可"流"；先 dot 后处理的输出不可"流"。拿不准就编译一次，报错的 `requires X bits` 会直接给出真实占用。

### L3.3 位级复验的最小流程

```python
# 大 shape 拆分路径 vs torch 参考 (小 shape 的 verify 走的是另一条分派路径, 覆盖不到)
ii, vv = impl(q, k, w, K, mode, True)
ri, rv = ref(q, k, w, K, mode, True)
assert ((ii.int() - ri.int()).abs() > 1).sum().item() == 0     # 索引逐位
fin = torch.isfinite(rv) & torch.isfinite(vv)
assert ( (vv[fin] - rv[fin]).abs() > 1e-3 + 1.22e-4 * rv[fin].abs() ).sum().item() == 0
```

**可替代方向**: 仅依赖小 shape verify 也可以，但会漏掉大 shape 分派路径的结构改动。

### L3.4 finalize 行合并模板（program 数 ÷R，小 shape 受益放大）

```python
# R=4 (UB 边界见 L1.5); 逐行版 program 数 ×4, 大行数时 finalize 达 2.4ms
@triton.jit
def finalize_kernel(v_ptr, i_ptr, oi_ptr, ov_ptr, S2P, K, ROWS,
                    BLOCK_K: tl.constexpr, R: tl.constexpr):
    pid = tl.program_id(0)
    row_vec = pid * R + tl.arange(0, R)
    valid_row = row_vec < ROWS
    kn = tl.arange(0, BLOCK_K)
    km = kn < K
    off = row_vec[:, None] * S2P + kn[None, :]
    v = tl.load(v_ptr + off, mask=valid_row[:, None] & km[None, :], other=float("-inf"))
    i = tl.load(i_ptr + off, mask=valid_row[:, None] & km[None, :], other=-1)
    oi = tl.where(v > float("-inf"), i.to(tl.int32), -1)          # 全位置写回 (L1.5)
    ov = tl.where(v > float("-inf"), v, float("-inf"))
    wmask = valid_row[:, None] & km[None, :]
    o_off = row_vec[:, None] * K + kn[None, :]
    tl.store(oi_ptr + o_off, oi, mask=wmask)
    tl.store(ov_ptr + o_off, ov, mask=wmask)
```

**可替代方向**: 逐行单 program（正确、无脏值风险，但 program 开销 ×4）；host 预填 + masked-store
（实测收益为负）。实测数据：case16 2399→1267µs、case12 309→176µs、case11 164→90µs。

---

## 性能基准

| 版本 | hsum 配置 | 16-shape vs op（核级几何平均） | 说明 |
|------|-----------|------------------------------|------|
| opt_iter_14（拆分 + 双路径分派） | R16/BJ128 | 0.362 | 大 shape 拆分路径建立 |
| opt_iter_16（tile 放大） | R16/BJ256 | 0.404 | profile 探查 + tile 放大，单 kernel -30% |
| opt_iter_17（finalize 行合并） | 同上 + finalize R=4 | 0.421 | program 数 ÷4，单 kernel -43~47% |
| **opt_iter_18（形状感知分派）** | 分派：rows≥256 且 work≥1M | **0.433** | 阈值 A/B 定档；#4 0.401→0.49、#5 0.367→0.48；任务 5.63→5.75× |
| **opt_iter_19（因果掩码三段 tile 切分）** | 同上 + j 循环三段化 + qk 父块对齐 | **0.516（全 ≥0.40）** | hsum 19.1→6.3ms(#16)；#11 0.35→0.42、#16 0.26→0.40；任务 5.72× |
| **opt_iter_20（两级排序 + L2 组调度）** | 同上 + A 组 4096 窄排 + batch 对齐组 | **0.526（全严格 >0.41）** | sort 16.9→11.4ms(#16)；#16 0.398→0.487；任务 5.71× |

关键结论：
1. 先归因（唯一 kernel 名）、再探查（变体矩阵 + 消融边际成本）、后放大（实测 UB 边界）是 dot+逐元素混合 kernel 的正确优化顺序，任何一步顺序颠倒都可能打错目标。
2. "载入→转换→dot"链的转换可分段搬运，UB 估算偏保守；"dot→后处理"链不可，估算即真实上限——两类行为必须实测区分。
3. tile 放大前先论证位级不变性（只动 N 维不动 K 归约），放大后做大 shape 位级复验。
4. 收尾类小 kernel（finalize/cast/fill）在 small shape 占比可达 5%+，行合并削减 program 数是最后一批
   可稳定兑现的收益；但合并后必须维持"输出全位置有定义"的契约（tl.where 全写回，勿 masked-store）。
5. profile 归因要用精确 per-kernel 分解复核"不可达"结论：opt_18 判 #11/#16 不可达的依据
   "sort 锁定 38-40%"被证伪——hsum（39.5%）才是第一大开销；因果掩码的有效列前缀结构
   （三段 tile 切分）是算术逐位不变的大 lever（L1.7 @ transformer-inference.md §6）。
6. 两个新陷阱（实测复现）：循环上界依赖 device load 会让编译器放弃预取流水（qk 反慢 39%，
   用 UNIFORM_LEN 标量算术上界修复）；0×NaN=NaN 穿透块对角 m1 结构零（跳过写区垃圾
   进入邻 kernel dot → 整 tile 污染，非确定性；用父块对齐写入上界修复，勿用 fp32 广播
   load 掩码——UB 溢出 304KB>192KB）。
7. sort 侧最后一块肉（opt_20）：aclnnSort 有 4096 宽甜点（耗时 0.37×/元素，实测）；
   参考 isfinite 掩码使 -inf 槽位列序无关 → eff≤W 的行只排 [0,W) 逐位一致；
   host 双 sort + 分部 finalize 落地（in-kernel 分裂布局 UB 溢出放弃）。
   多流重叠对"kernel 时长求和"考核零收益——L2 局部性（同 stream batch 对齐组调度）才有效。
