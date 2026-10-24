# AscendC Scatter 性能优化与实现选型指导（总览）

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

> Scatter 类算子（按轴 `dim` + 单值 `indices`，`y[..., indices[t], ...] = / += / *= updates[t]`）的性能关键有两条：
> **① 放弃 SIMD 的 UB-tile 范式，改用 SIMT 逐元素并行 + GM 直写**——写地址完全由运行期 `indices` 的值决定，tile 化收益低；
> **② 在六条并行/仲裁路径之间按 shape 与 reduce 语义选对一条**——选错就是数量级的差距（实测同一用例 12602us ↔ 3767us）。
> 本族属**多实现选型型**（SKILL.md 结构 B）：优化点在"选型"而非"调参"。

> **来源与可信度**：提炼自 [cann-bench](https://gitcode.com/cann/cann-bench) `tasks/level2/scatter` 的一份 direct-launch 实现，
> 在 **Ascend950PR 上实测评测总分 72.46（20/20 通过）、Scatter 算子榜第 1 / 21 个提交**，隐藏集 80/80。
> 文中所有性能数字均为该硬件上的**实测值**，非推算；推断性结论会显式标注。
> 与 CANN 官方 `ops-nn/index/scatter_elements_v2` 的 arch35 实现的差异见 §7。
> 适用：**DAV_3510（A5 / Ascend950）**。SIMT（`__simt_vf__` / `asc_vf_call` / `asc_atomic_*`）是该架构专有能力，910B 不适用。

> **和 Gather 族的分工**：两族都属"访存地址由运行期数据决定"，但**读侧与写侧的可优化空间不对称**——
> [Gather 族](../gather/guide.md)（读侧聚集）可以靠 Transpose / full-load 把不规则读**变回规则**，路径选择的核心是"要不要转、转得下吗、转了划算吗"；
> 直接 Scatter 的目标地址由 index 决定。本文六条路径针对 arch35 的直接写入和仲裁；当重复目标、随机写或确定性约束改变成本结构时，还应先读[跨架构的数据依赖写与重复目标归约](../common/indexed_write_reduction.md)，评估暂存分组及输出所有权。
> 同一个算子如果读写两侧都不规则（如 `index_put_` 的部分形态），两族都要看。SIMT 族（`../simt/`）是编程模型层的横切规则，与本族叠加使用。

## 文档结构

| 文件 | 内容 |
|------|------|
| **scatter_design.md**（本文） | 总览、六条实现速查、核心不变量、选路特征表、实测规律、约束、证伪清单（**决策树与 tiling 公式在下方 tiling/scatter/**） |
| [fiber_exclusive_design.md](fiber_exclusive_design.md) | 实现①：纤维独占——**零原子**，天然后写覆盖。满核时的最优解 |
| [element_parallel_design.md](element_parallel_design.md) | 实现②③④：元素级并行——打包键单趟仲裁 / 两趟胜者 / 直接原子（含 CAS 补原子乘） |
| [warp_last_design.md](warp_last_design.md) | 实现⑤：warp 内 bitonic 排序取段末，单趟且确定性、免 workspace |
| [generic_stride_design.md](generic_stride_design.md) | 实现⑥：通用路径——三张量非 dim 维范围不等时按逐维 stride 还原坐标 |
| [precision_and_traps.md](precision_and_traps.md) | 精度（fp32 累加器 / bf16 舍入模式）与四个只在特定环境暴露的工程陷阱 |
| [code/scatter_simt_common.h](code/scatter_simt_common.h) | 范式代码：①②③④ 的 VF 骨架 + reduce 组合 + CAS 原子乘 |
| [code/scatter_warp_last.h](code/scatter_warp_last.h) | 范式代码：⑤ warp bitonic 排序 + 段末写回 |
| [code/scatter_host_routing.cpp](code/scatter_host_routing.cpp) | 范式代码：Host 侧选路判据 + 线程数/block 数 tiling |
| **[tiling/scatter/](../../../ascendc-perf-optimize/references/tiling/scatter/fallback/tiling-flow.md)**（本 skill 之外） | **Tiling 建模侧**：六路径决策树、线程数/block 数/workspace 公式、阈值来源、TilingData 字段语义 |

---

## 0. 实现速查（六条路径）

记号：`dim` 归一后为 `d`；`inner = ∏_{i>d} idxShape[i]`、`outer = ∏_{i<d} idxShape[i]`、
`numFibers = outer × inner`、`sIdx/sUpd/sVar` 分别是三张量在 `d` 维的长度、`esize` = 值字节宽。

| # | 实现 | 冲突怎么消解 | 散射趟数 | 原子 | 适用 |
|---|------|------------|---------|------|------|
| ① | [**纤维独占**](fiber_exclusive_design.md) ★默认 | 线程输出集互不相交，**免原子** | 1 | 无 | `numFibers` 足以铺满核（`!starved`）；`reduce=multiply` 且规模小 |
| ② | [**打包键单趟仲裁**](element_parallel_design.md#2-打包键单趟仲裁) | `key=(j+1)<<16 \| bits16(v)` 的 32 位 `atomic_max` | 1 | 32 位 | `none` + `esize==2` + `sIdx+1 ≤ 65535` |
| ③ | [**两趟胜者**](element_parallel_design.md#3-两趟胜者) | 趟1 `atomic_max(win, i+1)` 裁决，趟2 比中才写 | 2(+1) | 32 位 | `none` + 键装不进 32 位（4/8 字节值、或 `sIdx` 过大） |
| ④ | [**直接原子**](element_parallel_design.md#4-直接原子需满足数值契约的-reduce) | 硬件原子消解写冲突；浮点累加顺序不固定 | 1 | 32 位 | `add` / `amin` / `amax` / `multiply`（乘用 CAS 循环补）；需满足数值与确定性契约 |
| ⑤ | [**warp 段末写回**](warp_last_design.md) | warp 内 15 级 bitonic 排序，同 index 段末者写 | 1 | 无 | `none` + `esize==4` + 键 ≤31 位 + `numFibers ≥ 1024` |
| ⑥ | [**通用逐维 stride**](generic_stride_design.md) | 同 ③/④，只是寻址按每维独立 stride | 同上 | 同上 | 三张量**非 dim 维范围不一致**（torch 只要求 `index.size(k) ≤ self.size(k)`） |

**本页六路径内的速查**：先核对数值顺序与确定性要求，再看纤维能否铺满核；能则考虑①，不能则按 `reduce` 分叉。允许非固定累加顺序的归约可考虑④；`none` 按键宽在②⑤③中选。三张量非 dim 维不等时优先⑥。六路径之外的暂存分组和输出所有权见上方通用指南。

---

## 1. 核心不变量：冲突只发生在"纤维"内

**这是全族所有优化的第一性依据，先理解它再看任何一条实现。**

更新元素的落点只把第 `dim` 维坐标换成 `index` 值，**其余坐标原样保留**。所以两个更新元素撞到同一个输出位置，
当且仅当它们的非 dim 坐标全同且 index 值相同 —— 冲突严格局限在一条**纤维**（固定非 dim 坐标、沿 dim 变化）内。

由此推出两个可直接变现的结论：

1. **按纤维分线程 ⇒ 各线程输出集合互不相交，全程不需要原子**（实现①）。且纤维内沿 `dim` 递增遍历，
   `reduce=none` 天然就是"后写覆盖"，与 CPU golden 逐位一致。
2. **元素级并行时，`atomic_max` 按 `j` 裁决出的胜者唯一**（实现②③）。
   因为 `dst = o·sVar·inner + ix·inner + c`（`0≤c<inner`、`0≤ix<sVar`）是一个**混合进制表示**，
   `(o, ix, c) ↔ dst` 一一对应 ⇒ 同一个 `dst` 只可能来自同一条纤维，而纤维内 `j` 互不相同 ⇒
   比 `j` 永远能分出胜负，胜者 = 最后写者。

**后写覆盖是硬要求，不是"能保证更好"**：评测判据 MARE 取**最大**相对误差，重复索引下选错一个胜者就整例挂。
CANN 内置的 `REDU_NONE` 是裸写、不提供这个保证（与 PyTorch"重复索引下 scatter 结果不确定"的文档约定一致）——
本族实现慢于内置的那部分开销，主要就是买这个保证（§7）。

## 2. 为什么不是 SIMD / UB-tile（范式级选型）

普通 elementwise / broadcast / reduce 的性能关键是"高效搬进 UB、向量指令算、搬出去"，地址是**规则的**。
Scatter 的写地址完全由 `indices` 的运行期数值决定：同一 tile 内的元素可能落到 `y` 的任意位置，
UB tile 既不能保证目标连续、也用不上 `DataCopy`/`Broadcast` 这类规则搬运指令。

因此正确的范式是**分工**：

```
阶段一（SIMD + MTE）：y = data 的整块拷贝 / dtype 转换 —— 规则、连续、高吞吐
阶段二（SIMT）      ：每个更新元素或每条纤维一个线程，只做"算地址 → 读改写一个标量"
```

**判定依据是"写地址是否数据依赖"，不是"这次要不要更快"。** 顺带一条实测反例：
半精度累加器的 cast 曾用 SIMT 标量循环做，改成 SIMD 的 MTE 搬运 + `Vector Cast` 后
**同一用例 1565us → 89us**。规则的顺序流量必须留给 SIMD，别图省事全塞进 SIMT。

## 3. 选路：特征 → 路径 对应关系

> 以下是"特征—路径"对应关系，**不是决策树**。
> 完整决策树（分支顺序、判据代码、位宽计算）与线程数 / block 数 / workspace 的推导公式
> 统一放在 Tiling 建模侧：**[tiling/scatter/fallback/tiling-flow.md](../../../ascendc-perf-optimize/references/tiling/scatter/fallback/tiling-flow.md)**。
> 本文只讲每条路径**为什么**这么设计、正确性依据是什么、实现要点在哪。

| 特征 | 路径 |
|---|---|
| 三张量非 dim 维范围**不等** | ⑥ 通用逐维 stride（**优先级最高，一票否决其余五条**） |
| `numFibers` 铺得满核（`!starved`） | ① 纤维独占 ★ 免原子、免 workspace |
| `reduce` 允许原子执行顺序带来的数值差异（`add`/`amin`/`amax`/`multiply`） | ④ 直接原子（乘用 CAS 循环补） |
| `none` + 4 字节值 + 键 ≤31 位 + 纤维数 ≥1024 | ⑤ warp 段末 |
| `none` + 2 字节值 + `sIdx+1 ≤ 65535` | ② 打包键单趟 |
| `none` + 其余 | ③ 两趟胜者 |
| fp16/bf16 的 `add`/`multiply` | 在上述路径之上**叠加** fp32 累加器（见 [precision_and_traps.md](precision_and_traps.md#1-半精度的累积型-reduce-必须在-fp32-域做)） |

两条容易踩的：

- ⚠ **②③④⑤ 四条快路的判据都必须显式带 `extentsMatch`**。漏掉会与⑥同时成立、输出错位——踩过的真实缺陷，见 [precision_and_traps.md 陷阱 2](precision_and_traps.md#4-陷阱-2快路判据必须排除通用路径)。
- ⚠ **不要因为"CAS 可能活锁"把 `multiply` 排除在④之外**。实测排除后某用例从 3767us 劣化到 12602us。

## 4. 两个真正决定性能的量

具体公式在 [tiling-flow.md §3](../../../ascendc-perf-optimize/references/tiling/scatter/fallback/tiling-flow.md#3-各路径的-tiling-推导)，这里只说为什么是这两个。

### 4.1 `starved`：纤维路径够不够铺满核

纤维路径的并行宽度**被 `numFibers` 死死卡住**——某用例只有 769 条纤维，就只能有 769 个线程在跑，其余核全空。
**这是本族最大的性能陷阱。** 元素级并行把并行宽度换成 `numel(indices)`（可达千万级），
代价是丢掉免原子、用原子或仲裁换回来 —— 所以只在 starved 时才划算，
**已经满核的 shape 留在①**，对它们元素级并行只是多一趟仲裁。

### 4.2 总线程数：吞吐几乎正比于它

见 §5 第一条。由此得到线程数的选法：**在能铺满核的前提下取最大的线程数，铺不满就取最小的**——
总线程数不变，但把访存发射摊到尽可能多的核上。
注意 `LAUNCH_BOUND` / `dim3` 要求编译期常量，所以候选值只能是固定几档、核内按 `switch` 展开。

## 5. 实测到的硬性规律（Ascend950PR）

| 规律 | 实测数据 | 用法 |
|---|---|---|
| **散射吞吐 ∝ 总线程数** | 每 block 8 / 32 / 128 / 512 线程 → 0.5 / 1.7 / 3.3 / 10~27 G updates/s | 是访存**延迟**受限不是带宽受限（顺序流式 pass 能跑 ~600 GB/s）。优先堆线程 |
| **32 位原子 ≈ 3× 64 位原子** | 11~18 G/s vs 4.7~5.0 G/s | 宁可拆两趟 32 位（③），也别用一趟 64 位。顺带顺序流量还减半（24B/元素 → 12B/元素） |
| **随机散射的写放大是数量级的** | 同 kernel、同指令数、同读模式，只换 index 分布：random 2805us / rowwise 583us（**4.81×**）/ constant 562us | 写地址 `y[ix*inner+c]`，一个 warp 内 32 线程 `c` 连续、`ix` 各异 → 落 32 条不同 cacheline，每条只用 2 字节。**换路径救不了**（②③写的是同一个地址式子），只能按 `ix` 分块重构 |
| **`__vector__` 是强制的** | 缺失时 `ld.lld` 报 `auto derivate failed` | 含 SIMT VF 的 `__global__` 核函数必须显式标 `__vector__`（`__attribute__((aiv))`）。纯 SIMD 核不需要 |

> 第三条的对照实验方法值得复用：**保持 kernel、指令数、读模式完全不变，只改输入数据的分布**，
> 就能把"访存局部性"从其它因素里干净地分离出来，不需要 msprof 指令级拆解。

## 6. 约束与适用范围

- **平台**：DAV_3510（950）。`__simt_vf__` / `asc_vf_call` / `LAUNCH_BOUND` / `asc_atomic_*` / `asc_shfl_*` 是该架构专有。
- **算子范围**：仅覆盖"按轴 `dim` + 单值 `indices`"的 Scatter 族——`Scatter` / `ScatterAdd` / `ScatterReduce`
  （对应 CANN `scatter_elements_v2`、ONNX `ScatterElements`、torch `Tensor.scatter_` / `scatter_add_` / `scatter_reduce_`）。
  **不覆盖** `ScatterNd` / `ScatterNdUpdate`（多维联合索引，寻址模型不同）、`ScatterUpdate`（整行替换、无原子竞争）、
  `scatter_pa_kv_cache`（PagedAttention 专用布局）、`scatter_list` —— 同名但寻址/一致性模型不同，别套本指导。
- **形状**：三张量 rank 相同且 ≤ 8；`indices.size(k) ≤ min(data.size(k), updates.size(k))`。
  非 dim 维范围不等由⑥覆盖，**不要 `TORCH_CHECK` 拒掉**（§8 方法论）。
- **reduce**：本实现支持 `none/add/multiply/amin/amax`。注意 **CANN 内置在 950 上只支持 `none/add/mul`**
  （`REGBASE_DTYPE_SUPPORT_LIST`，host 校验信息明写 `reduction must be in [none, add, mul]`），
  `max/min/mean` 仅 arch22（910B/910_93）分支接受；torch 侧 `scatter_reduce_` 在 op-plugin 中**完全没有 NPU 绑定**，
  会整算子回落 CPU。自研实现不受这些限制，但对比基线时要知道对面跑在哪。
- **代码为范式级**：`code/` 下三份摘录自真实工程并做了教学性精简（省略 dtype × idx × reduce 的全量派发宏、
  多份 launcher 声明），落地需接自己的注册/派发工程件。

## 7. 与 CANN 内置实现（`scatter_elements_v2` arch35）的差异

两者都选了 "SIMD clone + SIMT 散射" 的大范式，分歧在三处：

| 维度 | CANN 内置 | 本族实现 | 说明 |
|---|---|---|---|
| `reduce=none` 的重复索引 | 裸写，**谁赢不保证** | 严格后写覆盖（②③⑤ 三条仲裁路径） | 内置更快的那部分主要来自这里。要对齐 torch CPU golden 就必须付这个成本 |
| 并行划分 | 每个 `updates` 元素一个线程（固定） | **按 `starved` 在纤维/元素两种划分间动态选** | 内置在 `numFibers` 大时拿不到免原子的收益；本族在 starved 时才切元素级 |
| 多维寻址 | `GetUintDivMagicAndShift` + `Simt::UintDiv` quick-div，按 rank 特化 `SimtComputeDim1~8` | 快路径压成 `(o, j, c)` 三段、`outer==1` 时省一次除法；通用路径逐维取模 | **实测 quick-div 类标量优化在本族零收益**（§8），瓶颈在访存延迟不在标量流水 |

同步约束两者一致：阶段二在 GM 上原地写 `y`，**必须等阶段一的整块 clone 全部落盘**，
否则是"整块拷贝"与"局部原子写"的写后写竞争。内置用核内 `SyncAll()`，direct-launch 形态下靠 stream 上的 kernel 顺序保证。

## 8. 证伪清单（别重复试）

| 试过的 | 结果 |
|---|---|
| int64 → int32 索引运算宽度收窄 | **零收益**（+0.04 分）。连带 quick-div magic number、按 rank 特化 kernel 同属标量优化，**预期同样无用**——瓶颈是访存延迟 |
| 缩小 block 换更多核 | `inner` 大时会把合并访问拆碎，净收益 ≈ 0 |
| 硬件 fp16 原子加 | `asc_atomic_add(__gm__ half*)` **存在**（声明在 `simt_api/asc_fp16.h`，**不在** `device_atomic_functions.h`），但每次操作都舍回 fp16 —— 换用后 MARE 与软件版**逐位重现** `0.020576`，精度过不了。累加类必须走 fp32 累加器 |
| 放宽⑤的纤维数门槛 1024 → 256 | **明显更慢**：某用例（511 纤维、`sIdx=2048`）从③的 133.67us 劣化到⑤的 295.94us。511 条纤维只能铺 511×32=16352 线程，而③按元素级铺满核（约 105 万更新）；再叠每 warp 64 chunk × 15 步 bitonic = 960 步。**门槛 1024 是合理的** |
| 放宽⑤到 8 字节值 | 零收益（噪声内）。改动本身安全（`T` 只出现在最后一句赋值，排序全基于 uint32 key），但**无任何 int64 + 大纤维数的实测证据支持**，反而有相反证据。要启用先补实测 |
| 线程数 512 vs 1024 | 1024 净收益更好但不均匀（个别用例退化 8~24%）。按用例挑最优只能再多一点点，**且没有正当的形状判据**，不值得 |

### 一条方法论提醒

手上的公开用例**恰好全部满足"三张量非 dim 维相等"**，据此把一类合法输入直接 `TORCH_CHECK` 拒掉，
结果隐藏用例 77/80、correctness 门禁整体判失败、掉 15 分。

**算子的合法输入域要按语义规范推（`desc.md` / `proto.yaml` / torch 文档），不能按手上的用例推。**
