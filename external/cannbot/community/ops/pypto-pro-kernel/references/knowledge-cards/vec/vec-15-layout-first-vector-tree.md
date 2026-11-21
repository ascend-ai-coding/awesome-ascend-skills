---
type: "PyPTO Performance Optimization Card"
title: "布局先行 + 64-lane 向量树化：消除跨 lane 归约与标量 pack"
description: "把 VF 输入布局翻转为列=query 后，沿 key 轴的归约退化为跨 load 的 element-wise max/add 树，配合对齐写，整类消除跨 lane reduce 与 store_unalign 标量 pack。"
status: "stable"
tags: ["pypto-pro", "vec", "layout", "reduction", "softmax"]
item_id: "vec-15"
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: overlaps_with
      target:
        skill: pypto-pro-kernel
        id: "document:skill/patterns/vf-outer-reduce.md#output-ownership-before-blocking"
      scope: "改变生产布局后，让每条 lane 保留一个输出并沿外轴累加，减少横向归约与标量 pack。"
      boundary: "通用外轴所有权不证明 QK/PV 翻布局、DualModeSplitN 或多输出游标正确；这些变换需完整端到端验证。"
bound_hint: "VEC"
applicability: "VF 热点沿某一轴做归约（在线 softmax 的行 max / 行和、注意力分数统计等），分数矩阵布局可翻转为列=query（cube 侧把 Q·Kᵀ 改写为 K·Q̃ 类形式，acc 搬运按 N 维拆分），使每条 64-lane load 恰好覆盖一个归约行 × 64 个查询；归约结果无跨行交叉消费，翻布局后的寄存器与 Tile live set 可被容量容纳"
target_api_gate: "仅限 Ascend 950PR 或 950DT；依赖已核验的 vf.load_align/vf.store_align、vf.max/muls/add 树、vf.reduce_max/reduce_sum 与 pl.AccToVecMode.DualModeSplitN；旧写法所用 vf.store_unalign/unalign_reg_for_store tracker 语义须按当前版本复核"
---

# 技术卡片 vec-15：布局先行 + 64-lane 向量树化：消除跨 lane 归约与标量 pack

- **适用 bound**：VEC（联动 Cube 与调度）
- **一句话**：先翻布局（列=query），让"沿 key 的归约"从跨 lane reduce 退化为跨 load 的 element-wise max/add 树——一条 64-lane 指令同时推进 64 个 query 的计算，跨 lane 归约与标量 pack 两类指令因此有机会整体消失。

## 背景约定（先读这一段）

- **示例算子**（本卡所有例题的载体）：块内在线 softmax + 跨块行和归总——对分数矩阵 `S = Q·Kᵀ` 按块算在线 softmax：每块 key 求"行 max m_t（每个 query 一个）→ exp(x·scale − m_t) → 段和"；跨块把各段部分和按行归总成全局行和 sum（分子），最后 finalize 用全局行和归一。分多块累加时，每段部分和都以其计算时的行 max 为基准（softmax 每项 exp(x−m) 以 m 为基准，m 变大须乘 exp(旧m−新m) 才等价），"历史和 + 它算出时的 m"成对保存，称为一个"帧"。
- **UB / GM**：UB=片上向量内存（快、容量小、fresh-alloc 可能读出非 0 毒值如 0x6935FFFF）；GM=全局内存（大、慢）。所谓"常驻累加区"指在 UB 里开一块跨迭代存活的部分和缓冲，用前必须清零；"落盘 GM"指把中间结果写到 GM 工作区、finalize 时读回。
- **query / key**：注意力的两个矩阵轴；"沿 key 归约"指对每个 query 求 max（行 max）或求和（行和/段和），是 softmax 类算子的热点结构。
- **行=query 布局（旧）**：分数 tile 为 `[64, 128]`（64 个 query 行 × 128 个 key 列，行主序）。一条 64-lane load 只拿到 1 个 query 的部分 key 分数，行 max 必须跨 lane `reduce_max` 归约成标量。
- **列=query 布局（新）**：分数 tile 翻转为 `[128, 64]`（128 个 key 行 × 64 个 query 列）。一条 64-lane load = 1 个 key 行 × 64 个 query 的分数，lane i 恒对应 query i。
- **cube 侧配合**：把 `Q·Kᵀ` 改写为 `K·Q̃`（Q̃ 为 query 转置/重排后的乘数）即可直接产出列=query 布局；cube 累加矩阵搬入 UB 时用 `pl.AccToVecMode.DualModeSplitN`（按 N 维拆半的搬运模式，相对的 `DualModeSplitM` 产生行=query 布局）。
- **在线 softmax（flash）骨架**：维护运行状态"当前最大值 m 与部分和 sum"；每处理一块新 key，先更新 `m_new = max(m_old, m_t)`（m_t 为本块行 max），再按 `exp(m_old − m_new)` 重缩放历史和。

## 何时用（诊断特征）

以下特征可从当前源码、生成物或静态指令账本直接核对：

- VF 内层按 query 行循环（行=query 布局）：每行 1 次 load，随后逐行 reduce_max/reduce_sum、`store_unalign` 标量写回；循环次数 = query 行数。
- 静态指令账本或 profiler 中 reduce_max、store_unalign（含地址游标）计数与 query 行数同阶。
- UB 常驻累加区需要全尺寸清零、拷贝与重缩放（fresh-alloc 的 UB 可能读出非 0 毒值，如 0x6935FFFF，常驻区首用前必须清零）。
- 布局侧存在配合改写空间：cube 侧 matmul 可改为 K·Q̃ 形式，acc→UB 搬运模式可选。

## 何时不适用

- cube 侧无法配合翻布局：matmul 输出布局被上游冻结，或 K·Q̃ 改写与拆分搬运代价过高。
- 归约结果存在跨行/跨段交叉消费，无法表达为纯 element-wise 树。
- 翻布局后寄存器/Tile live set 超容量并引发 spill 或容量回退（先按 vec-12 评估拆分）。
- query 数不足 64、无法填满 lane 时，尾块须另行处理（可结合 vec-13 的统一 mask），收益需重估。
- 布局不能动、只想把 reduce 换成硬件树形归约时，用 vec-07 即可，不必引入本卡。

## 原理

列=query 后，"每个 query 的行 max"就是若干 64-lane 寄存器中对应 lane 的运行值：逐行 `reduce_max`（跨 lane 归约成标量）退化为"跨 load 的 element-wise max 树"，一条指令同拍推进 64 个 query。由此产生两组连锁消除：

1. **跨 lane 归约消失**：行 max 从"每 query 一次 reduce_max"变为"每拍每路一条 `vf.max` 的运行树"（树深 = key 行数/路数，尾接二叉归并）；行和从"每段每行 reduce_sum 取 lane0 标量"变为"每条 lane 对各自 query 的跨拍标量累加"（tile 视角即沿 key 行方向的列归约，lane 维零交互）。
2. **标量 pack 消失**：行和/段和按 lane 累加后本身就是 64-lane 全宽向量（每 lane 一个 query 的和），可直接 `store_align` 整向量写一个 `[段数, 64]` 块主序 tile 的对应行；旧写法"reduce 成标量 → store_unalign 标量写回 ×行数"的 pack 动作整个不存在了。

主要改法与机制一览：

| 项 | 改法 | 机制 |
|---|---|---|
| 布局先行 | 行=query → 列=query | 使能项：归约从跨 lane 退化为跨 load 的 element-wise 树 |
| 行 max | 逐行 reduce_max ×64 → 4 路运行 max + 二叉归并 3 条 | 树深 = key 行数/路数；零跨 lane 归约 |
| 流水 | 多路运算交错 + 双累加树 + 下一拍预取 | 相邻指令落不同寄存器组提双发率；双树压短加法关键路径；藏 load 延迟 |
| 行和落盘 | store_unalign 标量写回 ×行数×段数 → 块主序 store_align | 见上文"标量 pack 消失" |
| 累积 | UB 常驻累加区 → 行和内联累积 + 历史帧落 GM + finalize 一次性修正 | 用可预算的 GM 流量换掉常驻区及其清零/拷贝/重缩放指令链 |
| init 清零 | 全尺寸循环清零 → 少量向量 store | 常驻区取消后毒值规避从"清场"变成"不进场" |
| finalize | 共享修正因子提出外层循环；div → 倒数表乘法 | 同一 tile 帧的修正因子被多个列块共享时只算一次；除法摊销为循环外倒数表 + 块内 mul |
| 同步 | 每 tile 多对 sync → 两 store 排同队共享 1 对 + 全局读回前一次 bar_all | 主循环同步原语减半 |

## 怎么改（before / after）

以下为**嵌入片段**：所用 `vf.*` / `pl.*` API 已在目标 Ascend 950PR 工具链的同代 kernel 代码中核验存在下述形态，具体语义以当前版本 API 文档与生成物为准。片段是教学级最小示意：分片/交织的具体拆法按实际布局自定，保持"逐拍多行、树形归并、全宽写回"的语义即可。掩码/无效值处理与布局无关，不在本卡范围内，使用者按自身数值语义另行处理。

**共享前提（例题设定）**：64-lane FP32 VF；列=query 分数 tile `[128, 64]`（128 个 key 行 × 64 个 query 列，恰好占满 lane）；`scale` 为缩放标量；`preg`=FP32 全有效掩码域。

**before（行=query 逐行风格）——每个 query 一轮循环，逐行跨 lane 归约 / 标量写回：**

```python
# qk_tile：行主序 [64,128] 分数 tile（128 个 key 分一条 64-lane load 装 64 个，
# 实际为两次 load + 一次归并，此处为教学简化成一条示意）；tmax_tile：行 max 暂存 tile
# ureg 为 vf.unalign_reg_for_store 生成的地址游标，专供 store_unalign
# 递进使用（多 dst 共享会错位）
for qi in pl.range(0, 64):                     # 64 个 query 行
    r0 = vf.load_align(qk_tile, qi * 128)      # 该 query 的 key 分数（×2 次 load）
    s0 = vf.muls(r0, scale, preg)
    tmax_m = vf.reduce_max(s0, preg)           # 跨 lane 归约成 1 个标量（×2 + 归并）
    vf.store_unalign(tmax_tile, tmax_m, ureg, 1, post_update=True)  # 标量写回 ×64
```

**after pass1（行 max）——4 路运行 max + 二叉归并：**

```python
# src_b0..src_b3：[128,64] 分数按 4 组交织拆出的 4 个一维 Tile，
# 每组承载 32 个 key 行、组内行主序连续（行步长 64 元素；具体拆法自定，
# 只需保证 4 组同拍各读 1 个 key 行）
max0 = vf.full(-1e9, preg, dtype=pl.DT_FP32)   # max1..max3 同理
for it in pl.range(0, 32):                     # 每拍 4 个 key 行（每组 1 行）
    c0 = vf.load_align(src_b0, it * 64)        # c0..c3：4 组各自当前 key 行
    max0 = vf.max(max0, c0, preg)              # 64 个 query 的运行 max 一条指令推进
    # max1..max3 同理；4 路运算交错排布可提高双发率
m01 = vf.max(max0, max1, preg)                 # 二叉归并 3 条 → 64-lane 行 max
m23 = vf.max(max2, max3, preg)
tile_max = vf.muls(vf.max(m01, m23, preg), scale, preg)
vf.store_align(tile_max_tile, tile_max, preg)  # 全宽整向量一条对齐 store，零游标
```

**after pass2（行和/段和）——双累加树 + 全宽直写：**

```python
# 段 = 16 个连续 key；段和 = 段内各行 exp 值按 lane 累加 → 64 个 query 各自的段和
# （物理形态：每条 lane 对各自 query 跨拍标量累加，lane 维零交互）
sum_new = vf.full(0.0, preg, dtype=pl.DT_FP32)          # 全局行和（在线累积）
for seg in pl.range(0, 8):                              # 8 段覆盖 128 个 key
    acc0 = vf.full(0.0, preg, dtype=pl.DT_FP32)          # acc1 奇偶双树，段尾合并压短关键路径
    for it in pl.range(0, 4):                            # 段内 16 行 / 每拍 4 行
        c0 = vf.load_align(src_b0, (seg * 4 + it) * 64)  # 组内线性寻址
        cs0 = vf.muls(c0, scale, preg)                   # scale 先乘（与 pass1 归并后乘 scale 数学等价）
        e0 = vf.exp(vf.sub(cs0, tile_max, preg), preg)   # exp(x−tile_max)，两步组合
        acc0 = vf.add(acc0, e0, preg)                    # acc1 同理；下一拍 load 提前发起
        # 4 组交错：c1..c3 / cs1..cs3 / e1..e3 / add(acc1,...) 同构
    seg_sum = vf.add(acc0, acc1, preg)
    sum_new = vf.add(sum_new, seg_sum, preg)             # 行和归总内联进段循环
    vf.store_align(seg_sum_tile + seg * 64, seg_sum, preg)  # 直写 [8,64] 块主序 tile 第 seg 行
```

**before（store_unalign 标量 pack）→ after（块主序 store_align）：**

```python
# before：每行每段 1 条标量 store + 独立地址游标；64 行 × 8 段 = 512 次/子 tile
# （hist_tile：旧写法的段和暂存 tile，[64,1] DN 布局；ureg0：第 0 段的地址游标；
#   tsum：第 0 段的段和标量，即 reduce_sum 结果）
vf.store_unalign(hist_tile, tsum, ureg0, 1, post_update=True)
# after：段和本身是 64-lane 向量（每 lane 一个 query 的段和）
vf.store_align(seg_sum_tile + seg * 64, seg_sum, preg)   # 对齐直写，零游标
```

**配套写法（finalize 与同步，术语就地定义）**：

- **在线 softmax 状态**（骨架对应的运行时变量）：`m_run`=运行 max（64-lane，每条 lane 存自己 query 的），`sum_c`=全局行和，`m_t`=本 tile 行 max。每 tile：`m_new = max(m_run, m_t)`，重缩放因子 `corr = exp(m_old − m_new)`；首帧 `m_run=−1e9` 时 `exp(−1e9−m_new)=0`，等价于把空帧和直接置 0。
- **lazy 帧修正**：示例算子的行和要跨多块 key 归总，每段部分和以其计算时的行 max 为基准（见背景约定"帧"）。若每来一块更大的 key 就把历史和读回重缩放，代价是 O(段数²) 次乘加读改写。lazy 帧的做法：历史段行和不做在线重缩放，连同其计算时的行 max 一起落盘 GM；finalize 拿到全局最终 max 后对每段只乘一次 `exp(段落盘时的行 max − 最终 max)`，压成 O(段数)。适用于归一化推迟到末尾的场景；若需在线输出（每块后立即可用），仍须 eager 重缩放。
- **修正因子外提**：同一 tile 帧的修正因子被多个列块共享时，提到外层循环计算一次，内层只做 mul。
- **倒数表**：finalize 分解为"行块因子 × 列块因子"时，循环外用 `div(one_v, x)` 各预生成一份倒数，块内 BRC 广播读表 + mul 合成（除法延迟高且不能双发）。除零路径（0/0→NaN、非零/0→inf）必须与 golden 逐例对齐。
- **布局别名**：向 GM 落盘的小 tile（如 [1,64] 的历史 max）若以 DN 布局 tile 直 store，会把该 GM 区标记为 DN 布局，与后续 RowMajor 读回冲突——在同址再声明一个 RowMajor 别名 tile 专门用于 store。
- **同步合并**：同一 tile 内多次 GM store 排同一 MTE3 队列时，V→MTE3 与 MTE3→V 各只需 1 对 sync 事件；全局读回前一次性 `bar_all`。
- **尾块分派**：主路径假设满 64 query / 满 128 key；key 总数 S 非 128 整数倍时才进 tail 路径，边界处理只在 tail 做，主路径不付这笔指令税。

## 性能与验证指标

机制证据：下面是**一次单算子改造的静态指令账本**（行=query 逐行风格 → 列=query 向量树；例题配置：128×64 分数 tile、FP32、64 lane；每子 tile / 每 AIV（向量核）口径，按代码逐条累加的静态估算）。数字属于该次实验的材料，**未随卡片提交，未计编译器调度与双发实际收益，未在其它算子复现，不能作为当前算子的收益承诺**；容量类数字（176KB/2KB）仅为该配置下的量级参考。

| 阶段 | 行=query 逐行风格 | 列=query 向量树 | 降幅 |
|---|---|---|---|
| 初始化（常驻区清零等） | ~706 | ~4 | ~170× |
| softmax 主体（两遍在线） | ~2800 | ~1080 | ~2.6× |
| 其中跨 lane 归约 + 标量写回 | reduce_max ×64、reduce_sum ×768、store_unalign ×576 | reduce_sum ×32、store_align ×9 | 归约基本消失 |
| 在线累积 | ~42 | ~14 | 3× |
| finalize（每列块） | ~26 | ~17 | ~35% |
| 主循环同步原语 | 5/tile | 2/tile + 全局 bar_all 一次 | ~2.5× |
| UB 常驻占用 | 常驻累加区 176KB + 双缓冲 64KB | 常驻区释放 | UB 压力显著下降 |

正确性覆盖与验证方法：

- 每项优化单独做消融变体验证，无增益的组合果断回退；改变"每迭代产出什么/写到哪"的外提须先做逐元素覆盖等价性检查。
- CPU golden 逐位/容差双门禁 + NPU 精度对比全程保留；div→mul、lazy 帧修正均写出与 golden 的等价式和边界行为（除零路径 → NaN/inf；`corr = exp(−1e9−m) = 0` 等价直赋）。
- 端到端收益以当前算子实测为准：改造前后各跑一次 formal compare 与指标采集，按账户本记录结论。

## 技术限制与风险

- **归约结果语义**：max 树/累加树每条 lane 只服务自己的 query，若归约结果还要参与跨 query 的后续运算（如全局归一化因子），需另行统筹。
- **地址游标语义**：`store_unalign` 的 unalign_reg 是全局地址游标，多 dst 共享会错位；保留旧写法时须逐条复核。
- **禁止 host 预计算**：倒数表等数值查表一律在 kernel 内 V 侧生成，host 只做形状/分配/launch——表进 kernel 才能吃到 V 流水与缓存。
- **数值语义**：div→mul 的除零路径、lazy 帧修正的等价式，都要显式写出并经 golden 验证后采信。
- **布局污染**：DN 布局 tile 直 store 会把 GM 推成 Layout::DN，与 RowMajor 读回冲突；同址 RowMajor 别名 tile 规避。
- **mem_bar 纪律**：V 内"先写后读"必须 `vf.mem_bar(pl.MemBarMode.VST_VLD)` 显式排水；省略处必须显式论证依赖顺序。
- **容量权衡**：UB 常驻区换成 GM lazy 帧后，GM 流量进入预算（该实验约 2KB/tile）；释放出的 UB 空间需重新规划复用（倒数表、历史 max、段和等小缓冲）。
- **毒值**：UB fresh-alloc 可能读出非 0 毒值（如 0x6935FFFF）；取消常驻区后天然规避，保留常驻区的写法必须先全尺寸清零再首用。
- **外提纪律**：循环不变量外提（tile 级 / 组级 / 调用级）前先证"纯函数 + 无跨迭代依赖"。

## 与已有卡片的关系

- vec-07 保留 reduce、改用硬件树形归约；本卡通过翻布局让 reduce 需求整体消失，二者对同一处热点互斥，按布局可否翻转选择。
- vec-04 消除单次标量往返；本卡的"段和全宽直写 + store_align"是其布局级推广。
- vec-05 消除中间 UB 暂存；本卡的"GM lazy 帧 + 一次性修正"是其跨 tile 累积版本。
- vec-13 的尾块统一 mask 可用于本卡 fast/tail 分派中的 tail 路径。

## 参考资料

- 核查线索：目标版本官方 softmax / attention 类样例中，element-wise 树替代跨 lane reduce 与 BRC 广播查表的用法。
