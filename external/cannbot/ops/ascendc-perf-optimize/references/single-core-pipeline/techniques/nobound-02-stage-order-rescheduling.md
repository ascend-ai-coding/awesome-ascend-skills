# 技术卡片 nobound-02：三阶段顺序重排（计算发射优先 + 消费阶段延后一拍）

- **适用代际**：通用方法（arch22 / arch35）；具体 API、指标和阈值按目标核验，示例或实测仅代表所标平台。 RegBase / MicroAPI / VF 部分仅适用于 arch35。
- **适用 bound**：无 bound
- **一句话**：DB 已开却仍不并行时，根因往往是标量核（所有流水的唯一发射源、单发射顺序执行）被堵住、后续指令发不出去——把「消费异步计算结果的阶段」延后一拍写上一个 tile、把「大批量搬运发射」排在计算发射之后，不改一行计算逻辑就能把流水重叠拉上来。

## 何时用（诊断特征）

先用目标 msprof 指标判断流水重叠，再结合指令时间线或受控实验定位依赖。下面的 `PipeUtilization.csv` / `ResourceConflictRatio.csv` 字段仅在目标产物提供时使用，重叠因子见 `../metrics.md` §2.1；ratio 本身不能证明阻塞因果。arch22 的详细分析入口见 `ops-profiling`。以下指令级仿真示例仅用于 arch35 / Ascend950（`ops-simulator` 的支持范围），不作为 arch22 的执行前置：

```bash
npusim record ./<user_app> -s Ascend950 -g -o ./out   # cannsim 已更名 npusim，两名字同入口
npusim report -e ./out -n all                          # 逐核指令级 trace（chrome trace json）
```

- 【前置】**`InitBuffer` 已开 DB，但搬入与计算仍不重叠**（判据见 `../metrics.md` §3：MTE2 全在前、VECTOR 全在后 = DB 未生效）。DB 没开就先开 DB，本卡不是它的替代品。
- 【msprof】各单元利用率均不高（如 MTE2/VEC/SCALAR 都在 0.5 上下），无单一 bound，但 `aiv_time` 不达预期。
- 【msprof】**重叠因子低**：`Σ(各泳道耗时)/aiv_time` 只有 1.6 量级、达成率约 50%（算法与实测算例见 `../metrics.md` §2.1）。
- 【msprof】`aiv_mte3_wait_ratio`（或对应写回泳道的 wait 比例）**高于 0.4**——写回在等计算。
- 【仿真】trace 上**标量泳道**出现大块空窗（单次 0.2~2.6us 量级）。注意与 `../metrics.md` §3「盯 VECTOR 泳道找空泡」的默认读法不同，本卡盯的是 **SCALAR 泳道的空窗**，再看空窗期间谁在忙来定性（机制见下节）：
  - **计算泳道满负荷** → 消费阻塞
  - **搬运泳道满负荷、但每次只放行 1 条指令**（单条耗时可低至 1ns） → 发射阻塞

## 原理

普通算子（搬入→计算→搬出）只要 `InitBuffer` 开了 DB，三段靠 TQue 的 EnQue/DeQue 同步就会自动并行，**不用手排**。本卡针对的是 DB 已开、流水却仍然串行的情形。

原因是 RegBase kernel 里地址计算、DataCopy 发射、VF 参数打包全排在同一条标量指令流上。标量核一旦停住，后续指令根本发不出去——DB 给的两块 buffer 只是**允许**重叠，真正让它重叠的是指令发得出来。两种停法：

- **消费阻塞**：`DeQue()` 等的是**刚刚发出、还在跑**的那次异步计算。计算发出后自己要跑若干 us，紧跟着消费它就是在等这段尾巴。
- **发射阻塞**：一个 tile 要发的搬运指令数（token 数 × 每 token 笔数）**远超硬件指令队列深度**，队列写满后反压发射方，标量核走走停停；若计算发射排在其后，向量单元只能干等标量核脱身。

两者正交：计算提前发出后**仍要跑完**，所以消费阶段该延后还得延后。

## 怎么改（before / after）

```cpp
// before：搬入在前（先被队列反压）、写出紧跟计算（等刚发出的 VF）
while (cur.Valid()) {
    CopyIn(next);      // 发 num*N 条 DataCopy -> 队列满 -> 标量核被反压住
    Compute(cur);      // VF 迟迟才发出，向量单元干等
    CopyOut(cur);      // DeQue 等刚发出、还在跑的 VF
}

// after：计算发射优先 + 写出延后一拍
Tile pend;                          // 已算完、待写出的 tile；num==0 表示还没有
while (cur.Valid()) {
    const Tile next = NextTile(cur);
    Compute(cur);                   // VF 先发出去，自己跑
    if (next.Valid()) {
        CopyIn(next);               // 标量核此时被队列反压也无妨，与 VF 并行
    }
    if (pend.Valid()) {
        CopyOut(pend);              // 写上一拍的，其 VF 早已完成，DeQue 不阻塞
    }
    pend = cur;
    cur  = next;
}
if (pend.Valid()) { CopyOut(pend); }   // 收尾：最后一个 tile 还欠着
```

配套：输出队列 depth 取 2（`Compute` EnQue 时上一拍还挂着未 DeQue）；跨 tile 的索引/元数据滑窗要能同时容纳 pend / in-flight / prefetch **三个** tile。

## 性能数据

A5（arch3510，Ascend950PR，BlockDim 56）上的 RMSNorm+RoPE+KV Cache 分页散射写融合算子，T=10752、heads (8,1,1)。交替换包各两轮、每轮复核 kernel `.o` md5：

| 口径 | 改前 | 仅延后写出 | 两处都改 | 总变化 |
| --- | --- | --- | --- | --- |
| msprof Task Duration | 58.99 us | 52.86 us | **39.61 us** | **-32.9%** |
| back-to-back | 54.46 us | 46.51 us | 33.99 us | -37.6% |
| L2 冲脏后 p50 | 91.85 us | 89.99 us | 89.57 us | -2.5% |

机制侧：`aiv_mte3_wait_ratio` 0.47 → 0.06；重叠因子 1.63 → 1.90 → **2.58**；`aiv_mte2_ratio` 0.497 → 0.814（搬运接近饱和，瓶颈已从编排转为搬运量本身）。

逐 shape：T=64 / T=512 中性，T=4096 **-24.6%**，T=10752 **-35.3%**。

## 红线（Stop rules）

- **别把手排当通法**：先确认 DB 已开且确实没生效，再动手。手排会把 tile 边界与 pend/in-flight/prefetch 的状态摊进 `Process`，可读性代价不小；DB 能自动重叠的算子不该走这条路。
- **两处正交、别只做一个**：在计算发射优先的基础上把写出改回不延后，实测 39.83 → 42.72us（**+7.3%**）、`mte3_wait` 从 0.05 弹回 0.22~0.28。
- **先确认 depth 改大不额外占 UB**：物理 buffer 数由 `InitBuffer(que, BUFFER_NUM, size)` 决定，queue depth 只约束同时挂起的句柄数。若两者绑定，加 depth 会压小 tile 粒度、tile 数变多，每 tile 固定开销（参数打包等）跟着乘，可能得不偿失。
- **元数据窗口要跟着放宽**：写出延后一拍后同时有三个 tile 需要读索引，kernel 与 host 两侧同名的 `static_assert` 必须**一起**从 2 倍改 3 倍（只改一侧不会立刻报错，但注释与实现失配会误导后人）。
- **小 shape 无收益**：`T < coreNum × ubFactor` 时每核只有一个 tile，流水根本没形成，实测中性；别拿小 case 判定收益。
- **buffer 复用安全性依赖 TQue 的 FIFO 分配**（`CopyIn` 取到的是上一轮释放的那块，而非计算正在读的那块）。这一点文档无保证，须用「结果与改前逐位一致」实测坐实。
- **仿真只认相对形态，别认绝对时间**：仿真平台核数可能与实机不同（本例 32 vs 实机 56，单核工作量差 75%）、DDR 模型也非 HBM，绝对 us 系统性偏悲观。本卡全部判据只依赖「空窗里谁在忙」这类相对占用，收益结论一律回上板 msprof。
- **报收益要说清口径**：L2 被冲脏时（真实部署里调用间夹着别的算子）搬运绝对量成为主导，编排收益被稀释到 -2.5%。

## 关联

- 源：`../no-bound.md`。
- **与 nobound-01 的边界**：nobound-01 是在**既定顺序内**填气泡（开 PingPong、Preload 提前取数、UnitFlag 减同步）；本卡改的是**三阶段本身的先后顺序**。两者可叠加——先用本卡把顺序排对，再用 nobound-01 的手段填剩余气泡。另外 nobound-01 的红线"PingPong 使能时可用 UB 需除以 2"针对的是物理 buffer 数；本卡红线补充了 **queue depth 与物理 buffer 数是两回事**，depth 加大不一定占 UB，取决于 `InitBuffer` 的 `BUFFER_NUM`。
- 定位手段：CANN Simulator 指令级 trace，采集命令见「何时用」，泳道与指标口径见 `../metrics.md`。
- 相邻卡片：vec-09（外层 for 下沉进 VF，同样减少标量发射负担）、vec-08（依赖链型空泡，与本卡的编排型空泡区分）。
