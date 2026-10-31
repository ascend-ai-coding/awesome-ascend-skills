# 第 2 层：流水建模——AIC+AIV 延迟线流水（方法 + SFA 实例）

本篇是最佳实践库第 2 层**流水建模**：讲一种优化手段——把"AIC 做 GEMM、AIV 做布局变换+逐元素"的混合核从单循环串行改造成**双核独立延迟线流水**。§1-§2 与 §5-§6 是方法（任何同形态混合核适用），§3-§4、§7 用 SparseFlashAttention（SFA）作贯穿实例，§8 是边界。

> 实例证据：CANNBot-DSL `SparseFlashAttention_DSL`，实测相对单循环基线 1.85×，CUBE busy 93.2% → 99.4%，370 TFLOPS，seed 逐位一致，1650/1650 满频；37 步消融见 §7。复验（同协议 msprof op）：acc_check max_abs=4.8828e-04，与首轮结果一致；采集产物按实例工程 logs/ 归档。
> 参考实现：[整类投放件 M0-M4](../assemble/index.md)（逐模块独立文件，出处行号见各模块标注）；
> tiling 形状与容量的推导见 [容量方程](3-capacity-equations.md)；
> 表达层落地规则（视图声明/双代形态/动态 region/unit_flag）见
> [胶水：实施惯用法](../assemble/glue-idioms.md)——按本文实施结构机制前**必读**。

## 1. 优化手段：双核独立延迟线流水

| 项目 | 内容 |
|---|---|
| 解决的问题 | AIC（GEMM）+ AIV（gather/布局变换/逐元素）混合核两侧互相等待：供数侧不上节奏则 CUBE 出洞，CUBE 太快则供数环反压 |
| 核心手段 | 双核独立延迟线流水：每核按 wall tick 推进自己的阶段队列，跨核只通过环形缓冲交接，任一 pipe 不等自己的依赖、只等跨核握手 |
| 适用场景 | "Gather → GEMM → 逐元素(softmax/exp) → GEMM" 链等一切 AIC 做 GEMM、AIV 做布局变换+逐元素的混合核 |
| 判定信号 | PipeTimeline 显示 CUBE busy 明显低于 AIC 窗口（出洞）或 AIV union 低于窗口（断流）；一侧 SCALAR 发射流 ≈100% 但 CUBE <97% |
| 收益边界 | 每加一条 AIV 指令都直接吃 CUBE 占用率（两侧差 <1% 时为共同贴顶）——AIV 侧变重时平衡翻转为 vec-bound |

## 2. 建模规则（方法）

### 2.1 阶段划分

按计算链把每核切成阶段队列，每 tick 各推进一步；阶段粒度对齐数据节拍（tile/chunk），两核各持一条延迟线。

> SFA 实例（tick = 一个 128 行 KV tile 的处理节拍，实测 1.245µs/tile）：
>
> ```
> AIV(wall tick w):  I(w) → G(w-1) → S(w-3) → U(w-4)     [延迟线深 5]
> AIC(wall tick w):  QK(w-2) → PV(w-3)                    [相位落后 AIV ~1 tick，自锁定]
>
> I = 装载本 tile 的 128 个稀疏索引（GM→UB，int32）
> G = 离散 gather 16 行×4 chunk（GM→UB stage0）
> S = softmax（吃 qk_ub，产 P̃）
> U = 输出更新（吃 pv_halves）
> ```

### 2.2 滞后推导：滞后不是自由参数，由"数据何时就绪"逐级推出

通式：某阶段滞后 = 生产侧完成 tick + 跨核往返 + 缓冲裕量。

> SFA 实例：
>
> | 滞后 | 推导 |
> |---|---|
> | G 滞后 1 | 索引 DMA（256B）必须先于 KV 行 DMA → G 比 Plan 晚 1 tick |
> | S 滞后 3 | QK(X) 在 C 侧于 wall X+2 写完 qk_ub → S 最早 X+3 = 往返 2 + 1 格缓冲 |
> | U 滞后 4 | p_channel depth1 把 PV 拖到 X+3 末 → U = X+4 = 3 + 1 |

### 2.3 深度推导：深度 = 跨越的握手数 + 裕量

隔代缓冲深 2；跨多拍交接按在途拍数；同 tick 交接可故意收紧为锁相点（省存储，代价是强耦合）。

> SFA 实例：
>
> | 缓冲 | 深度 | 推导 |
> |---|---|---|
> | KV 环（L1，right_lower/upper） | 3 | G 领先 QK 1 tick + 抖动 2；且 3×128KB 恰好装下 L1 |
> | qk_ub | 2 | 隔代：C 写第 T 代、V 读 T−1 代 |
> | p_channel（L1）/ pv_ub | 1 | 同 tick 交接（故意收紧 = 锁相点），省 L1/UB |
> | stage0 / nz_staging | 2 | 同 tick 内 DMA→VF 双缓冲重叠 |
> | l0a / l0b | 4 | MMAD 背靠背与 MTE1 供数解耦 4 个 chunk 窗 |

### 2.4 同步点最少化

跨核握手压到必要最小，且放在"等待方其他 pipe 仍有排队工作"的位置，亚 tick 等待被吸收。

> SFA 实例：每 tile 仅 2 个跨核握手（KV→QK、P→PV）。

## 3. SFA 实例：kernel 程序序（每核主循环）

AIV 侧（每 tick，chunk 粒度 16 行 ×4）：

```
I: idx_buf ← GM 索引（提前 1 tick 预载，ping-pong）
G: 索引排序成对 → MTE2 乱序 gather 16 行 → stage0[chunk%2]（下一 chunk DMA 与本 chunk VF 重叠）
   → VECTOR nd2nz: stage0(ND) → nz_staging(NZ 16×16 分形)
   → mem_copy 推 L1 KV 环槽 → 通知 AIC
S: qk_ub.consume() → fused softmax（scale+max+exp+sum 一趟）→ P̃ bf16 NZ 写 p_ub
   → mem_copy 进 L1 p_channel → 通知 AIC
U: pv_halves.consume() → 在线更新 O（见 §4.7）→ 末 tile 归一化写 GM
```

AIC 侧（每 tile）：

```
QK: 等 KV 环 → 8 个 64 特征 chunk 逐个 L1→L0A/L0B → mmad 累加（init=chunk0）
    → fixpipe score 按 M 轴劈给两个 AIV 的 qk_ub
PV: 等 p_channel → P 装入 L0A，V 从 KV 环槽转置装 L0B
    → 8 个 D-chunk 各自独立 L0C mmad → fixpipe 到 pv_halves[0/1]
```

关键程序序细节：每 query 首个 QK tick 内依次 `load_q_tail(本query) → QK → consume 下块 KV → load_q_wide(下一query, 另一slot)`——宽段 Q 预取距离设计 5 tick，实测领先首用 2.1~3.5µs。

## 4. SFA 实例：模块实现要点（INDEX / GATHER / ND2NZ / QK / SOFTMAX / PV / UPDATE_O）

### 4.1 INDEX（索引装载）

- idx_buf (1,128) int32 ×2 ping-pong；**提前 1 tick** 装载，gather 不等 softmax。
- 两个相邻索引读出后排序 min/max，拼成一条 2 行 DMA 的地址对；无效索引（-1）后缀触发二分定位有效末尾（有效前缀先检查一次可跳过二分——实测 −79µs，见 §7 step14）。

### 4.2 GATHER（离散 KV 收集）

- 16 行/chunk，每 AIV 4 chunk；stage0 双缓冲让 chunk c+1 的 MTE2 与 chunk c 的 nd2nz 重叠（这是 AIV 关键路径）。
- gather 是乱序 2KB 小读：每 tick 128.5KB（KV 128KB + 索引 0.5KB），MTE2 实测 1.01µs/tile。
- 尾行处理：`local_rows` 不足一个 chunk 时先清零 stage0 再 gather 有效行（无效索引行输出 0）。

### 4.3 ND2NZ（布局变换，AIV 代劳）

- **AIV 侧做 nd2nz**（ds 风格）：`vload` ND 行 → `vstore_strided` NZ 16×16 分形 → mem_copy 推 L1。CUBE 前不允许出现数据整形（NZ 布局端到端）。
- 两 bank 交织：一次 load 相邻两个 128 元素列块再分别 strided store，落进不同 256B UB bank 组，避免 bank 冲突串行化。
- nz_staging 双缓冲：nd2nz(chunk c) 与 L1 推送(chunk c−1) 重叠。
- 对比（旧 token-wise 路径）：AIV 聚合后写 GM workspace、AIC 再 MTE2 Nd2Nz——多一次 144KB GM 往返。AIV 直推 L1 是本流水的核心结构改造之一。

### 4.4 QK（AIC）

- QK = Q[64,512]×K[512,128]→score[64,128]（收缩维 512，输出列=S2）；不是 [64,128]×[128,S2]（那是 PV）。
- 分层：DSL 8 次 [64,64]×[64,128] chunk 累加（MM_FEATURE_CHUNK=64）；硬件每条 MMAD = M64N128K128（2 chunk 融合），每 tile 8 条 × 0.1556µs。
- `set_mmad_direction("m")` + `unit_flag` 2/3 控制链式发射；L0A/L0B 四槽与 MTE1 解耦。
- fixpipe `score.partition(dim=0)` 把 64 行 score 按 M 轴劈给两个 AIV 的 qk_ub（每 AIV 32 行）。
- Q 驻留策略：**每 query 一载（64KB）+ query 内 5-tile 复用 + 跨 query 重载**；宽段 (32,384)×2 半块双代 ping-pong，尾段 (32,128) 单代。跨 query 真正驻留的只有 softmax 状态。

### 4.5 SOFTMAX（AIV，融合单趟）

- 一行 128 fp32 score = 2 个 VL=64 向量集（v0/v1）；行内 max → `vexp_sub` → exp。
- **写 P 的免 shuffle 打包**：偶/奇两半 fp32 分别 cast bf16（`RegLayout.ZERO/ONE` 错位）再 `bitwise_or` 拼成一个 128 元素 bf16 向量。
- 融合失败教训：softmax 只读 score、去掉写回、fold 重做缩放（省一次写）→ 实测 +6.3µs 无收益。收益绑定访存结构，不要盲目减一次写。
- 状态族：sm_max/sm_sum/sm_exp 各 2 槽 + tmp_new_max/tmp_sum；`se = exp(m_old−m_new)`、`l ← l_old·se + rowsum(P̃)` 一次向量化更新 32 行。状态槽 = query_id % 2（query i 尾部与 query i+1 头部重叠）。

### 4.6 PV（AIC）

- PV = P[64,128]×V[128,512]→O[64,512]；V 来自同一 L1 KV 槽的 nope 区（MLA-absorb K/V 同源）。
- 8 个 D-chunk 各自独立 L0C（`init=True` 每个 chunk 独立，不是跨 chunk 累加），先发第一个 V 切片再等 P（V 搬运在跨核 wait 之前发出）。
- 结果 `pv_l0c[chunk].partition(dim=0)` fixpipe 按行劈给两个 AIV。

### 4.7 UPDATE_O（AIV，O 原地累加）

**O 没有独立累加缓冲，驻留在 pv_halves 通道内**（每 AIV 两块，各 (64,256) fp32=64KB）：

```
pv_halves[half]  (64,256) fp32, depth 1
┌─────────────────────────┐
│ rows 0-31:  acc 区       │ ← O 驻留 UB，跨 5 tile 不动，只有 AIV 碰（原地 vmadd）
│ rows 32-63: pv 区        │ ← 每 tile AIC fixp 新写入（首 tile offset (0,0)，其余 (1,0)）
└─────────────────────────┘
```

- 数学（FlashAttention 在线更新）：`O ← O·exp(m_old−m_new) + P̃·V`（中间 tile），`O ← O/l`（仅末 tile 归一化）。新 PV 块不需要搬基准——P̃ 用新 max 计算天然以 m_new 为基准。
- 之所以一块就够：L0C 64 行结果经 `partition(dim=0)` 按行劈给两个 AIV，每 AIV 每 tick 只收 32 行 = acc 区大小；旧 O 32 行 + 新 PV 32 行 = 64 行恰好一个通道。
- 首 tile：fixpipe 直接写 acc 区（offset (0,0)），落点即累加器零拷贝；vector 只做 1-lane 读写**确认所有权**（ack 通道，否则 producer 下 tick 阻塞）。编译器曾拒绝此跨核所有权路径——必须按"first 优先组织分支"通过控制路径校验。
- 末 tile：`(acc·se + cur)/l` 归一化（sum=0 安全除 1），cast bf16 写回 pv 区 → mem_copy 本 AIV 的 32 行到 GM。halves 沿 D 维切（half0=特征 0-255，half1=256-511），各写 GM 对应列。
- 实测收益（§7 step17）：去掉 32×512 FP32 复制（−4.2%）。

## 5. 验收判据（方法；"本例实测"列为 SFA 实例）

| 验收项 | 判据 | 本例实测 |
|---|---|---|
| CUBE busy / AIC 窗口 | ≥99%（与规格无关，最硬的一条） | 99.4% |
| AIV union / 窗口 | ≥99%（无真空闲） | SCALAR 98.8%、MTE2 89.2% |
| 平衡 | \|T_C−T_V\|/max < 1% | \|1.245−1.25\|/1.25 = 0.4% |
| 满频 | Current==Rated | 1650/1650 |
| 精度 | seed 逐位一致 | 逐位一致 |
| 复测 | ±1µs | 两卡复测偏差 <0.1% |

> SFA 实例标定：T_C = 16.78 MFLOP ÷ RF(13.5 TFLOPS/核) = 1.243µs；T_V = max(MTE2 1.01, VECTOR 0.69, MTE3 0.90, SCALAR 1.25) = 1.25µs。RF 由板上 CUBE 忙时反推，与 npusim MMAD 形状（每条 2.10 MFLOP × 8 条/tile）交叉验证一致。

> cube bound 不是"算子算得多"，而是调度把其他一切都藏进了 CUBE 的影子里。判定只有一条：CUBE busy ≈ AIC 窗口 ≈ kernel 时长。

## 6. 方法论七步（从零到 cube bound）

1. **定协议**：msprof op 单 kernel、满频门禁、同卡同协议对比、中位数/复测确认（±1µs）；性能只认 msprof Task Duration。
2. **找节拍器**：pipe_summary 算各 pipe busy/窗口；≈100% 的那条就是 pacing pipe（SFA 例：AIV SCALAR 发射流）。
3. **画事件图**：按 §2 建立逐 tick 事件表，标出跨核交接点与环深。
4. **算内存预算**：三级台账（见[容量方程](3-capacity-equations.md)）→ 找"双贴顶"还是"单边瓶颈"。
5. **归因用消融**：一次只改一个机制，Task Duration 仲裁；警惕负收益（教训见 §7）。
6. **再平衡**：谁忙占比低就往谁挪工作/减谁的发射流，每轮回测直到 CUBE ≥99%。
7. **验收**：按 §5 六条全过才可宣称 cube bound。

## 7. SFA 实例：实测优化阶梯（单循环基线 → 最终形态 1.85×，37 步消融，全部逐位一致）

按机制分族汇总（收益为相对上一步的 Task Duration 变化；完整账见实例工程 OPTIMIZATION.txt）：

| 机制族 | 关键步 | 收益 | 要点 |
|---|---|---|---|
| 结构改造：单循环 → 双核独立流水 | step2 拆 KV 为两个 64 行 depth3 环 | −0.7% | 起点小：真正收益在后续机制叠加 |
| 预取/依赖前移 | step6 提前一拍装索引+成对地址描述符 | −9.6% | 旧版每拍 VECTOR 生成描述符阻塞 MTE2；把依赖前移 |
| PV 分半交接 | step8 两个 256 列 UB 环、分离 QK/PV L0C | −8.3% | pv_halves 结构雏形 |
| Q 双代 ping-pong | step9 宽段 384 提前载 + 尾段切换载 | −12.9% | L1 预算内最大可藏量 |
| Cube 指令链 | step10b unit_flag/mmad 方向；step11 D 向 64 分块、L0 四槽 | −0.8% | 纯改循环/编号（step7）无收益，必须改真实依赖 |
| 索引路径减负 | step12 只预取原始索引、标量直接发 DMA | −1.8% | 调 gather 前置依赖 |
| 有效末尾快速路径 | step14 有效前缀先检查免二分 | −15.6% | 单步最大收益之一 |
| O 原地累加 | step17 首 tile fixp 直写 accumulator 半区 | −4.2% | 见 §4.7；编译器所有权校验是坑 |
| 状态槽收紧 | step23 max/sum/alpha 各两槽；step25 连续 Buffer+tile 偏移 | −0.4% | 覆盖 SM→U 一拍生命周期 |
| L1 两半均衡 | step27 Q 上下 32 head、L1 对称安排、L0A 物理布局 | −0.05% | 最终形态 |

**负收益教训（消融纪律的证据）**：softmax 去写回 fold 重做（+0.7%）；固定 L0B 四槽、交错分核、均衡连续分核、缩 idx 槽 128→64、Ofast/unroll/缓存策略对齐——均无收益或变慢，**不采纳**；每步只改一个机制，Task Duration 仲裁。

## 8. 通用性边界

- **可复用**：延迟线结构（I→G→S→U / QK→PV）、滞后=往返+裕量、深度=握手数+裕量、O 原地累加、NZ 端到端、七步方法论、消融纪律。
- **必须重推导**：滞后值（新跨核往返 latency）、环深（新在途 tile 数）、状态槽数（query 交错度）、TILE 尺寸平衡点（CUBE µs/tile vs AIV µs/tick）。
- **会失效**：AIV 侧工作变重（RoPE/复杂 mask）→ vec-bound，应把工作移回 AIC 或简化 AIV 发射流；K 不整除 TILE_S2 时 tail 不得破坏节拍；描述符类机制在 AIV 发射流贴满时是净负担（本例 −16µs 消融）。
