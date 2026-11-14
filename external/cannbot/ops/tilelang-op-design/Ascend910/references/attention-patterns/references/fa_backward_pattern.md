# Attention FA Backward（FAG）Pattern

一句话：FA 反向模式处理的是"以前向统计量 + dy 重建 P 并反传 dq/dk/dv"的算子形态；它的独特性在于 **P 直接重建（非在线 softmax）、dS 双消费（dS 与 dS^T）、三段式 Pre/Main/Post 的 fp32 workspace 累加结构**，而不是前向的 online-softmax 流水。

## 先读这个

命中信号：输入含 `dy`（输出梯度）、`softmax_max`/`softmax_sum`（前向统计量）、`attention_in`（前向输出），输出为 `dq/dk/dv` 三个梯度张量。

本模式只负责：FA 反向的计算图组织与 CV 融合结构。布局（TND/Paged）、mask/causal、GQA 等仍由各专项模式文档决定——反向只是给它们套上了"梯度计算"的外壳。

参照实现（生产 arch22 路线）：`output/<FAG 生产 re-host 归档>/`（S1s2Bn2gs1s2 泛化路径 re-host + design/ 设计复刻）。

参照实现（TileLang 生成路线）：`output/<SDPA backward 任务归档>/`（SDPA backward /
FlashAttentionBwd，C/V 分核 workspace-pass 分解；含 native/fp32 双精度通路对照、
判据口径审计与调优归档）。

> **平台范围**：本文生产路线口径只覆盖 **arch22（Atlas A2/A3，910B）** 分支
> （生产算子另有 arch35/950、arch92 分支，参数与模板集可能不同）；
> TileLang 生成路线部分按 910B 上的生成任务结构固化。

## 1. 识别条件

- 输入签名含 `dy, softmax_max, softmax_sum, attention_in`（或 `softmax_in` 统计量变体），输出 `(dq, dk, dv)`。
- `softmax_max/sum` 形状 `(B, Nq, Sq, 8)` fp32，只用 `[..., :1]`——存在即反向的强信号。
- 与前向的关键区别：**反向不做在线 softmax**，直接用前向统计量重建 P，无 running max/sumexp 状态。

## 2. 核心规则

### 计算流程（6 步，5 个 matmul）

```text
S  = Q @ K^T * scale           (Cube)
P  = exp(S - softmax_max) / softmax_sum    (Vector，mask 位置置零)
dP = dy @ V^T                  (Cube)
sfmg = rowsum(dy ∘ attention_in)           (Vector，FlashSoftmaxGrad)
dS = P ∘ (dP - sfmg)           (Vector，不折 scale)
dq += dS @ K ; dk += dS^T @ Q ; dv += P^T @ dy   (Cube，fp32 累加；
                                    scale 由 Post 施加于 dq/dk，dv 不乘)
```

> **scale 约定（生产 kernel 口径）**：生产实现 dS **不折** scale，workspace
> 累加原始 dS 乘积，Post 写回时对 dq/dk 施加一次 `Muls(scale)`（`needMuls`：
> TYPE_Q/TYPE_K 为 true、TYPE_V 为 false）、dv 不乘。等价约定是 dS 折入 scale
> （`= P∘(dP−sfmg)·scale`）且 dq/dk 不再乘——TileLang 生成路线即采用后者
> （V 侧 axpy 折入）。两种约定数学等价但**只可取其一**，混用即 dq/dk 得 scale²。

### 三段式结构（Pre / Main / Post）

```text
Pre : 清零 fp32 workspace（dq/dk/dv 累积区；fp32 dtype 时直接清 dvGm）
Main: 上述 6 步的 CV 融合主计算（核间 BN2GS1 合并轴，核内 S1/S2 切分，
      CV 双子图 ping-pong + Cube 双发 preload）
Post: fp32 workspace → Muls(scale) + Cast → 按 numel 全量写回 dq/dk/dv
      （Muls 仅作用于 dq/dk：needMuls = (type != TYPE_V)；dv 数学上无 scale）
```

### 块级伪代码（CV 融合骨架，泛化模板 S1s2Bn2gs1s2 视角）

```text
# 核间切分：BN2GS1 合并轴 → block 分配到 (b, n2, g, s1o)
# 核内切分：Q/dy/attention_in 按 S1 切 s1o 块；K/V 按 S2 切 s2o 块
# AIC:AIV = 1:2；CV 双子图 SubGrapA/B 按 slot = s2o % 2 ping-pong

# ---- Pre（AIV）----
zero(ws_dq); zero(ws_dk); zero(ws_dv)        # fp32 GM workspace（dq/dk/dv 累积区）
# fp32 dtype 特例：改为清 dvGm（Main 直写），Post 跳过 dv（CAST_DV 条件）

# ---- Main（CV 融合主计算）----
for s1o in S1_tiles:                          # 本核分到的 S1 块
    for s2o in S2_tiles:                      # KV 全长循环（累加方向）
        # Cube 通道（C1 同窗口双发，搬运覆盖 V1 的 Vector 时长）
        S  = gemm(Q[s1o],   K[s2o],  transB=True, init=True)     # 独立产出，L0C
        dP = gemm(dy[s1o],  V[s2o],  transB=True, init=True)     # 独立产出，L0C
        # Vector 通道（V1：P 重建 + sfmg + dS 合成）
        P    = exp(S * scale - softmax_max[s1o]) / softmax_sum[s1o]  # mask 位置置零
        sfmg = rowsum(dy[s1o] * attention_in[s1o])   # D 方向按 sfmgd(outer/inner/tail) 切分
        dS   = P * (dP - sfmg)                       # UB；不折 scale（约定 B，scale 由 Post 施加于 dq/dk）；双消费源，不物化转置
        # Cube 通道（dq 与 dk/dv 累加轴不同，init 极性分别对待）
        ws_dq[s1o] += gemm(dS,   K[s2o], init=(s2o == 0))  # dq 沿 s2o（内层）累加
        ws_dk[s2o] += gemm(dS^T, Q[s1o], init=True)         # dk 沿 s1o（外层）累加：每 tile 独立产出
        ws_dv[s2o] += gemm(P^T,  dy[s1o], init=True)        # dv 同 dk；^T 由 transpose 标志消化
        # GQA：dk/dv 按 kv head（N2）输出，G 组内归约经 ws_dk/ws_dv 多核/多块累加

# ---- Post（AIV）----
# 写回范围恰好铺满 numel（qPostBlockTotal == qSize）→ host 侧可用 at::empty
dq = cast(muls(ws_dq)) -> GM                  # fp32 → 输出 dtype，全量覆写
dk = cast(muls(ws_dk)) -> GM
dv = cast(ws_dv) -> GM                        # dv 不乘 scale；fp32 特例：CAST_DV 不成立，此步跳过
```

关键规则：

- **转置全部由 mma transpose 标志消化**（K^T、V^T、dS^T、P^T），禁止显式物化转置张量。
- **init 语义（写错极性即丢累加或残留脏数据）**：S/dP 每个 (s1o, s2o) tile 独立产出，`init=True`（L0C 不复用）。**累加轴分两条，不可混写**：dq 沿 s2o（内层 KV 循环，及 GQA 的 g）跨 tile 累加，首 tile `init=True`、后续 `init=False`，即 `init=(s2o == 0)`；dk/dv 沿 s1o（外层 Q 循环）累加——内层循环对每个 s2o 只经过一次，`init=(s2o == 0)` 对 s2o>0 恒为 False 会让 L0C 复用上一 kv 块的结果（跨块污染）。两种合法写法：① 每 tile `init=True` 独立产出、经 `ws_dk/ws_dv +=` 累加进 GM workspace（本伪代码采用）；② L0C 跨 s1o 累加 `init=(s1o == 0)` 且仅在最后一个 s1o 写回。GM 累加与 L0C 累加二者只取其一，混用即重复计数。
- **dS 双消费**：dq 用 dS、dk 用 dS^T——同一份 dS workspace 被两个 mma 以不同转置位读取。
- **fp32 workspace 累加**：Cube 输入 fp16/bf16、L0C/GM workspace fp32；Post 才 cast 回输出 dtype。这是精度契约。
- **输出全量覆写 → host 可 `at::empty`**：Post 的写回范围恰好铺满 numel（qPostBlockTotal == qSize == b·n1·s1·d），host 侧 `at::zeros` 是纯冗余（冗余 ZerosLike kernel 会额外占用 device 时间）。fp32 特例：dv 由 Pre 清 dvGm + Main 直写，Post 跳过（CAST_DV 条件）。
- **GQA**：dk/dv 按 kv head（N2）输出，G 组内归约经 workspace 多核/多块累加；golden 侧以 `repeat_interleave` + `view-sum` 对齐语义。
- **mask 语义**：`atten_mask` bool，**True 表示屏蔽**；causal 用上三角 mask 表达（对应生产 sparse_mode=0 + 显式 mask，而非 next_tockens 稀疏路径）。

### CV 配比与流水（A2 训练系列）

- AIC:AIV = 1:2（`KERNEL_TYPE_MIX_AIC_1_2`，kernel 入口可证）；CV 数据量配比 1:16
  （Cube 基本块 128×1024 fp16 vs Vector 8×1024 fp32 预算；S1 nRatio=1、S2 ratio=8
  组成——**来源生产 tiling 层，具体数值以所对接版本为准**）。
- Cube 双发（CV preload）：C1 同窗口连发 mm(S) 与 mm(dP)，覆盖 V1 的 Vector 计算时长；SubGrapA/B 双通道 ping-pong 隐藏 CV 交互。
- sfmg 的 D 方向切分（sfmgd outer/inner/tail）：大 value_d 时多轮累加。

### 精度通路路由（native 16-bit cube-input）

Cube 输入精度是 FAG 的重要性能开关：

- **fp16/bf16 输入 → native 通路**（16-bit cube-input + fp32 累加）：生产 FAG 在 16-bit case 上即走此通路。"禁止为提速降级精度"红线的放行条件：保留 fp32 workspace 累加、fp32 输入仍走 fp32 入口、接口/张量/dtype 全不变。
- **fp32 输入 → fp32 入口**：相对 native 有明显 wrapper 代价，且无任何合规 16-bit 配置可回收（见下）。
- **判据依赖（路由前必读）**：native 16-bit 通路在纯相对误差判据（无 atol 绝对下限）下容易 FAIL——dq/dk/dv 含大量近零/相消元素，相对误差被放大；验收须采用带 atol 的混合判据。验收判据未确认前不得默认走 native。
- **16-bit 中间补偿不可行**：P/dS hi-lo 残差补偿代价 ≡ fp32 入口（2×16-bit + 2×MMAD），零收益；只降 dV 通路的 16-bit 子集在相对判据下仍会 FAIL。

### TileLang 生成路线（workspace-pass 分解，按实际生成任务结构固化）

生产三段式之外，TileLang 路线按 **C/V 分核 + GM workspace 中转**的 pass 分解组织
（SDPA backward / FlashAttentionBwd 类任务即此形态），不做单 kernel CV 融合：

- **核分解**（5 matmul 分布到 4 个 Cube/V 核，核间以 GM workspace 交接，
  launch 边界即全设备同步）：
  - `fag_c1`（纯 Cube）：grid = B·Hq·s1ob·s2ob，逐 tile 产出
    `S = Q@K^T`、`dP = dy@V^T`（transpose_B，init=True）→ `ws_S/ws_dP`（fp32，
    per-tile [BM, BN]）；
  - `fag_v1`（纯 Vector）：同 grid，`T.Scope("V")` 下每 AIV（vid）持
    SUB = BM/2 行；P 重建（含 softcap 的 1−tanh² 因子与 mask）、
    `sfmg = rowsum(dy∘attention_in)`、`dS = P∘(dP−sfmg)[∘(1−t²)]·scale`
    （约定 A，scale 折入 dS）→ `ws_P/ws_dS`（输入 dtype，pad 区恒 0）；
  - `fag_c2dq`（纯 Cube）：grid = B·Hq·s1ob（每核一个 (b, hq, s1o) 行块），
    核内 `T.serial` 循环 s2ob 个 KV 块，`dq += dS[tile] @ K[s2o]` 走 L0C fp32
    累加 `init=(t == 0)`，循环结束一次写出；
  - `fag_c2kv`（纯 Cube）：grid = B·Hkv·s2ob（每核一个 (b, hkv, s2o) 列块），
    核内 `gi × s1o` 双重 `T.serial` 循环，`dk/dv += dS^T @ Q / P^T @ dy`
    （transpose_A，L0C fp32 累加 `init=(s1o == 0)`，每个 gi 重新 init），
    逐 gi 部分和写出 `ws_dk[gi]/ws_dv[gi]`；
  - `fag_cast_2d` / `fag_cast_g2d`（纯 Vector，dq / dk·dv 各一次）：
    fp32 ws → 输出 dtype；dk/dv 的 `cast_g2d` 顺带完成 GQA 组内归约
    （fp32 求和 + 最终单次舍入）。
  - D 维 16 对齐另有条件 `pad_copy` 核（见 host 契约）。
- **布局 BNSD** `[B,H,S,D]`（host `transpose(1,2)` 整理；与生产路线同轴序，
  无 BHSD 形态）。kernel 以原生 dtype 运行；仅 S/dP 与 dq/dk/dv 累加区为
  fp32，P/dS 以输入 dtype 存 ws——与 golden 逐 op 实体化 dtype 中间张量的
  语义镜像。
- **tile**：BM = BN = 64（s1/s2 块）；V1 每 AIV 持 SUB = BM/2 = 32 行。
- **mask（hi/lo 对角界编译进 kernel）**：`masked ⇔ (c−r) > hi | (c−r) < lo |
  r ≥ Sq | c ≥ Skv`（比较时加 d0 = Skv−Sq 平移）；每个 (hi, lo, softcap)
  配置编译一个 kernel 实例并缓存。被掩/越界位置经
  `select(valid ? s : −40000)` 置为 −40000（exp 下溢为 0），**不走算术运算**
  ——GM/L1 未初始化区垃圾（含 NaN）不传播，且防护 `0×NaN=NaN` 漏进 dS；
  pad 区 P/dS 恒 0，保证 C2 全块读取安全。**无 GM 物化 mask、无全 q 遍历**。
- **行尾处理**：不做 host 行零填充——GM→L1 的 `T.copy(pad_value=0)` 吸收
  行 OOB，配合上述谓词；仅 D%16≠0 时以 `pad_copy` 核做 D 维 16 对齐零填充。
  **scale 按真实 D（1/sqrt(dim_real)）计算，不按 pad 后维度**。
- **精度镜像（该路线核心技巧）**：golden 逐 op 实体化 dtype 中间张量；V1 在
  每个实体化点做 fp32→dtype→fp32 单次舍入（`CAST_RINT` + 经 GM `ws_rnd`
  往返，防编译器消除来回 cast；GM 往返处须 `barrier_all`）。
- **累加结构（全程无 atomic_add）**：dq/dk/dv 的跨块累加全部在 C2 核内
  `T.serial` 循环 + L0C fp32 `init` 语义完成；GQA 组内归约 = C2kv 核内 gi
  循环（逐 gi 部分和物化）+ cast_g2d 组内 fp32 求和。host 侧 ws 与输出均
  `torch.empty`（无 Pre 清零；覆写/累加完备性由 C2 的 init 与全块写出保证）。
- **pass_configs**：C1/C2 核全套 4 pass（`AUTO_SYNC + MEMORY_PLANNING +
  AUTO_CV_COMBINE + AUTO_CV_SYNC`——对纯 C/V 核 AUTO_CV_* 无害）；纯 V 核
  （V1/cast）`@tilelang.jit` 默认配置即可。

host 准备契约（kernel 入口前置条件）：

- BSHD→BNSD `transpose+contiguous`（纯张量变换）；`softmax_max/sum`
  `[B,Hq,Sq,8]` 取 `[..., 0]` 转 fp32 contiguous 传入。
- scale 标量**先按输入 dtype 舍入**再传 kernel（镜像 aclnn Muls 语义，保证
  bitwise 对齐 golden）。
- (causal, window_left, window_right, softcap) → (hi, lo) 编译配置：每配置
  编译并缓存一套 kernel（配置是缓存 key 的一部分）。
- 全部 workspace（ws_S/ws_dP/ws_P/ws_dS/ws_rnd/ws_dq/ws_dk/ws_dv）与输出
  `torch.empty` 预分配；D%16≠0 时先跑 `pad_copy`。

## 3. 模板路由（生产多模板体系，设计侧复现的价值点）

泛化模板（S1s2Bn2gs1s2）支持所有 shape 但小 shape 结构成本高
（标量/同步控制主导、Cube 低载、核间不均严重）：

| 条件 | 路由 | 收益来源 |
| --- | --- | --- |
| S1<1024 && S2<1024 && (B·N2·2>CoreNum \|\| G==1) | S1s2Bn2（核间 B/N2） | 小 S 减少 CV 交互 |
| G·S1·S2≤32KB && G==1 && S2<1536 | Ngs1s2Bn | 核内切 N2.i 加大 CV 基本块 |
| N2·G·S1·S2≤64·128 | Bngs1s2B | 极小内层，B.i 入 CV 块 |
| S1≥1024 or S2≥1024 | SameAB（Cube 主核） | V0/V1 复用左矩阵 + L0C 累加 |
| 其他 | S1s2Bn2gs1s2（泛化兜底） | 全 shape 可用 |

路由阈值（1024/1536/32KB/64·128 等）为生产 tiling 版本相关数值，**以所对接的
生产版本为准**；模板的结构收益定性结论（小 S 减 CV 交互、大 S 复用左矩阵 +
L0C 累加、极小内层 B.i 入块）不受版本影响。

设计口径（定性）：小/中 S 的泛化模板结构成本（标量/同步控制主导、Cube 低载、
核间不均）靠上表模板路由规避，而非核内标量优化消除；大 S（S1/S2 ≥ 1024）路由
Cube 主核模板；非常规 head dim 的 shape 泛化路径适应性更好。

## 4. 边界

- 本模式只覆盖 BNSD / keep_prob=1.0 / 无 pse / sparse_mode=0 / 无 rope 的
  最简反向形态；TND（varlen）、sink、rope、dScale 量化属 V4 接口的扩展参数域，
  分别按 tnd_pattern / sink_pattern 等专项文档组合。
- dropout（keep_prob<1）会在 dP/dV 路径插入 dropmask 生成与施加，不在本模式内。
- NZ 输出（mm12IsNZOut/mm345IsNZOut）仅在特定 s2/d 组合命中，泛化设计保持 ND 即可。
