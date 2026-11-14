# Attention FA Backward Pattern

一句话：FA 反向模式定义"由前向统计量（softmax_max/softmax_sum）与前向输出
重计算 P，经 5 个 matmul 产出 dq/dk/dv 三个梯度"；实现上 dS 与 P 各有双消费
（正向 + 转置视图），转置一律用 matmul transpose 标志消化，梯度以 fp32
workspace 累加后 cast 写回。

## 先读这个

命中信号：输入是 `q/k/v/dy` + 前向统计量 `softmax_max`/`softmax_sum`（或前向
输出 `attention_in`），输出是 `dq/dk/dv`。原文件名 `dsl2Ascendc_FA_Bwd.md`，
对应生产 FAG 算子（arch22 / Atlas A2\A3 训练系列）。

本模式只负责：反向梯度流的计算结构、转置消费规则、fp32 累加与写回覆写语义。
前向 flash 主循环、mask/causal 可见性语义由对应 forward 模式决定，本模式只
复用不重定义。

## 1. 识别条件

- 输入含 dy（前向输出梯度）+ 前向统计量（softmax_max / softmax_sum），
  或注意力前向输出 attention_in（FlashSoftmaxGrad 的输入）。
- 输出是三个梯度 dq/dk/dv（不是单个 dX）。
- 布局 BNSD；keep_prob=1.0、无 pse 时为最简形态。
- GQA 语义：dk/dv 按 kv head 数（N2）输出，kernel 对 G 组内归约。
- 区分点：元素级反向（如逐元素算子的 grad）只有一个 dX 且 matmul 退化为
  外积/对角结构；FA 反向是完整 5-matmul 计算图。

## 2. 核心规则

### 计算流（6 步，P 重计算式反向）

```text
1. P  = SimpledSoftmax(Mask(Q @ K^T) * scale)        # 重计算前向 P，驻留 UB
2. dP = dy @ V^T
3. dS = P ∘ (dP - FlashSoftmaxGrad(dy, attention_in))          # 不折 scale（主线 = 约定 B，生产口径）
4. dq = dS @ K                                                 # 原始 dS 累加，scale 由 Post 施加
5. dk = dS^T @ Q                                               # 同上
6. dv = P^T @ dy                                               # dv 不乘 scale
```

- 5 个 matmul（步骤 1/2/4/5/6）为 Cube 侧；softmax 重建、FlashSoftmaxGrad、
  逐元素减/乘为 Vector 侧，典型 CV 融合形态。
- **scale 在 dq/dk 路径只出现一次，两种等价约定只可取其一，本文主线采用
  约定 B（生产 arch22 口径）**：约定 B（主线）——dS 不折 scale，Post 写回时
  对 dq/dk 施加一次 `Muls(scale)`（`needMuls`：dq/dk 为 true、dv 为 false）——
  生产 arch22 kernel 即此口径。等价替代是约定 A——scale 折入 dS
  （`= P∘(dP−FlashSoftmaxGrad)·scale`）、dq/dk 直写、Post 仅保留 Cast——
  TileLang 生成路线可采用（V 侧 axpy 折入）。混用约定（dS 折 scale 又保留
  Post Muls）使 dq/dk 得 scale²，是 D 类精度错误的来源。
- **转置不物化**：dS 有两个消费者（dq 用 dS、dk 用 dS^T），P 同理（dv 用
  P^T）。转置通过 matmul 的 transpose 标志消化，禁止显式物化转置张量。

### 三段式组织（Pre → Main → Post）

```text
Pre : fp32 workspace 清零（dq/dk/dv 累积区）
Main: 6 步主计算（CV 协同），dq/dk/dv 以 fp32 累加进 workspace
Post: workspace fp32 → Cast → 按 numel 全量覆写 dq/dk/dv
      （约定 B：写回前对 dq/dk Muls(scale) 一次、dv 不乘，needMuls=type!=TYPE_V）
```

- **host 侧输出可以 `at::empty` 而不必 `at::zeros`**——成立条件是 Pre 清零 +
  Main 累加 + Post 全量覆写三者保证每个输出元素都被写。漏覆写表现为跨 case
  污染型数值错。
- fp32 特例：Post 不写 dv，dv 由 Pre 清零 + Main 直接累加覆盖。

### 模板路由（按 shape，序号越小越泛化）

- **泛化模板**（核间 BN2GS1、核内 S1/S2）支持所有 shape，但小 shape 下 CV
  同步与标量控制成本占绝对主导，非计算瓶颈。
- S1<1024 && S2<1024 且 B·N2 大（或 G==1）→ 路由小 S 模板（核间 B/N2）。
- S1≥1024 or S2≥1024 → Cube 主核模板（V0/V1 复用左矩阵、L0C 累加，省带宽）。
- 确定性计算走独立模板（避免多核同地址累加）。

## 3. 地址/形状公式

```text
q/dy:   [B, N1, S1, D]        # N1 = query heads
k/v:    [B, N2, S2, D]        # N2 = kv heads，K==V 共享存储
dq:     [B, N1, S1, D]
dk/dv:  [B, N2, S2, D]        # 按 kv head 数输出

scale = 1 / sqrt(真实 D)      # D=64 → 0.125；不得按 pad 后维度

# GQA 组内归约（dk/dv 按 kv head 输出，G 组 query head 的梯度累加）
group_size = N1 / N2
kv_head    = q_head / group_size

# 转置消费（不物化，matmul transpose 标志消化）
dq = dS   @ K          # dS  正向视图
dk = dS^T @ Q          # dS  转置视图（同一 buffer）
dv = P^T  @ dy         # P   转置视图（同一 buffer）

# fp32 累加区（Pre 清零、Main 累加、Post 覆写写回）
workspace_dq/dk/dv: [B, N*, S*, D] fp32
```

causal mask：以上三角表达（True=屏蔽），与生产 sparse_mode=0 语义对齐。

## 4. 与其它模式的边界

```text
forward attention 主循环(mask/causal/softmax) -> P 重计算复用其语义 -> 5-matmul 梯度流 -> fp32 累加 -> cast 覆写写回
```

- Mask/Causal：反向的 mask 语义复用前向；本模式不重定义可见性，只在重计算 P
  时沿用同一 mask。
- Head Sharing / GQA：只决定 kv_head 与组内归约；梯度计算图本身不变。
- MLA / Paged KV：决定 KV 的存储与寻址（latent/page），反向只改梯度流，
  不改变 KV 消费路径的维度规则。
- 区分：FA 反向 ≠ 逐元素反向；5 个 matmul 的计算图结构是本模式的本质特征。

## 5. 检查点

- P 重计算 + dS/P 的双消费（dS/dS^T、P/P^T）通过 matmul transpose 标志消化，
  不物化转置张量。
- fp32 workspace 累加 + Post cast/scale 覆写写回；确认全量覆写后 host 用
  `at::empty` 替代 `at::zeros`。
- `scale = 1/sqrt(真实 D)`，不按 pad 后维度；scale 在 dq/dk 路径只出现一次
  （主线约定 B：Post Muls 施加于 dq/dk、dv 除外；替代约定 A：折入 dS、Post 仅 Cast），
  两约定不得混用，否则 dq/dk 多一个 scale²。
- GQA：dk/dv 按 N2 输出 + 组内归约语义对齐 golden。
- causal 以上三角 mask 表达（True=屏蔽），与生产 sparse_mode=0 语义对齐。
- 多 dtype/mask 入口拆分独立构建 target（单入口二进制有注册大小上限）。
- **dq 的相对误差指标陷阱**：dq = dS @ K 且 dS 每行和≈0（softmax 梯度性质），
  dq 是消减残差，golden 值普遍很小、近零元素占比高。逐元素相对指标
  `mean(|diff|/(|golden|+eps))` 会被近零 golden 严重放大，16-bit 下容易超阈。
  判定 dq 改用 `mean(|diff|)/mean(|golden|)`、
  绝对误差 + matched_ratio，或以生产算子输出为 golden；不要为通过该指标把
  dq 链路改 fp32。
- **判据口径三项（D 类失败疑似判据 artifact 时先查判据再动 kernel）**：
  ① gauge symmetry——把生产标杆跑当前判据，标杆同样 FAIL ⇒ 判据口径问题
  而非 kernel bug；② 超阈分布形态——呈 dV/dQ/dK 梯度嵌套且超阈轴为相对误差
  而非绝对误差时，是纯相对判据对小值/相消桶零余量所致；③ golden 数据生成方式——
  CPU RNG 极端值可使 fp32 参考自身 inf，先用 NPU 生成数据复现再定性。
- 16-bit 输入通路（16-bit mma 输入 + fp32 累加）允许启用，但红线是：保留
  fp32 workspace 累加、fp32 输入仍走 fp32 入口、接口/张量/dtype 全不变；
  验收判据须带 atol 混合判据（纯相对判据下 16-bit 容易 FAIL）。
