---
id: X4
origin: discovered
discovered_round: 1
discovered_from: round_1/parallel_3
base_speedup: 0.5745x
---

# Strategy X4: Concat-Fused Strided DMA + Two-Level Skv Reduction

## 核心思路
针对 decode/MTP attention（单行/小批量 Q、长 Skv）的 **DMA-launch-latency-bound** 瓶颈（非 BW）：
1. **Concat 折叠进 K-tile DMA**：一个 tile 用两条 MTE2（nope-tile / rope-tile）以 strided dst 交错写入连续 `[T,D_qk]` UB（ND2NZ 式 packed load）。N_kv=1 时两个 burst 背靠背=一次物理传输，host 侧 `torch.cat` / 逐 key 双 DMA 消失。
2. **Skv sweep 两级向量归约**：`vmul(q,k) -> vcadd -> vcadd` 把每个 dot 直接写进 `score[r]`，去掉逐 key 标量部分和、逐 key DMA、逐 key sync。
3. **单遍 FlashAttention-2 online softmax**：同 tile 的 V 重叠加载并在同遍累加到 fp32 `o_f`，4 遍 -> 1 遍；标量 exp 用 1-lane `vexp`（不依赖 libm）。

本质是把 decode 的 per-key DMA/sync 从 O(Skv) 降到 O(tiles)，直击 launch-latency 项；fp32 online-softmax 与 2-pass 代数等价，精度不损。

## 适用场景
- 算子：decode/MTP 阶段 attention（单行/小批量 Q、长 KV 序列、GQA N_kv=1 尤佳；concat 型 Q/K，如 MLA 的 nope+rope）。
- 瓶颈：DMA launch latency / per-key sync 主导（大量短 MTE2 + 高同步频率），而非 BW 或 compute。
- 硬件：Ascend 910B 系列 vector 核（dav-c220-vec），UB 放得下 K-tile+V-tile+fp32 状态（tile<=64 @fp16 约 150KB / 192KB）。

## 实现要点
- K-tile 用两条带 strided dst 的 MTE2 拼 nope+rope，dst stride 按 D_qk 交错，保证 32B 对齐。
- dot 用 `vmul->vcadd->vcadd` 两级归约直接落 `score[r]`，禁用逐 key 标量累加。
- online softmax：fp32 m/l/o 驻留 UB 跨 tile rescale（`o_f *= exp(m_old - m_new)`），exp 用 1-lane `vexp`；同 tile V 与 score 重叠加载。
- 多 repeat 的 `vmul/vcadd/vaxpy` 间存在 RAW 依赖，`pipe_barrier(PIPE_V)` 不可省（R1 删 barrier 精度崩，已验证）。
- tile 不宜过大：64->80 因 K/V 流量主导 + 二次 vexp 开销反而退化 ~23%（R2 已验证）；tile=64 为甜区。

## 来源
自动发现于第 1 轮进化，算子 mla，round_1/parallel_3（open_exploration）：decode/MTP 0.11-0.27x -> 0.25-0.79x（per-case ~2.3-3.9x），geomean 0.5745x（20/20 valid）。
