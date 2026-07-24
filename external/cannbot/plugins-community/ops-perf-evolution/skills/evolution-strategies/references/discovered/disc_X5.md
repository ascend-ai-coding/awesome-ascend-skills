---
id: X5
origin: discovered
discovered_round: 2
discovered_from: round_2/parallel_2
base_speedup: 3.527x
---

# Strategy X5: GQA-Decode GEMV->GEMM (M=N_q)

## 核心思路
对 GQA decode/MTP（N_kv=1，一个 token 的全部 N_q 个 query head 共享同一个 KV head），把 decode 注意力的 M 维取为 **N_q（128/64）而非 S**。于是一块 = 一个 (b,s) token、grid=B·S、M=N_q，QK^T=`Q[N_q,D_qk]@K^T[D_qk,S_kv]` 与 PV=`P[N_q,S_kv]@V[S_kv,d_nope]` 都成为稠密 cube GEMM（M≥64），替代 vec 路径里 N_q 条独立的归约型 GEMV。这是把 decode 从 AIV/GEMV 换到 AIC/GEMM 的**计算域切换**，由 N_kv=1 不变量保证 M=N_q 足够大以喂饱 cube。

## 适用场景
- 算子：GQA/MLA decode 或 MTP（S∈{1,2}）且 N_kv=1（或 N_kv≪N_q）。
- 瓶颈：decode 受 vec GEMV 归约吞吐制约（N_q 条长度 S_kv 的归约既欠用向量单元又欠用 HBM），cube 16 TFLOPS fp16 空闲。
- 不适用：prefill（S 大，已是 GEMM）、MHA（N_kv=N_q，M=N_q 但 K/V 不能 head 共享流一次）、N_kv>1 的 GQA（K/V 需按 KV head 重复，退化回多 GEMM）。

## 实现要点
- 新增 sibling decode kernel（C-V 1AIC:2AIV FFTS），prefill 路径字节不变；plugin 在 prefill 前加 `use_decode_mix=(S∈{1,2} && 对齐)` 分发。
- AIV：按 head marshal Q（layout-aware），透明 fp32 行 softmax（无 max-shift，score 为 O(1)），因果尾 mask 跨 head 一致（mask 仅依赖 s）。
- AIC：QK^T 的 K 每 token 流一次（N-chunk NC=256）；PV 做 K-tile（KC=256）+ N-tile（NV=128，使 cL0[N_q,128]=64KB≤L0C 128KB）；V 用 proven 子列 nd2nz 流一次，cube fixpipe 直接落 y（无 o_scr 往返）。
- 3 个 one-shot FFTS flag（MARSHAL/CS/VP）；EVENT_ID 复用 prefill 验证过的 0..7。
- Trade-off：AIC/AIV 串行（FFTS）、Q 每 NC-chunk 重载、半核轮流空闲、L0C 限 NV=128（P 重流 ndv=4×），但 cube 吞吐 + 内存最优流式主导 → decode 11–33×。

## 与既有策略的关系
- X4 是最佳 VEC 骨架；P*/D*/A* 卡在单一计算域内做 tiling/buffer/vectorize。X5 改变 decode 的**计算域**（AIV→AIC、GEMV→GEMM），是结构性 reformulation，不是任何单卡的组合。

## 来源
自动发现于第 2 轮进化，算子 mla，round_2/parallel_2（open_exploration），geomean 3.527x（父 x0 best-vec 0.5745x），decode 11–33x，20/20 valid。
