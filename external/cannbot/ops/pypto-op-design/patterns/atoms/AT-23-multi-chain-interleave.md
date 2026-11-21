---
type: pattern/atom
title: Unroll-and-Jam 多链交错（外层无依赖轴并行化）
description: 将外层无依赖轴（head/batch/seq-group）的 G 个迭代展开后并入内层循环体，各链持独立累积器，把单一跨 loop 串行依赖改写为 G 条可重叠调度的并行链。
tags:
- loop-interleave
flow_pattern: []
examples:
- FlashAttentionMHA
---

## AT-23: Unroll-and-Jam 多链交错（外层无依赖轴并行化）

**描述**: 内层循环含跨 loop 状态（如 online softmax 的 running max/sum/加权和）且相邻迭代串行依赖，外层存在无依赖轴（head/batch/seq-group）时，将外层 G 个迭代展开并入内层循环体，各链持独立累积器，把「1 条串行链 × N 迭代」改写为「G 条并行链 × N 迭代」，供调度器重叠执行。

**CV 排布**: 不改变 C1-V1-C2-V2 各阶段计算流，属循环结构变换，作用于 SK-01 等骨架的外层无依赖轴。

**适用条件**（同时满足）:
1. 内层循环含跨 loop 状态，相邻迭代串行依赖；
2. 泳道 Wait Schedule / Wait Predecessor 占比 > 50% E2E，AIV/AIC 利用率偏低；
3. 外层存在无跨迭代依赖的轴可供展开。

**计算流**（HEAD_GROUP=G）:
```python
HEAD_GROUP = 4                          # 每 episode 交错的链数
head_groups = NUM_HEADS // HEAD_GROUP

for b in pypto.loop(batch):
    ...
    for hg in pypto.loop(head_groups):                      # 外层无依赖轴，展开并入内层
        h_off0 = (hg * HEAD_GROUP + 0) * HEAD_DIM
        h_off1 = (hg * HEAD_GROUP + 1) * HEAD_DIM
        ...
        for qt in pypto.loop(q_tiles):
            oi0 = pypto.tensor([Q_TILE, HEAD_DIM], pypto.DT_FP32, "oi0")   # 各链独立累积器
            mi0 = pypto.tensor([Q_TILE, 1], pypto.DT_FP32, "mi0")
            li0 = pypto.tensor([Q_TILE, 1], pypto.DT_FP32, "li0")
            oi1 = pypto.tensor(...); mi1 = ...; li1 = ...                  # head 1..G-1 同理

            for kt in pypto.loop(k_tiles, unroll_list=[2]):   # 候选 2 / 1，初始设计只选一个值
                # C1: QK^T（G 链独立）
                S0 = pypto.matmul(q_t0, k_t0, pypto.DT_FP32, b_trans=True)
                S1 = pypto.matmul(q_t1, k_t1, pypto.DT_FP32, b_trans=True)
                ...
                # V1: online softmax 块内统计（G 链独立）
                m0_j, P0_bf16, l0_j = v1(S0)
                ...
                # C2: P@V（G 链独立）
                O0_j = pypto.matmul(P0_bf16, v_t0, pypto.DT_FP32)
                ...
                # V2: online 合并（G 链独立 update，一链等待时其余 G-1 链 compute 可执行）
                if pypto.is_loop_begin(kt):
                    oi0[:] = O0_j; li0[:] = l0_j; mi0[:] = m0_j
                    ...
                else:
                    pypto_v2_merge(oi0, mi0, li0, m0_j, l0_j, O0_j)         # 不含 is_loop_begin/end
                    pypto_v2_merge(oi1, mi1, li1, m1_j, l1_j, O1_j)
                    ...
                if pypto.is_loop_end(kt):
                    pypto_stage_out(oi0, li0, mi0, o, l, m, ..., h_off0, h0, valid_q)
                    ...
```

**约束**（必须遵守）:
- 累积器用 `pypto.tensor` 纯声明，禁止 `pypto.full` 显式物化（G 份 `[Q_TILE, D]` FP32 会溢出 UB）。
- merge 逻辑可抽 Layer H 子函数，但不得含 `pypto.is_loop_begin/end`（谓词须留 JIT body 内）。
- G 收益递减，受核数与 UB 约束，按实测定 G。

**禁用反模式**:
- 内层无跨 loop 状态时套用（纯并行内层用 `unroll_list` 即可）。
- 外层轴存在跨迭代依赖时交错（破坏语义）。
- `pypto.full` 显式物化多份累积器。

**实例化参数**:
| 参数 | 说明 | 典型值 |
|------|------|--------|
| `HEAD_GROUP`（G） | 每 episode 交错的链数 | 2 / 4 |
| 累积器 dtype | 各链跨 loop 状态 dtype | DT_FP32 |
| `unroll_list` | 内层展开候选集合，与本模板正交；初始设计只选一个值 | 2 / 1 |

**实测收益**: 910 FlashAttentionMHA（[b=2,n=8,s=4096,d=128]）：HEAD_GROUP=2 -12.4%（3009.68→2635.26 us）、HEAD_GROUP=4 累计 -14.5%。

**使用算子**: FlashAttentionMHA；可推广至「外层无依赖轴 + 内层跨 loop 状态」的 attention/recurrent 骨架。

**与其他模板的关系**:
- 是 AT-01（Online Softmax）、AT-14（Recurrent Update）的多实例并行化包装，单链逻辑不变。
- 与 AT-21（sg_set_scope 合图）、`unroll_list` 正交叠加，三者分别解决串行链的落地 / 内层流水 / 多链填隙。



---
